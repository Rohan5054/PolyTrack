#!/usr/bin/env python3
"""Inspect a WebAssembly module and report its interface.

**Run this on your own machine, on a module you obtained yourself.** It reads a
local ``.wasm`` file; it does not download anything.

This is the first concrete step of Stage 0. Before you can write a physics
binding you need to know what the module actually exposes: which functions can
be called, what memory it wants, and -- critically -- what *imports* it expects
the host to provide. Emscripten-generated modules typically demand a pile of JS
glue (syscalls, ``emscripten_*`` helpers, math shims). Every import you cannot
satisfy from Python is a reason to host the module in Node instead, which is
the single biggest architecture decision in this project.

Usage::

    python tools/inspect_wasm.py path/to/physics.wasm
    python tools/inspect_wasm.py path/to/physics.wasm --json report.json

What to look for in the output:

* **Exports containing "step", "tick", "update", "simulate"** -- the physics
  step entry point.
* **Exports containing "init", "reset", "spawn"** -- setup and episode reset.
* **An exported ``memory``** -- car state almost certainly lives in linear
  memory rather than being returned; you will read it out by offset.
* **Import count.** Near zero means Python hosting via wasmtime is realistic.
  Dozens of ``env.*`` / ``wasi_*`` imports means budget for a Node host process.
* **Threading imports** (``memory`` marked shared, ``__wait``/``__notify``)
  -- PolyTrack's car simulation is multithreaded, and multithreading is the
  hardest case for deterministic replay. Plan to run it single-threaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from wasmtime import Engine, Module
except ImportError:  # pragma: no cover
    sys.exit("wasmtime not installed. Run: pip install wasmtime")

# Substrings that hint at what a function does. Ordered by how strongly they
# suggest the physics step entry point.
INTERESTING = {
    "step": "physics step candidate",
    "tick": "physics step candidate",
    "simulate": "physics step candidate",
    "update": "physics step candidate",
    "advance": "physics step candidate",
    "frame": "frame counter or step",
    "reset": "episode reset candidate",
    "init": "setup candidate",
    "spawn": "setup candidate",
    "input": "input injection candidate",
    "key": "input injection candidate",
    "control": "input injection candidate",
    "state": "state accessor candidate",
    "car": "state accessor candidate",
    "position": "state accessor candidate",
    "velocity": "state accessor candidate",
    "checkpoint": "progress accessor candidate",
    "finish": "progress accessor candidate",
    "track": "track loading candidate",
    "load": "track loading candidate",
    "save": "state snapshot candidate",
    "restore": "state snapshot candidate",
    "seed": "DETERMINISM RISK: seeded randomness",
    "random": "DETERMINISM RISK: randomness",
    "rand": "DETERMINISM RISK: randomness",
    "time": "DETERMINISM RISK: wall-clock dependency",
    "now": "DETERMINISM RISK: wall-clock dependency",
    "date": "DETERMINISM RISK: wall-clock dependency",
}


def classify(name: str) -> list[str]:
    lowered = name.lower()
    return sorted({note for key, note in INTERESTING.items() if key in lowered})


def describe_type(item) -> str:
    """Best-effort rendering of an import/export type across wasmtime versions."""
    try:
        params = ", ".join(str(p) for p in item.params)
        results = ", ".join(str(r) for r in item.results)
        return f"({params}) -> ({results})"
    except AttributeError:
        return type(item).__name__


def inspect(path: Path) -> dict:
    engine = Engine()
    data = path.read_bytes()
    module = Module(engine, data)

    exports = []
    for export in module.exports:
        entry = {
            "name": export.name,
            "kind": type(export.type).__name__,
            "signature": describe_type(export.type),
            "notes": classify(export.name),
        }
        exports.append(entry)

    imports = []
    for imp in module.imports:
        name = imp.name or ""
        entry = {
            "module": imp.module,
            "name": name,
            "kind": type(imp.type).__name__,
            "signature": describe_type(imp.type),
        }
        imports.append(entry)

    return {
        "file": str(path),
        "size_bytes": len(data),
        "n_exports": len(exports),
        "n_imports": len(imports),
        "exports": exports,
        "imports": imports,
    }


def report(info: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {info['file']}  ({info['size_bytes']:,} bytes)")
    print(f"  {info['n_exports']} exports, {info['n_imports']} imports")
    print(f"{'=' * 72}\n")

    flagged = [e for e in info["exports"] if e["notes"]]
    if flagged:
        print("EXPORTS OF INTEREST")
        print("-" * 72)
        for entry in flagged:
            print(f"  {entry['name']}")
            print(f"      {entry['signature']}")
            for note in entry["notes"]:
                marker = "  !!" if note.startswith("DETERMINISM") else "    "
                print(f"    {marker} {note}")
        print()

    memories = [e for e in info["exports"] if "Memory" in e["kind"]]
    if memories:
        print("EXPORTED MEMORY (car state almost certainly lives here)")
        print("-" * 72)
        for entry in memories:
            print(f"  {entry['name']}: {entry['signature']}")
        print()
    else:
        print("NOTE: no exported memory found. If state is not returned by value,")
        print("      the module may keep it behind accessor functions instead.\n")

    if info["imports"]:
        print("IMPORTS THE HOST MUST PROVIDE")
        print("-" * 72)
        by_module = Counter(i["module"] for i in info["imports"])
        for mod, count in by_module.most_common():
            print(f"  {mod}: {count}")
        print()
        print("  First 20:")
        for entry in info["imports"][:20]:
            print(f"    {entry['module']}.{entry['name']}  {entry['signature']}")
        if len(info["imports"]) > 20:
            print(f"    ... and {len(info['imports']) - 20} more")
        print()

    print("VERDICT")
    print("-" * 72)
    n_imports = info["n_imports"]
    if n_imports == 0:
        print("  Self-contained: host it directly from Python via wasmtime.")
        print("  This is the best case for throughput and for the RL stack.")
    elif n_imports <= 15:
        print(f"  {n_imports} imports: Python hosting is probably viable.")
        print("  Stub each import and see which are actually called at runtime;")
        print("  many Emscripten imports are never hit on the happy path.")
    else:
        print(f"  {n_imports} imports: reimplementing this glue in Python is a")
        print("  large job. Prefer hosting the module in Node.js (which already")
        print("  has the generated glue) and talking to Python over a pipe or")
        print("  shared memory. Batch many steps per round trip to amortise IPC.")

    risky = [
        e for e in info["exports"]
        if any(n.startswith("DETERMINISM") for n in e["notes"])
    ]
    if risky:
        print()
        print("  Determinism flags raised on: " + ", ".join(e["name"] for e in risky))
        print("  Check whether these are seeded from wall-clock. Anything that is")
        print("  must be pinned before results are reproducible.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path, help="path to a local .wasm file")
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    args = parser.parse_args()

    if not args.wasm.exists():
        return f"no such file: {args.wasm}"  # type: ignore[return-value]

    info = inspect(args.wasm)
    report(info)

    if args.json:
        args.json.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"Full report written to {args.json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
