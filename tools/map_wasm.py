#!/usr/bin/env python3
"""Identify what a minified WASM module's exports actually do.

``probe_wasm.py`` shows that the module loads. This one works out *which export
is the step function* and *where the car state lives*, by calling candidates and
diffing linear memory before and after.

Three improvements over the probe that matter:

**Better stubs.** The probe returns 0 from everything. That is wrong for
``emscripten_resize_heap``, where 0 means *allocation failed* -- so the module
retries, and one call balloons into tens of thousands. Here, ``(i32) -> i32``
imports return 1 (success) by default.

**Per-import attribution.** You see exactly which import a call hit, not just a
total. That is how you tell a clock read from a memcpy.

**Fuel metering.** Every call is bounded, so a wrong guess about an argument
cannot hang the process. Calling unknown functions with invented arguments is
otherwise a good way to hit an infinite loop.

Usage::

    python tools/map_wasm.py polytrack_physics.wasm
    python tools/map_wasm.py polytrack_physics.wasm --fuel 500000000

Safe: wasmtime sandboxes the module with no file or network access, traps are
caught, and execution is fuel-bounded.
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

try:
    from wasmtime import (
        Config, Engine, Func, FuncType, Instance, Memory, Module, Store, Trap,
    )
except ImportError:
    sys.exit("wasmtime not installed. Run: pip install wasmtime")

CALLS: Counter[str] = Counter()

#: Plausible neutral arguments per WASM value type, used when poking unknown
#: functions. Zeros for ints (usually a null pointer or "no flags"), and a
#: sixtieth of a second for f32, since that is what a physics dt would be.
SAMPLE_ARGS = {"i32": 0, "i64": 0, "f32": 1.0 / 60.0, "f64": 1.0 / 60.0}


def signature_string(func_type: FuncType) -> str:
    params = ", ".join(str(p) for p in func_type.params)
    results = ", ".join(str(r) for r in func_type.results)
    return f"({params}) -> {results if results else '()'}"


def make_stub(store: Store, name: str, func_type: FuncType) -> Func:
    """A recording stub with sensible return values.

    ``(i32) -> i32`` returns 1 rather than 0: if this import is
    ``emscripten_resize_heap``, 0 reads as failure and sends the module into a
    retry loop that looks like a module bug but is really a stub bug.
    """
    params = [str(p) for p in func_type.params]
    results = [str(r) for r in func_type.results]

    def stub(*args):
        CALLS[name] += 1
        if not results:
            return None
        values = []
        for result in results:
            if result == "i32" and params == ["i32"]:
                values.append(1)  # success, for resize_heap-shaped imports
            elif result in ("f32", "f64"):
                values.append(0.0)  # fixed clock: never real time
            else:
                values.append(0)
        return values[0] if len(values) == 1 else values

    return Func(store, func_type, stub)


def snapshot(memory: Memory, store: Store) -> bytes:
    return bytes(memory.read(store))


def diff_regions(before: bytes, after: bytes, gap: int = 32) -> list[tuple[int, int]]:
    """Byte ranges that changed, merging runs separated by less than ``gap``."""
    if len(before) != len(after):
        return [(0, len(after))]
    changed = [i for i in range(min(len(before), len(after))) if before[i] != after[i]]
    if not changed:
        return []
    regions: list[list[int]] = [[changed[0], changed[0] + 1]]
    for index in changed[1:]:
        if index - regions[-1][1] <= gap:
            regions[-1][1] = index + 1
        else:
            regions.append([index, index + 1])
    return [(start, end) for start, end in regions]


def interpret(data: bytes, start: int, end: int, limit: int = 6) -> list[str]:
    """Read a changed region as floats, to spot physics-looking values."""
    notes: list[str] = []
    for offset in range(start - (start % 4), min(end, start + limit * 8), 4):
        if offset + 8 <= len(data):
            f64 = struct.unpack_from("<d", data, offset)[0]
            if 1e-6 < abs(f64) < 1e6:
                notes.append(f"    +{offset}  f64 {f64:.6g}")
                continue
        if offset + 4 <= len(data):
            f32 = struct.unpack_from("<f", data, offset)[0]
            if 1e-6 < abs(f32) < 1e6:
                notes.append(f"    +{offset}  f32 {f32:.6g}")
    return notes[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    parser.add_argument("--fuel", type=int, default=200_000_000,
                        help="execution budget per call")
    args = parser.parse_args()

    if not args.wasm.exists():
        print(f"no such file: {args.wasm}")
        return 1

    config = Config()
    config.consume_fuel = True
    store = Store(Engine(config))
    module = Module(store.engine, args.wasm.read_bytes())

    externs = []
    import_names = []
    for imp in module.imports:
        full = f"{imp.module}.{imp.name}"
        import_names.append(full)
        externs.append(make_stub(store, full, imp.type))

    store.set_fuel(args.fuel)
    instance = Instance(store, module, externs)
    exports = instance.exports(store)

    memory = next(
        (v for v in exports._extern_map.values() if isinstance(v, Memory)), None
    )
    if memory is None:
        print("No exported memory; cannot diff state.")
        return 1

    functions = [
        (name, item)
        for name, item in exports._extern_map.items()
        if isinstance(item, Func)
    ]

    print(f"\n{'=' * 74}")
    print(f"  {args.wasm.name}")
    print(f"  memory: {memory.data_len(store):,} bytes "
          f"({memory.size(store)} pages)")
    print(f"{'=' * 74}")

    def call(name, func, arguments, label=""):
        """Call one export, bounded by fuel, and report memory changes."""
        CALLS.clear()
        store.set_fuel(args.fuel)
        before = snapshot(memory, store)
        try:
            result = func(store, *arguments)
        except Trap as exc:
            print(f"  {name}{arguments}  trapped: {str(exc).splitlines()[0]}")
            return None
        except Exception as exc:
            print(f"  {name}{arguments}  error: {str(exc)[:70]}")
            return None
        after = snapshot(memory, store)
        regions = diff_regions(before, after)
        total = sum(end - start for start, end in regions)

        summary = f"  {name}{tuple(arguments)}"
        if label:
            summary += f"  [{label}]"
        print(f"{summary}  ->  {result}")
        print(f"      {total:,} bytes changed in {len(regions)} region(s)")
        if CALLS:
            hits = ", ".join(f"{k}x{v}" for k, v in CALLS.most_common(4))
            print(f"      imports: {hits}")
        for start, end in regions[:3]:
            print(f"      region {start:,}..{end:,} ({end - start} bytes)")
            for line in interpret(after, start, end):
                print(line)
        return after

    # 1. Run global constructors first. Anything called before this may see
    #    uninitialised state and behave nonsensically.
    print("\n1. GLOBAL CONSTRUCTORS")
    print("-" * 74)
    for name, func in functions:
        ftype = func.type(store)
        if not list(ftype.params) and not list(ftype.results):
            call(name, func, [], "likely __wasm_call_ctors")
            break

    # 2. Poke every export with neutral arguments and watch memory.
    print("\n2. EXPORTS, WITH MEMORY DIFF")
    print("-" * 74)
    for name, func in functions:
        ftype = func.type(store)
        params = [str(p) for p in ftype.params]
        if len(params) > 8:
            print(f"  {name}: {len(params)} params -- skipped, too many to guess")
            continue
        arguments = [SAMPLE_ARGS[p] for p in params]
        call(name, func, arguments)

    # 3. The question that decides whether any of this can be reproduced.
    print("\n3. DETERMINISM CHECK")
    print("-" * 74)
    clock = [
        n for n, imp in zip(import_names, module.imports)
        if isinstance(imp.type, FuncType) and signature_string(imp.type) == "() -> f64"
    ]
    if clock:
        print(f"  Clock-shaped import: {clock[0]}")
        print("  Watch the 'imports:' lines above. If it appears on the export")
        print("  you believe is the step function, the physics reads a clock and")
        print("  the stub must feed it a frame-derived value, never real time.")
    else:
        print("  No clock-shaped import.")

    print("\n  Best step-function candidates take an f32 (a dt) plus one or two")
    print("  i32s (an input bitmask), return void, and change a small, stable")
    print("  region of memory each call.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
