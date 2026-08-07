#!/usr/bin/env python3
"""Instantiate a WASM module with stubbed imports and watch what it does.

``inspect_wasm.py`` reads the module statically. This one *runs* it: every import
is replaced with a stub that records each call, so you learn which imports the
module actually touches, in what order, and how often. On a minified module --
where every export is a single letter and static analysis tells you almost
nothing -- this is how you find the step function.

Two questions it answers, both blocking for Stage 0:

1. **Which imports are real dependencies?** Emscripten declares many that are
   never hit on the happy path. Every one you can leave as a no-op stub is one
   less reason to host the module in Node.
2. **Does anything read a clock?** An import with signature ``() -> f64`` is
   almost certainly ``emscripten_date_now`` or ``emscripten_get_now``. If the
   physics calls it while stepping, runs will not reproduce and the whole
   project is built on sand. The stub here returns a **fixed** value and counts
   the calls, so you can see the dependency instead of inheriting it silently.

Usage::

    python tools/probe_wasm.py polytrack_physics.wasm
    python tools/probe_wasm.py polytrack_physics.wasm --call-exports

``--call-exports`` additionally tries every zero-argument export to see which
ones run cleanly. It is safe: the module is sandboxed by wasmtime, has no file
or network access, and traps are caught.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from wasmtime import Func, FuncType, Instance, Module, Store, Trap, ValType
except ImportError:
    sys.exit("wasmtime not installed. Run: pip install wasmtime")

#: Signature -> what an Emscripten import of that shape usually is.
SIGNATURE_HINTS = {
    "() -> f64": "CLOCK (emscripten_date_now / get_now) -- DETERMINISM RISK",
    "(i32) -> i32": "memory growth (emscripten_resize_heap) or sbrk",
    "(i32, i32, i32, i32) -> i32": "syscall (fd_write, openat, ...)",
    "() -> ()": "abort / trap handler",
    "(i32) -> ()": "exit, or a logging callback",
    "(i32, i32, i32) -> ()": "memcpy-like, or a structured callback",
}

CALLS: Counter[str] = Counter()


def signature_string(func_type: FuncType) -> str:
    params = ", ".join(str(p) for p in func_type.params)
    results = ", ".join(str(r) for r in func_type.results)
    return f"({params}) -> {results if results else '()'}"


def make_stub(store: Store, name: str, func_type: FuncType) -> Func:
    """Build a recording stub that returns deterministic zeros.

    Returning a constant matters as much as recording the call: a clock stub
    that returned real time would let non-determinism in through the back door
    while looking like it worked.
    """
    results = list(func_type.results)

    def stub(*args):
        CALLS[name] += 1
        if not results:
            return None
        zeros = [0.0 if "f" in str(r) else 0 for r in results]
        return zeros[0] if len(zeros) == 1 else zeros

    return Func(store, func_type, stub)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    parser.add_argument("--call-exports", action="store_true",
                        help="try calling zero-argument exports")
    args = parser.parse_args()

    if not args.wasm.exists():
        print(f"no such file: {args.wasm}")
        return 1

    store = Store()
    module = Module(store.engine, args.wasm.read_bytes())

    # -- imports -------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  {args.wasm.name}")
    print(f"{'=' * 70}\n")
    print("IMPORTS (stubbed)")
    print("-" * 70)

    externs = []
    import_names = []
    for imp in module.imports:
        full = f"{imp.module}.{imp.name}"
        import_names.append(full)
        func_type = imp.type
        if not isinstance(func_type, FuncType):
            print(f"  {full}: {type(func_type).__name__} -- NOT a function, "
                  "needs a real object (memory/table/global)")
            externs.append(None)
            continue
        sig = signature_string(func_type)
        hint = SIGNATURE_HINTS.get(sig, "")
        marker = "  !!" if "DETERMINISM" in hint else "    "
        print(f"  {full:<10} {sig:<32}{marker}{hint}")
        externs.append(make_stub(store, full, func_type))

    if any(e is None for e in externs):
        print("\nSome imports are not functions. Provide them explicitly before")
        print("instantiating -- see the wasmtime docs on Memory and Table.")
        return 1

    # -- instantiate ---------------------------------------------------
    print("\nINSTANTIATING")
    print("-" * 70)
    try:
        instance = Instance(store, module, externs)
        print("  OK -- the module loads with stubs alone.")
        print("  Python hosting is viable; you do not need Node.")
    except Trap as exc:
        print(f"  TRAPPED during instantiation: {exc}")
        print("  A stub returning 0 was probably not acceptable. Check which")
        print("  import was called last in the tally below.")
        return 1
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return 1

    # -- exports -------------------------------------------------------
    print("\nEXPORTS")
    print("-" * 70)
    exports = instance.exports(store)
    functions: list[tuple[str, Func]] = []
    for name, item in exports._extern_map.items():
        if isinstance(item, Func):
            sig = signature_string(item.type(store))
            functions.append((name, item))
            print(f"  {name:<8} func  {sig}")
        else:
            print(f"  {name:<8} {type(item).__name__}")

    if CALLS:
        print("\n  imports called during instantiation:")
        for name, count in CALLS.most_common():
            print(f"    {name}: {count}")

    # -- optionally poke the exports -----------------------------------
    if args.call_exports:
        print("\nCALLING ZERO-ARGUMENT EXPORTS")
        print("-" * 70)
        for name, func in functions:
            func_type = func.type(store)
            if len(list(func_type.params)) != 0:
                continue
            before = sum(CALLS.values())
            try:
                result = func(store)
                touched = sum(CALLS.values()) - before
                extra = f"  (triggered {touched} import calls)" if touched else ""
                print(f"  {name}()  ->  {result}{extra}")
            except Trap as exc:
                print(f"  {name}()  ->  trapped: {str(exc).splitlines()[0]}")
            except Exception as exc:
                print(f"  {name}()  ->  error: {exc}")

    # -- what to do next -----------------------------------------------
    print("\nWHAT TO LOOK FOR")
    print("-" * 70)
    clocks = [n for n, imp in zip(import_names, module.imports)
              if isinstance(imp.type, FuncType)
              and signature_string(imp.type) == "() -> f64"]
    if clocks:
        print(f"  Clock-shaped import: {', '.join(clocks)}")
        print("  Stub this with a FIXED value (or a frame counter) in the real")
        print("  backend. If it is called during stepping, real time would leak")
        print("  into the physics and runs would not reproduce.")
    else:
        print("  No clock-shaped import. One less determinism risk.")

    print("\n  The step function is the export you call once per frame. Look for")
    print("  one taking a small number of i32 args (the input bitmask, maybe a")
    print("  dt) and returning void or a status i32. Find it by calling")
    print("  candidates and watching the exported memory change.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
