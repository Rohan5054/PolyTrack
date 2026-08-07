#!/usr/bin/env python3
"""Drive the step function and find where car state lives.

This is the tool that decides whether Stage 0 is achievable. It answers three
questions, in the order they matter:

1. **Is stepping deterministic?** Run the same inputs twice from a fresh
   instance and compare memory byte for byte. If this fails, nothing else in the
   project is worth building.
2. **Which argument is the input mask?** Step with different values and see
   which one changes the trajectory.
3. **Where are position and velocity?** Track every 4-byte offset across many
   frames and rank them by how *smoothly* they change. Position integrates
   velocity, so it moves continuously; counters jump, flags toggle, and noise is
   erratic. Smoothness separates them.

Every experiment gets a **fresh instance**. That is not fussiness: calling
``stackRestore(0)`` or similar in an earlier experiment corrupts the stack
pointer and every later call silently misbehaves, which is exactly how a
mapping session produces confident nonsense.

Usage::

    python tools/step_probe.py polytrack_physics.wasm
    python tools/step_probe.py polytrack_physics.wasm --step n --frames 180
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import Counter
from pathlib import Path

try:
    from wasmtime import Config, Engine, Func, FuncType, Instance, Memory, Module, Store, Trap
except ImportError:
    sys.exit("wasmtime not installed. Run: pip install wasmtime")

DT = 1.0 / 60.0
FUEL = 500_000_000


class Session:
    """One isolated instantiation. Never reuse across experiments."""

    def __init__(self, wasm_bytes: bytes, clock_value: float = 0.0) -> None:
        config = Config()
        config.consume_fuel = True
        self.store = Store(Engine(config))
        self.module = Module(self.store.engine, wasm_bytes)
        self.calls: Counter[str] = Counter()
        self.clock_value = clock_value

        externs = []
        for imp in self.module.imports:
            externs.append(self._stub(f"{imp.module}.{imp.name}", imp.type))
        self.store.set_fuel(FUEL)
        self.instance = Instance(self.store, self.module, externs)
        self.exports = self.instance.exports(self.store)
        self.memory: Memory = next(
            v for v in self.exports._extern_map.values() if isinstance(v, Memory)
        )

    def _stub(self, name: str, func_type: FuncType) -> Func:
        params = [str(p) for p in func_type.params]
        results = [str(r) for r in func_type.results]

        def stub(*args):
            self.calls[name] += 1
            if not results:
                return None
            values = []
            for result in results:
                if result in ("f32", "f64"):
                    # A frozen clock. Never real time -- that is the whole point.
                    values.append(self.clock_value)
                elif result == "i32" and params == ["i32"]:
                    values.append(1)  # resize_heap-shaped: report success
                else:
                    values.append(0)
            return values[0] if len(values) == 1 else values

        return Func(self.store, func_type, stub)

    def func(self, name: str) -> Func:
        return self.exports._extern_map[name]

    def call(self, name: str, *args):
        self.store.set_fuel(FUEL)
        return self.func(name)(self.store, *args)

    def run_ctors(self) -> None:
        """Call the zero-arg, zero-result export -- the global constructors."""
        for name, item in self.exports._extern_map.items():
            if isinstance(item, Func):
                ftype = item.type(self.store)
                if not list(ftype.params) and not list(ftype.results):
                    self.call(name)
                    return

    def snapshot(self) -> bytes:
        return bytes(self.memory.read(self.store))


def step_many(session: Session, step: str, frames: int, mask: int, third: int = 0):
    """Step ``frames`` times, returning a memory snapshot after each."""
    shots = []
    for _ in range(frames):
        session.call(step, DT, mask, third)
        shots.append(session.snapshot())
    return shots


def smoothness(series: list[float]) -> float:
    """How continuously a value evolves. Lower is smoother.

    Mean absolute second difference, normalised by the value's range. Position
    and velocity integrate, so their second difference is small; counters,
    flags, and hash-like values jump around.
    """
    if len(series) < 3:
        return float("inf")
    spread = max(series) - min(series)
    if spread == 0:
        return float("inf")  # constant: not state we care about
    second = [
        abs(series[i + 2] - 2 * series[i + 1] + series[i])
        for i in range(len(series) - 2)
    ]
    return (sum(second) / len(second)) / spread


def find_state(shots: list[bytes], limit: int = 12) -> list[tuple[int, str, float, list[float]]]:
    """Rank 4-byte offsets by how smoothly they evolve across frames."""
    first, last = shots[0], shots[-1]
    changed = [
        i for i in range(0, min(len(first), len(last)) - 8, 4)
        if first[i : i + 4] != last[i : i + 4]
    ]

    scored = []
    for offset in changed:
        for fmt, kind in (("<f", "f32"), ("<d", "f64")):
            size = struct.calcsize(fmt)
            try:
                series = [struct.unpack_from(fmt, s, offset)[0] for s in shots]
            except struct.error:
                continue
            if not all(abs(v) < 1e12 and v == v for v in series):
                continue  # reject NaN/inf: reinterpreted garbage
            if len(set(series)) < 3:
                continue
            score = smoothness(series)
            if score < 0.35:  # empirically, real physics sits well below this
                scored.append((offset, kind, score, series))
            break

    scored.sort(key=lambda entry: entry[2])
    # Suppress only genuine overlaps: a single f64 is readable at several
    # offsets, but two adjacent f32s are two real fields -- which is exactly
    # what a position/velocity struct looks like, so this radius must be the
    # value's own width and no wider.
    kept: list[tuple[int, str, float, list[float]]] = []
    for entry in scored:
        width = 4 if entry[1] == "f32" else 8
        if any(abs(entry[0] - k[0]) < width for k in kept):
            continue
        kept.append(entry)
        if len(kept) >= limit:
            break
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wasm", type=Path)
    parser.add_argument("--step", default="n", help="export to use as the step function")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    if not args.wasm.exists():
        print(f"no such file: {args.wasm}")
        return 1
    blob = args.wasm.read_bytes()

    print(f"\n{'=' * 74}")
    print(f"  {args.wasm.name} -- stepping '{args.step}' for {args.frames} frames")
    print(f"{'=' * 74}")

    # -- 1. determinism ------------------------------------------------
    print("\n1. DETERMINISM  (the gate)")
    print("-" * 74)
    try:
        a = Session(blob); a.run_ctors(); shots_a = step_many(a, args.step, args.frames, 1)
        b = Session(blob); b.run_ctors(); shots_b = step_many(b, args.step, args.frames, 1)
    except Trap as exc:
        print(f"  trapped: {str(exc).splitlines()[0]}")
        print(f"  '{args.step}' may not be the step function, or needs setup first.")
        return 1
    except KeyError:
        print(f"  no export named '{args.step}'. Use --step to pick another.")
        return 1

    if shots_a[-1] == shots_b[-1]:
        print(f"  PASS -- two fresh runs of {args.frames} frames produced")
        print("  byte-identical memory. The simulation is reproducible.")
    else:
        differing = sum(1 for x, y in zip(shots_a[-1], shots_b[-1]) if x != y)
        print(f"  FAIL -- {differing:,} bytes differ between identical runs.")
        print("  Something non-deterministic is leaking in. Check the import")
        print("  tally below before trusting any measured time.")
    if a.calls:
        print(f"  imports called during stepping: "
              f"{', '.join(f'{k}x{v}' for k, v in a.calls.most_common())}")
    else:
        print("  imports called during stepping: none")
        print("  No clock, no syscalls -- nothing external can perturb a run.")

    # -- 2. which argument is the input mask? --------------------------
    print("\n2. INPUT SENSITIVITY  (which argument drives the car?)")
    print("-" * 74)
    baseline = shots_a[-1]
    print("  baseline is arg2=1, arg3=0")
    for label, mask, third in (("arg2=2", 2, 0), ("arg2=4", 4, 0),
                               ("arg2=0", 0, 0), ("arg3=1", 0, 1)):
        try:
            session = Session(blob); session.run_ctors()
            shots = step_many(session, args.step, args.frames, mask, third)
            differing = sum(1 for x, y in zip(baseline, shots[-1]) if x != y)
            verdict = "changes trajectory" if differing else "no effect"
            print(f"  {label:<10} {differing:>7,} bytes differ from arg2=1   {verdict}")
        except Trap:
            print(f"  {label:<10} trapped")

    # -- 3. where is the car? ------------------------------------------
    print("\n3. STATE CANDIDATES  (smoothest-evolving offsets)")
    print("-" * 74)
    found = find_state(shots_a)
    if not found:
        print("  Nothing evolving smoothly. Either the car is not spawned yet")
        print("  (a track must be loaded first) or the step function is wrong.")
    else:
        print("  offset      type   smoothness   first -> last")
        for offset, kind, score, series in found:
            print(f"  {offset:>9,}  {kind}   {score:>9.4f}   "
                  f"{series[0]:+.4g} -> {series[-1]:+.4g}")
        print("\n  Lowest scores are the best position/velocity candidates.")
        print("  Confirm by stepping with throttle held and checking the value")
        print("  accelerates, then put the offsets into WasmLayout.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
