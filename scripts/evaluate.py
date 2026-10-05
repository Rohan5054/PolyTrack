#!/usr/bin/env python3
"""Evaluate a policy on a track library, against the scripted driver.

    python scripts/evaluate.py runs/gen1/agent.pt --library tracks_heldout.npz

The measurement the generalisation claim rests on. Run it on the library the
policy trained on and on a held-out one built from disjoint seeds; the gap
between them is the result.

Three things it does that a single greedy lap does not.

**Several attempts per track.** Greedy is *not* deterministic here: IQN samples
fresh quantile fractions on every forward pass, so twenty greedy laps are twenty
different laps. One lap is a sample, not a measurement, and the fastest of
twenty is how `arms/arm1-policy` was produced.

**A baseline on the same track.** `polytrack.pursuit` crawls, and a policy that
cannot beat a crawl has not learned to race. Reporting the policy alone leaves
no way to tell a good number from a bad one, and the scripted driver is the only
floor this project has.

**The same observation the policy trained on.** The environment is built with a
`RoadMap`, so the rays measure the real collision geometry. Evaluating without
one silently feeds the network a different vector from the one it learned on --
the reference-line fallback -- and the result would be meaningless.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--library", type=Path, default=Path("tracks_heldout.npz"))
    parser.add_argument("--attempts", type=int, default=5,
                        help="greedy laps per track; greedy is not deterministic")
    parser.add_argument("--limit-factor", type=float, default=4.0,
                        help="episode budget as a multiple of par")
    parser.add_argument("--baseline", action="store_true",
                        help="also drive each track with the scripted driver")
    parser.add_argument("--tracks", type=int, default=0,
                        help="evaluate only the first N tracks (0 = all)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from polytrack.priority import lift_background_qos

    lift_background_qos()

    import torch

    torch.set_num_threads(2)

    from polytrack.corridor import RoadMap
    from polytrack.env import PolyTrackEnv
    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.pursuit import PurePursuit

    if not args.checkpoint.exists():
        print(f"  no checkpoint at {args.checkpoint}")
        return 2
    agent = IQNAgent.load(args.checkpoint)
    library = TrackLibrary.load(args.library)
    assets = GeometryAssets.load()
    entries = list(library)
    if args.tracks:
        entries = entries[: args.tracks]

    print(f"\n  {args.checkpoint} at {agent.steps:,} gradient steps")
    print(f"  {args.library}: {len(entries)} tracks, {args.attempts} attempts each")
    restriction = agent.config.allowed_actions
    print(f"  actions: {'all 9' if restriction is None else restriction}\n")
    print(f"    {'track':<16}{'finished':>10}{'best':>10}{'par':>9}"
          f"{'best/par':>10}{'driver':>10}")

    finished_any = 0
    finish_rates: list[float] = []
    ratios: list[float] = []
    beat_driver = 0
    driver_compared = 0

    for entry in entries:
        code = entry.track_code()
        road = RoadMap.build(code, assets)
        times: list[int] = []
        for _ in range(args.attempts):
            env = PolyTrackEnv(
                track=entry.line,
                backend=WasmPhysics(code, assets=assets),
                road=road,
                par_frames=entry.par_frames,
                time_limit_ms=entry.par_frames * args.limit_factor,
            )
            observation, _ = env.reset()
            info: dict = {}
            while True:
                observation, _, terminated, truncated, info = env.step(
                    agent.act(observation, epsilon=0.0)
                )
                if terminated or truncated:
                    break
            if info.get("finished"):
                times.append(int(env.elapsed_frames))

        rate = len(times) / args.attempts
        finish_rates.append(rate)
        finished_any += bool(times)

        driver_text = "--"
        if args.baseline:
            result = PurePursuit(entry.line.centerline, corner_speed=9.0).drive(
                WasmPhysics(code, assets=assets),
                limit_ms=int(entry.par_frames * 8),
            )
            if result.finished:
                driver_compared += 1
                driver_text = f"{result.frames / 1000:.2f}s"
                if times and min(times) < result.frames:
                    beat_driver += 1

        if times:
            best = min(times)
            ratio = best / entry.par_frames
            ratios.append(ratio)
            print(f"    {entry.name:<16}{rate:>9.0%}{best / 1000:>9.2f}s"
                  f"{entry.par_frames / 1000:>8.2f}s{ratio:>10.2f}"
                  f"{driver_text:>10}")
        else:
            print(f"    {entry.name:<16}{rate:>9.0%}{'--':>10}"
                  f"{entry.par_frames / 1000:>8.2f}s{'--':>10}{driver_text:>10}")

    print(f"\n  tracks finished at least once : {finished_any}/{len(entries)}"
          f" ({finished_any / max(len(entries), 1):.0%})")
    print(f"  mean finish rate per attempt  : {statistics.mean(finish_rates):.0%}")
    if ratios:
        print(f"  median best time / par        : {statistics.median(ratios):.2f}")
    if args.baseline and driver_compared:
        print(f"  beat the scripted driver on   : {beat_driver}/{driver_compared}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
