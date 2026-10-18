#!/usr/bin/env python3
"""Separate "unfamiliar road" from "unfamiliar reference line".

    python scripts/measure_line_confound.py runs/gen1/agent.pt \
        --band runs/support/support_band.csv

`measure_support.py` finds official tracks 147x further outside the training
distribution than unseen generated ones, and **95% of those violations are
lookahead features**. That number cannot carry a claim about track geometry on
its own, because the lookahead points and the lateral offset are measured
against the reference line, and the two libraries get their lines from
different places: a generated track's line is its own centreline, an official
track's is a human lap. A racing line cuts corners. Some of the violation is
therefore racing-line-against-centreline rather than road-against-road, and
from the support measurement alone there is no way to tell how much.

This isolates it. Take a track the policy **trained on**, keep the road exactly
as it is, and swap only the line -- from the generator's centreline to a line
derived from a lap the policy itself drove, which is how every official track's
line was built. Same geometry, same policy, same band; one variable.

* If the relined track reads as out of support, the lookahead signal is about
  line provenance and the geometry claim does not follow from it.
* If it stays inside support, the line is not what moved and the official
  tracks' violations are about the road.

The laps come from `runs/<run>/best_generated_<i>.json`, which training writes
for every track it completes, so this needs no new driving.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from measure_support import feature_names, groups, rollout


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=Path("runs/gen1/agent.pt"))
    parser.add_argument("--band", type=Path,
                        default=Path("runs/support/support_band.csv"))
    parser.add_argument("--library", type=Path,
                        default=Path("tracks_generated.npz"))
    parser.add_argument("--laps", type=Path, default=Path("runs/gen1"),
                        help="directory of best_generated_<i>.json laps")
    parser.add_argument("--tracks", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit-factor", type=float, default=4.0)
    parser.add_argument("--sample-ms", type=int, default=20)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("runs/support"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from polytrack.priority import lift_background_qos

    lift_background_qos()

    import torch

    torch.set_num_threads(2)

    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackEntry, TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.reference import sample_replay

    agent = IQNAgent.load(args.checkpoint)
    assets = GeometryAssets.load()
    library = TrackLibrary.load(args.library)
    names = feature_names()

    rows = list(csv.DictReader(args.band.open(encoding="utf-8")))
    if [r["feature"] for r in rows] != names:
        print(f"  {args.band} does not match the observation layout")
        return 2
    low = np.array([float(r["train_p0.1"]) for r in rows])
    high = np.array([float(r["train_p99.9"]) for r in rows])
    margin = 1e-3 * np.maximum(high - low, 1.0)
    low_b, high_b = low - margin, high + margin

    laps = {}
    for path in sorted(args.laps.glob("best_generated_*.json")):
        lap = json.loads(path.read_text())
        laps.setdefault(lap["track"], (path, lap))

    print(f"\n  {args.checkpoint} at {agent.steps:,} gradient steps")
    print(f"  band from {args.band}")
    print(f"  {len(laps)} laps available in {args.laps}\n")

    print(f"    {'track':<16}{'line':>12}{'length':>9}{'any OOS':>9}"
          f"{'mean OOS':>10}{'lookahead share':>17}")

    paired = []
    for entry in library:
        if entry.name not in laps or len(paired) >= args.tracks:
            continue
        _, lap = laps[entry.name]

        # The relined twin: same code, same road, line taken from a driven lap.
        trajectory = sample_replay(
            WasmPhysics(entry.track_code(), assets=assets),
            Recording.deserialize(lap["recording"]),
            int(lap["frames"]),
            sample_ms=args.sample_ms,
        )
        relined = TrackEntry(
            code=entry.code,
            line=trajectory.to_track(entry.name, spacing=args.spacing),
            par_frames=int(lap["frames"]),
        )

        measured = {}
        for label, candidate in (("centreline", entry), ("driven lap", relined)):
            best = None
            for _ in range(args.attempts):
                obs, arcs, info = rollout(
                    candidate, agent, assets, args.limit_factor
                )
                if best is None or arcs[-1] > best[1][-1]:
                    best = (obs, arcs, info)
            obs, arcs, _ = best
            outside = (obs < low_b[None, :]) | (obs > high_b[None, :])
            per_frame = outside.sum(axis=1)
            total = max(int(outside.sum()), 1)
            look = int(outside[:, groups()["lookahead"]].sum()) / total
            measured[label] = (obs, outside, per_frame, candidate.line.length, look)
            print(f"    {entry.name:<16}{label:>12}{candidate.line.length:>8.0f}m"
                  f"{(per_frame > 0).mean():>9.0%}{per_frame.mean():>10.1f}"
                  f"{look:>17.0%}", flush=True)
        paired.append((entry.name, measured))

    if not paired:
        print(f"  no track in {args.library} had a lap in {args.laps}")
        return 1

    # --- pooled comparison ------------------------------------------------
    print(f"\n  {len(paired)} tracks, each driven on both lines\n")
    print(f"    {'line':<14}{'frames with any OOS':>21}{'features per frame':>20}")
    pooled = {}
    for label in ("centreline", "driven lap"):
        per_frame = np.concatenate([m[label][2] for _, m in paired])
        outside = np.concatenate([m[label][1] for _, m in paired], axis=0)
        pooled[label] = (per_frame, outside)
        print(f"    {label:<14}{(per_frame > 0).mean():>20.0%}"
              f"{per_frame.mean():>20.2f}")

    print(f"\n    {'group':<16}{'centreline':>14}{'driven lap':>14}")
    with (args.out / "line_confound.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["group", "centreline_frames_outside",
                         "driven_lap_frames_outside"])
        for group, sl in groups().items():
            shares = []
            for label in ("centreline", "driven lap"):
                shares.append(pooled[label][1][:, sl].any(axis=1).mean())
            writer.writerow([group, f"{shares[0]:.6g}", f"{shares[1]:.6g}"])
            if any(s > 0 for s in shares):
                print(f"    {group:<16}{shares[0]:>13.0%}{shares[1]:>13.0%}")

    centre = pooled["centreline"][0].mean()
    driven = pooled["driven lap"][0].mean()
    print(f"\n  {'=' * 64}")
    print(f"  swapping ONLY the reference line moves the observation from "
          f"{centre:.2f}")
    print(f"  to {driven:.2f} features out of support per frame, on the same roads.")
    print(f"  Official tracks read 9.2. Line provenance explains "
          f"{min(driven / 9.2, 1.0):.0%} of that at most.")
    print(f"  {'=' * 64}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
