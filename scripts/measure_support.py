#!/usr/bin/env python3
"""Measure where an official-track observation leaves the training distribution.

    python scripts/measure_support.py

`arms/generalisation/README.md` reports 0 of 4 official tracks and explains it
as distribution distance. That explanation is an inference, and this turns it
into a measurement: roll the policy out on the tracks it trained on to get a
per-feature support band, roll it out on the official tracks, and report where
and by how much the observation leaves that band.

The point is the ordering. If the observation leaves support **before** the car
stops making progress, the policy was handed inputs it had never seen and then
failed -- which is distribution distance. If it stalls while still inside
support, the inference in that README is wrong and the failure is the policy's.

Support is the band between the 0.1st and 99.9th percentile of what the policy
actually saw while driving its training library, not the min and max: one
outlying frame in twenty thousand would otherwise widen a feature's band enough
to swallow anything. Features that never varied at all collapse to a point,
which is not a defect of the estimator -- it is the finding. A flat track with
no walls and no jumps never lifts a wheel.

Writes a CSV of per-feature bands and per-frame violation counts so the figure
can be drawn from the artifact rather than from a re-run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# The observation layout, from ObservationBuilder.size. Named here because a
# violation is only interpretable if you know which feature violated.
LAYOUT = [
    ("vel_fwd", 1), ("vel_left", 1), ("vel_up", 1),
    ("speed", 1),
    ("angvel_x", 1), ("angvel_y", 1), ("angvel_z", 1),
    ("airborne", 1),
    ("wheel_fl", 1), ("wheel_fr", 1), ("wheel_rl", 1), ("wheel_rr", 1),
    ("lateral_offset", 1),
    ("lookahead", 40),
    ("rays", 9),
]


def feature_names() -> list[str]:
    names: list[str] = []
    for name, count in LAYOUT:
        if count == 1:
            names.append(name)
        else:
            names.extend(f"{name}[{i}]" for i in range(count))
    return names


def groups() -> dict[str, slice]:
    out: dict[str, slice] = {}
    start = 0
    for name, count in LAYOUT:
        out[name] = slice(start, start + count)
        start += count
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, nargs="?",
                        default=Path("arms/generalisation/agent.pt"))
    parser.add_argument("--train-library", type=Path,
                        default=Path("tracks_generated.npz"))
    parser.add_argument("--official-library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("--control-library", type=Path,
                        default=Path("tracks_heldout.npz"),
                        help="unseen tracks from the TRAINING distribution; the "
                             "control that makes an out-of-support count mean "
                             "something")
    parser.add_argument("--support-tracks", type=int, default=0,
                        help="training tracks to roll out for the band (0 = all)")
    parser.add_argument("--band", type=Path, default=None,
                        help="reuse a support_band.csv instead of re-driving the "
                             "48 training laps that produced it (~30 min)")
    parser.add_argument("--attempts", type=int, default=3,
                        help="greedy attempts per official track; the furthest is kept")
    parser.add_argument("--limit-factor", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("runs/support"))
    return parser.parse_args(argv)


def rollout(entry, agent, assets, limit_factor):
    """Drive one greedy lap, returning every observation and the arc at each."""
    from polytrack.corridor import RoadMap
    from polytrack.env import PolyTrackEnv
    from polytrack.physics.wasm import WasmPhysics

    code = entry.track_code()
    road = RoadMap.build(code, assets)
    backend = WasmPhysics(code, assets=assets)
    env = PolyTrackEnv(
        track=entry.line,
        backend=backend,
        road=road,
        par_frames=entry.par_frames,
        time_limit_ms=entry.par_frames * limit_factor,
    )
    try:
        observation, _ = env.reset()
        observations = [observation]
        arcs = [0.0]
        info: dict = {}
        while True:
            observation, _, terminated, truncated, info = env.step(
                agent.act(observation, epsilon=0.0)
            )
            observations.append(observation)
            arcs.append(float(info.get("arc", 0.0)))
            if terminated or truncated:
                break
        return np.asarray(observations), np.asarray(arcs), info
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()


def main(argv=None) -> int:
    args = parse_args(argv)

    from polytrack.priority import lift_background_qos

    lift_background_qos()

    import torch

    torch.set_num_threads(2)

    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets

    if not args.checkpoint.exists():
        print(f"  no checkpoint at {args.checkpoint}")
        return 2

    agent = IQNAgent.load(args.checkpoint)
    assets = GeometryAssets.load()
    train = list(TrackLibrary.load(args.train_library))
    official = list(TrackLibrary.load(args.official_library))
    control = list(TrackLibrary.load(args.control_library))
    if args.support_tracks:
        train = train[: args.support_tracks]

    names = feature_names()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"\n  {args.checkpoint} at {agent.steps:,} gradient steps")
    print(f"  support from {len(train)} tracks of {args.train_library}")
    print(f"  control  : {len(control)} unseen tracks of {args.control_library}")
    print(f"  tested on: {len(official)} tracks of {args.official_library}\n")

    # --- the support band -------------------------------------------------
    if args.band is not None:
        rows = list(csv.DictReader(args.band.open(encoding="utf-8")))
        if [r["feature"] for r in rows] != names:
            print(f"  {args.band} does not match the observation layout")
            return 2
        low = np.array([float(r["train_p0.1"]) for r in rows])
        high = np.array([float(r["train_p99.9"]) for r in rows])
        print(f"  band reused from {args.band} ({len(rows)} features)")
    else:
        collected = []
        for entry in train:
            obs, arcs, info = rollout(entry, agent, assets, args.limit_factor)
            collected.append(obs)
            flag = "finished" if info.get("finished") else info.get("failure", "--")
            print(f"    {entry.name:<16}{len(obs):>7} steps  {arcs[-1]:>8.1f} m"
                  f"  {flag}", flush=True)
        train_obs = np.concatenate(collected, axis=0)
        print(f"\n  {train_obs.shape[0]:,} training frames, "
              f"{train_obs.shape[1]} features")
        low = np.percentile(train_obs, 0.1, axis=0)
        high = np.percentile(train_obs, 99.9, axis=0)
    # A feature that never moved has a zero-width band. Widen every band by a
    # hair so float noise alone cannot register as a violation; the margin is
    # small enough that a wheel leaving the ground still counts.
    width = high - low
    margin = 1e-3 * np.maximum(width, 1.0)
    low_b, high_b = low - margin, high + margin

    degenerate = [names[i] for i in range(len(names)) if width[i] <= 1e-6]
    print(f"  features that never varied in training ({len(degenerate)}): "
          f"{', '.join(degenerate) if degenerate else 'none'}\n")

    if args.band is None:
        with (args.out / "support_band.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(["feature", "train_p0.1", "train_p99.9",
                             "train_min", "train_max"])
            mins, maxs = train_obs.min(axis=0), train_obs.max(axis=0)
            for i, name in enumerate(names):
                writer.writerow([name, f"{low[i]:.6g}", f"{high[i]:.6g}",
                                 f"{mins[i]:.6g}", f"{maxs[i]:.6g}"])

    # --- both libraries, through one code path ----------------------------
    # The control exists so the test is falsifiable. An out-of-support count on
    # the official tracks is only evidence if unseen tracks from the *training*
    # distribution -- which this policy finishes 94% of the time -- score near
    # zero on the same band. If they do not, the band is too tight and every
    # number below is an artifact of the estimator.
    def analyse(entries, label):
        print(f"\n  {label}\n")
        print(f"    {'track':<16}{'steps':>7}{'reached':>9}{'outcome':>10}"
              f"{'any OOS':>9}{'mean OOS':>10}{'first OOS':>11}{'% of run':>10}")
        collected, summary = [], []
        for entry in entries:
            best = None
            for _ in range(args.attempts):
                obs, arcs, info = rollout(entry, agent, assets, args.limit_factor)
                if best is None or arcs[-1] > best[1][-1]:
                    best = (obs, arcs, info)
            obs, arcs, info = best
            collected.append(obs)

            outside = (obs < low_b[None, :]) | (obs > high_b[None, :])
            per_frame = outside.sum(axis=1)
            first = int(np.argmax(per_frame > 0)) if per_frame.any() else -1
            end_arc = float(arcs[-1])
            first_arc = float(arcs[first]) if first >= 0 else float("nan")
            fraction = first_arc / end_arc if end_arc > 0 else float("nan")
            outcome = "finished" if info.get("finished") else info.get("failure", "--")

            offenders = np.argsort(-outside.sum(axis=0))[:6]
            offender_text = ", ".join(
                f"{names[i]} ({outside[:, i].mean():.0%})"
                for i in offenders if outside[:, i].any()
            )

            print(f"    {entry.name:<16}{len(obs):>7}{end_arc:>8.0f}m{outcome:>10}"
                  f"{(per_frame > 0).mean():>9.0%}{per_frame.mean():>10.1f}"
                  f"{(f'step {first}' if first >= 0 else 'never'):>11}"
                  f"{fraction:>10.0%}")
            if offender_text:
                print(f"      {offender_text}", flush=True)
            summary.append(((per_frame > 0).mean(), per_frame.mean()))

            name = entry.name.replace(" ", "_")
            with (args.out / f"violations_{name}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as fh:
                writer = csv.writer(fh)
                writer.writerow(["step", "arc_m", "features_out_of_support"])
                for step, (arc, count) in enumerate(zip(arcs, per_frame)):
                    writer.writerow([step, f"{arc:.3f}", int(count)])

        any_oos = float(np.mean([s[0] for s in summary]))
        mean_oos = float(np.mean([s[1] for s in summary]))
        print(f"\n    frames with any feature out of support : {any_oos:.0%}")
        print(f"    features out of support per frame      : {mean_oos:.1f} of 62")

        # Which *kind* of feature leaves support decides what the result means.
        # The lookahead points and the lateral offset are both measured against
        # the reference line, and an official track's line comes from a human
        # lap while a generated track's is its own centreline -- so violations
        # there are confounded with racing-line-against-centreline and cannot
        # carry the claim alone. The car's own dynamics are not confounded:
        # nothing about where the line runs makes a car roll or drop.
        pooled = np.concatenate(collected, axis=0)
        outside = (pooled < low_b[None, :]) | (pooled > high_b[None, :])
        print(f"\n    {'group':<16}{'share of violations':>21}"
              f"{'frames outside':>16}")
        total = max(int(outside.sum()), 1)
        per_group = []
        for group, sl in groups().items():
            count = int(outside[:, sl].sum())
            frames = float(outside[:, sl].any(axis=1).mean())
            per_group.append((group, count / total, frames))
            if count:
                print(f"    {group:<16}{count / total:>20.0%}{frames:>16.0%}")
        with (args.out / f"groups_{label.split()[0].lower()}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.writer(fh)
            writer.writerow(["group", "share_of_violations", "frames_outside"])
            for group, share, frames in per_group:
                writer.writerow([group, f"{share:.6g}", f"{frames:.6g}"])
        return pooled, any_oos, mean_oos

    control_obs, control_any, control_mean = analyse(
        control, f"CONTROL -- unseen, same distribution ({args.control_library})"
    )
    official_obs, official_any, official_mean = analyse(
        official, f"TEST -- official tracks ({args.official_library})"
    )

    # --- the headline features, side by side ------------------------------
    print(f"\n  per-group range: training band, control, official\n")
    print(f"    {'group':<16}{'training band':>22}{'control':>22}{'official':>22}")
    for name, sl in groups().items():
        def span(obs):
            return f"[{obs[:, sl].min():.3g}, {obs[:, sl].max():.3g}]"
        band = f"[{low[sl].min():.3g}, {high[sl].max():.3g}]"
        print(f"    {name:<16}{band:>22}{span(control_obs):>22}"
              f"{span(official_obs):>22}")

    print(f"\n  {'=' * 64}")
    print(f"  control {control_mean:.1f} features out of support per frame, "
          f"official {official_mean:.1f}")
    if control_mean > 0:
        print(f"  official tracks are {official_mean / control_mean:.1f}x "
              f"further outside the training distribution")
    print(f"  {'=' * 64}")
    print(f"\n  artifacts in {args.out}/\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
