#!/usr/bin/env python3
"""Arm 3: continuous trajectory optimisation. No policy, no Arm 2 artifact.

    python scripts/arm3_run.py --minutes 90

Sweeps a window of spline knots along the lap, refitting each by cross-entropy
against arc reached at a fixed horizon, with the prefix frozen behind a saved
anchor. Seeded by projecting the human record onto the spline -- shared
substrate, available to every arm.

The seed is a *starting point*, not a lap: projecting the 19.025 record onto any
fixed block grid perturbs its input timing, and this track does not tolerate
that. Measured: quantising the record to 50/25/10/5 ms grids reaches 26, 349,
477 and 468 m of 1747 before leaving the road. Zero timing error replays exactly
(``reference.sample_replay`` cuts at input boundaries); five milliseconds does
not. So the sweep is repairing a diverged lap, not polishing a good one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="Winter 5")
    p.add_argument("--library", type=Path, default=Path("tracks.npz"))
    p.add_argument("--block-ms", type=int, default=10)
    p.add_argument("--knot-spacing", type=int, default=1, help="blocks between knots")
    p.add_argument("--window", type=int, default=40, help="knots optimised at once")
    p.add_argument("--stride", type=int, default=20, help="knots between window starts")
    p.add_argument("--horizon", type=int, default=300, help="blocks scored per attempt")
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--generations", type=int, default=4)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--passes", type=int, default=8)
    p.add_argument("--backoff", type=int, default=400,
                   help="knots to start earlier after a pass that gains nothing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", type=Path, default=None,
                   help="knots .npy from an earlier run")
    p.add_argument("-o", "--out", type=Path, default=Path("runs/arm3/winter_5.json"))
    return p.parse_args(argv)


#: Metres of arc -- or, once a lap finishes, milliseconds -- that a pass has
#: to gain before it counts as progress rather than noise.
MEANINGFUL_GAIN = 1.0


def main(argv=None) -> int:
    args = parse_args(argv)

    import numpy as np

    from polytrack.priority import lift_background_qos

    lift_background_qos()

    from polytrack.actions import mask_to_action
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.spline import SplineCEM, SteeringSpline
    from polytrack.tas import format_time

    entry = TrackLibrary.load(args.library).by_name(args.track)
    assets = GeometryAssets.load()
    track = entry.line

    records = json.loads(Path("records.json").read_text())
    human = min(
        (r for r in records if r.get("recording")), key=lambda r: r["frames"]
    ) if False else next(r for r in records if r["frames"] == 19025)
    target = int(human["frames"])

    masks: list[int] = []
    for mask, ticks in Recording.deserialize(human["recording"]).runs(target):
        masks.extend([mask] * ticks)

    block = args.block_ms
    n_blocks = target // block + 40
    masks = masks[:target] + [masks[-1]] * (n_blocks * block - target)
    seq = np.array([mask_to_action(masks[b * block]) for b in range(n_blocks)])

    spline = SteeringSpline(
        n_blocks=n_blocks, block_frames=block, knot_spacing=args.knot_spacing
    )
    knots = spline.fit(seq)
    if args.resume is not None and args.resume.exists():
        loaded = np.load(args.resume)
        if loaded.shape == knots.shape:
            knots = loaded
            print(f"  resuming from {args.resume}")
        else:
            print(f"  ignoring {args.resume}: {loaded.shape} != {knots.shape}")

    backend = WasmPhysics(entry.track_code(), assets=assets)
    cem = SplineCEM(
        backend, track, spline,
        population=args.population, seed=args.seed,
    )

    print(f"\n  {args.track}: {track.length:.0f} m, par {format_time(entry.par_frames)}")
    print(f"  spline: {n_blocks} blocks of {block} ms, {spline.n_knots} knots "
          f"every {args.knot_spacing * block} ms")
    print(f"  seeded by projecting the {target} human record (shared substrate)")
    print(f"  window {args.window} knots, stride {args.stride}, "
          f"horizon {args.horizon * block} ms, "
          f"population {args.population} x {args.generations} generations\n")

    started = time.monotonic()
    deadline = started + args.minutes * 60.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    baseline = cem.evaluate(knots)
    print(f"    seed: {'FINISH ' + format_time(baseline.frames) if baseline.finished else f'{baseline.arc:7.1f} m'}"
          f"   [{time.monotonic() - started:.0f}s]", flush=True)
    best_full = baseline

    stalled = 0
    for sweep in range(args.passes):
        if time.monotonic() > deadline:
            break
        print(f"\n  -- pass {sweep + 1} --", flush=True)
        # Start where the run breaks -- sweeping from knot zero spends most of a
        # pass re-optimising an opening straight that already works -- but back
        # off further each time a pass gains nothing. The block where progress
        # *stops* is not the block that caused it: a car that clips a wall keeps
        # crawling forward for seconds afterwards, so the frontier sits well
        # past the damage and a sweep anchored there can never undo it. Six
        # passes of this run started at the same knot 300 past the crash and
        # returned the same 992.4 m, which is the arc-greedy failure the beam
        # search hit and is recorded in CLAUDE.md.
        frontier = best_full.peak_block // args.knot_spacing
        first = max(0, frontier - args.window - stalled * args.backoff)
        first -= first % args.stride
        # Nothing past the frontier is reachable, so every window there scores
        # identically and costs a full evaluation to say so.
        last = min(spline.n_knots - args.window - 1, frontier + 2 * args.window)
        print(f"    sweeping knots {first}-{last} "
              f"(progress stops at block {best_full.peak_block}"
              f"{f', backed off {stalled * args.backoff}' if stalled else ''})",
              flush=True)
        anchor = None
        for j in range(first, last, args.stride):
            if time.monotonic() > deadline:
                print("    budget spent", flush=True)
                break
            anchor = cem.anchor_at(knots, j, previous=anchor) if j > 0 else None
            attempt = cem.optimise(
                knots,
                (j + 1, min(j + 1 + args.window, spline.n_knots)),
                anchor=anchor,
                horizon=args.horizon,
                generations=args.generations,
            )
            knots = attempt.knots
            if j % (args.stride * 5) == 0:
                mins = (time.monotonic() - started) / 60.0
                print(f"    knot {j:>4}/{spline.n_knots}  "
                      f"arc {attempt.arc:7.1f} m  {mins:5.1f} min", flush=True)
                np.save(args.out.with_suffix(".knots.npy"), knots)

        full = cem.evaluate(knots)
        tag = (f"FINISH {format_time(full.frames)}" if full.finished
               else f"{full.arc:.1f} m")
        print(f"    end of pass {sweep + 1}: {tag}", flush=True)
        # "Improved" has to mean improved by an amount worth another pass.
        # Comparing keys directly made a pass that gained a hundredth of a
        # millimetre reset the back-off, so the sweep re-ran the same six
        # unreachable windows instead of moving to where the damage was.
        gained = (
            full.key[0] > best_full.key[0]
            or full.key[1] > best_full.key[1] + MEANINGFUL_GAIN
        )
        if full.key > best_full.key:
            best_full = full
        stalled = 0 if gained else stalled + 1
        np.save(args.out.with_suffix(".knots.npy"), knots)
        if full.finished:
            break

    print(f"\n  {cem.rollouts:,} rollouts, {cem.frames_simulated:,} frames simulated")
    if not best_full.finished:
        print(f"  did not finish: reached {best_full.arc:.1f} m of {track.length:.0f} m")
        return 1

    # Re-verify on a fresh backend, exactly as the gate would.
    from polytrack.verify import verify

    actions = spline.actions(best_full.knots)
    tas_masks: list[int] = []
    from polytrack.actions import ACTION_TO_MASK
    for a in actions:
        tas_masks.extend([ACTION_TO_MASK[int(a)]] * block)
    recording = Recording.from_masks(tas_masks[: best_full.frames])
    fresh = WasmPhysics(entry.track_code(), assets=assets)
    result = verify(fresh, recording, best_full.frames)
    print(f"  replay check: {result.describe()}")
    if not result.passed:
        return 1

    args.out.write_text(json.dumps({
        "track": args.track,
        "frames": int(best_full.frames),
        "recording": recording.serialize(),
        "arm": "arm3-independent",
        "method": "separable CEM over a steering spline, no policy",
        "rollouts": cem.rollouts,
        "frames_simulated": cem.frames_simulated,
    }, indent=1))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
