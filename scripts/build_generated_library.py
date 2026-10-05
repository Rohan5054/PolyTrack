#!/usr/bin/env python3
"""Build a training library from procedurally generated tracks.

    python scripts/build_generated_library.py --count 32 -o tracks_generated.npz

The piece that turns the generator into training data, and the thing that
**retires the reference-line bootstrap**. For the official tracks a library
entry needs a lap somebody drove: the route comes from replaying it, and the par
time is that lap's time. A generated track supplies the route by construction,
and this script supplies the par by driving it with `polytrack.pursuit`.

Three things happen per track, and two of them are judgement calls worth
stating plainly.

**Drivability.** A self-avoiding walk can lay down geometry nothing can drive.
Each track is driven by the scripted driver and rejected if it never finishes.
The driver is retried at slower corner speeds first: on a sample of 24, the one
track it failed at 11 m/s was a three-piece U-turn it finished at 9. Rejecting
on a single speed would have thrown away a good track.

**Par.** A generated track has no human lap, so there is no honest "time to
beat". What there is, is the scripted driver's time -- which is a crawl, and an
upper bound rather than a target. Par is therefore that time scaled by
``--par-factor``, and **the factor is a guess**: the driver corners at 11 m/s
where the car can do far more. It matters because the finish bonus is
``par / elapsed`` capped at 3, so a par set to the crawl time saturates for any
lap a third as long and switches off time pressure entirely over the whole
useful range. Once a policy has driven these tracks, rebuild with par set from
the best known lap, exactly as the official tracks do it.

**Width.** 6.57 m, the half-width of a `Straight`'s road surface, rather than
the 5.0 the official library defaults to. With the real backend this is only an
observation scale -- the module decides what is off-track and never consults it.

## Vary the tracks, or the distribution is not one

Length and corner density are sampled per track rather than fixed. A first
build held both constant and produced 48 tracks of 572-576 m, every one with
four checkpoints and a scripted-driver time within half a second of every
other. A policy can learn *that* without learning to drive, and a
generalisation result measured on it would not mean much -- held-out tracks
drawn from a one-point distribution are the training tracks with different
corners.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Half-width of a Straight's road surface, from the collision mesh.
ROAD_HALF_WIDTH = 6.57


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=32,
                        help="tracks to accept before stopping")
    parser.add_argument("--pieces", default="20,60",
                        help="piece count, or a 'low,high' range sampled per track")
    parser.add_argument("--checkpoints", type=int, default=3)
    parser.add_argument("--turn-chance", default="0.2,0.5",
                        help="turn probability, or a 'low,high' range")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=0,
                        help="give up after this many seeds (0 = count * 3)")
    parser.add_argument("--par-factor", type=float, default=0.5,
                        help="par = scripted driver's time * this. A guess; see "
                             "the module docstring")
    parser.add_argument("--corner-speeds", default="11,9,7",
                        help="retry a track at each of these before rejecting")
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--smoothing", type=int, default=3)
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("tracks_generated.npz"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    from polytrack.priority import lift_background_qos

    lift_background_qos()

    from polytrack.generator import generate
    from polytrack.library import TrackEntry, TrackLibrary
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.pursuit import PurePursuit
    from polytrack.track import Track
    from polytrack.trackcode import encode

    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable as exc:
        print(f"  {exc}")
        return 2

    def _range(text, cast):
        parts = [cast(v) for v in str(text).split(",")]
        return (parts[0], parts[-1])

    pieces_range = _range(args.pieces, int)
    turn_range = _range(args.turn_chance, float)
    speeds = [float(s) for s in args.corner_speeds.split(",")]
    max_seeds = args.max_seeds or args.count * 3
    entries: list[TrackEntry] = []
    rejected = 0
    started = time.monotonic()

    print(f"\n  generating {args.count} drivable tracks of {args.pieces} pieces")
    print(f"  corner speeds tried: {speeds} m/s, par factor {args.par_factor}\n")

    for seed in range(args.first_seed, args.first_seed + max_seeds):
        if len(entries) >= args.count:
            break
        # Sampled from the seed itself, so a library is reproducible from its
        # seed range alone.
        choose = random.Random(seed * 2_654_435_761)
        pieces = choose.randint(*pieces_range)
        turn_chance = choose.uniform(*turn_range)
        track = generate(
            seed, pieces=pieces, checkpoints=args.checkpoints,
            turn_chance=turn_chance,
        )

        order = {cell: index for index, cell in enumerate(track.cells)}
        checkpoint_samples = sorted(
            order[(p.x, p.y, p.z)]
            for p in track.code.parts
            if p.checkpoint_order is not None
        )
        line = Track.from_trajectory(
            track.route,
            spacing=args.spacing,
            smoothing=args.smoothing,
            width=ROAD_HALF_WIDTH,
            checkpoint_samples=checkpoint_samples,
            name=f"Generated {seed}",
        )

        # Drive the resampled line, not the raw cell centres. That line is what
        # the agent is scored against, and smoothing the right angles out of it
        # is worth a third of the lap: 28.6 s against 41.2 s on one sample. Par
        # taken from the raw route would be calibrated against a path nothing
        # ever follows.
        result = None
        for speed in speeds:
            driver = PurePursuit(line.centerline, corner_speed=speed)
            result = driver.drive(
                WasmPhysics(track.code, assets=assets), limit_ms=180_000
            )
            if result.finished:
                break
        if result is None or not result.finished:
            rejected += 1
            print(f"    seed {seed:>4}: rejected, reached "
                  f"{result.reached if result else 0}/{result.of if result else 0}",
                  flush=True)
            continue
        par_frames = max(int(result.frames * args.par_factor), 1)
        entries.append(
            TrackEntry(
                code=encode(track.code, header=True),
                line=line,
                par_frames=par_frames,
            )
        )
        print(f"    seed {seed:>4}: {line.length:6.0f} m, "
              f"{len(line.checkpoint_indices)} cp, driver "
              f"{result.frames / 1000:6.2f} s -> par {par_frames / 1000:6.2f} s",
              flush=True)

    if not entries:
        print("\n  no drivable tracks; nothing written")
        return 1

    library = TrackLibrary(entries)
    library.save(args.out)
    minutes = (time.monotonic() - started) / 60.0

    lengths = [e.line.length for e in entries]
    pars = [e.par_frames / 1000.0 for e in entries]
    corners = [
        sum(1 for part in e.track_code().parts if part.part_id == 1)
        for e in entries
    ]
    print(f"  corners {min(corners)}-{max(corners)} per track")
    print(f"\n  accepted {len(entries)}, rejected {rejected}, in {minutes:.0f} min")
    print(f"  length {min(lengths):.0f}-{max(lengths):.0f} m, "
          f"par {min(pars):.1f}-{max(pars):.1f} s")
    print(f"  wrote {args.out}\n")
    print(library.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
