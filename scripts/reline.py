#!/usr/bin/env python3
"""Rebuild one track's reference line from a faster lap.

    python scripts/reline.py arms/arm2-search/winter_5.json \
        --track "Winter 5" -o tracks_18786.npz

A reference line is a *route*, taken from a lap that was actually driven, and on
Winter 5 the one in `tracks.npz` is the 19.025 human record. Search has since
produced a lap 239 ms faster along the same road, so the line every future run is
measured against -- and, with `--line-weight`, pulled toward -- is now the slower
of the two.

This writes a new library with that track's line derived from a given lap
instead. It reads the track *code* out of the existing library rather than the
game's `.track` files, so it needs no `POLYTRACK_TRACKS`; and it never writes to
the input, because `tracks.npz` is frozen for the duration of Phase 2 and holding
Arm 2 output would contaminate Arm 3. See EXPERIMENT.md.

Two things change together, and both matter:

* **Progress** is measured along the new route, so `arc` means distance along a
  faster line.
* **The observation** -- lateral offset and the lookahead points -- describes that
  line, so a policy trained against it is being shown a better target than one
  trained against a 19.025 route.

`par_frames` comes from the lap too, which is what scales the finish bonus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lap", type=Path, help="a JSON lap with frames + recording")
    parser.add_argument("--track", required=True)
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--sample-ms", type=int, default=20)
    args = parser.parse_args(argv)

    from polytrack.library import TrackEntry, TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.reference import sample_replay
    from polytrack.tas import format_time

    if args.out.resolve() == args.library.resolve():
        print(f"  refusing to overwrite {args.library}: it is the frozen "
              "substrate.\n  Write the new line to a different file.")
        return 2

    library = TrackLibrary.load(args.library)
    target = library.by_name(args.track)
    lap = json.loads(args.lap.read_text())
    frames = int(lap["frames"])
    assets = GeometryAssets.load()

    print(f"\n  {target.name}: reference line {format_time(target.par_frames)} "
          f"-> {format_time(frames)} from {args.lap.name}")

    started = time.perf_counter()
    trajectory = sample_replay(
        WasmPhysics(target.track_code(), assets=assets),
        Recording.deserialize(lap["recording"]),
        frames,
        sample_ms=args.sample_ms,
    )
    line = trajectory.to_track(target.name, spacing=args.spacing)
    print(f"  {line.length:.1f} m, {line.n_checkpoints} cp "
          f"(was {target.line.length:.1f} m, {target.line.n_checkpoints} cp)"
          f"   [{time.perf_counter() - started:.0f}s]")

    rebuilt = TrackLibrary([
        TrackEntry(code=entry.code, line=line, par_frames=frames)
        if entry.name == target.name else entry
        for entry in library
    ])
    rebuilt.save(args.out)
    print(f"\n{rebuilt.describe()}")
    print(f"\n  wrote {args.out}. Pass it as --library to train.py, "
          "search_run.py or report.py.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
