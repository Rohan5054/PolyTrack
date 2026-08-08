#!/usr/bin/env python3
"""Build the track library the training loop reads.

    python tools/extract_records.py -o records.json
    POLYTRACK_TRACKS=path/to/tracks python scripts/build_library.py

For every recording that matches an available track, this replays the run
through the physics module and resamples it into a reference line, then writes
the lot to ``tracks.npz``. It costs one full replay per track -- about 45 seconds
for a 20-second run -- and that is exactly why it is done once and cached rather
than at collector start-up.

Where a track has several runs, the fastest one wins: it is the better racing
line and the harder par time.

The output embeds the game's track codes, so it is derived from Kodub's content.
It is gitignored; keep it local.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack.library import TrackEntry, TrackLibrary
from polytrack.physics.assets import GeometryAssets
from polytrack.physics.wasm import WasmPhysics
from polytrack.recording import Recording
from polytrack.reference import ReferenceLineError, sample_replay
from polytrack.trackcode import PREFIX, decode


def load_tracks(root: str | None) -> dict[str, tuple[str, object]]:
    """Index the raw code text and the decoded track by leaderboard id."""
    if not root:
        return {}
    tracks: dict[str, tuple[str, object]] = {}
    for path in Path(root).rglob("*.track"):
        text = path.read_text(encoding="utf-8").strip()
        if not text.startswith(PREFIX):
            continue  # legacy PolyTrack1; a separate decoder
        try:
            track = decode(text)
        except Exception as exc:
            print(f"  ! {path.name}: {exc}")
            continue
        tracks[track.track_id] = (text, track)
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="?", default=Path("records.json"))
    parser.add_argument("-o", "--out", type=Path, default=Path("tracks.npz"))
    parser.add_argument(
        "--tracks", default=os.environ.get("POLYTRACK_TRACKS"),
        help="directory of .track files (or set POLYTRACK_TRACKS)",
    )
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--sample-ms", type=int, default=20)
    args = parser.parse_args()

    if not args.records.exists():
        print(f"no {args.records}. Run tools/extract_records.py first.")
        return 2
    available = load_tracks(args.tracks)
    if not available:
        print("No tracks available. Pass --tracks or set POLYTRACK_TRACKS: a")
        print("recording identifies its track only by id.")
        return 2

    # Fastest run per track: the better line and the harder par time.
    best: dict[str, dict] = {}
    for record in json.loads(args.records.read_text()):
        current = best.get(record["track_id"])
        if current is None or record["frames"] < current["frames"]:
            best[record["track_id"]] = record

    assets = GeometryAssets.load()
    print(f"geometry {assets.version}, {len(assets)} parts\n")

    entries: list[TrackEntry] = []
    for track_id, record in sorted(best.items(), key=lambda kv: kv[1]["frames"]):
        found = available.get(track_id)
        if found is None:
            continue
        code, track = found
        label = track.name or track_id[:12]

        started = time.perf_counter()
        try:
            trajectory = sample_replay(
                WasmPhysics(track, assets=assets),
                Recording.deserialize(record["recording"]),
                record["frames"],
                sample_ms=args.sample_ms,
            )
        except ReferenceLineError as exc:
            print(f"  SKIP  {label}: {exc}")
            continue
        line = trajectory.to_track(label, spacing=args.spacing)
        elapsed = time.perf_counter() - started

        entries.append(
            TrackEntry(code=code, line=line, par_frames=record["frames"])
        )
        print(f"  {label:<14} {line.length:7.1f} m  {line.n_checkpoints} cp  "
              f"par {record['frames'] / 1000:6.3f} s   [{elapsed:.0f}s]")

    if not entries:
        print("\nNo record matched an available track.")
        return 1

    library = TrackLibrary(entries)
    library.save(args.out)
    size = args.out.stat().st_size / 1024
    print(f"\nwrote {args.out} ({size:.0f} KB)\n")
    print(library.describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
