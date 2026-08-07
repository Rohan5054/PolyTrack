#!/usr/bin/env python3
"""Run the acceptance gate: replay real runs and check the finish frames.

    python tools/extract_records.py -o records.json
    POLYTRACK_TRACKS=path\\to\\tracks python scripts/run_gate.py records.json

Each record pairs an input recording with the exact frame the run finished on.
Replaying one through the backend and landing on that frame -- not near it -- is
the only evidence that this project simulates PolyTrack rather than something
that resembles it. See ``polytrack/verify.py`` for the loop, which is the game's
own.

Exit status is 0 only if every run verifies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polytrack.physics.assets import GeometryAssets
from polytrack.physics.wasm import WasmPhysics
from polytrack.recording import Recording
from polytrack.trackcode import PREFIX, decode


def load_tracks(root: str | None) -> dict:
    """Index every track we can find by its leaderboard id."""
    if not root:
        return {}
    tracks = {}
    for path in Path(root).rglob("*.track"):
        text = path.read_text(encoding="utf-8").strip()
        if not text.startswith(PREFIX):
            continue  # legacy PolyTrack1; a separate decoder
        try:
            track = decode(text)
        except Exception as exc:
            print(f"  ! {path.name}: {exc}")
            continue
        tracks[track.track_id] = track
    return tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="JSON from tools/extract_records.py")
    parser.add_argument(
        "--tracks", default=os.environ.get("POLYTRACK_TRACKS"),
        help="directory of .track files (or set POLYTRACK_TRACKS)",
    )
    parser.add_argument("--track-code", help="a single track code, if not in --tracks")
    args = parser.parse_args()

    records = json.loads(args.records.read_text())
    tracks = load_tracks(args.tracks)
    if args.track_code:
        track = decode(args.track_code)
        tracks[track.track_id] = track
    if not tracks:
        print("No tracks available. Pass --tracks or --track-code: a recording")
        print("identifies its track only by id, so the geometry has to come from")
        print("somewhere.")
        return 2

    assets = GeometryAssets.load()
    print(f"geometry {assets.version}, {len(assets)} parts, "
          f"massOffset {assets.mass_offset}\n")

    passed = failed = skipped = 0
    for record in records:
        track = tracks.get(record["track_id"])
        if track is None:
            print(f"  SKIP  {record['track_id'][:16]}…  track not available")
            skipped += 1
            continue

        recording = Recording.deserialize(record["recording"])
        target = record["frames"]
        label = f"{track.name or track.track_id[:12]} ({target / 1000:.3f} s)"

        backend = WasmPhysics(track, assets=assets)
        started = time.perf_counter()
        # Imported here so a failure to build the backend reports first.
        from polytrack.verify import verify

        result = verify(backend, recording, target)
        elapsed = time.perf_counter() - started

        print(f"  {result.describe()}")
        print(f"        {label}  [{elapsed:.1f}s wall, "
              f"{target / max(elapsed, 1e-9) / 1000:.2f}x realtime]")
        if result.passed:
            passed += 1
        else:
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    if failed:
        print(
            "\nA failure here means the simulation is not the game's. Check, in\n"
            "this order: the track decoded to the right geometry; every part the\n"
            "track uses is in geometry.npz and matched its checksum; the start\n"
            "transform; the mountain terrain; the part ordering in the buffer."
        )
    return 0 if failed == 0 and passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
