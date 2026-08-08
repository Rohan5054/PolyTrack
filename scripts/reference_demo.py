#!/usr/bin/env python3
"""Build reference lines from real runs and drive an environment on them.

    python tools/extract_records.py -o records.json
    POLYTRACK_TRACKS=path/to/tracks python scripts/reference_demo.py records.json

For each recording that matches an available track, this replays the run through
PolyTrack's own physics module, resamples where the car went into a reference
line, and then runs a short rollout of a naive policy against it. The point is to
show the full stack standing up on real physics and real geometry: track code ->
collision meshes -> module -> trajectory -> reference line -> observation ->
reward.

Nothing here is trained, and the rollout policy is deliberately stupid. What the
numbers are for is calibration -- the human run's speed profile is the first
honest scale the reward coefficients have ever had to work against, since the
stand-in's speeds were invented.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from polytrack.env import PolyTrackEnv
from polytrack.physics.assets import GeometryAssets
from polytrack.recording import Recording
from polytrack.reference import reference_line
from polytrack.tas import format_time
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
        except Exception:
            continue
        tracks[track.track_id] = track
    return tracks


def rollout(env: PolyTrackEnv, seconds: float) -> dict:
    """Hold throttle, steer at the first lookahead point. No cleverness."""
    observation, _ = env.reset()
    steps = int(seconds * 1000.0 / env.control_interval_ms)
    total = 0.0
    info: dict = {}
    for _ in range(steps):
        lateral = float(observation[13 + 1])  # y of the nearest lookahead point
        action = 4 if lateral > 0.02 else 5 if lateral < -0.02 else 3
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return {"reward": total, **info}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, nargs="?", default=Path("records.json"))
    parser.add_argument(
        "--tracks", default=os.environ.get("POLYTRACK_TRACKS"),
        help="directory of .track files (or set POLYTRACK_TRACKS)",
    )
    parser.add_argument("--rollout-seconds", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=2.0)
    args = parser.parse_args()

    if not args.records.exists():
        print(f"no {args.records}. Run tools/extract_records.py first.")
        return 2
    tracks = load_tracks(args.tracks)
    if not tracks:
        print("No tracks available. Pass --tracks or set POLYTRACK_TRACKS: a")
        print("recording identifies its track only by id.")
        return 2

    assets = GeometryAssets.load()
    records = json.loads(args.records.read_text())

    # One run per track: the reference line only needs the best of them, and
    # each one costs a full replay.
    best: dict[str, dict] = {}
    for record in records:
        current = best.get(record["track_id"])
        if current is None or record["frames"] < current["frames"]:
            best[record["track_id"]] = record

    shown = 0
    for track_id, record in sorted(best.items(), key=lambda kv: kv[1]["frames"]):
        track = tracks.get(track_id)
        if track is None:
            continue
        shown += 1

        started = time.perf_counter()
        line, backend, trajectory = reference_line(
            track,
            Recording.deserialize(record["recording"]),
            record["frames"],
            assets=assets,
            spacing=args.spacing,
        )
        build_seconds = time.perf_counter() - started

        seconds = record["frames"] / 1000.0
        speeds = trajectory.speed_kmh
        print(f"\n  {track.name or track_id[:12]}")
        print(f"    recorded    {format_time(record['frames'])}  "
              f"({len(trajectory)} samples, built in {build_seconds:.1f}s)")
        print(f"    line        {line.length:.1f} m over {len(line.centerline)} "
              f"points, {line.n_checkpoints} checkpoints")
        print(f"    pace        {line.length / seconds * 3.6:.0f} km/h average, "
              f"{speeds.max():.0f} km/h peak")

        env = PolyTrackEnv(track=line, backend=backend,
                           time_limit_ms=record["frames"] * 3)
        result = rollout(env, args.rollout_seconds)
        # How far the human had got by the same point in the run, for scale.
        at = int(np.searchsorted(trajectory.frames, env.elapsed_frames))
        human_arc, _ = line.project_many(trajectory.positions[at : at + 1])
        print(f"    naive       {result['arc']:.0f} m in "
              f"{env.elapsed_frames / 1000.0:.1f} s, reward {result['reward']:.0f}"
              f"   (the run itself: {float(human_arc[0]):.0f} m)")

    if not shown:
        print("No record in records.json matches an available track.")
        return 1
    print(f"\n{shown} track(s). The reference line is one lap, not the centre of")
    print("the road: 'lateral offset' is distance from that lap. Generated tracks")
    print("will supply their own route and will not need a lap driven for them.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
