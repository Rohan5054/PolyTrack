#!/usr/bin/env python3
"""Where does one lap lose time to another?

    python scripts/compare_laps.py --track "Winter 5" \
        --reference records.json:19025 --lap runs/searched/seed_19420.json

The question search should be pointed at, and the one a total time cannot
answer. A lap does not lose time evenly: the agent's 19.420 on Winter 5 is 395 ms
off the 19.025 human lap, and that deficit is two clusters -- 250-500 m and
1000-1300 m -- with the final third dead level. A uniform refinement sweep spent
ninety window positions on the opening acceleration and found nothing, which is
what should have been predicted.

Reports, at each 100 m of the reference line, when each run got there and how
much the gap grew over the previous 100 m. The ``dgap`` column is the one to
read: it is time lost *in that stretch*, not carried in from earlier.

Both runs are sampled with :func:`polytrack.reference.sample_replay`, which cuts
at input boundaries. Naively holding one mask per sample perturbs a human
recording -- inputs change mid-block -- and it drives off the road. That is not a
subtle failure; it just silently compares a lap against something that never
happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(spec: str):
    """``path`` for an exported run, or ``records.json:FRAMES`` to pick one."""
    path, _, frames = spec.partition(":")
    blob = json.loads(Path(path).read_text())
    if isinstance(blob, list):
        if not frames:
            options = ", ".join(str(r["frames"]) for r in blob)
            raise SystemExit(f"{path} holds several runs; pick one: {options}")
        wanted = int(frames)
        for record in blob:
            if record["frames"] == wanted:
                return record["recording"], wanted
        raise SystemExit(f"no run of {wanted} frames in {path}")
    return blob["recording"], int(blob["frames"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True)
    parser.add_argument("--reference", required=True,
                        help="the faster lap, e.g. records.json:19025")
    parser.add_argument("--lap", required=True, help="the lap being judged")
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("--every", type=float, default=100.0, metavar="M")
    args = parser.parse_args(argv)

    import numpy as np

    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.reference import sample_replay
    from polytrack.track import ProgressTracker

    entry = TrackLibrary.load(args.library).by_name(args.track)
    assets = GeometryAssets.load()

    def profile(spec):
        serialised, frames = load(spec)
        backend = WasmPhysics(entry.track_code(), assets=assets)
        trajectory = sample_replay(
            backend, Recording.deserialize(serialised), frames, sample_ms=50
        )
        tracker = ProgressTracker(entry.line)
        tracker.reset()
        arcs = np.array([tracker.update(p)[0] for p in trajectory.positions])
        return (
            arcs,
            np.array(trajectory.frames),
            np.array(trajectory.speed_kmh),
            trajectory.final_frame,
        )

    ref_arc, ref_frame, ref_speed, ref_end = profile(args.reference)
    lap_arc, lap_frame, lap_speed, lap_end = profile(args.lap)

    print(f"\n  {entry.name}: reference {ref_end / 1000:.3f} s, "
          f"lap {lap_end / 1000:.3f} s, {lap_end - ref_end:+,} ms\n")
    print("     arc      ref      lap     gap    dgap   ref km/h  lap km/h")

    previous = 0.0
    worst: list[tuple[float, float]] = []
    for metres in np.arange(args.every, entry.line.length, args.every):
        i = int(np.searchsorted(ref_arc, metres))
        j = int(np.searchsorted(lap_arc, metres))
        if i >= len(ref_frame) or j >= len(lap_frame):
            break
        gap = float(lap_frame[j] - ref_frame[i])
        delta = gap - previous
        previous = gap
        worst.append((delta, float(metres)))
        print(f"  {metres:6.0f} m {ref_frame[i] / 1000:7.2f}s "
              f"{lap_frame[j] / 1000:7.2f}s {gap:+6.0f} {delta:+7.0f}   "
              f"{ref_speed[i]:7.0f}   {lap_speed[j]:7.0f}")

    worst.sort(reverse=True)
    if worst:
        print("\n  Worst stretches, and the --focus window that covers each:")
        for delta, metres in worst[:4]:
            if delta <= 0:
                break
            j = int(np.searchsorted(lap_arc, metres))
            start = lap_frame[max(j - int(args.every / 20), 0)] / 1000
            print(f"    {metres - args.every:.0f}-{metres:.0f} m: {delta:+.0f} ms"
                  f"   --focus {start - 0.5:.1f},{lap_frame[j] / 1000 + 0.5:.1f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
