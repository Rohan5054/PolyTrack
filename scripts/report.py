#!/usr/bin/env python3
"""Where is a training run, and is it still getting better?

    python scripts/report.py                 # the newest run, from its log
    python scripts/report.py runs/foo        # a particular one
    python scripts/report.py --eval          # also make the agent drive

Reading the log answers "is it improving". ``--eval`` answers the different and
more interesting question "what can it actually do", by loading the checkpoint
and driving every track greedily -- no exploration noise, no epsilon. That is
the number to judge the run on, and it is not the one the training log shows:
the logged mean mixes all the actors, and the noisiest is taking a random action
a quarter of the time, so it can never complete a lap however good the policy
gets.

Nothing here writes anything. It is safe to run against a live run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def newest_run() -> Path | None:
    candidates = glob.glob("runs/*/history.json")
    if not candidates:
        return None
    return Path(max(candidates, key=os.path.getmtime)).parent


def curve(history: list[dict], rows: int = 10) -> None:
    """The run compressed to a readable number of rows, with a sparkline."""
    step = max(1, len(history) // rows)
    sample = history[::step] + [history[-1]]

    laps = [r.get("mean_fraction") or 0.0 for r in history]
    blocks = " .:-=+*#%@"
    peak = max(laps) or 1.0
    spark = "".join(blocks[min(int(v / peak * 9), 9)] for v in laps[::step])

    print(f"  lap  {spark}  (peak {peak:.3f})\n")
    # "fastest", not "best": these two columns are the median and minimum of the
    # last hundred episodes, so they move in both directions and are nothing to
    # do with the all-time record printed above. Reading one as the other makes
    # a run that is improving look like it went backwards a second.
    print("      time    steps   trans/s  ratio      lap   greedy   fin%"
          "   median  fastest   failures")
    for r in sample:
        greedy = r.get("greedy_fraction")
        # Runs started before a column existed simply do not have it.
        def opt(key, fmt="{:.3f}"):
            value = r.get(key)
            return fmt.format(value) if value is not None else "--"

        print(
            f"    {r['seconds'] / 60:6.1f}m {r['gradient_steps']:>8,} "
            f"{r['transitions_per_second']:>8.1f} "
            f"{r.get('achieved_ratio', 0):>6.2f} "
            f"{(r.get('mean_fraction') or 0):>8.3f} "
            f"{opt('greedy_fraction'):>8} "
            f"{(r.get('finish_rate') or 0) * 100:>5.0f} "
            f"{opt('finish_median'):>8} "
            f"{opt('finish_best'):>8}   "
            f"{_failures(r)}"
        )


def _failures(row: dict) -> str:
    counts = row.get("failures") or {}
    total = sum(counts.values()) or 1
    return " ".join(f"{k}={v * 100 // total}%" for k, v in sorted(counts.items()))


def verdict(history: list[dict]) -> None:
    """Still climbing, flat, or going backwards -- and the exploit check."""
    if len(history) < 8:
        print("\n  Too few log rows to judge a trend. Give it 10 minutes.")
        return

    half = len(history) // 2
    older = [r.get("mean_fraction") or 0.0 for r in history[half : half + len(history) // 4 or 1]]
    recent = [r.get("mean_fraction") or 0.0 for r in history[-len(history) // 4 or -1 :]]
    before, after = float(np.mean(older)), float(np.mean(recent))
    change = after - before

    print()
    if change > max(0.02 * before, 0.005):
        # Extrapolate, with the caveat that RL curves flatten and this one will.
        minutes = history[-1]["seconds"] / 60 - history[half]["seconds"] / 60
        rate = change / max(minutes, 1e-9)
        remaining = (0.9 - after) / rate if rate > 0 else float("inf")
        print(f"  CLIMBING. Lap fraction {before:.3f} -> {after:.3f} over the "
              f"last {minutes:.0f} min.")
        if 0 < remaining < 6000:
            print(f"  At this rate 0.9 is ~{remaining / 60:.1f} h away, but the "
                  "rate will fall -- the\n  last stretch of a track is always "
                  "the expensive part. Treat it as a floor.")
    elif change < -max(0.05 * before, 0.005):
        print(f"  FALLING. Lap fraction {before:.3f} -> {after:.3f}.")
        _exploit_check(history)
    else:
        print(f"  FLAT. Lap fraction {before:.3f} -> {after:.3f}. Either it has "
              "learned what it\n  can from these four tracks, or something is "
              "stuck -- run with --eval to see\n  which.")

    # Averaged over the last few rows: since these are now per-interval rates
    # rather than since-block-start ones, a single row is noisy enough to raise
    # a false alarm on its own.
    recent_ratios = [r["achieved_ratio"] for r in history[-5:] if "achieved_ratio" in r]
    ratio = float(np.mean(recent_ratios)) if recent_ratios else None
    if ratio is not None and ratio < 0.3:
        print(f"\n  Learner is starved: {ratio:.2f} gradient steps per "
              "transition. Most of what\n  the collectors produce is never "
              "learned from. Fewer workers or a smaller\n  batch would both "
              "help.")


def _exploit_check(history: list[dict]) -> None:
    """Progress falling while failures shift is the signature to watch for.

    It is what the first real run did: it found that failing quickly cost less
    than staying alive, and learned to drive off the line at once. Reward
    exploits look like improvement on every axis except the one you care about.
    """
    def share(row, key):
        counts = row.get("failures") or {}
        return counts.get(key, 0) / max(sum(counts.values()), 1)

    first, last = history[len(history) // 2], history[-1]
    shift = share(last, "off_line") - share(first, "off_line")
    if shift > 0.2 and (last.get("mean_return") or 0) >= (first.get("mean_return") or 0):
        print("\n  WARNING: off_line failures rose "
              f"{shift * 100:.0f} points while return did not fall.")
        print("  That is the reward-exploit signature -- the agent is being paid "
              "for something\n  other than driving. Stop and check the reward "
              "before spending more hours.")


def evaluate(run: Path, library_path: Path, only=None) -> None:
    """Drive every track with the checkpoint, greedily. The real answer."""
    from polytrack.env import PolyTrackEnv
    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.tas import format_time

    checkpoint = run / "agent.pt"
    if not checkpoint.exists():
        print(f"\n  no {checkpoint} to evaluate")
        return

    import torch

    torch.set_num_threads(2)
    agent = IQNAgent.load(checkpoint)
    library = TrackLibrary.load(library_path)
    if only:
        library = library.select(only)
    assets = GeometryAssets.load()

    print(f"\n  Greedy evaluation at {agent.steps:,} gradient steps "
          "(no exploration)\n")
    print("      track            lap      time     par    outcome")
    for entry in library:
        backend = WasmPhysics(entry.track_code(), assets=assets)
        env = PolyTrackEnv(
            track=entry.line, backend=backend,
            time_limit_ms=entry.par_frames * 3, par_frames=entry.par_frames,
        )
        observation, _ = env.reset()
        info: dict = {}
        while True:
            observation, _, terminated, truncated, info = env.step(
                agent.act(observation, epsilon=0.0)
            )
            if terminated or truncated:
                break
        fraction = info.get("arc", 0.0) / entry.line.length
        outcome = "FINISHED" if info.get("finished") else info.get("failure", "?")
        print(f"    {entry.name:<14} {fraction * 100:5.1f}%  "
              f"{format_time(env.elapsed_frames)}  "
              f"{format_time(entry.par_frames)}   {outcome}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=None)
    parser.add_argument("--eval", action="store_true",
                        help="load the checkpoint and drive each track greedily")
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("--track", action="append", default=None, metavar="NAME",
                        help="evaluate only this track; repeat for several")
    parser.add_argument("--rows", type=int, default=10)
    args = parser.parse_args()

    run = args.run or newest_run()
    if run is None:
        print("no runs found. Start one with scripts/train.py")
        return 2
    path = run / "history.json"
    if not path.exists():
        print(f"no {path}")
        return 2
    history = json.loads(path.read_text())
    if not history:
        print(f"{path} is empty -- the run has not logged yet")
        return 1

    last = history[-1]
    print(f"\n  {run.name}")
    print(f"  {last['seconds'] / 60:.0f} min, {last['gradient_steps']:,} gradient "
          f"steps, {last['transitions']:,} transitions, "
          f"{last.get('finishes', 0)} finish(es)")
    if last.get("best"):
        print(f"  record: {last['best']}   (all-time; the table's two time "
              "columns are the last 100 episodes)")
    print()
    curve(history, args.rows)
    verdict(history)

    if args.eval:
        evaluate(run, args.library, args.track)
    else:
        print("\n  --eval loads the checkpoint and drives each track greedily, "
              "which is the\n  only honest measure of what it can do. Costs a "
              "lap per track.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
