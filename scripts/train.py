#!/usr/bin/env python3
"""Train an IQN policy against PolyTrack's own physics.

    POLYTRACK_TRACKS=path/to/tracks python scripts/build_library.py
    python scripts/train.py --workers 6

    python scripts/train.py --smoke      # 90 seconds, checks the plumbing

Collectors drive real episodes in their own processes; this process owns the
replay buffer and the gradient steps. The simulator is the bottleneck by a wide
margin -- about eleven transitions a second per collector -- so the learner
spends most of its time waiting, and the useful knob is ``--workers``, not
anything about the network.

**Expect the first useful behaviour in hours, not minutes.** A 20-second track is
about 400 decisions, so an episode costs ~35 seconds of wall clock per collector.

Checkpoints land in ``runs/<name>/``, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from polytrack.learning.collector import CollectorConfig, CollectorPool, EpisodeStats
from polytrack.learning.iqn import IQNAgent, IQNConfig
from polytrack.learning.replay import PrioritisedReplay
from polytrack.library import TrackLibrary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("--name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="an agent.pt to continue from. The replay buffer is not saved, so "
             "it refills before gradient steps restart",
    )
    # 128 rather than 256: the learner shares a machine with the collectors and
    # a step at 256 costs 68 ms, which cannot keep up with what they produce.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument(
        "--learn-after", type=int, default=5_000,
        help="transitions to collect before the first gradient step",
    )
    parser.add_argument(
        "--replay-ratio", type=float, default=1.0,
        help="gradient steps per transition collected. Higher than the usual "
             "0.25 because the simulator, not the learner, is the constraint",
    )
    parser.add_argument("--control-interval-ms", type=float, default=50.0)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    parser.add_argument("--checkpoint-every", type=int, default=20_000)
    parser.add_argument("--log-every", type=float, default=20.0)
    parser.add_argument("--learner-threads", type=int, default=2)
    parser.add_argument(
        "--smoke", action="store_true",
        help="two workers, 90 seconds, tiny buffer -- proves the loop runs",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.workers = min(args.workers, 2)
        args.max_seconds = 90.0
        args.learn_after = 200
        args.capacity = 20_000
        args.batch_size = 64
        args.log_every = 10.0
        args.name = "smoke"

    import torch

    torch.set_num_threads(args.learner_threads)

    if not args.library.exists():
        print(f"no {args.library}. Build it first:")
        print("  POLYTRACK_TRACKS=... python scripts/build_library.py")
        return 2
    library = TrackLibrary.load(args.library)

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    iqn_config = IQNConfig(lr=args.lr, gamma=args.gamma)
    collector_config = CollectorConfig(
        library_path=str(args.library.resolve()),
        control_interval_ms=args.control_interval_ms,
        n_step=args.n_step,
        gamma=args.gamma,
        iqn_config=iqn_config,
    )

    history: list[dict] = []
    elapsed_offset = 0.0
    if args.resume:
        agent = IQNAgent.load(args.resume)
        # Carry the previous block's log forward so a run split across several
        # sittings still plots as one curve. gradient_steps is already
        # continuous because it lives on the agent; seconds would restart at
        # zero without this, which makes the graph lie about the pace.
        earlier = args.resume.parent / "history.json"
        if earlier.exists():
            history = json.loads(earlier.read_text())
            elapsed_offset = history[-1]["seconds"] if history else 0.0
        print(f"\n  resumed from {args.resume} at {agent.steps:,} gradient steps"
              f" / {elapsed_offset / 60:.0f} min")
    else:
        agent = IQNAgent(iqn_config)
    replay = PrioritisedReplay(
        capacity=args.capacity, obs_dim=agent.config.obs_dim
    )

    print(f"\n  {args.name}")
    print(library.describe())
    print(f"\n  {args.workers} collectors, batch {args.batch_size}, "
          f"n-step {args.n_step}, replay ratio {args.replay_ratio}")
    print(f"  writing to {out}\n")

    pool = CollectorPool(args.workers, collector_config)
    pool.start()

    started_steps = agent.steps
    started = time.monotonic()
    last_log = started
    collected = 0
    losses: deque = deque(maxlen=200)
    episodes: deque = deque(maxlen=100)
    finishes = 0
    best_frames: dict[str, int] = {}
    pending_steps = 0.0

    try:
        while agent.steps - started_steps < args.max_steps:
            if time.monotonic() - started > args.max_seconds:
                break

            for transition in pool.drain_transitions():
                replay.add(*transition)
                collected += 1
                pending_steps += args.replay_ratio

            for item in pool.drain_stats():
                if isinstance(item, str):  # a worker died
                    print("\n" + item)
                    raise SystemExit(1)
                episodes.append(item)
                if item.finished:
                    finishes += 1
                    previous = best_frames.get(item.track)
                    if previous is None or item.frames < previous:
                        best_frames[item.track] = item.frames
                        print(f"    * {item.track}: finished in "
                              f"{item.frames / 1000:.3f} s "
                              f"(worker {item.worker}, eps {item.epsilon:.3f})")

            if len(replay) >= args.learn_after and pending_steps >= 1.0:
                for _ in range(int(min(pending_steps, 64))):
                    indices, weights, *batch = replay.sample(
                        args.batch_size, agent.steps
                    )
                    td_errors, loss = agent.learn(batch, weights)
                    replay.update_priorities(indices, td_errors)
                    losses.append(loss)
                    pending_steps -= 1.0
                    if agent.steps % max(args.checkpoint_every, 1) == 0:
                        agent.save(out / "agent.pt")
                if agent.steps % 200 == 0:
                    pool.broadcast(agent.weights())
            else:
                # Nothing to learn from yet, or the collectors are behind. This
                # is the normal state: the simulator sets the pace.
                time.sleep(0.02)

            now = time.monotonic()
            if now - last_log >= args.log_every:
                # Two clocks: this sitting, which sets the collection rate, and
                # the run as a whole, which is what the curve is plotted against.
                block = now - started
                elapsed = block + elapsed_offset
                row = {
                    "seconds": round(elapsed, 1),
                    "gradient_steps": agent.steps,
                    "transitions": collected,
                    "buffer": len(replay),
                    "transitions_per_second": round(collected / max(block, 1e-9), 1),
                    # Gradient steps actually achieved per transition. If this
                    # sits well under --replay-ratio the learner is the
                    # bottleneck, not the collectors, and most of what they
                    # produce is never learned from.
                    "achieved_ratio": round(
                        (agent.steps - started_steps) / max(collected, 1), 3
                    ),
                    "loss": round(float(np.mean(losses)), 4) if losses else None,
                    "episodes": len(episodes),
                    "finishes": finishes,
                    "mean_return": (
                        round(float(np.mean([e.reward for e in episodes])), 1)
                        if episodes else None
                    ),
                    "mean_fraction": (
                        round(float(np.mean([e.fraction for e in episodes])), 3)
                        if episodes else None
                    ),
                    "failures": {
                        reason: sum(1 for e in episodes if e.failure == reason)
                        for reason in sorted({e.failure for e in episodes})
                    },
                    "best": {k: v / 1000.0 for k, v in best_frames.items()},
                }
                history.append(row)
                print(
                    f"  {elapsed / 60:6.1f} min | "
                    f"{agent.steps:>7,} steps | "
                    f"{collected:>7,} trans ({row['transitions_per_second']:.1f}/s) | "
                    f"ratio {row['achieved_ratio']:.2f} | "
                    f"loss {row['loss'] if row['loss'] is not None else '--':>8} | "
                    f"lap {row['mean_fraction'] if row['mean_fraction'] is not None else '--':>5} | "
                    f"return {row['mean_return'] if row['mean_return'] is not None else '--':>7} | "
                    f"{row['episodes']} eps {row['failures']}"
                )
                (out / "history.json").write_text(json.dumps(history, indent=1))
                last_log = now
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        pool.close()
        agent.save(out / "agent.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    block = time.monotonic() - started
    print(f"\n  this block: {collected:,} transitions in {block / 60:.1f} min "
          f"({collected / max(block, 1e-9):.1f}/s)")
    print(f"  total: {agent.steps:,} gradient steps over "
          f"{(block + elapsed_offset) / 60:.0f} min")
    print(f"  {finishes} finish(es)"
          + (f": {best_frames}" if best_frames else ""))
    print(f"\n  continue with:")
    print(f"    python scripts/train.py --workers {args.workers} "
          f"--resume {out / 'agent.pt'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
