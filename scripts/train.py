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

from polytrack.actions import Action
from polytrack.learning.collector import CollectorConfig, CollectorPool, EpisodeStats
from polytrack.learning.iqn import IQNAgent, IQNConfig
from polytrack.learning.replay import PrioritisedReplay
from polytrack.library import TrackLibrary

#: The three throttle actions. Everything else -- coast and brake, straight or
#: turning -- is removed by ``--forward-only``. Steering is untouched: the claim
#: the human records support is about the longitudinal input only.
FORWARD_ONLY = (
    int(Action.THROTTLE),
    int(Action.THROTTLE_LEFT),
    int(Action.THROTTLE_RIGHT),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument(
        "--track", action="append", default=None, metavar="NAME",
        help="train on this track only; repeat for several. Four tracks split "
             "the data four ways, so one track gets four times the episodes -- "
             "worth it while the question is still whether it can get round",
    )
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
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--forward-only", action="store_true",
        help="restrict the policy to the three throttle actions -- no coasting, "
             "no braking. Correct on Winter 5, where the 19.025 human lap holds "
             "throttle for 100%% of its frames and never brakes; wrong on every "
             "other track in records.json, which brake for 8-21%% of the lap. "
             "Sticks to the checkpoint, so resumes and evaluations inherit it",
    )
    actions.add_argument(
        "--all-actions", action="store_true",
        help="clear a restriction carried in a resumed checkpoint",
    )
    parser.add_argument("--control-interval-ms", type=float, default=50.0)
    parser.add_argument(
        "--time-limit-factor", type=float, default=3.0,
        help="episode budget as a multiple of the track's par time. 3x is right "
             "while the question is whether the car can get round at all; once "
             "it finishes regularly it is mostly waste, because a timeout costs "
             "the whole budget and a finish costs par. Under --forward-only it "
             "is worse still: a car that always accelerates keeps gaining ground "
             "while it pinballs off walls, so the stall cut never fires and the "
             "run burns the full budget at a fifth of par pace",
    )
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
    if args.track:
        library = library.select(args.track)

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    elapsed_offset = 0.0
    resumed_best: dict[str, int] = {}
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
            # Best times have to come forward too. Restarting the tracking each
            # block made the reported best jump back up whenever a block's first
            # finish was slower than an earlier block's -- so the headline number
            # was not the best at all, just the best since the last Ctrl+C.
            for row in history:
                for name, seconds in (row.get("best") or {}).items():
                    frames = int(round(seconds * 1000))
                    if frames < resumed_best.get(name, 1 << 62):
                        resumed_best[name] = frames
        print(f"\n  resumed from {args.resume} at {agent.steps:,} gradient steps"
              f" / {elapsed_offset / 60:.0f} min")
    else:
        agent = IQNAgent(IQNConfig(lr=args.lr, gamma=args.gamma))

    # An action restriction is part of the checkpoint, so a resume keeps whatever
    # it was trained under unless told otherwise. Changing it here rather than at
    # construction means --forward-only can be turned on partway through a run.
    if args.forward_only:
        agent.restrict(FORWARD_ONLY)
    elif args.all_actions:
        agent.restrict(None)

    restriction = agent.config.allowed_actions
    if restriction is not None and len(library) > 1:
        print("\n  WARNING: an action restriction is set on a run covering "
              f"{len(library)} tracks.\n  It is a per-track claim -- the human "
              "records brake for 8-21% of the lap on\n  every track except "
              "Winter 5. Pass --track to train one, or --all-actions.\n")

    collector_config = CollectorConfig(
        library_path=str(args.library.resolve()),
        track_names=tuple(args.track or ()),
        control_interval_ms=args.control_interval_ms,
        time_limit_factor=args.time_limit_factor,
        n_step=args.n_step,
        gamma=args.gamma,
        # The agent's config, not one rebuilt from the CLI. A resumed learner
        # takes its config from the checkpoint, so building the collectors' from
        # the flags instead let the two disagree -- and they must not: an action
        # the collectors can take but the learner masks is a transition the
        # critic is asked to value for a policy that never plays it.
        iqn_config=agent.config,
    )
    replay = PrioritisedReplay(
        capacity=args.capacity, obs_dim=agent.config.obs_dim
    )

    print(f"\n  {args.name}")
    print(library.describe())
    print(f"\n  {args.workers} collectors, batch {args.batch_size}, "
          f"n-step {args.n_step}, replay ratio {args.replay_ratio}")
    print(f"  episode budget {args.time_limit_factor:g}x par")
    print("  actions: " + (
        "all 9" if restriction is None
        else ", ".join(Action(a).name.lower() for a in restriction)
    ))
    print(f"  writing to {out}\n")

    pool = CollectorPool(args.workers, collector_config)
    pool.start()
    # Immediately, and before any collection. A worker builds its own network
    # from scratch and only replaces it when the learner broadcasts -- which
    # does not happen until the buffer holds --learn-after transitions. On a
    # resumed run that means the first ~100 seconds are driven by a random
    # policy and land in the buffer as if they meant something.
    pool.broadcast(agent.weights())

    started_steps = agent.steps
    started = time.monotonic()
    last_log = started
    collected = 0
    last_log_steps = agent.steps
    last_log_collected = 0
    last_broadcast = agent.steps
    losses: deque = deque(maxlen=200)
    episodes: deque = deque(maxlen=100)
    finishes = 0
    best_frames: dict[str, int] = dict(resumed_best)
    #: Fastest of this sitting, deliberately *not* carried forward: the point is
    #: to always come away with the best lap the collectors managed, whether or
    #: not it beat the record.
    session_best: dict[str, int] = {}
    #: Lowest median finish time seen, for keeping the best checkpoint rather
    #: than only the most recent one.
    best_median = float("inf")
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
                    # Keep the fastest lap of this sitting, not only laps that
                    # beat the all-time record. The collectors' epsilon noise is
                    # by far the best lap generator in this repo -- it has
                    # produced 19.126 on Winter 5 where a policy-guided beam
                    # search over seven runs and 2.5M simulated frames managed
                    # 19.405 -- and every one of those laps used to be discarded
                    # unless it beat the record outright. A search is only as
                    # good as the line it is seeded from, and this is where the
                    # good lines come from.
                    if item.recording and item.frames < session_best.get(
                        item.track, 1 << 62
                    ):
                        session_best[item.track] = item.frames
                        stem = item.track.replace(" ", "_").lower()
                        (out / f"fast_{stem}.json").write_text(json.dumps({
                            "track": item.track,
                            "frames": item.frames,
                            "recording": item.recording,
                            "gradient_steps": agent.steps,
                            "epsilon": item.epsilon,
                        }, indent=1))
                    previous = best_frames.get(item.track)
                    if previous is None or item.frames < previous:
                        best_frames[item.track] = item.frames
                        print(f"    * {item.track}: finished in "
                              f"{item.frames / 1000:.3f} s "
                              f"(worker {item.worker}, eps {item.epsilon:.3f})")
                        # Keep the run, not just the number. Without this a
                        # record lap exists only as a line in a log and can
                        # never be watched or verified.
                        if item.recording:
                            stem = item.track.replace(" ", "_").lower()
                            (out / f"best_{stem}.json").write_text(json.dumps({
                                "track": item.track,
                                "frames": item.frames,
                                "recording": item.recording,
                                "gradient_steps": agent.steps,
                                "epsilon": item.epsilon,
                            }, indent=1))

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
                # Compared against the last broadcast, not tested for divisibility.
                # The inner loop advances ``steps`` by up to 64 at a time and this
                # runs once afterwards, so ``steps % 200 == 0`` was only ever true
                # when a multiple of 64 happened to land on a multiple of 200 --
                # every 1600 steps rather than every 200. Under a thermally
                # throttled learner doing 4.5 steps/s that is a six-minute-old
                # policy in every collector, so the buffer fills with exploration
                # from a network the learner has long since moved past.
                if agent.steps - last_broadcast >= 200:
                    last_broadcast = agent.steps
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
                # Rates over the log interval, not since the block began. The
                # cumulative version is dominated for a long time by the
                # --learn-after warmup, during which the learner does nothing at
                # all, so a resumed run reads as starved for the best part of an
                # hour after it has fully recovered. That is not a cosmetic
                # difference: it is the number report.py raises the alarm on.
                window = max(now - last_log, 1e-9)
                finished_recently = [e.frames for e in episodes if e.finished]
                window_steps = agent.steps - last_log_steps
                window_trans = collected - last_log_collected
                last_log_steps, last_log_collected = agent.steps, collected
                row = {
                    "seconds": round(elapsed, 1),
                    "gradient_steps": agent.steps,
                    "transitions": collected,
                    "buffer": len(replay),
                    "transitions_per_second": round(window_trans / window, 1),
                    "steps_per_second": round(window_steps / window, 1),
                    # Gradient steps actually achieved per transition. If this
                    # sits well under --replay-ratio the learner is the
                    # bottleneck, not the collectors, and most of what they
                    # produce is never learned from.
                    "achieved_ratio": round(window_steps / max(window_trans, 1), 3),
                    "cumulative_ratio": round(
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
                    # The mean mixes every actor, and the noisiest runs at
                    # epsilon 0.4 -- a quarter of its actions are random, so it
                    # can never complete a lap however good the policy is. The
                    # near-greedy worker is the one whose progress answers "can
                    # this thing drive yet", so it gets its own column.
                    "greedy_fraction": (
                        round(float(np.mean(
                            [e.fraction for e in episodes
                             if e.epsilon == min(x.epsilon for x in episodes)]
                        )), 3) if episodes else None
                    ),
                    # How fast the laps that *did* finish were. "best" only moves
                    # on a record, so a run whose median finish is coming down
                    # from 30 s to 22 s looks identical in the log to one that is
                    # doing nothing -- and that is exactly the window where you
                    # want to know whether to keep paying for electricity.
                    "finish_median": (
                        round(float(np.median(finished_recently)) / 1000, 3)
                        if finished_recently else None
                    ),
                    "finish_best": (
                        round(min(finished_recently) / 1000, 3)
                        if finished_recently else None
                    ),
                    "finish_rate": round(len(finished_recently) / max(len(episodes), 1), 3),
                    "failures": {
                        reason: sum(1 for e in episodes if e.failure == reason)
                        for reason in sorted({e.failure for e in episodes})
                    },
                    "best": {k: v / 1000.0 for k, v in best_frames.items()},
                }
                history.append(row)

                # Keep the best-median checkpoint alongside the latest one.
                # Training on this project has regressed as well as improved: one
                # block moved the median +487 ms across its four quartiles, and
                # because agent.pt is overwritten every save, the policy from
                # before that block no longer exists anywhere. A second file
                # costs one write per improvement and makes a bad block
                # survivable instead of terminal.
                median = row["finish_median"]
                if median is not None and median < best_median:
                    best_median = median
                    agent.save(out / "best_agent.pt")
                    (out / "best_agent.json").write_text(json.dumps({
                        "gradient_steps": agent.steps,
                        "finish_median": median,
                        "finish_best": row["finish_best"],
                        "finish_rate": row["finish_rate"],
                        "seconds": row["seconds"],
                    }, indent=1))
                print(
                    f"  {elapsed / 60:6.1f} min | "
                    f"{agent.steps:>7,} steps ({row['steps_per_second']:.0f}/s) | "
                    f"{collected:>7,} trans ({row['transitions_per_second']:.0f}/s) | "
                    f"ratio {row['achieved_ratio']:.2f} | "
                    f"loss {row['loss'] if row['loss'] is not None else '--':>8} | "
                    f"lap {row['mean_fraction'] if row['mean_fraction'] is not None else '--':>5} "
                    f"(greedy {row['greedy_fraction'] if row['greedy_fraction'] is not None else '--':>5}) | "
                    f"fin {row['finish_rate'] * 100:>3.0f}% "
                    f"{row['finish_median'] if row['finish_median'] is not None else '--':>6} "
                    f"med / {row['finish_best'] if row['finish_best'] is not None else '--':>6} fastest | "
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
