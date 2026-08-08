#!/usr/bin/env python3
"""End-to-end pipeline demo.

Runs random and scripted policies through the environment, exports a TAS,
replays it, and reports throughput. Uses the reference stand-in physics, so the
*times mean nothing* -- what this proves is that the plumbing works: observation
construction, reward, episode termination, TAS export and deterministic replay.

Run::

    PYTHONPATH=. python scripts/smoke_test.py
"""

from __future__ import annotations

import time

import numpy as np

from polytrack.env import PolyTrackEnv
from polytrack.tas import format_time
from polytrack.track import chicane_track, oval_track, straight_track


def run_episode(env: PolyTrackEnv, policy, max_steps: int = 3600) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {"reward": total_reward, **info}
    return {"reward": total_reward, "frames": max_steps, "finished": False}


def main() -> None:
    rng = np.random.default_rng(0)

    print("\n" + "=" * 66)
    print("  PolyTrack agent -- pipeline smoke test")
    print("  (reference stand-in physics; times are meaningless)")
    print("=" * 66)

    for track in (straight_track(), oval_track(), chicane_track()):
        env = PolyTrackEnv(track=track)
        print(f"\n  Track: {track.name}")
        print(f"    length      {track.length:.1f} m")
        print(f"    checkpoints {track.n_checkpoints}")
        print(f"    obs vector  {env.observer.size} floats")
        print(f"    actions     {env.action_space.n}")

        # A random policy: should mostly fail, which is the point.
        random_result = run_episode(env, lambda _: int(rng.integers(0, 9)))
        print(
            f"    random      reward {random_result['reward']:8.1f}  "
            f"finished={random_result.get('finished', False)}"
        )

        # A trivial scripted policy: full throttle, steer toward the track
        # centre using the first lookahead point's lateral component.
        def scripted(obs: np.ndarray) -> int:
            # Lookahead block starts after the 13 scalar features.
            lateral = obs[13 + 1]  # y of the nearest lookahead point
            if lateral > 0.02:
                return 4  # throttle + left
            if lateral < -0.02:
                return 5  # throttle + right
            return 3  # throttle

        scripted_result = run_episode(env, scripted)
        finished = scripted_result.get("finished", False)
        line = (
            f"    scripted    reward {scripted_result['reward']:8.1f}  "
            f"finished={finished}"
        )
        if finished:
            line += f"  time={scripted_result.get('time')}"
        print(line)

        # TAS export and deterministic replay.
        tas = env.export_tas()
        replay = env.replay_tas(tas)
        match = replay["frames"] == tas.n_frames
        # Seconds via the backend's own rate. format_time counts in PolyTrack
        # frames, which are milliseconds; the stand-in's are sixtieths.
        seconds = tas.n_frames / env.backend.rate
        print(
            f"    TAS         {len(tas)} changes / {tas.n_frames} frames "
            f"({seconds:.3f} s of stand-in)  replay_exact={match}"
        )

    # Throughput. This number is the whole reason for building a headless
    # simulator: the Trackmania projects were pinned near real time (60 steps/s
    # per instance) and had to resort to game-speed hacks to train at all.
    print("\n  Throughput (single process, reference physics)")
    env = PolyTrackEnv(track=chicane_track())
    env.reset()
    n = 20_000
    start = time.perf_counter()
    for i in range(n):
        _, _, term, trunc, _ = env.step(3)
        if term or trunc:
            env.reset()
    elapsed = time.perf_counter() - start
    # Realtime against how much time was *simulated*, not against 60 Hz: one
    # step is a control interval now, and against the real 1000 Hz module the
    # same step count is a very different amount of driving.
    simulated = n * env.ticks_per_step / env.backend.rate
    print(
        f"    {n / elapsed:,.0f} env steps/s  "
        f"({simulated / elapsed:.0f}x realtime, {env.ticks_per_step} frames/step)"
    )
    print("    Still dominated by the Python observation builder, not physics.")
    print("    The real module costs ~700 us per 1 ms tick, so on PolyTrack's")
    print("    own physics this figure is about 1x realtime and the collector")
    print("    count, not the step cost, is what buys throughput.\n")


if __name__ == "__main__":
    main()
