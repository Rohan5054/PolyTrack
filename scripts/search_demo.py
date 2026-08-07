#!/usr/bin/env python3
"""Search demo: the policy-plus-search hybrid, in miniature.

Runs three approaches on the same track and compares finish times:

1. **Flat out** -- hold throttle. The naive baseline.
2. **Scripted** -- a hand-written centerline-follower. Stands in for what a
   trained policy would provide: competent, generalises, not optimal.
3. **Beam search** -- test-time optimisation against the deterministic model.
4. **CEM, seeded from the scripted run** -- the actual hybrid: refine a good
   line rather than search from noise.

This is the design the research review landed on. Yosh's Trackmania agent
reached human-record level with RL, but going past it needed test-time search
and exact-state exploitation. Point 4 is that idea at small scale.

Uses the reference stand-in physics, so **the times are meaningless in
absolute terms**. What is meaningful is the *ordering* and the relative gaps:
those tell you the search layer is doing real work.

Run::

    PYTHONPATH=. python scripts/search_demo.py
"""

from __future__ import annotations

import time

import numpy as np

from polytrack.actions import ACTION_TO_MASK
from polytrack.physics.reference import ReferencePhysics
from polytrack.search import BeamSearch, CEMSearch, RefinementSearch
from polytrack.tas import TAS, format_time
from polytrack.track import ProgressTracker, Track, chicane_track


def run_policy(
    track: Track, policy, block_frames: int = 1
) -> tuple[TAS | None, int | None]:
    """Roll a policy out and return its TAS and finish frame.

    ``block_frames`` must match whatever the search layer uses. This is not a
    detail: a policy that acts every frame produces a run the block-based
    searchers cannot represent. Quantizing one such rollout to 6-frame blocks
    changed 47 of its 408 frames and the run stopped finishing entirely -- so
    both CEM and refinement were silently handed a broken seed and reported
    times worse than the run they were given.

    Keep action repeat consistent across policy and search, or convert
    explicitly and re-verify.
    """
    sim = ReferencePhysics(track)
    state = sim.reset()
    tracker = ProgressTracker(track)
    tracker.reset()
    masks: list[int] = []

    for _ in range(3600 // block_frames):
        action = policy(state, tracker.arc)
        mask = ACTION_TO_MASK[action]
        for _ in range(block_frames):
            masks.append(mask)
            result = sim.step(mask)
            state = result.state
            tracker.update(state.position)
            if result.finished:
                return TAS.from_frame_masks(masks), result.state.frame
            if result.terminated:
                return TAS.from_frame_masks(masks), None
    return TAS.from_frame_masks(masks), None


def flat_out(state, arc) -> int:
    return 3


def make_scripted(track: Track):
    """Aim at a point ahead on the centerline; lift off when badly misaligned."""

    def policy(state, arc) -> int:
        target = track.point_at(arc + 20.0)
        to_target = target - state.position
        right = np.cross(state.forward, np.array([0.0, 0.0, 1.0]))
        lateral = float(np.dot(to_target, right))
        heading = to_target / (np.linalg.norm(to_target) + 1e-9)
        throttle = 3 if float(np.dot(heading, state.forward)) > 0.9 else 0
        if lateral < -0.5:
            return throttle + 1
        if lateral > 0.5:
            return throttle + 2
        return throttle

    return policy


def show(label: str, frames: int | None, baseline: int | None, extra: str = "") -> None:
    if frames is None:
        print(f"  {label:<28} did not finish   {extra}")
        return
    line = f"  {label:<28} {format_time(frames):>9}"
    if baseline is not None and frames != baseline:
        delta = (frames - baseline) / 60.0
        line += f"   {delta:+.3f}s"
    else:
        line += "            "
    print(line + f"   {extra}")


def main() -> None:
    track = chicane_track()
    print("\n" + "=" * 72)
    print(f"  Search demo -- track '{track.name}', {track.length:.0f} m, "
          f"{track.n_checkpoints} checkpoints")
    print("  (reference stand-in physics: compare the gaps, not the times)")
    print("=" * 72 + "\n")

    _, flat_frames = run_policy(track, flat_out)
    show("1. flat out", flat_frames, None)

    scripted_tas, scripted_frames = run_policy(
        track, make_scripted(track), block_frames=6
    )
    show("2. scripted (6-frame blocks)", scripted_frames, flat_frames)

    t0 = time.perf_counter()
    beam = BeamSearch(
        ReferencePhysics(track), track, beam_width=48, block_frames=6
    ).run(max_frames=1800)
    beam_time = time.perf_counter() - t0
    show(
        "3. beam search",
        beam.frames if beam.finished else None,
        flat_frames,
        f"{beam.n_rollouts:,} rollouts, {beam_time:.1f}s",
    )

    if scripted_tas is not None and scripted_frames is not None:
        n_blocks = scripted_frames // 6 + 12
        t0 = time.perf_counter()
        cem = CEMSearch(
            ReferencePhysics(track), track, n_blocks=n_blocks, block_frames=6,
            population=96, seed=0,
        ).run(iterations=25, prior=scripted_tas)
        show(
            "4. CEM seeded from (2)",
            cem.frames if cem.finished else None,
            flat_frames,
            f"{cem.n_rollouts:,} rollouts, {time.perf_counter() - t0:.1f}s",
        )

        t0 = time.perf_counter()
        refined = RefinementSearch(
            ReferencePhysics(track), track, block_frames=6, window_blocks=8, seed=0,
        ).run(scripted_tas, iterations=600)
        show(
            "5. refinement of (2)",
            refined.frames if refined.finished else None,
            flat_frames,
            f"{refined.n_rollouts:,} rollouts, {time.perf_counter() - t0:.1f}s",
        )

    print("\n  Verifying the searched run replays identically...")
    if beam.tas is not None:
        clean = ReferencePhysics(track)
        clean.reset()
        final = None
        for mask in beam.tas.expand():
            final = clean.step(mask)
            if final.finished or final.terminated:
                break
        exact = final is not None and final.state.frame == beam.frames
        print(f"    replay frame {final.state.frame} vs searched {beam.frames}"
              f"  -> {'EXACT' if exact else 'MISMATCH'}")
        print("\n    This is the same check to run against the real game once the")
        print("    WASM backend exists: encode the TAS to a ghost and confirm the")
        print("    finish frame matches. A searched run that does not replay is a")
        print("    run that only ever existed inside the search.\n")


if __name__ == "__main__":
    main()
