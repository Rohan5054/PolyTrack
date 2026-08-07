"""Beam search over action sequences.

Expands candidates in lockstep, block by block, keeping the most promising beam
at each depth. Because every candidate at a given depth has stepped an identical
number of frames, ranking by centerline progress is exactly ranking by speed --
no heuristic value function required, which is the luxury a perfect model buys.

Two-phase use is the intended pattern:

1. **Coarse pass** -- large ``block_frames`` (6-10), wide beam. Finds the racing
   line and, usually, a first completion.
2. **Refinement pass** -- :func:`refine` re-searches short windows of that
   solution at ``block_frames=1``, which is where the last tenths live.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..actions import N_ACTIONS
from ..physics.base import PhysicsBackend
from ..track import Track
from .base import (
    ActionPrior,
    Candidate,
    SearchResult,
    expand_block,
    prune,
    rollout,
    uniform_prior,
)


def beam_search(
    backend: PhysicsBackend,
    track: Track,
    *,
    beam_width: int = 24,
    block_frames: int = 6,
    max_frames: int = 3600,
    prior: ActionPrior | None = None,
    top_actions: int | None = None,
    arc_bucket: float = 1.5,
    speed_bucket: float = 3.0,
    progress_callback: Callable[[int, list[Candidate]], None] | None = None,
) -> SearchResult:
    """Search for a fast input sequence.

    Args:
        beam_width: Candidates carried to the next depth. Cost is linear in this.
        block_frames: Frames each action is held for. Smaller is finer and
            exponentially more expensive.
        max_frames: Give up after this many frames without finishing.
        prior: Optional action-preference function -- pass a trained policy here
            to guide the search. This is the hybrid design: the policy proposes,
            the search disposes.
        top_actions: Expand only the ``k`` highest-prior actions per candidate.
            With a good policy this is the single biggest cost saving available;
            with the uniform prior it just truncates arbitrarily, so leave it
            unset unless a real prior is supplied.
        progress_callback: Called as ``(depth, beam)`` after each expansion.

    Returns:
        The best finishing trajectory found, or the furthest-progressing one if
        nothing finished.
    """
    prior = prior or uniform_prior
    if top_actions is None:
        top_actions = N_ACTIONS

    start = rollout(backend, track, [])
    beam: list[Candidate] = [start]
    best_finish: Candidate | None = None
    evaluations = 0
    depth = 0

    while beam and depth * block_frames < max_frames:
        children: list[Candidate] = []

        for parent in beam:
            weights = np.asarray(prior(parent), dtype=np.float64)
            # Highest-prior actions first; ties broken by action index so the
            # search is deterministic and reproducible run to run.
            order = np.argsort(-weights, kind="stable")[:top_actions]

            for action in order:
                child = expand_block(backend, track, parent, int(action), block_frames)
                evaluations += child.frame - parent.frame

                if child.finished:
                    # Frames elapsed is the objective; keep only the quickest.
                    if best_finish is None or child.frame < best_finish.frame:
                        best_finish = child
                    continue  # a finished run has nothing left to expand
                if not child.terminated:
                    children.append(child)

        # A finish already found bounds the search: any candidate that cannot
        # beat it is dead weight.
        if best_finish is not None:
            children = [c for c in children if c.frame < best_finish.frame]

        beam = prune(
            children, beam_width, arc_bucket=arc_bucket, speed_bucket=speed_bucket
        )
        depth += 1
        if progress_callback is not None:
            progress_callback(depth, beam)

    if best_finish is not None:
        return SearchResult(
            masks=best_finish.masks,
            frames=best_finish.frame,
            finished=True,
            arc=best_finish.arc,
            evaluations=evaluations,
        )

    if not beam:
        return SearchResult(masks=[], frames=0, finished=False, arc=0.0,
                            evaluations=evaluations)

    furthest = max(beam, key=lambda c: c.arc)
    return SearchResult(
        masks=furthest.masks,
        frames=furthest.frame,
        finished=False,
        arc=furthest.arc,
        evaluations=evaluations,
    )


def refine(
    backend: PhysicsBackend,
    track: Track,
    masks: list[int],
    *,
    window_frames: int = 60,
    stride: int = 30,
    beam_width: int = 16,
    block_frames: int = 2,
) -> SearchResult:
    """Locally re-search overlapping windows of an existing solution.

    Slides a window along the trajectory, re-searching each window at finer
    granularity while holding everything after it fixed, and keeps a change only
    if it genuinely improves the outcome.

    This is the segmented optimisation that took Yosh's Trackmania agent past
    the human world record: the coarse policy or search finds the line, and
    per-segment refinement finds the hundredths. Windows overlap (``stride`` <
    ``window_frames``) because an improvement often needs a slightly earlier
    entry than the window boundary allows.
    """
    best = list(masks)
    baseline = _evaluate(backend, track, best)
    evaluations = 0

    start = 0
    while start < len(best):
        prefix = best[:start]
        suffix = best[start + window_frames:]

        # Replay the prefix once, then search only inside the window.
        anchor = rollout(backend, track, prefix)
        if anchor.terminated or anchor.finished:
            break

        local = _windowed_beam(
            backend,
            track,
            anchor,
            window_frames=window_frames,
            beam_width=beam_width,
            block_frames=block_frames,
        )
        evaluations += local["evaluations"]

        for window_masks in local["candidates"]:
            # Resume from the anchor rather than replaying the prefix: the
            # prefix is identical for every trial in this window, so replaying
            # it each time is exactly the O(depth^2) cost that save/restore
            # exists to avoid.
            tail = window_masks + suffix
            outcome = _evaluate_from(backend, track, anchor.blob, tail)
            evaluations += outcome["stepped"]
            if _better(outcome, baseline):
                best = prefix + tail
                baseline = outcome

        start += stride

    return SearchResult(
        masks=best,
        frames=baseline["frames"],
        finished=baseline["finished"],
        arc=baseline["arc"],
        evaluations=evaluations,
    )


def _windowed_beam(
    backend: PhysicsBackend,
    track: Track,
    anchor: Candidate,
    *,
    window_frames: int,
    beam_width: int,
    block_frames: int,
) -> dict:
    """Beam search confined to a window, returning the window's input segments."""
    beam = [
        Candidate(
            blob=anchor.blob,
            masks=[],
            arc=anchor.arc,
            speed=anchor.speed,
            frame=anchor.frame,
        )
    ]
    evaluations = 0
    steps = max(window_frames // block_frames, 1)

    for _ in range(steps):
        children: list[Candidate] = []
        for parent in beam:
            for action in range(N_ACTIONS):
                child = expand_block(backend, track, parent, action, block_frames)
                evaluations += child.frame - parent.frame
                if child.finished or not child.terminated:
                    children.append(child)
        if not children:
            break
        beam = prune(children, beam_width)

    return {
        "candidates": [c.masks for c in beam],
        "evaluations": evaluations,
    }


def _evaluate(backend: PhysicsBackend, track: Track, masks: list[int]) -> dict:
    """Replay a full input sequence from the start and report the outcome."""
    candidate = rollout(backend, track, masks)
    return {
        "frames": candidate.frame,
        "finished": candidate.finished,
        "arc": candidate.arc,
        "terminated": candidate.terminated,
        "stepped": candidate.frame,
    }


def _evaluate_from(
    backend: PhysicsBackend, track: Track, blob: bytes, masks: list[int]
) -> dict:
    """Evaluate a tail of inputs resuming from a saved state.

    ``frames`` is still the absolute finish frame, because the restored state
    carries its own frame counter -- so results stay directly comparable with
    :func:`_evaluate`. ``stepped`` reports only the frames actually simulated,
    which is what the evaluation budget should be charged.
    """
    before = backend.load_state(blob).frame
    candidate = rollout(backend, track, masks, from_blob=blob)
    return {
        "frames": candidate.frame,
        "finished": candidate.finished,
        "arc": candidate.arc,
        "terminated": candidate.terminated,
        "stepped": max(candidate.frame - before, 0),
    }


def _better(trial: dict, baseline: dict) -> bool:
    """Is ``trial`` an improvement?

    Finishing beats not finishing; among finishers, fewer frames wins; among
    non-finishers, more progress wins. Ordering these explicitly matters -- a
    naive "more arc is better" comparison would happily prefer a non-finishing
    run that overshot the finish line.
    """
    if trial["terminated"] and not trial["finished"]:
        return False
    if trial["finished"] and not baseline["finished"]:
        return True
    if not trial["finished"] and baseline["finished"]:
        return False
    if trial["finished"]:
        return trial["frames"] < baseline["frames"]
    return trial["arc"] > baseline["arc"] + 1e-9
