"""Trajectory search primitives.

This is the half of the design that the research review argued is PolyTrack's
real advantage over Trackmania. Because the physics is deterministic,
frame-stepped and cheap, and because the backend can save and restore state, an
input sequence can be *optimised directly* rather than only learned.

The evidence for needing both halves is Yosh's Trackmania result: reinforcement
learning got his agent to roughly human-world-record pace, but going past it
required test-time search and exploiting exact game state. A reactive policy
generalises to unseen tracks; a search finds the frame-perfect line on a known
one. Neither alone gets you a record.

Search deliberately talks to a :class:`~polytrack.physics.base.PhysicsBackend`
directly rather than going through the Gymnasium environment. It has no use for
observations -- it sees true state -- and skipping the observation builder is a
large constant-factor saving on the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..actions import ACTION_TO_MASK, N_ACTIONS
from ..physics.base import PhysicsBackend
from ..tas import TAS, format_time
from ..track import Track


@dataclass(slots=True)
class SearchResult:
    """Outcome of a search."""

    masks: list[int]
    """Per-frame input masks, ready for :meth:`TAS.from_frame_masks`."""

    frames: int
    """Frames elapsed. When ``finished``, this is the finish time."""

    finished: bool
    arc: float
    """Centerline progress reached, in metres."""

    evaluations: int = 0
    """Physics frames stepped during the search. The cost metric that matters."""

    def to_tas(self) -> TAS:
        return TAS.from_frame_masks(self.masks)

    @property
    def time(self) -> str:
        return format_time(self.frames)

    def __repr__(self) -> str:
        status = f"finished in {self.time}" if self.finished else f"reached {self.arc:.1f}m"
        return f"SearchResult({status}, {self.evaluations:,} frames evaluated)"


@dataclass(slots=True)
class Candidate:
    """One partial trajectory being carried through a search.

    ``blob`` is the saved simulator state, which is what makes breadth-first
    expansion affordable: without save/restore every candidate would have to be
    replayed from frame zero on each expansion, turning an O(depth) search into
    an O(depth^2) one.
    """

    blob: bytes
    masks: list[int]
    arc: float
    speed: float
    frame: int
    finished: bool = False
    terminated: bool = False

    def key(self, arc_bucket: float, speed_bucket: float) -> tuple[int, int]:
        """Coarse identity, used to prune near-duplicate candidates.

        Plain top-k beam search collapses: the k best trajectories at a given
        depth tend to be minor variations of one line, so the beam explores one
        idea very thoroughly and every other idea not at all. Bucketing on
        position *and* speed keeps genuinely different approaches alive -- most
        importantly the slow-in-fast-out line that loses ground early and wins
        it back through the corner exit.
        """
        return (int(self.arc / arc_bucket), int(self.speed / speed_bucket))


#: Signature of an optional action-prior function: given the candidate's arc and
#: speed, return a preference weight per action. This is the hook for the hybrid
#: design -- a trained policy supplies priors, and search only expands the
#: actions the policy considers plausible.
ActionPrior = Callable[[Candidate], np.ndarray]


def rollout(
    backend: PhysicsBackend,
    track: Track,
    masks: Sequence[int],
    *,
    from_blob: bytes | None = None,
    max_frames: int | None = None,
) -> Candidate:
    """Step a backend through ``masks`` and return the resulting candidate.

    Args:
        from_blob: Resume from a saved state instead of resetting. This is the
            whole point of the save/restore requirement.
        max_frames: Stop early regardless of how many masks remain.
    """
    if from_blob is None:
        state = backend.reset()
        arc, _ = track.project(state.position)
        speed = state.speed
        frame = backend.frame
        finished = terminated = False
    else:
        state = backend.load_state(from_blob)
        arc, _ = track.project(state.position)
        speed = state.speed
        frame = backend.frame
        finished = state.finished
        terminated = False

    applied: list[int] = []
    for i, mask in enumerate(masks):
        if max_frames is not None and i >= max_frames:
            break
        result = backend.step(mask)
        applied.append(mask)
        state = result.state
        frame = state.frame
        finished = result.finished
        terminated = result.terminated
        if finished or terminated:
            break

    arc, _ = track.project(state.position)
    return Candidate(
        blob=backend.save_state(),
        masks=applied,
        arc=arc,
        speed=state.speed,
        frame=frame,
        finished=finished,
        terminated=terminated,
    )


def expand_block(
    backend: PhysicsBackend,
    track: Track,
    parent: Candidate,
    action: int,
    block_frames: int,
) -> Candidate:
    """Hold one action for ``block_frames`` frames from a parent candidate.

    Searching one frame at a time gives a branching factor of 9 per frame, which
    is hopeless past a few dozen frames. Holding an action for a block trades
    some optimality for a tractable tree -- and matches how the inputs actually
    look, since human and TAS runs both hold keys for many frames at a time.
    Shrink ``block_frames`` in a refinement pass once a coarse line is found.
    """
    mask = ACTION_TO_MASK[action]
    backend.load_state(parent.blob)

    masks: list[int] = []
    state = None
    finished = terminated = False
    for _ in range(block_frames):
        result = backend.step(mask)
        masks.append(mask)
        state = result.state
        if result.finished or result.terminated:
            finished, terminated = result.finished, result.terminated
            break

    assert state is not None, "block_frames must be at least 1"
    arc, _ = track.project(state.position)
    return Candidate(
        blob=backend.save_state(),
        masks=parent.masks + masks,
        arc=arc,
        speed=state.speed,
        frame=state.frame,
        finished=finished,
        terminated=terminated,
    )


def prune(
    candidates: list[Candidate],
    beam_width: int,
    *,
    arc_bucket: float = 1.5,
    speed_bucket: float = 3.0,
) -> list[Candidate]:
    """Keep the best ``beam_width`` candidates, one per (position, speed) bucket.

    All candidates at a given search depth have stepped the *same number of
    frames*, so ranking them by centerline progress is an apples-to-apples
    comparison -- the fastest trajectory is simply the one that got furthest in
    equal time. That property is why the search expands in lockstep rather than
    letting candidates run to different depths.
    """
    alive = [c for c in candidates if not c.terminated]
    alive.sort(key=lambda c: c.arc, reverse=True)

    kept: list[Candidate] = []
    seen: set[tuple[int, int]] = set()
    for candidate in alive:
        bucket = candidate.key(arc_bucket, speed_bucket)
        if bucket in seen:
            continue
        seen.add(bucket)
        kept.append(candidate)
        if len(kept) >= beam_width:
            break

    # If bucketing was so aggressive that the beam starved, top up with the
    # best remaining candidates regardless of bucket.
    if len(kept) < beam_width:
        remaining = [c for c in alive if c not in kept]
        kept.extend(remaining[: beam_width - len(kept)])
    return kept


def uniform_prior(_: Candidate) -> np.ndarray:
    """Default prior: every action equally worth trying."""
    return np.ones(N_ACTIONS, dtype=np.float64)
