"""Trajectory search over input sequences.

This is the half of the design that a reactive policy cannot do, and the half
that PolyTrack makes practical in a way Trackmania never did.

The reasoning: PolyTrack's physics is deterministic and frame-stepped, and a
backend that implements save/restore gives us a perfect model with random
access. That turns "find the fastest run on this track" from a learning problem
into an *optimisation* problem. The F1TENTH survey (Evans et al. 2024) found
that optimisation-and-tracking beat end-to-end RL on pure lap time; Yosh's
Trackmania agent reached human-record level with RL but needed test-time search
and exact-state exploitation to go past it.

So: the policy generalises and discovers, the search optimises.

Two methods here, with different strengths:

* :class:`BeamSearch` -- expands frame blocks breadth-first, keeping the best
  candidates at each depth. Because every candidate at a given depth has burned
  exactly the same number of frames, maximising progress at each depth *is*
  minimising time to reach a given point on the track. Needs save/restore.
* :class:`CEMSearch` -- samples whole input sequences from a per-block
  categorical distribution, keeps the elite fraction, refits. Needs only
  reset/step, so it works against a backend whose save/restore is not yet
  trustworthy, and it parallelises trivially.

Both emit :class:`~polytrack.tas.TAS` objects, so their output can be verified
in the real game like any other run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .actions import ACTION_TO_MASK, N_ACTIONS
from .physics.base import PhysicsBackend
from .tas import TAS, format_time
from .track import ProgressTracker, Track


@dataclass(slots=True)
class Candidate:
    """One partial trajectory under consideration."""

    blob: bytes  # simulator state after the last block
    frame: int
    checkpoint: int
    arc: float
    actions: tuple[int, ...]  # one action per block, not per frame
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    finished: bool = False
    terminated: bool = False

    @property
    def score(self) -> tuple[int, float]:
        """Progress, ordered. Checkpoints dominate arc length.

        Checkpoint count comes first because arc length alone is ambiguous on
        tracks that cross or double back on themselves -- a car that has cut a
        corner can show a higher arc than one that legitimately passed the
        checkpoint.
        """
        return (self.checkpoint, self.arc)


@dataclass(slots=True)
class SearchResult:
    """Outcome of a search."""

    tas: TAS | None
    frames: int | None
    finished: bool
    n_rollouts: int
    n_frames_simulated: int
    best_arc: float
    history: list[float] = field(default_factory=list)
    #: Ticks per second of the backend that produced ``frames``. PolyTrack's
    #: 1000 is the default because a frame count only reads out as a time at all
    #: when a frame is a millisecond; the stand-in's 60 Hz frames have to be
    #: converted, and a result that does not carry its own rate cannot be.
    rate: float = 1000.0

    @property
    def time(self) -> str:
        if self.frames is None:
            return "--"
        return format_time(round(self.frames * 1000.0 / self.rate))

    def summary(self) -> str:
        status = f"finished in {self.time}" if self.finished else (
            f"did not finish (reached {self.best_arc:.1f} m)"
        )
        return (
            f"{status} | {self.n_rollouts:,} rollouts, "
            f"{self.n_frames_simulated:,} frames simulated"
        )


class BeamSearch:
    """Breadth-first search over blocks of held inputs, pruned to a beam.

    Args:
        backend: Any :class:`PhysicsBackend`. Must implement save/restore.
        track: Used to measure progress; the backend is not asked for it, so
            this works against backends that expose nothing but car state.
        beam_width: Candidates kept at each depth. The main cost/quality knob.
        block_frames: Frames each chosen action is held for. This is the single
            most important parameter -- searching one action per frame is
            hopeless (9^N), while holding for ~6 frames (0.1 s) makes a
            10-second run about 100 decisions.
        actions: Restrict the action set if you know some are never useful.
    """

    def __init__(
        self,
        backend: PhysicsBackend,
        track: Track,
        beam_width: int = 48,
        block_frames: int = 6,
        actions: tuple[int, ...] | None = None,
        dedup_position: float = 1.0,
        dedup_velocity: float = 2.0,
        window: float = 60.0,
    ) -> None:
        self.backend = backend
        self.track = track
        self.beam_width = beam_width
        self.block_frames = block_frames
        self.actions = actions or tuple(range(N_ACTIONS))
        self.dedup_position = dedup_position
        self.dedup_velocity = dedup_velocity
        self.window = window
        self._rollouts = 0
        self._frames = 0

    def run(
        self,
        max_frames: int = 3600,
        on_depth: Callable[[int, Candidate], None] | None = None,
    ) -> SearchResult:
        """Search until a candidate finishes or the frame budget runs out.

        Returns as soon as any candidate finishes. Because all candidates at a
        given depth have used identical frame counts, the first finisher is the
        fastest at that block granularity -- refining it further is the job of
        :meth:`polish`.
        """
        self._rollouts = 0
        self._frames = 0

        state = self.backend.reset()
        arc, _ = self.track.project(state.position)
        beam = [
            Candidate(
                blob=self.backend.save_state(),
                frame=0,
                checkpoint=state.checkpoint,
                arc=arc,
                actions=(),
                position=state.position.copy(),
                velocity=state.velocity.copy(),
            )
        ]
        history: list[float] = []
        best_arc = arc
        max_depth = max_frames // self.block_frames

        for depth in range(max_depth):
            children: list[Candidate] = []
            for parent in beam:
                for action in self.actions:
                    child = self._expand(parent, action)
                    if child.terminated:
                        continue
                    if child.finished:
                        # First finisher wins: identical frame budget across
                        # the whole depth means nothing still in the beam can
                        # do better without searching another depth.
                        return self._result(child, history, finished=True)
                    children.append(child)

            if not children:
                break

            beam = self._prune(children)
            best = beam[0]
            best_arc = max(best_arc, best.arc)
            history.append(best.arc)
            if on_depth is not None:
                on_depth(depth, best)

        best = max(beam, key=lambda c: c.score) if beam else None
        if best is None:
            return SearchResult(
                None, None, False, self._rollouts, self._frames, best_arc,
                rate=self.backend.rate,
            )
        return self._result(best, history, finished=False)

    def _expand(self, parent: Candidate, action: int) -> Candidate:
        """Run one block of frames from a parent state with one held action."""
        self.backend.load_state(parent.blob)
        mask = ACTION_TO_MASK[action]
        result = None
        for _ in range(self.block_frames):
            result = self.backend.step(mask)
            self._frames += 1
            if result.finished or result.terminated:
                break
        self._rollouts += 1

        assert result is not None
        # Project relative to the parent's arc: each branch tracks its own
        # progress, which is what makes this correct on looping tracks.
        arc, _ = self.track.project(
            result.state.position, near_arc=parent.arc, radius=self.window
        )
        return Candidate(
            blob=self.backend.save_state(),
            frame=result.state.frame,
            checkpoint=result.state.checkpoint,
            arc=arc,
            actions=parent.actions + (action,),
            position=result.state.position.copy(),
            velocity=result.state.velocity.copy(),
            finished=result.finished,
            terminated=result.terminated,
        )

    def _prune(self, children: list[Candidate]) -> list[Candidate]:
        """Keep the best candidates, discarding near-duplicate states.

        Deduplication matters more than beam width here. Without it the beam
        fills with dozens of trajectories that reached effectively the same
        place at the same speed by trivially different routes, and the search
        explores one narrow corridor instead of genuinely different lines.
        """
        children.sort(key=lambda c: c.score, reverse=True)
        kept: list[Candidate] = []
        seen: set[tuple] = set()
        for child in children:
            key = self._bucket(child)
            if key in seen:
                continue
            seen.add(key)
            kept.append(child)
            if len(kept) >= self.beam_width:
                break
        return kept

    def _bucket(self, candidate: Candidate) -> tuple:
        """Coarse state fingerprint used for deduplication.

        Reads the position and velocity captured when the candidate was
        created. An earlier version reloaded the saved state and stepped the
        simulator just to read these back -- which both cost an extra physics
        frame per candidate and fingerprinted the state one frame *after* the
        one being compared.
        """
        pos = np.round(candidate.position / self.dedup_position).astype(int)
        vel = np.round(candidate.velocity / self.dedup_velocity).astype(int)
        return (candidate.checkpoint, *pos.tolist(), *vel.tolist())

    def _result(
        self, candidate: Candidate, history: list[float], finished: bool
    ) -> SearchResult:
        return SearchResult(
            tas=self.to_tas(candidate),
            frames=candidate.frame,
            finished=finished,
            n_rollouts=self._rollouts,
            n_frames_simulated=self._frames,
            best_arc=candidate.arc,
            history=history,
            rate=self.backend.rate,
        )

    def to_tas(self, candidate: Candidate) -> TAS:
        """Expand a block-level action list into a frame-level TAS."""
        masks: list[int] = []
        for action in candidate.actions:
            masks.extend([ACTION_TO_MASK[action]] * self.block_frames)
        return TAS.from_frame_masks(masks[: candidate.frame])


class CEMSearch:
    """Cross-entropy method over block-level action sequences.

    Maintains a categorical distribution over actions for each block, samples
    whole sequences, keeps the elite fraction, and refits. Slower to converge
    than beam search on a fresh track, but it needs only reset/step -- which
    makes it the right tool while a new backend's save/restore is still
    unproven, and the right tool for parallelising across many worker processes.

    It is also the natural way to *refine* a policy rollout: seed the
    distribution from the policy's own actions and let CEM polish it.
    """

    def __init__(
        self,
        backend: PhysicsBackend,
        track: Track,
        n_blocks: int,
        block_frames: int = 6,
        population: int = 96,
        elite_frac: float = 0.15,
        smoothing: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.track = track
        self.n_blocks = n_blocks
        self.block_frames = block_frames
        self.population = population
        self.n_elite = max(2, int(population * elite_frac))
        self.smoothing = smoothing
        self.rng = np.random.default_rng(seed)
        self._rollouts = 0
        self._frames = 0

    def run(self, iterations: int = 20, prior: TAS | None = None) -> SearchResult:
        """Optimise for a fixed number of CEM iterations.

        Args:
            prior: Optional starting sequence -- typically a policy rollout.
                The distribution is biased toward it, so CEM refines a good
                line rather than searching from uniform noise.
        """
        self._rollouts = 0
        self._frames = 0
        probs = np.full((self.n_blocks, N_ACTIONS), 1.0 / N_ACTIONS)
        best_score = (-1, -math.inf)
        best_sequence: np.ndarray | None = None
        best_frames: int | None = None
        best_finished = False
        history: list[float] = []

        if prior is not None:
            probs = self._seed_from(prior)
            # Evaluate the prior directly so it is the incumbent from the very
            # first iteration. Otherwise CEM spends its early budget rediscovering
            # a run it was handed for free.
            best_sequence = self._sequence_from(prior)
            checkpoint, arc, frames, finished = self._evaluate(best_sequence)
            best_score = (checkpoint + 1, -frames) if finished else (checkpoint, arc)
            best_frames = frames
            best_finished = finished

        for _ in range(iterations):
            sequences = self._sample(probs)
            if best_sequence is not None:
                # Elitism: carry the incumbent into every population.
                #
                # Without this, CEM can finish *worse* than the run it was
                # seeded from. A soft seed gives each block ~86% chance of
                # keeping its prior action, which across 79 blocks is a 1-in-
                # 100,000 chance of ever reproducing the seed -- and one wrong
                # block mid-corner ends the run. Re-inserting the best-so-far
                # makes non-regression structural rather than lucky.
                sequences[0] = best_sequence

            scored = []
            for sequence in sequences:
                checkpoint, arc, frames, finished = self._evaluate(sequence)
                # Finishers are ranked by speed; non-finishers by distance.
                key = (
                    (checkpoint + 1, -frames) if finished else (checkpoint, arc)
                )
                scored.append((key, finished, frames, sequence))

            scored.sort(key=lambda item: item[0], reverse=True)
            top_key, top_finished, top_frames, top_sequence = scored[0]
            if top_key > best_score:
                best_score = top_key
                best_sequence = top_sequence
                best_frames = top_frames
                best_finished = top_finished

            elite = np.stack([item[3] for item in scored[: self.n_elite]])
            probs = self._refit(probs, elite)
            history.append(float(best_score[1]))

        tas = None
        if best_sequence is not None:
            masks = np.repeat(
                [ACTION_TO_MASK[a] for a in best_sequence], self.block_frames
            )
            tas = TAS.from_frame_masks(masks[:best_frames].tolist())

        return SearchResult(
            tas=tas,
            frames=best_frames,
            finished=best_finished,
            n_rollouts=self._rollouts,
            n_frames_simulated=self._frames,
            best_arc=float(best_score[1]) if not best_finished else 0.0,
            history=history,
            rate=self.backend.rate,
        )

    def _seed_from(self, prior: TAS, confidence: float = 0.9) -> np.ndarray:
        """Bias the initial distribution toward an existing run.

        ``confidence`` is the per-block probability of keeping the prior's
        action. It must be high: the chance of reproducing the whole prior is
        ``confidence ** n_blocks``, so 0.86 across 79 blocks is effectively
        zero. Even at 0.9 the seed is rarely sampled intact, which is why
        elitism in :meth:`run` carries the incumbent forward regardless.
        """
        from .actions import mask_to_action

        spread = (1.0 - confidence) / (N_ACTIONS - 1)
        probs = np.full((self.n_blocks, N_ACTIONS), spread)
        masks = list(prior.expand(self.n_blocks * self.block_frames))
        for block in range(self.n_blocks):
            index = block * self.block_frames
            if index < len(masks):
                probs[block, :] = spread
                probs[block, mask_to_action(masks[index])] = confidence
        return probs / probs.sum(axis=1, keepdims=True)

    def _sequence_from(self, tas: TAS) -> np.ndarray:
        """Read a TAS back into a block-level action sequence."""
        from .actions import mask_to_action

        masks = list(tas.expand(self.n_blocks * self.block_frames))
        return np.array(
            [
                mask_to_action(masks[b * self.block_frames])
                if b * self.block_frames < len(masks)
                else 0
                for b in range(self.n_blocks)
            ]
        )

    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Draw ``population`` action sequences from the per-block categoricals."""
        cumulative = np.cumsum(probs, axis=1)
        draws = self.rng.random((self.population, self.n_blocks, 1))
        return (draws > cumulative[None, :, :]).sum(axis=2)

    def _evaluate(self, sequence: np.ndarray) -> tuple[int, float, int, bool]:
        state = self.backend.reset()
        # A fresh tracker per rollout: progress is path-dependent, so it cannot
        # be shared between candidates that took different routes.
        progress = ProgressTracker(self.track)
        progress.update(state.position)
        result = None
        for action in sequence:
            mask = ACTION_TO_MASK[int(action)]
            for _ in range(self.block_frames):
                result = self.backend.step(mask)
                progress.update(result.state.position)
                self._frames += 1
                if result.finished or result.terminated:
                    break
            if result is not None and (result.finished or result.terminated):
                break
        self._rollouts += 1

        assert result is not None
        arc = progress.arc
        return (
            result.state.checkpoint,
            arc,
            result.state.frame,
            result.finished,
        )

    def _refit(self, probs: np.ndarray, elite: np.ndarray) -> np.ndarray:
        """Move the distribution toward the elite sample's action frequencies."""
        counts = np.zeros_like(probs)
        for block in range(self.n_blocks):
            observed = np.bincount(elite[:, block], minlength=N_ACTIONS)
            counts[block] = observed
        empirical = (counts + 1.0) / (counts.sum(axis=1, keepdims=True) + N_ACTIONS)
        # Smoothing keeps the distribution from collapsing onto the first good
        # sequence it finds, which is the classic CEM failure mode.
        return self.smoothing * empirical + (1.0 - self.smoothing) * probs


class RefinementSearch:
    """Segment-wise hill climbing on an existing run.

    The right tool for the last tenths, and the one that matches how TAS work is
    actually done by hand: take a run that already works, perturb a short
    contiguous stretch of it, keep the change if the run got faster, discard it
    otherwise. Repeat, walking the window along the track.

    This succeeds where whole-sequence CEM struggles because it never risks the
    parts of the run that are already good. Resampling every block at once means
    a single bad choice anywhere ruins the whole attempt, and the probability of
    that grows with track length; perturbing eight blocks at a time keeps the
    failure surface small no matter how long the run is.

    It is also the natural consumer of a trained policy: hand it the policy's
    rollout and it polishes it, which is the "learned policy plus test-time
    search" split the whole design rests on.
    """

    def __init__(
        self,
        backend: PhysicsBackend,
        track: Track,
        block_frames: int = 6,
        window_blocks: int = 8,
        seed: int = 0,
    ) -> None:
        self.backend = backend
        self.track = track
        self.block_frames = block_frames
        self.window_blocks = window_blocks
        self.rng = np.random.default_rng(seed)
        self._rollouts = 0
        self._frames = 0

    def run(self, prior: TAS, iterations: int = 400) -> SearchResult:
        """Polish ``prior``. Returns the best run found, never worse than the input."""
        self._rollouts = 0
        self._frames = 0

        n_blocks = math.ceil(prior.n_frames / self.block_frames) + 4
        sequence = self._sequence_from(prior, n_blocks)
        best_key, best_frames, best_finished = self._evaluate(sequence)
        history: list[float] = []

        for _ in range(iterations):
            candidate = sequence.copy()
            start = int(self.rng.integers(0, max(n_blocks - self.window_blocks, 1)))
            span = min(self.window_blocks, n_blocks - start)
            # Perturb a contiguous window. Contiguity matters: racing inputs are
            # correlated in time, so scattering single-block mutations mostly
            # produces noise that any good line immediately rejects.
            candidate[start : start + span] = self.rng.integers(0, N_ACTIONS, size=span)

            key, frames, finished = self._evaluate(candidate)
            if key > best_key:
                sequence, best_key, best_frames, best_finished = (
                    candidate,
                    key,
                    frames,
                    finished,
                )
            history.append(float(best_key[1]))

        masks = np.repeat(
            [ACTION_TO_MASK[int(a)] for a in sequence], self.block_frames
        )
        return SearchResult(
            tas=TAS.from_frame_masks(masks[:best_frames].tolist()),
            frames=best_frames,
            finished=best_finished,
            n_rollouts=self._rollouts,
            n_frames_simulated=self._frames,
            best_arc=float(best_key[1]),
            history=history,
            rate=self.backend.rate,
        )

    def _sequence_from(self, tas: TAS, n_blocks: int) -> np.ndarray:
        from .actions import mask_to_action

        masks = list(tas.expand(n_blocks * self.block_frames))
        return np.array(
            [
                mask_to_action(masks[b * self.block_frames])
                if b * self.block_frames < len(masks)
                else 0
                for b in range(n_blocks)
            ]
        )

    def _evaluate(self, sequence: np.ndarray) -> tuple[tuple, int, bool]:
        state = self.backend.reset()
        progress = ProgressTracker(self.track)
        progress.reset()
        progress.update(state.position)
        result = None
        for action in sequence:
            mask = ACTION_TO_MASK[int(action)]
            for _ in range(self.block_frames):
                result = self.backend.step(mask)
                progress.update(result.state.position)
                self._frames += 1
                if result.finished or result.terminated:
                    break
            if result is not None and (result.finished or result.terminated):
                break
        self._rollouts += 1

        assert result is not None
        frames = result.state.frame
        if result.finished:
            # Finishers rank above everything, fastest first.
            key = (result.state.checkpoint + 1, -float(frames))
        else:
            key = (result.state.checkpoint, progress.arc)
        return key, frames, result.finished
