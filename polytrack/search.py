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
    #: Full car state and offset from the line, kept only so an action prior can
    #: be asked what to try from here. ``None`` when no prior is in use.
    state: object | None = None
    lateral: float = 0.0
    #: Consecutive blocks this branch has gained no ground. A car in a ditch is
    #: just a car in a ditch as far as the module is concerned -- it never
    #: reports ``terminated`` -- so this is the only thing that tells a beam its
    #: candidate is dead rather than merely slow.
    stalled: int = 0

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
        prior: ``f(state, arc, lateral) -> array of N_ACTIONS values``, highest
            first. A trained policy, normally. Only the ``top_k`` best-rated
            actions are expanded from each parent, which is what makes the full
            nine-action space affordable: the branching factor stops being a
            property of the game and becomes a parameter.
        top_k: How many of the prior's suggestions to expand. Ignored without a
            prior. Keep it above one -- at one the search only ever replays the
            policy, and the whole point is to look where the policy would not.
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
        prior: Callable | None = None,
        top_k: int = 3,
        stall_blocks: int = 8,
        stall_metres: float = 0.5,
        reserve: int = 0,
        reserve_band: float = 8.0,
    ) -> None:
        self.backend = backend
        self.track = track
        self.beam_width = beam_width
        self.block_frames = block_frames
        self.actions = actions or tuple(range(N_ACTIONS))
        self.dedup_position = dedup_position
        self.dedup_velocity = dedup_velocity
        self.window = window
        self.prior = prior
        self.top_k = max(1, int(top_k))
        #: Drop a branch after this many consecutive blocks gaining less than
        #: ``stall_metres``. Zero disables it.
        self.stall_blocks = int(stall_blocks)
        self.stall_metres = float(stall_metres)
        #: Extra candidates kept from *behind* the leader, one per
        #: ``reserve_band`` metres. Zero disables it.
        #:
        #: This is insurance against the way beam search actually dies. On
        #: Winter 5 the whole beam reached 891.8 m and stuck: the trap was a
        #: corner entered too fast, and every candidate that got there quickest
        #: got there fastest. Pruning on arc alone actively prefers the doomed
        #: ones, because a stuck branch *holds* its arc while a slower branch
        #: that would have survived is simply behind. Reserving a few places for
        #: candidates further back keeps a slower entry into the same corner
        #: alive, which is the one thing that gets through it.
        self.reserve = int(reserve)
        self.reserve_band = float(reserve_band)
        self._rollouts = 0
        self._frames = 0
        #: The surviving candidates after the last :meth:`descend`.
        self.beam: list[Candidate] = []
        #: The candidate that crossed the line, if one did.
        self.winner: Candidate | None = None

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
        arc, lateral = self.track.project(state.position)
        root = Candidate(
            blob=self.backend.save_state(),
            frame=0,
            checkpoint=state.checkpoint,
            arc=arc,
            actions=(),
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            state=state,
            lateral=lateral,
        )
        return self.descend(root, max_frames // self.block_frames, on_depth)

    def descend(
        self,
        root: Candidate,
        max_depth: int,
        on_depth: Callable[[int, Candidate], None] | None = None,
    ) -> SearchResult:
        """Expand from ``root`` for at most ``max_depth`` blocks.

        Split out of :meth:`run` so a beam can start from a state restored part
        way through a known lap instead of from the start line. Searching four
        seconds of a corner is a different proposition from searching a whole
        track: the failure that killed the Winter 5 run needed a long way to
        develop, and a short window simply does not give it room.

        The surviving beam is left on :attr:`beam` afterwards, because a windowed
        search wants every candidate to splice back into the lap and time to the
        finish, not just the one that got furthest inside the window.
        """
        beam = [root]
        self.beam = beam
        history: list[float] = []
        best_arc = root.arc

        for depth in range(max_depth):
            # Two passes over the depth, because a saved state is 24 MB against
            # 14 ms to make one. Scoring every child with its blob attached would
            # hold beam_width * len(actions) of them at once -- 3.5 GB at the
            # default width -- so pass one keeps only the numbers, and pass two
            # re-runs the survivors to materialise their states. That costs one
            # extra block of simulation per surviving candidate, about a third
            # more, and takes peak memory from 4x the beam to 2x.
            scored: list[tuple[Candidate, int, int]] = []
            finishers: list[tuple[int, int, int]] = []
            for index, parent in enumerate(beam):
                for action in self._actions_from(parent):
                    outcome = self._simulate(parent, action)
                    if outcome.terminated:
                        continue
                    if self.stall_blocks and outcome.stalled >= self.stall_blocks:
                        # Going nowhere and has been for a while. Left in, it
                        # keeps its arc and therefore its rank, and crowds out
                        # branches that are still driving.
                        continue
                    if outcome.finished:
                        finishers.append((outcome.frame, index, action))
                        continue
                    scored.append((outcome, index, action))

            if finishers:
                # The earliest finisher in the depth, not the first one found.
                # Candidates cross the line at different frames *within* a block,
                # so "first expanded" is an ordering accident worth up to a whole
                # block -- 50 ms, which is more than the margin these searches
                # are run to find.
                _, index, action = min(finishers)
                winner = self._materialise(beam[index], action)
                self.winner = winner
                return self._result(winner, history, finished=True)

            if not scored:
                break

            beam = self._prune_scored(scored, beam)
            self.beam = beam
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

    def _actions_from(self, parent: Candidate) -> tuple[int, ...]:
        """Which actions to expand from here -- all of them, or the prior's best."""
        if self.prior is None or parent.state is None:
            return self.actions
        values = np.asarray(
            self.prior(parent.state, parent.arc, parent.lateral), dtype=np.float64
        )
        # Restricted actions come back as -inf from a masked agent, and argsort
        # would happily rank them; drop them rather than letting the search
        # expand a branch the policy has never valued.
        allowed = [a for a in self.actions if np.isfinite(values[a])]
        if not allowed:
            return self.actions
        allowed.sort(key=lambda a: values[a], reverse=True)
        return tuple(allowed[: self.top_k])

    def _simulate(self, parent: Candidate, action: int) -> Candidate:
        """One block from a parent state, scored but with no state saved.

        ``blob`` is deliberately empty. This is the hot path -- ``beam_width``
        times ``top_k`` calls per depth -- and saving a state it will almost
        certainly throw away is the single most expensive thing the search could
        do.
        """
        self.backend.load_state(parent.blob)
        result = self.backend.step_many(ACTION_TO_MASK[action], self.block_frames)
        self._frames += result.state.frame - parent.frame
        self._rollouts += 1

        # Project relative to the parent's arc: each branch tracks its own
        # progress, which is what makes this correct on looping tracks.
        arc, lateral = self.track.project(
            result.state.position, near_arc=parent.arc, radius=self.window
        )
        moved = arc - parent.arc >= self.stall_metres
        return Candidate(
            blob=b"",
            frame=result.state.frame,
            checkpoint=result.state.checkpoint,
            arc=arc,
            actions=parent.actions + (action,),
            position=result.state.position.copy(),
            velocity=result.state.velocity.copy(),
            finished=result.finished,
            terminated=result.terminated,
            state=result.state,
            lateral=lateral,
            stalled=0 if moved else parent.stalled + 1,
        )

    def _materialise(self, parent: Candidate, action: int) -> Candidate:
        """Re-run a surviving child and keep its state this time.

        Sound only because the backend is deterministic: the same block from the
        same restored state must produce the same result it produced in the
        scoring pass. That is the same property the acceptance gate checks, and
        if it ever stopped holding this search would be the least of it.
        """
        child = self._simulate(parent, action)
        child.blob = self.backend.save_state()
        return child

    def _prune_scored(
        self, scored: list[tuple[Candidate, int, int]], beam: list[Candidate]
    ) -> list[Candidate]:
        """Pick the survivors from blob-less probes, then re-run only those.

        Deduplication needs a state fingerprint, and pass one did keep one --
        position and velocity are a handful of floats, unlike the 24 MB blob --
        Deduplication matters more than beam width does. Without it the beam
        fills with dozens of trajectories that reached effectively the same place
        at the same speed by trivially different routes, and the search explores
        one narrow corridor instead of genuinely different lines.
        """
        scored.sort(key=lambda item: item[0].score, reverse=True)
        chosen: list[tuple[int, int]] = []
        seen: set[tuple] = set()
        for probe, index, action in scored:
            key = self._bucket(probe)
            if key in seen:
                continue
            seen.add(key)
            chosen.append((index, action))
            if len(chosen) >= self.beam_width:
                break

        if self.reserve and chosen:
            # Places held for candidates *behind* the leader, one per band, so a
            # corner that kills everything arriving at full speed still has a
            # slower arrival left to try. See the note on ``reserve``.
            leader = scored[0][0].arc
            picked = set(chosen)
            bands: set[int] = set()
            for probe, index, action in scored:
                if (index, action) in picked:
                    continue
                band = int((leader - probe.arc) / self.reserve_band)
                if band <= 0 or band in bands:
                    continue
                key = self._bucket(probe)
                if key in seen:
                    continue
                seen.add(key)
                bands.add(band)
                chosen.append((index, action))
                if len(bands) >= self.reserve:
                    break

        return [self._materialise(beam[i], a) for i, a in chosen]

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
        actions: tuple[int, ...] | None = None,
        on_progress: Callable[[int, int, int, bool], None] | None = None,
        focus_blocks: tuple[int, int] | None = None,
    ) -> None:
        self.backend = backend
        self.track = track
        self.block_frames = block_frames
        self.window_blocks = window_blocks
        self.rng = np.random.default_rng(seed)
        #: Actions a perturbation may draw from. Narrow it to what the run being
        #: polished actually uses -- proposing a brake into a lap that never
        #: brakes just wastes evaluations on candidates that will be rejected.
        self.actions = actions or tuple(range(N_ACTIONS))
        self.on_progress = on_progress
        #: Restrict the sweep to ``[first, last)`` blocks. A uniform sweep spends
        #: equal effort everywhere, which is the wrong shape: on Winter 5 the
        #: agent's 395 ms deficit against the human lap is two clusters, 250-500 m
        #: and 1000-1300 m, and the last third is already dead level. Ninety
        #: window positions over the opening acceleration found nothing, which is
        #: exactly what should have been predicted.
        #:
        #: Focusing is also cheaper per evaluation, not just better aimed: the
        #: cost of an attempt is the lap *remaining* after the anchor, so windows
        #: late in a run are the cheap ones.
        self.focus_blocks = focus_blocks
        #: Share of proposals that hold one action across the whole window rather
        #: than drawing each block independently. See :meth:`_propose`.
        self.hold_fraction = 0.5
        #: How far a candidate may fall behind the incumbent at the same frame
        #: before it is abandoned. Zero would prune lines that dip behind and
        #: come back with more speed in hand, which is most of what a good
        #: perturbation does; too generous and every hopeless line runs to the
        #: end of the lap.
        self.abandon_metres = 6.0
        self._rollouts = 0
        self._frames = 0
        self._abandoned = 0

    def run(self, prior: TAS, iterations: int = 400) -> SearchResult:
        """Polish ``prior``. Returns the best run found, never worse than the input.

        The window sweeps along the track in order rather than landing at random,
        which is what makes an **anchor** possible: everything before the window
        is frozen once the sweep has passed it, so the state at the window's start
        can be saved once and restored for every attempt there. Evaluating a
        change near the end of a lap then costs the end of a lap, not a whole one.

        On Winter 5 that is the difference between 14.4 s per evaluation and an
        average of about 7 -- and, more to the point, it is what lets the last
        few seconds of a run be polished at a couple of seconds a try.
        """
        self._rollouts = 0
        self._frames = 0
        self._abandoned = 0

        n_blocks = math.ceil(prior.n_frames / self.block_frames) + 4
        sequence = self._sequence_from(prior, n_blocks)
        best_key, best_frames, best_finished, trace = self._evaluate(sequence, 0, None)
        history: list[float] = []

        stride = max(self.window_blocks // 2, 1)
        starts = list(range(0, max(n_blocks - self.window_blocks, 1), stride))
        if self.focus_blocks is not None:
            first, last = self.focus_blocks
            starts = [s for s in starts if first <= s < last] or starts
        # With fewer evaluations than window positions, thin the positions rather
        # than running out partway: a sweep that only ever polishes the first two
        # seconds of a lap is worse than a coarse one that covers all of it.
        if len(starts) > iterations:
            keep = np.linspace(0, len(starts) - 1, iterations).round().astype(int)
            starts = [starts[i] for i in sorted(set(keep.tolist()))]
        per_window = max(iterations // max(len(starts), 1), 1)
        anchor: tuple[bytes, float] | None = None
        anchor_at = 0
        used = 0

        for start in starts:
            if used >= iterations:
                break
            # Advance the anchor over the blocks the sweep has just frozen.
            anchor = self._advance(sequence, anchor, anchor_at, start)
            anchor_at = start
            span = min(self.window_blocks, n_blocks - start)
            for _ in range(per_window):
                if used >= iterations:
                    break
                candidate = sequence.copy()
                candidate[start : start + span] = self._propose(span)
                key, frames, finished, candidate_trace = self._evaluate(
                    candidate, start, anchor, reference=trace, from_block=start + span
                )
                used += 1
                if key > best_key:
                    sequence, best_key, best_frames, best_finished = (
                        candidate, key, frames, finished,
                    )
                    trace = candidate_trace
                history.append(float(best_key[1]))
            if self.on_progress is not None:
                self.on_progress(start, len(starts), best_frames, best_finished)

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

    def _propose(self, span: int) -> np.ndarray:
        """Actions for one window. Contiguous, and biased toward holding.

        Drawing every block independently is the obvious move operator and it is
        the wrong one here. Measured on Winter 5: the 19.025 human lap makes 85
        input changes with a median hold of 102 ms, while the agent's 19.420
        makes 198 -- it saws at the wheel to stay on line where the human carries
        one arc. Independent per-block draws generate more sawing, never less, so
        three sweeps found nothing at all, including ninety attempts aimed
        squarely at the stretch losing the most time.

        So half the proposals are a single action held across the window. That is
        the shape of the change that would actually help, and it costs nothing to
        offer.
        """
        if self.rng.random() < self.hold_fraction:
            return np.full(span, self.rng.choice(self.actions))
        return self.rng.choice(self.actions, size=span)

    def _advance(
        self,
        sequence: np.ndarray,
        anchor: tuple[bytes, float] | None,
        anchor_at: int,
        start: int,
    ) -> tuple[bytes, float] | None:
        """Move the anchor forward to ``start``, simulating only what is new.

        Called once per window position, so the whole sweep pays for the prefix
        exactly once between them rather than once per evaluation.
        """
        if start <= 0:
            return None
        progress = ProgressTracker(self.track)
        if anchor is None:
            state = self.backend.reset()
            progress.reset()
            progress.update(state.position)
            begin = 0
        else:
            blob, arc = anchor
            self.backend.load_state(blob)
            progress.reset(arc)
            begin = anchor_at
        for action in sequence[begin:start]:
            result = self.backend.step_many(
                ACTION_TO_MASK[int(action)], self.block_frames
            )
            progress.update(result.state.position)
            self._frames += self.block_frames
            if result.finished or result.terminated:
                # The frozen prefix already ends the run, so no window past here
                # can change anything. Keep the anchor where it was.
                return anchor
        return self.backend.save_state(), progress.arc

    def _evaluate(
        self,
        sequence: np.ndarray,
        start: int = 0,
        anchor: tuple[bytes, float] | None = None,
        reference: dict[int, float] | None = None,
        from_block: int = 0,
    ) -> tuple[tuple, int, bool, dict[int, float]]:
        """Run ``sequence`` from the anchor, or from the line if there is none.

        ``reference`` is the incumbent's arc at each block index, and every block
        is exactly ``block_frames`` long, so block index and frame number line up
        between the two runs. That makes an ordinary branch-and-bound possible:
        once a candidate is clearly behind the run it is trying to beat *at the
        same frame*, nothing it does later can be worth the ten seconds it would
        take to find out. Most perturbations are worse, so abandoning them early
        is most of the search's throughput.
        """
        progress = ProgressTracker(self.track)
        if anchor is None:
            state = self.backend.reset()
            progress.reset()
            progress.update(state.position)
            begin = 0
        else:
            blob, arc = anchor
            self.backend.load_state(blob)
            progress.reset(arc)
            begin = start

        trace: dict[int, float] = dict(reference or {})
        result = None
        block = begin
        for action in sequence[begin:]:
            result = self.backend.step_many(
                ACTION_TO_MASK[int(action)], self.block_frames
            )
            progress.update(result.state.position)
            self._frames += self.block_frames
            block += 1
            trace[block] = progress.arc
            if result.finished or result.terminated:
                break
            if (
                reference is not None
                and block > from_block
                and block in reference
                and progress.arc < reference[block] - self.abandon_metres
            ):
                # Behind, and not by a rounding error. A candidate can dip behind
                # and recover with more speed in hand, which is why the margin is
                # metres rather than zero -- but past it, this is a slower line.
                self._rollouts += 1
                self._abandoned += 1
                return (-1, progress.arc), 0, False, trace
        self._rollouts += 1

        if result is None:  # the window sat past the end of the sequence
            return (0, progress.arc), 0, False, trace
        frames = result.state.frame
        if result.finished:
            # Finishers rank above everything, fastest first.
            key = (result.state.checkpoint + 1, -float(frames))
        else:
            key = (result.state.checkpoint, progress.arc)
        return key, frames, result.finished, trace
