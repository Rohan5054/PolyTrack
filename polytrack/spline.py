"""Arm 3: continuous trajectory optimisation over a steering spline.

Deliberately a different *decomposition* from :mod:`polytrack.search`, not a
variation on it. Beam search treats a lap as a tree of discrete per-block action
choices and explores it under a learned prior. This treats a lap as a smooth
scalar function of time -- how hard the car is turning, and when -- and optimises
the handful of numbers that define that curve with a derivative-free population
method. No policy, no checkpoint and no learned component appears in this file.

Three things follow from the change of variables:

* **Smoothing is available, but Winter 5 will not take it.** A knot every few
  blocks cannot express single-block flicker, which reads like a free win. It is
  not one here. Projecting the human record onto the spline at a two-block
  spacing reproduces **97.8%** of its actions and still reaches only 72 m of
  1747, against 477 m for the unsmoothed sequence -- a 2.2% difference in
  actions costing 400 m. On this track the useful setting is therefore
  ``knot_spacing=1``, where the parameterisation is a continuous *relaxation* of
  the input sequence rather than a smooth curve through it. The smoothing is
  real and may pay on a more forgiving track; it does not pay here, and saying
  otherwise would be selling the idea rather than reporting it.
* **Dimension is set by knot spacing, not lap length.** A 19-second lap is 380
  blocks at 50 ms but only ~50 knots at four blocks apart, and a population
  method's cost scales with the parameter count.
* **The hypothesis class is restricted, and that cuts both ways.** A lap the
  spline cannot represent is unreachable however good the optimiser is. Whether
  a competitive lap is *in* the class is an empirical question about the track,
  and it has to be measured rather than assumed -- see ``fit`` below.

The optimiser is a **separable** cross-entropy method: one independent Gaussian
per knot, refit from the elite fraction each generation. That is not full CMA-ES
-- no covariance between knots, no evolution path -- and it is named for what it
does rather than for the more impressive relative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import ACTION_TO_MASK

#: The forward-only steering vocabulary: hold throttle, and turn or do not.
#: Winter 5's human record holds throttle for 100.0% of its frames and never
#: brakes, so the throttle axis carries no information on this track and is not
#: parameterised. On a track that brakes this vocabulary is wrong.
STEER_LEFT, STEER_STRAIGHT, STEER_RIGHT = 4, 3, 5


@dataclass(frozen=True, slots=True)
class SteeringSpline:
    """Knot values on a uniform grid -> one action per block.

    The signal is piecewise linear between knots and thresholded into three
    actions. Thresholding rather than rounding is what makes the parameterisation
    continuous where it matters: moving a knot slightly changes *when* a turn
    starts and ends, by sliding the crossing point, long before it changes which
    actions appear at all.
    """

    n_blocks: int
    block_frames: int = 50
    #: Blocks between knots. The one real knob: it trades what the class can
    #: represent against how many parameters the optimiser has to fit.
    knot_spacing: int = 4
    #: Where the signal crosses into a turn. Not 0.5: a knot fitted from a block
    #: that steered for half its span lands near 0.5, and it should steer.
    threshold: float = 0.35

    @property
    def knot_blocks(self) -> np.ndarray:
        """Block index of each knot. Runs past the end so the last block interpolates."""
        return np.arange(0, self.n_blocks + self.knot_spacing, self.knot_spacing)

    @property
    def n_knots(self) -> int:
        return int(len(self.knot_blocks))

    def actions(self, knots: np.ndarray) -> np.ndarray:
        """Per-block actions for a knot vector."""
        signal = np.interp(
            np.arange(self.n_blocks), self.knot_blocks, np.asarray(knots, float)
        )
        out = np.full(self.n_blocks, STEER_STRAIGHT, dtype=int)
        out[signal < -self.threshold] = STEER_LEFT
        out[signal > self.threshold] = STEER_RIGHT
        return out

    def fit(self, actions: np.ndarray) -> np.ndarray:
        """Least-squares knots for an existing action sequence.

        Fitting rather than sampling because the knots are a coarser grid than
        the blocks: every knot's value is pinned by the blocks either side of it,
        and a nearest-block reading throws that away. The basis is the same
        linear interpolation :meth:`actions` uses, so this is the exact
        projection of the sequence onto the spline's span -- which is what makes
        the round-trip error below an honest statement about the class rather
        than about the fitting.
        """
        actions = np.asarray(actions)
        signal = np.zeros(self.n_blocks, dtype=float)
        signal[actions == STEER_LEFT] = -1.0
        signal[actions == STEER_RIGHT] = 1.0

        # Interpolation weights: row per block, column per knot, two non-zeros.
        blocks = np.arange(self.n_blocks, dtype=float)
        knots = self.knot_blocks.astype(float)
        basis = np.zeros((self.n_blocks, self.n_knots))
        right = np.clip(np.searchsorted(knots, blocks, side="right"), 1, self.n_knots - 1)
        left = right - 1
        span = knots[right] - knots[left]
        frac = (blocks - knots[left]) / span
        rows = np.arange(self.n_blocks)
        basis[rows, left] = 1.0 - frac
        basis[rows, right] = frac

        # Ridge, not plain least squares: knots past the end of the sequence are
        # unconstrained and a bare solve leaves them arbitrary.
        gram = basis.T @ basis + 1e-6 * np.eye(self.n_knots)
        return np.linalg.solve(gram, basis.T @ signal)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One evaluated knot vector."""

    key: tuple
    knots: np.ndarray
    frames: int
    finished: bool
    arc: float
    #: Block at which peak arc was last improved -- where the run stopped making
    #: progress, which is where a sweep should start rather than at block zero.
    peak_block: int = 0


class SplineCEM:
    """Separable cross-entropy method over :class:`SteeringSpline` knots.

    Two phases, because a lap that does not finish and a lap that does are
    different optimisation problems:

    * **Advance.** While no finishing lap exists, anchor at the furthest knot
      boundary the incumbent still survives, and optimise the next few knots to
      push the frontier further down the track. Scored on arc at a fixed
      horizon, which is the same "equal frames, so furthest is fastest" argument
      the beam search uses and needs no coefficients.
    * **Polish.** Once a lap finishes, sweep a window along it and minimise the
      finish frame, keeping a change only if the lap got faster. Bounded below
      by its input.

    Anchoring is what makes either affordable: a full lap costs ~23 s to
    simulate, so an optimiser that re-drives from frame zero every evaluation
    can afford a few hundred of them in an hour. Restoring a saved state costs
    17 ms. Anchors land on knot boundaries so that every block before the anchor
    depends only on frozen knots -- with linear interpolation a block's value
    reads the knot either side of it, so anchoring mid-span would let a "frozen"
    prefix change under the optimiser.
    """

    def __init__(
        self,
        backend,
        track,
        spline: SteeringSpline,
        *,
        population: int = 16,
        elite_frac: float = 0.25,
        smoothing: float = 0.7,
        sigma0: float = 0.6,
        sigma_floor: float = 0.05,
        seed: int = 0,
        on_progress=None,
    ) -> None:
        self.backend = backend
        self.track = track
        self.spline = spline
        self.population = population
        self.n_elite = max(2, int(population * elite_frac))
        self.smoothing = smoothing
        self.sigma0 = sigma0
        self.sigma_floor = sigma_floor
        self.rng = np.random.default_rng(seed)
        self.on_progress = on_progress
        self._rollouts = 0
        self._frames = 0

    # -- evaluation ---------------------------------------------------------

    def evaluate(
        self,
        knots: np.ndarray,
        *,
        anchor: tuple[bytes, float, int] | None = None,
        horizon: int | None = None,
    ) -> Attempt:
        """Drive ``knots`` from the anchor, to the finish or to ``horizon`` blocks.

        ``horizon`` bounds the advance phase: there is no point simulating to the
        end of a lap that is being scored on where it got to.
        """
        from .track import ProgressTracker

        actions = self.spline.actions(knots)
        progress = ProgressTracker(self.track)
        if anchor is None:
            state = self.backend.reset()
            progress.reset()
            progress.update(state.position)
            begin = 0
        else:
            blob, arc, begin = anchor
            self.backend.load_state(blob)
            progress.reset(arc)

        stop = len(actions) if horizon is None else min(begin + horizon, len(actions))
        result, peak, peak_block = None, progress.arc, begin
        for offset, action in enumerate(actions[begin:stop]):
            result = self.backend.step_many(
                ACTION_TO_MASK[int(action)], self.spline.block_frames
            )
            progress.update(result.state.position)
            if progress.arc > peak:
                peak, peak_block = progress.arc, begin + offset
            self._frames += self.spline.block_frames
            if result.finished or result.terminated:
                break
        self._rollouts += 1

        if result is None:
            return Attempt((-1, peak), np.asarray(knots), 0, False, peak, begin)
        if result.finished:
            key = (result.state.checkpoint + 1, -float(result.state.frame))
        else:
            key = (result.state.checkpoint, peak)
        return Attempt(
            key, np.asarray(knots), int(result.state.frame),
            bool(result.finished), float(peak), int(peak_block),
        )

    def anchor_at(self, knots: np.ndarray, knot_index: int, previous=None):
        """Save the state at a knot boundary, driving ``knots`` to reach it.

        ``previous`` is an earlier anchor on the same sweep. Given one, only the
        blocks between the two are simulated. This is not a micro-optimisation:
        rebuilding from frame zero costs the whole prefix, so a sweep that does
        it at every window position pays O(n^2) in lap length and spends more
        time re-driving the opening straight than optimising. Advancing is sound
        because the sweep moves forward and a window only ever edits knots ahead
        of the anchor, so the prefix is frozen by the time it is reused.
        """
        from .track import ProgressTracker

        block = int(self.spline.knot_blocks[knot_index])
        if block <= 0:
            return None
        actions = self.spline.actions(knots)
        progress = ProgressTracker(self.track)
        if previous is not None and previous[2] <= block:
            blob, arc, begin = previous
            self.backend.load_state(blob)
            progress.reset(arc)
        else:
            state = self.backend.reset()
            progress.reset()
            progress.update(state.position)
            begin = 0
        for action in actions[begin:block]:
            result = self.backend.step_many(
                ACTION_TO_MASK[int(action)], self.spline.block_frames
            )
            progress.update(result.state.position)
            self._frames += self.spline.block_frames
            if result.finished or result.terminated:
                return None
        return self.backend.save_state(), progress.arc, block

    # -- the optimiser ------------------------------------------------------

    def optimise(
        self,
        knots: np.ndarray,
        window: tuple[int, int],
        *,
        anchor=None,
        horizon: int | None = None,
        generations: int = 6,
        incumbent: Attempt | None = None,
    ) -> Attempt:
        """Refit knots ``window`` by CEM. Never returns worse than ``incumbent``."""
        lo, hi = window
        best = incumbent or self.evaluate(knots, anchor=anchor, horizon=horizon)
        mean = np.asarray(best.knots, float).copy()
        sigma = np.full(hi - lo, self.sigma0)

        for _ in range(generations):
            samples = np.repeat(mean[None, :], self.population, axis=0)
            samples[:, lo:hi] += self.rng.normal(
                0.0, sigma, size=(self.population, hi - lo)
            )
            np.clip(samples, -1.5, 1.5, out=samples)
            attempts = [
                self.evaluate(s, anchor=anchor, horizon=horizon) for s in samples
            ]
            attempts.sort(key=lambda a: a.key, reverse=True)
            elite = np.array([a.knots[lo:hi] for a in attempts[: self.n_elite]])
            # Refit, smoothed. Without smoothing the distribution collapses onto
            # the first lucky elite set and stops exploring.
            mean[lo:hi] = (
                self.smoothing * elite.mean(axis=0)
                + (1.0 - self.smoothing) * mean[lo:hi]
            )
            sigma = np.maximum(
                self.smoothing * elite.std(axis=0)
                + (1.0 - self.smoothing) * sigma,
                self.sigma_floor,
            )
            if attempts[0].key > best.key:
                best = attempts[0]
            if self.on_progress is not None:
                self.on_progress(best)
        return best

    @property
    def rollouts(self) -> int:
        return self._rollouts

    @property
    def frames_simulated(self) -> int:
        return self._frames
