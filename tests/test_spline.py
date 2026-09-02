"""Arm 3's continuous parameterisation and the optimiser over it.

The properties here are the ones that make the arm's result mean anything: that
the projection onto the spline basis is exact when it can be, that a knot the
optimiser is not allowed to touch cannot change the frozen prefix, and that the
optimiser is bounded below by what it was handed.

The anchor test is the load-bearing one. Anchors are what make the sweep
affordable -- a full lap is ~23 s of simulation and restoring a state is 17 ms
-- and they are only sound if every block before the anchor depends solely on
knots behind it. With linear interpolation a block reads the knot either side of
it, so an anchor placed mid-span would let a supposedly frozen prefix move under
the optimiser, and every score after it would be measured against a different
run than the one that produced the anchor.
"""

from __future__ import annotations

import numpy as np

from polytrack.physics.reference import ReferencePhysics
from polytrack.spline import (
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    SplineCEM,
    SteeringSpline,
)
from polytrack.track import chicane_track


def test_unit_spacing_reproduces_any_sequence_exactly():
    """At one knot per block the basis spans every sequence, so the fit is exact.

    This is not a formality: it is the only setting at which Winter 5's seed
    survives. A two-block spacing reproduces 97.8% of the human record's actions
    and still loses 400 m of progress, so "close" is not a useful category here.
    """
    rng = np.random.default_rng(0)
    actions = rng.choice([STEER_LEFT, STEER_STRAIGHT, STEER_RIGHT], size=200)
    spline = SteeringSpline(n_blocks=200, knot_spacing=1)
    assert np.array_equal(spline.actions(spline.fit(actions)), actions)


def test_a_coarser_spline_cannot_express_single_block_flicker():
    """The smoothing is real -- it is simply not free, which is the point."""
    actions = np.full(60, STEER_STRAIGHT)
    actions[::2] = STEER_LEFT  # alternating, one block on one block off
    coarse = SteeringSpline(n_blocks=60, knot_spacing=6)
    recovered = coarse.actions(coarse.fit(actions))
    assert not np.array_equal(recovered, actions)
    # ...and it fails by smoothing the flicker away, not by inventing new turns.
    assert set(np.unique(recovered)) <= {STEER_STRAIGHT, STEER_LEFT}


def test_moving_a_knot_slides_a_turn_boundary_before_changing_any_action():
    """Continuity is what a population method needs to have a gradient to follow.

    A parameterisation where small knot changes did nothing until they flipped a
    whole block would give CEM a flat objective almost everywhere.
    """
    spline = SteeringSpline(n_blocks=40, knot_spacing=4, threshold=0.35)
    knots = np.zeros(spline.n_knots)
    knots[4:6] = 1.0
    wide = spline.actions(knots)
    knots[4] = 0.5
    narrow = spline.actions(knots)
    assert (narrow == STEER_RIGHT).sum() < (wide == STEER_RIGHT).sum()
    assert (narrow == STEER_RIGHT).sum() > 0


def test_an_anchor_at_a_knot_boundary_freezes_the_prefix():
    """Knots after the anchor must not be able to move a block before it."""
    spline = SteeringSpline(n_blocks=60, block_frames=4, knot_spacing=5)
    anchor_knot = 6
    anchor_block = int(spline.knot_blocks[anchor_knot])

    rng = np.random.default_rng(1)
    knots = rng.normal(0, 0.8, spline.n_knots)
    before = spline.actions(knots)[:anchor_block]
    for _ in range(20):
        perturbed = knots.copy()
        perturbed[anchor_knot + 1:] = rng.normal(0, 0.8, spline.n_knots - anchor_knot - 1)
        assert np.array_equal(spline.actions(perturbed)[:anchor_block], before)


def test_cem_is_bounded_below_by_the_run_it_was_given():
    """The optimiser may not return something worse than its incumbent.

    Without this an arm's "result" is a coin flip against its own seed, and the
    lineage in `arms/` stops meaning anything.
    """
    track = chicane_track()
    backend = ReferencePhysics(track)
    spline = SteeringSpline(n_blocks=40, block_frames=4, knot_spacing=4)
    cem = SplineCEM(backend, track, spline, population=6, seed=3)

    knots = np.zeros(spline.n_knots)
    incumbent = cem.evaluate(knots, horizon=40)
    best = cem.optimise(
        knots, (1, 5), horizon=40, generations=2, incumbent=incumbent
    )
    assert best.key >= incumbent.key


def test_evaluate_reports_where_progress_stopped():
    """The sweep starts at the break, so the break has to be reported."""
    track = chicane_track()
    backend = ReferencePhysics(track)
    spline = SteeringSpline(n_blocks=50, block_frames=4, knot_spacing=5)
    cem = SplineCEM(backend, track, spline, population=4)
    attempt = cem.evaluate(np.zeros(spline.n_knots), horizon=50)
    assert 0 <= attempt.peak_block < 50
    assert attempt.arc >= 0.0


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
