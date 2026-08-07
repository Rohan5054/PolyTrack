"""Search-layer tests, and regression tests for two progress-tracking bugs.

Both bugs below were found by beam search rather than by the unit tests, which
is worth noting: search is an adversarial consumer of a reward signal. It will
locate any way to score highly that does not involve driving fast, and it does
so within seconds. Anything that survives a search is much more likely to be
correct than anything that merely survives a scripted rollout.
"""

from __future__ import annotations

import numpy as np

from polytrack.actions import ACTION_TO_MASK
from polytrack.physics.reference import ReferencePhysics
from polytrack.search import BeamSearch, CEMSearch
from polytrack.tas import PHYSICS_HZ
from polytrack.track import (
    ProgressTracker,
    chicane_track,
    oval_track,
    straight_track,
)

DEMO_TRACKS = (straight_track, chicane_track, oval_track)


# -- regression: closed-loop start line ---------------------------------------

def test_reversing_at_the_start_does_not_report_a_full_lap():
    """A car just behind the start line must not project onto the track's end.

    On a closed loop the start and finish are the same point in space. An
    unwindowed projection reports a car a centimetre behind the line as being at
    arc == track length -- a completed lap. Beam search found this in under a
    second and returned a "winning run" consisting of one frame of reverse.
    """
    track = oval_track()
    tracker = ProgressTracker(track)
    tracker.reset()

    just_behind = track.centerline[0] - track.start_forward * 0.01
    arc, _ = tracker.update(just_behind)
    assert arc < track.length * 0.1, (
        f"reversing 1 cm reported arc {arc:.1f} of {track.length:.1f} m"
    )


def test_progress_cannot_jump_further_than_the_car_can_travel():
    """Arc must advance no faster than the car physically moves.

    This is the invariant that makes progress reward trustworthy. If arc can
    leap, both the reward and the checkpoint logic can be gamed without driving.
    """
    for factory in DEMO_TRACKS:
        track = factory()
        sim = ReferencePhysics(track)
        state = sim.reset()
        tracker = ProgressTracker(track)
        tracker.reset()
        previous = tracker.update(state.position)[0]

        rng = np.random.default_rng(3)
        for _ in range(400):
            result = sim.step(ACTION_TO_MASK[int(rng.integers(0, 9))])
            arc = tracker.update(result.state.position)[0]
            travelled = result.state.speed / PHYSICS_HZ
            assert abs(arc - previous) <= travelled + 1.0, (
                f"{track.name}: arc moved {abs(arc - previous):.2f} m in one "
                f"frame while the car travelled {travelled:.2f} m"
            )
            previous = arc
            if result.finished or result.terminated:
                break


def test_checkpoints_cannot_all_trigger_in_a_few_frames():
    """Collecting every checkpoint almost instantly means progress is broken."""
    track = oval_track()
    sim = ReferencePhysics(track)
    sim.reset()
    for _ in range(10):
        result = sim.step(ACTION_TO_MASK[6])  # reverse off the line
        assert not result.finished, "reversing off the line completed the track"


# -- regression: physics calibration ------------------------------------------

def test_demo_tracks_are_completable():
    """A competent policy must be able to finish every demo track.

    Guards the reference physics parameters. An earlier steering falloff left
    the car with roughly 3 deg/s of authority at 200 km/h, which made every
    corner a wall -- the tracks were silently impossible, and search failures
    looked like search bugs.
    """
    for factory in (straight_track, chicane_track):
        track = factory()
        sim = ReferencePhysics(track)
        state = sim.reset()
        tracker = ProgressTracker(track)
        tracker.reset()

        finished = False
        for _ in range(3600):
            arc = tracker.arc
            # Steer toward a point on the centerline ahead of the car, and lift
            # off when pointing badly wrong. Deliberately unsophisticated.
            target = track.point_at(arc + 20.0)
            to_target = target - state.position
            right = np.cross(state.forward, np.array([0.0, 0.0, 1.0]))
            lateral = float(np.dot(to_target, right))
            aligned = float(np.dot(to_target / (np.linalg.norm(to_target) + 1e-9),
                                   state.forward))
            throttle = 3 if aligned > 0.9 else 0
            if lateral < -0.5:
                action = throttle + 1
            elif lateral > 0.5:
                action = throttle + 2
            else:
                action = throttle
            result = sim.step(ACTION_TO_MASK[action])
            state = result.state
            tracker.update(state.position)
            if result.finished:
                finished = True
                break
            if result.terminated:
                break
        assert finished, f"{track.name} could not be completed by a basic policy"


# -- search behaviour ---------------------------------------------------------

def test_beam_search_finds_a_finishing_run():
    track = chicane_track()
    result = BeamSearch(
        ReferencePhysics(track), track, beam_width=24, block_frames=6
    ).run(max_frames=1800)
    assert result.finished, result.summary()
    assert result.tas is not None


def test_search_output_replays_identically():
    """The critical property: a searched TAS must reproduce its own result.

    Search operates through save/restore. If restore is lossy, search finds
    trajectories that exist only inside the search and vanish on a clean replay.
    This test is the same check that must later be run against the real game.
    """
    track = chicane_track()
    result = BeamSearch(
        ReferencePhysics(track), track, beam_width=24, block_frames=6
    ).run(max_frames=1800)
    assert result.tas is not None

    clean = ReferencePhysics(track)
    clean.reset()
    final = None
    for mask in result.tas.expand():
        final = clean.step(mask)
        if final.finished or final.terminated:
            break

    assert final is not None
    assert final.finished, "searched run did not finish on a clean replay"
    assert final.state.frame == result.frames, (
        f"replay took {final.state.frame} frames, search reported {result.frames}"
    )


def test_beam_search_beats_flat_out_throttle():
    """Search must actually be worth running: it should beat the naive line."""
    track = chicane_track()

    naive = ReferencePhysics(track)
    naive.reset()
    naive_frames = None
    for i in range(3600):
        result = naive.step(ACTION_TO_MASK[3])
        if result.finished:
            naive_frames = i
            break
        if result.terminated:
            break

    searched = BeamSearch(
        ReferencePhysics(track), track, beam_width=24, block_frames=6
    ).run(max_frames=1800)
    assert searched.finished

    if naive_frames is not None:
        assert searched.frames <= naive_frames, (
            f"search ({searched.frames}) was slower than flat-out ({naive_frames})"
        )


def test_cem_search_runs_and_reports_progress():
    track = straight_track(length=120.0)
    result = CEMSearch(
        ReferencePhysics(track), track, n_blocks=40, block_frames=6,
        population=48, seed=1,
    ).run(iterations=6)
    assert result.n_rollouts == 48 * 6
    assert result.history, "CEM recorded no progress history"


def test_cem_improves_over_its_own_first_iteration():
    """The distribution must actually be learning something."""
    track = chicane_track()
    result = CEMSearch(
        ReferencePhysics(track), track, n_blocks=70, block_frames=6,
        population=64, seed=5,
    ).run(iterations=12)
    assert result.history[-1] >= result.history[0], (
        f"CEM regressed: {result.history[0]:.1f} -> {result.history[-1]:.1f}"
    )


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
