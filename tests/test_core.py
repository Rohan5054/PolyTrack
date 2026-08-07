"""Core invariants.

The determinism and save/load tests are the important ones. They encode the two
properties the whole project rests on: if the simulator is not reproducible, no
measured time means anything; if save/restore is lossy, the search layer silently
produces trajectories that do not replay.
"""

from __future__ import annotations

import numpy as np

from polytrack.actions import (
    ACTION_TO_MASK,
    LEFT,
    RIGHT,
    UP,
    canonicalize,
    keys_to_mask,
    mask_to_action,
    mask_to_keys,
)
from polytrack.env import PolyTrackEnv
from polytrack.physics.base import PhysicsBackend
from polytrack.physics.reference import ReferencePhysics
from polytrack.tas import TAS, InputChange, format_time
from polytrack.track import chicane_track, oval_track, straight_track


def test_action_mask_roundtrip():
    for action, mask in enumerate(ACTION_TO_MASK):
        assert mask_to_action(mask) == action
        assert keys_to_mask(mask_to_keys(mask)) == mask


def test_opposing_inputs_cancel():
    # Matches the v0.6.0 changelog: left and right cancel rather than favouring one.
    assert canonicalize(LEFT | RIGHT) == 0
    assert canonicalize(UP | LEFT | RIGHT) == UP


def test_tas_text_roundtrip():
    original = TAS(
        [InputChange(0, UP), InputChange(120, UP | RIGHT), InputChange(300, 0)],
        n_frames=450,
    )
    assert TAS.parse(original.to_text()) == original


def test_tas_preserves_run_length_past_last_input_change():
    """A run that holds throttle to the finish still knows when it ended."""
    tas = TAS.from_frame_masks([UP] * 500)
    assert len(tas) == 1  # one change, at frame 0
    assert tas.last_frame == 0
    assert tas.n_frames == 500
    assert TAS.parse(tas.to_text()).n_frames == 500
    assert len(list(tas.expand())) == 500


def test_tas_rejects_length_shorter_than_inputs():
    try:
        TAS([InputChange(0, UP), InputChange(90, 0)], n_frames=10)
    except ValueError as exc:
        assert "shorter" in str(exc)
    else:
        raise AssertionError("expected ValueError for truncated n_frames")


def test_tas_expand_matches_changes():
    tas = TAS([InputChange(0, UP), InputChange(3, UP | LEFT)])
    assert list(tas.expand(6)) == [UP, UP, UP, UP | LEFT, UP | LEFT, UP | LEFT]


def test_tas_from_dense_masks_collapses_runs():
    dense = [UP, UP, UP, UP | LEFT, UP | LEFT, 0]
    tas = TAS.from_frame_masks(dense)
    assert len(tas) == 3  # only the three transitions are stored
    assert list(tas.expand(len(dense))) == dense


def test_tas_parser_rejects_bad_keys():
    try:
        TAS.parse("0,wq")
    except ValueError as exc:
        assert "wq" in str(exc) or "q" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown key")


def test_reference_backend_satisfies_protocol():
    assert isinstance(ReferencePhysics(straight_track()), PhysicsBackend)


def test_physics_is_deterministic():
    """Identical inputs from reset must give bit-identical trajectories."""
    inputs = [UP] * 40 + [UP | LEFT] * 25 + [UP] * 40

    def run():
        sim = ReferencePhysics(chicane_track())
        sim.reset()
        return [sim.step(m).state.position.copy() for m in inputs]

    first, second = run(), run()
    for a, b in zip(first, second):
        assert np.array_equal(a, b), "trajectory diverged between identical runs"


def test_save_load_state_is_exact():
    """Restoring a saved state must continue the run identically.

    A backend that fails this looks fine under plain rollout and desyncs only
    under search -- the worst possible failure mode, so it is asserted directly.
    """
    sim = ReferencePhysics(chicane_track())
    sim.reset()
    for _ in range(30):
        sim.step(UP)

    snapshot = sim.save_state()
    continued = [sim.step(UP | LEFT).state.position.copy() for _ in range(25)]

    sim.load_state(snapshot)
    replayed = [sim.step(UP | LEFT).state.position.copy() for _ in range(25)]

    for a, b in zip(continued, replayed):
        assert np.array_equal(a, b), "state restore was lossy"


def test_track_projection_is_monotonic_along_centerline():
    track = straight_track(length=100.0)
    arcs = [track.project(np.array([x, 0.0, 0.0]))[0] for x in range(0, 100, 10)]
    assert arcs == sorted(arcs)
    assert arcs[-1] > arcs[0]


def test_track_projection_reports_lateral_offset():
    track = straight_track(length=100.0)
    _, offset = track.project(np.array([50.0, 6.0, 0.0]))
    assert abs(offset - 6.0) < 1e-6


def test_batched_projection_matches_single():
    """project_many must agree with project. It is an optimisation, not a variant."""
    track = chicane_track()
    rng = np.random.default_rng(7)
    points = rng.uniform(-20, 120, size=(60, 3))
    points[:, 2] = 0.0

    arcs, offsets = track.project_many(points)
    for point, arc, offset in zip(points, arcs, offsets):
        ref_arc, ref_offset = track.project(point)
        assert abs(arc - ref_arc) < 1e-9
        assert abs(offset - ref_offset) < 1e-9


def test_windowed_projection_matches_full_scan_near_the_car():
    """Windowing must not change the answer for points inside the window.

    The window is a speed optimisation that skips distant segments. If it ever
    disagrees with the full scan for a nearby point, observations and rewards
    quietly drift apart from the geometry they claim to describe.
    """
    track = chicane_track()
    rng = np.random.default_rng(11)
    for _ in range(20):
        centre_arc = float(rng.uniform(0.0, track.length))
        origin = track.point_at(centre_arc)
        probes = origin[None, :] + rng.uniform(-15, 15, size=(25, 3))
        probes[:, 2] = 0.0

        full_arc, full_off = track.project_many(probes)
        win_arc, win_off = track.project_many(probes, near_arc=centre_arc, radius=80.0)
        assert np.allclose(full_arc, win_arc, atol=1e-9)
        assert np.allclose(full_off, win_off, atol=1e-9)


def test_points_at_matches_point_at():
    track = oval_track()
    arcs = np.linspace(0.0, track.length, 25)
    batched = track.points_at(arcs)
    for arc, row in zip(arcs, batched):
        assert np.allclose(track.point_at(float(arc)), row, atol=1e-9)


def test_lookahead_returns_points_ahead():
    track = chicane_track()
    points = track.lookahead(0.0, np.array([10.0, 20.0, 40.0]))
    assert points.shape == (3, 3)
    # Successive lookahead points move further along the track.
    d1 = np.linalg.norm(points[1] - points[0])
    assert d1 > 0


def test_env_observation_shape_matches_space():
    env = PolyTrackEnv(track=oval_track())
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all(), "observation contains NaN or inf"


def test_env_step_returns_finite_reward():
    env = PolyTrackEnv(track=chicane_track())
    env.reset()
    for action in [3, 4, 3, 5, 3]:
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert np.isfinite(obs).all()
        if terminated or truncated:
            break


def test_progress_reward_is_not_payable_twice():
    """Re-covering ground already driven must earn nothing.

    Without a high-water mark, an agent can oscillate back and forth over one
    stretch of track and farm progress reward forever instead of finishing --
    a standard exploit in progress-shaped racing rewards.
    """
    env = PolyTrackEnv(track=straight_track(length=400.0))
    env.reset()
    for _ in range(50):
        env.step(3)  # throttle forward

    # Brake. The car keeps coasting forward for a while before it reverses, so
    # the high-water mark legitimately rises during the first part of this.
    for _ in range(120):
        env.step(6)
    high_water = env._previous_arc

    arc_now, _ = env.track.project(env.backend._position)
    assert arc_now < high_water, "car did not actually move backwards"

    # Now re-drive the ground already covered: it must pay nothing.
    regained = 0.0
    for _ in range(30):
        _, reward, term, trunc, _ = env.step(3)
        regained += reward
        if term or trunc:
            break
    assert regained < 0, "re-covering old ground produced positive reward"


def test_env_exports_replayable_tas():
    """An episode must round-trip: play it, export it, replay it, same result."""
    env = PolyTrackEnv(track=chicane_track())
    env.reset()
    actions = [3, 3, 4, 3, 5, 3, 3, 4, 3]
    for action in actions:
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    tas = env.export_tas()

    replay_env = PolyTrackEnv(track=chicane_track())
    outcome = replay_env.replay_tas(tas)
    assert outcome["frames"] == len(env._input_log)


def test_format_time_matches_game_style():
    assert format_time(0) == "0:00.000"
    assert format_time(60) == "0:01.000"
    assert format_time(3690) == "1:01.500"


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
