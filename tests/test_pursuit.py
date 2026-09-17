"""The scripted driver: the drivability filter and the floor a policy must beat.

Steering is tested with synthetic car states, not the stand-in. The stand-in
lies in the x-y plane while PolyTrack is y-up, so the driver cannot steer there
at all and would still report finishing its demo chicane -- a pass that proves
nothing. The steering *convention* itself is a measured fact about the real
module, and the one module-backed test here re-measures it.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from polytrack.actions import ACTION_TO_MASK, Action
from polytrack.physics.reference import ReferencePhysics
from polytrack.pursuit import PurePursuit
from polytrack.track import straight_track

_SKIPPED: list[str] = []

RIGHT = {Action.THROTTLE_RIGHT, Action.BRAKE_RIGHT}
LEFT = {Action.THROTTLE_LEFT, Action.BRAKE_LEFT}
BRAKING = {Action.BRAKE, Action.BRAKE_LEFT, Action.BRAKE_RIGHT}


def _car(position, forward, speed):
    forward = np.asarray(forward, dtype=np.float64)
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float64),
        forward=forward,
        velocity=forward * speed,
    )


def _l_route():
    """East for four pieces, then north. World units, y-up, road in x-z."""
    east = [(20.0 * i, 0.0, 0.0) for i in range(5)]
    north = [(80.0, 0.0, 20.0 * i) for i in range(1, 5)]
    return np.array(east + north)


def test_a_route_needs_two_points():
    try:
        PurePursuit(np.zeros((1, 3)))
    except ValueError:
        return
    raise AssertionError("accepted a one-point route")


def test_the_corner_is_found_where_the_route_turns():
    driver = PurePursuit(_l_route())
    assert list(np.flatnonzero(driver._bends)) == [4]


def test_a_target_toward_plus_z_while_facing_plus_x_steers_right():
    """The measured convention, as the controller applies it.

    Facing +x, THROTTLE_RIGHT turns the real car toward +z. A target at +z must
    therefore produce a right-hand action -- the sign every earlier controller
    had backwards.
    """
    route = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 40.0), (0.0, 0.0, 80.0)])
    driver = PurePursuit(route)
    action = driver.act(_car((0, 0, 0), (1, 0, 0), speed=5.0))
    assert action in RIGHT, action


def test_a_target_toward_minus_z_steers_left():
    route = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -40.0), (0.0, 0.0, -80.0)])
    action = PurePursuit(route).act(_car((0, 0, 0), (1, 0, 0), speed=5.0))
    assert action in LEFT, action


def test_it_brakes_for_a_corner_ahead_and_not_on_a_straight():
    route = _l_route()
    on_the_straight = PurePursuit(route, corner_lookahead=2)
    on_the_straight.target = 1
    fast = _car((0, 0, 0), (1, 0, 0), speed=20.0)
    assert on_the_straight.act(fast) not in BRAKING

    # The driver advances its target as the car arrives, it does not search for
    # the nearest point -- so a car placed part way along needs its target set,
    # exactly as one that had driven there would have it.
    nearing_the_corner = PurePursuit(route, corner_lookahead=2)
    nearing_the_corner.target = 3
    fast_near = _car((55, 0, 0), (1, 0, 0), speed=20.0)
    assert nearing_the_corner.act(fast_near) in BRAKING


def test_the_time_limit_is_milliseconds_whatever_the_backend_rate():
    """A backend knows its own rate; nothing above it may assume one.

    The stand-in ticks at 60 Hz, so two seconds is 120 of its frames, not 2000.
    Steering is irrelevant here -- this is about the clock only.
    """
    track = straight_track(length=2000.0)
    backend = ReferencePhysics(track)
    result = PurePursuit(track.centerline).drive(
        backend, limit_ms=2000, interval_ms=50
    )
    assert not result.finished
    assert 120 <= result.frames <= 123, result.frames


def test_throttle_right_turns_toward_plus_z_on_the_real_module():
    """Re-measure the convention the controller is built on."""
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.physics.wasm import WasmPhysics, WasmUnavailable
    from polytrack.trackcode import Environment, Part, TrackCode

    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return
    parts = [Part(0, 0, 0, 5, 3, 0, 0, None, 0)]
    parts += [Part(4 * i, 0, 0, 0, 3, 0, 0, None, None) for i in range(1, 12)]
    code = TrackCode(
        parts=tuple(parts), environment=Environment.SUMMER, sun_direction=90,
        name=None, author=None, last_modified=None, payload=b"",
    )
    try:
        backend = WasmPhysics(code, assets=assets)
    except WasmUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return

    state = backend.reset()
    for _ in range(20):
        state = backend.step_many(ACTION_TO_MASK[int(Action.THROTTLE)], 50).state
    before = np.asarray(state.forward, dtype=np.float64)
    for _ in range(6):
        state = backend.step_many(
            ACTION_TO_MASK[int(Action.THROTTLE_RIGHT)], 50
        ).state
    after = np.asarray(state.forward, dtype=np.float64)
    assert before[0] > 0.99, "the spawn should face +x at rotation 3"
    assert after[2] > 0.1, f"right turned toward {after}, not +z"


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
    if _SKIPPED:
        print(f"\nmodule-backed tests skipped: {_SKIPPED[0]}")
    sys.exit(1 if failures else 0)
