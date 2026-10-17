"""The real physics backend.

Split in two. The first half tests the pieces that surround the module -- the
world construction, the quaternion arithmetic, the record parser -- and runs
anywhere. The second half drives the actual module and needs two files that are
deliberately not in the repo, because both are derived from Kodub's game:

    polytrack_physics.wasm      extracted from the desktop build's app.asar
    geometry.npz                built by tools/extract_geometry.mjs

Those tests skip with a note when the files are absent, and run automatically
when they are present. Set ``POLYTRACK_WASM`` / ``POLYTRACK_GEOMETRY`` to point
elsewhere.

The gate this project actually turns on -- replaying a known recording and
reproducing its finish frame exactly -- is not here. It is a separate harness
against a real ghost; see PROJECT_STATUS.md. What is here is everything that
must hold before that gate is even worth running.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from polytrack.trackcode import PART_SIZE, ROTATION_QUATERNIONS, Part, TrackCode
from polytrack.physics.world import (
    _TERRAIN_TABLE,
    _TableCycle,
    angular_velocity,
    forward_vector,
    mountain_vertices,
    quaternion_multiply,
    rotate_vector,
    track_bounds,
    up_vector,
)

_SKIPPED: list[str] = []


def _backend():
    """A backend on a small synthetic track, or ``None`` if the files are absent."""
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.physics.wasm import WasmPhysics, WasmUnavailable

    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None
    try:
        return WasmPhysics(_flat_track(assets), assets=assets)
    except WasmUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None


def _flat_track(assets) -> TrackCode:
    """A start piece and a run of straights: enough to drive on, cheap to build."""
    start_id = next(p.id for p in assets.parts if p.name == "Start")
    straight_id = next(p.id for p in assets.parts if p.name == "Straight")
    parts = [Part(0, 0, 0, start_id, 0, 0, 0, None, 0)]
    parts += [Part(0, 0, -z, straight_id, 0, 0, 0, None, None) for z in range(1, 25)]
    return TrackCode(parts=tuple(parts), name="Straight line")


# ------------------------------------------------------------------- world --


def test_the_half_turn_is_not_exactly_a_half_turn():
    """``Euler(0, pi, 0)`` carries a 6.1e-17 residue in w, and it must be kept.

    Rounding it to a clean ``(0, 1, 0, 0)`` moves the spawn point by a few parts
    in 1e17 -- invisible, and exactly the kind of difference that makes a replay
    diverge a thousand frames later instead of failing outright.
    """
    from polytrack.physics.world import _HALF_TURN_Y

    x, y, z, w = _HALF_TURN_Y
    assert (x, y, z) == (0.0, 1.0, 0.0)
    assert w != 0.0 and abs(w) < 1e-15


def test_quaternion_multiply_matches_three_js_operand_order():
    """``a * b`` is not ``b * a``; the client composes placement then half turn."""
    a = ROTATION_QUATERNIONS[2][1]
    b = ROTATION_QUATERNIONS[4][3]
    assert quaternion_multiply(a, b) != quaternion_multiply(b, a)
    # Composition must agree with applying the rotations in sequence.
    v = (0.3, -1.7, 2.1)
    composed = rotate_vector(v, quaternion_multiply(a, b))
    sequential = rotate_vector(rotate_vector(v, b), a)
    assert np.allclose(composed, sequential, atol=1e-12)


def test_forward_and_up_are_orthonormal():
    for axis in range(6):
        for rotation in range(4):
            q = ROTATION_QUATERNIONS[axis][rotation]
            forward, up = forward_vector(q), up_vector(q)
            assert abs(np.linalg.norm(forward) - 1.0) < 1e-12
            assert abs(np.linalg.norm(up) - 1.0) < 1e-12
            assert abs(float(np.dot(forward, up))) < 1e-12


def test_angular_velocity_recovers_a_known_rotation():
    """Including past a half turn, where the small-angle form silently fails."""
    rate = 1000.0
    for angle in (0.01, 0.5, math.pi * 0.9):
        half = angle / 2
        previous = (0.0, 0.0, 0.0, 1.0)
        current = (0.0, math.sin(half), 0.0, math.cos(half))
        omega = angular_velocity(previous, current, rate)
        assert np.allclose(omega, [0.0, angle * rate, 0.0], atol=1e-9)
    # An unchanged orientation must give exactly zero, not a denormal.
    q = ROTATION_QUATERNIONS[3][2]
    assert np.array_equal(angular_velocity(q, q, rate), np.zeros(3))


def test_terrain_table_starts_at_index_one():
    """The client pre-increments, so the first draw is table[1].

    Off by one here shifts every mountain vertex, which no test of ours would
    otherwise notice and which the acceptance gate would report as a plain
    desync.
    """
    cycle = _TableCycle()
    assert cycle.next() == _TERRAIN_TABLE[1]
    assert cycle.next() == _TERRAIN_TABLE[2]
    # And it wraps rather than running off the end.
    for _ in range(len(_TERRAIN_TABLE)):
        cycle.next()
    assert 0.0 <= cycle.next() <= 1.0


def test_mountains_are_reproducible_and_clear_of_the_track():
    """Same track, same terrain -- and terrain far enough out to never be hit."""
    parts = tuple(
        Part(x, 0, z, 0, 0, 0, 0, None, None) for x in range(-20, 21) for z in (-8, 8)
    )
    track = TrackCode(parts=parts)
    first, offset = mountain_vertices(track)
    second, offset_again = mountain_vertices(track)
    assert np.array_equal(first, second) and offset == offset_again
    assert first.dtype == np.float32 and first.size % 9 == 0  # whole triangles

    min_x, min_z, max_x, max_z = track_bounds(track)
    reach = max(abs(max_x - min_x), abs(max_z - min_z)) * PART_SIZE / 2
    radial = np.hypot(first[0::3], first[2::3])
    assert radial.min() > reach + 100.0


def test_an_empty_track_produces_no_terrain_and_no_crash():
    vertices, offset = mountain_vertices(TrackCode(parts=()))
    assert vertices.size > 0  # the 200 m floor still applies
    assert offset == (0.0, 0.0, 0.0)


def test_parts_are_ordered_the_way_the_client_orders_them():
    """Canonical order is (partId, x, y, z, rotation, axis, colour, orders).

    The client binary-inserts into sorted lists rather than appending, so a
    track built in a different order still hands the module the same buffer.
    The procedural generator will rely on this.
    """
    shuffled = TrackCode(
        parts=(
            Part(5, 0, 1, 7, 0, 0, 0, None, None),
            Part(1, 0, 0, 3, 0, 0, 0, None, None),
            Part(0, 0, 0, 7, 0, 0, 0, None, None),
            Part(0, 0, 0, 3, 1, 0, 0, None, None),
        )
    )
    ordered = shuffled.ordered_parts()
    assert [(p.part_id, p.x, p.rotation) for p in ordered] == [
        (3, 0, 1), (3, 1, 0), (7, 0, 0), (7, 5, 0),
    ]
    # The buffer follows the same order, 19 bytes each.
    buffer = shuffled.physics_buffer()
    assert len(buffer) == 4 * 19
    assert buffer[0] == 3 and buffer[19 * 2] == 7


# ------------------------------------------------------------------ record --


#: Offset of ``nCollisionImpulses`` in a not-yet-finished buffer: the i32 carId,
#: then frames, speedKmh, flags, nextCheckpointIndex, position, quaternion.
_IMPULSE_COUNT_AT = 4 + 3 + 4 + 1 + 2 + 12 + 16


def _synthetic_record(
    *, finished: bool, contact: tuple[bool, ...], impulses: int = 2
) -> bytes:
    """Build a record byte-for-byte the way the module writes one."""
    flags = 1 | (2 if finished else 0)
    for index, touching in enumerate(contact):
        if touching:
            flags |= 8 << index
    out = bytearray()
    out += (0).to_bytes(4, "little")  # carId prefix
    out += (1234).to_bytes(3, "little")  # frames
    out += struct.pack("<f", 42.5)  # speedKmh
    out.append(flags)
    if finished:
        out += (1234).to_bytes(3, "little")  # finishFrames
    out += (3).to_bytes(2, "little")  # nextCheckpointIndex
    out += struct.pack("<3f", 1.0, 2.0, 3.0)
    out += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0)
    out.append(impulses)
    out += struct.pack(f"<{impulses}f", *(0.25 * (n + 1) for n in range(impulses)))
    for touching in contact:
        if touching:
            out += struct.pack("<6f", *range(6))
    for _ in range(4):
        out += struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    out += struct.pack("<f", -0.25)
    out.append(1 | 32)
    return bytes(out)


def test_record_parses_at_both_extents():
    from polytrack.physics.wasm import has_finished, parse_car_record

    for finished in (False, True):
        for contact in ((False,) * 4, (True,) * 4, (True, False, True, False)):
            raw = _synthetic_record(finished=finished, contact=contact)
            record = parse_car_record(raw[4:])
            assert record.frames == 1234
            assert record.has_finished is finished
            assert record.finish_frames == (1234 if finished else None)
            assert record.next_checkpoint_index == 3
            assert record.position == (1.0, 2.0, 3.0)
            assert record.wheel_contact == contact
            assert record.collision_impulses == (0.25, 0.5)
            assert record.quaternion == (0.0, 0.0, 0.0, 1.0)
            assert record.steering == -0.25
            assert record.brake_light_enabled is True
            assert record.n_bytes == len(raw) - 4
            # The cheap flag read the game's verifier uses must agree.
            assert has_finished(raw) is finished


def test_the_full_record_fits_the_buffer_the_game_allocates():
    from polytrack.physics.wasm import STATE_BUFFER_SIZE, parse_car_record

    # Maximum extent: finished, all four wheels down, four collision impulses.
    raw = _synthetic_record(finished=True, contact=(True,) * 4, impulses=4)
    assert len(raw) == STATE_BUFFER_SIZE
    assert parse_car_record(raw[4:]).n_bytes == STATE_BUFFER_SIZE - 4


def test_a_drifted_cursor_is_caught_rather_than_returning_noise():
    """The impulse count is bounded by 4, which makes it a usable tripwire."""
    from polytrack.physics.wasm import parse_car_record

    raw = bytearray(_synthetic_record(finished=False, contact=(False,) * 4))
    assert raw[_IMPULSE_COUNT_AT] == 2  # the offset is where we think it is
    raw[_IMPULSE_COUNT_AT] = 63
    try:
        parse_car_record(bytes(raw[4:]))
    except ValueError:
        return
    raise AssertionError("a drifted record parsed without complaint")


# ----------------------------------------------------------------- backend --


def test_the_module_passes_its_own_determinism_self_test():
    """``testDeterminism`` is the check the client runs before submitting a run.

    It tests the host, not us. A false here means nothing this backend produces
    is worth anything, whatever else passes.
    """
    backend = _backend()
    if backend is None:
        return
    assert backend.test_determinism() is True


def test_the_car_spawns_where_the_client_would_put_it():
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.physics.world import start_transform

    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable:
        return
    track = _flat_track(assets)
    position, quaternion = start_transform(track, assets)

    start = assets[track.parts[0].part_id]
    assert start.start_offset is not None
    # Placement is identity here, so the whole transform is the half turn: the
    # car faces -Z and sits one start-offset ahead of the tile centre.
    assert np.allclose(forward_vector(quaternion), [0.0, 0.0, -1.0], atol=1e-12)
    assert abs(position[1] - start.start_offset[1]) < 1e-9
    assert abs(position[2] + start.start_offset[2]) < 1e-9


def test_the_same_inputs_give_the_same_trajectory_from_a_fresh_instance():
    """Determinism across instances, which is what replay verification needs."""
    from polytrack.physics.wasm import WasmPhysics

    if _backend() is None:
        return
    from polytrack.physics.assets import GeometryAssets

    assets = GeometryAssets.load()
    track = _flat_track(assets)
    masks = [1, 1, 3, 3, 9, 1, 1, 5, 5, 1]

    def run():
        backend = WasmPhysics(track, assets=assets)
        return [backend.step_many(mask, 20).info["record"] for mask in masks]

    assert run() == run()


def test_holding_an_input_is_the_same_as_stepping_it_tick_by_tick():
    """``step_many`` skips the per-tick parse; it must not skip anything else."""
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics

    if _backend() is None:
        return
    assets = GeometryAssets.load()
    track = _flat_track(assets)

    batched = WasmPhysics(track, assets=assets)
    one_at_a_time = WasmPhysics(track, assets=assets)
    for mask in (1, 3, 1, 9):
        expected = None
        for _ in range(25):
            expected = one_at_a_time.step(mask)
        actual = batched.step_many(mask, 25)
        assert actual.info["record"] == expected.info["record"]
        assert actual.state.frame == expected.state.frame
        assert np.array_equal(actual.state.velocity, expected.state.velocity)


def test_restoring_a_saved_state_reproduces_the_rest_of_the_run():
    """The whole point of save/restore: a resumed branch must not drift.

    A save that misses hidden state passes every shallow check and then desyncs
    only under search, so this replays a long tail rather than one step.
    """
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics

    if _backend() is None:
        return
    assets = GeometryAssets.load()
    backend = WasmPhysics(_flat_track(assets), assets=assets)

    for mask in (1, 1, 3):
        backend.step_many(mask, 200)
    blob = backend.save_state()
    frame_at_save = backend.frame

    tail = [backend.step_many(mask, 100).info["record"] for mask in (1, 9, 5, 1, 3)]

    restored = backend.load_state(blob)
    assert backend.frame == frame_at_save
    assert np.allclose(restored.position, backend.record.position)

    again = [backend.step_many(mask, 100).info["record"] for mask in (1, 9, 5, 1, 3)]
    assert tail == again


def test_the_car_actually_drives():
    """A guard against a backend that runs, reports, and simulates nothing.

    An earlier probe reported a clean determinism PASS on a call sequence that
    did nothing at all, so "it returned a well-formed record" is not evidence.
    """
    backend = _backend()
    if backend is None:
        return
    result = backend.step_many(1, 3000)  # three seconds of throttle
    assert result.info["has_started"]
    assert result.info["speed_kmh"] > 50.0
    travelled = np.linalg.norm(result.state.position - backend._start_position)
    assert travelled > 20.0
    # Speed from the record and speed from the finite difference must agree.
    assert abs(np.linalg.norm(result.state.velocity) - result.info["speed_kmh"] / 3.6) < 1.0


def test_building_and_closing_backends_does_not_grow_memory():
    """A rebuilt backend must hand its linear memory back.

    Refcounting does not do it: the imports are Funcs over bound methods, the
    Instance holds the Funcs and the module holds the Instance, and that cycle
    runs through wasmtime's C types where the cycle collector cannot follow it.
    Measured before the fix, build-and-drop cost 29 MB of private commit each,
    linearly, with del and an explicit gc.collect() -- so bounding the
    collector's environment cache converted a large fixed footprint into an
    unbounded leak, one eviction per episode.

    Checked by proxy rather than by reading the OS: after close(), the module's
    store and instance must be gone, which is what frees the memory. The
    end-to-end memory measurement lives in the commit message; this is the part
    that can run anywhere.
    """
    backend = _backend()
    if backend is None:
        return
    module = backend.module
    assert module.store is not None and module.instance is not None

    backend.close()
    assert module.store is None, "close() must release the store"
    assert module.instance is None, "close() must release the instance"
    assert not module._bound, "close() must drop the bound export cache"

    backend.close()  # idempotent: a cache may close an entry more than once


def test_the_compiled_module_is_shared_between_backends():
    """Compiling per backend is the other half of the cost: 226 ms -> 141 ms."""
    from polytrack.physics.wasm import _COMPILED, _resolve_wasm, WasmUnavailable

    first = _backend()
    if first is None:
        return
    try:
        resolved = _resolve_wasm(None)
    except WasmUnavailable:
        return
    assert resolved in _COMPILED
    engine, module = _COMPILED[resolved]

    second = _backend()
    assert second is not None
    assert _COMPILED[resolved][1] is module, "the module should be compiled once"
    # Separate stores, though -- save/restore and determinism depend on it.
    assert first.module.store is not second.module.store


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
