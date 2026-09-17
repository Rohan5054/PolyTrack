"""Procedural layout, checked against the real module rather than against itself.

Generated tracks with corners drive to the finish under a scripted driver. That
was not true for most of this module's life, and the two tests that would have
caught it are the ones added last: a single corner built at every rotation, and
whole generated tracks driven end to end. Straight-only tracks drove fine the
whole time, because a straight is symmetric under the half turn the original
model got wrong -- so every check that used only straights passed under a wrong
model. Keep the corner tests.

One earlier check is worth recording as a warning too. An adjacency test that
measured, for each piece, the furthest any vertex reached along the connecting
direction reported "0 broken connections" on a track the car could not drive.
It was measuring extent, not openness -- a barrier reaches a boundary exactly as
well as an opening does -- so it confirmed nothing while looking like proof. The
road-surface footprint check below is the version that actually asks the
question, by keeping only upward-facing triangles.
"""

from __future__ import annotations

import numpy as np

from polytrack.actions import ACTION_TO_MASK, Action
from polytrack.generator import (
    CHECKPOINT,
    FINISH,
    START,
    STEP,
    _HEADINGS,
    generate,
)
from polytrack.physics.world import PART_SIZE
from polytrack.trackcode import decode, encode

_SKIPPED: list[str] = []


def _assets():
    """Real collision geometry, or ``None`` when the game files are absent."""
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable

    try:
        return GeometryAssets.load()
    except GeometryUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None


# -- layout --------------------------------------------------------------------

def test_the_walk_never_reuses_a_cell():
    """Two pieces in one cell is not a crossover, it is two roads in one place."""
    for seed in range(8):
        track = generate(seed, pieces=40)
        assert len(set(track.cells)) == len(track.cells), seed


def test_every_step_moves_exactly_one_cell_along_one_axis():
    """A piece spans four grid units, so anything else is a gap or an overlap."""
    for seed in range(8):
        track = generate(seed, pieces=40)
        for before, after in zip(track.cells, track.cells[1:]):
            delta = [after[i] - before[i] for i in range(3)]
            moved = [d for d in delta if d != 0]
            assert moved == [STEP] or moved == [-STEP], (seed, delta)


def test_the_route_is_world_space_cell_centres():
    """What a reference line is built from, and the reason to generate at all."""
    track = generate(3, pieces=30)
    assert track.route.shape == (len(track.cells), 3)
    assert np.allclose(track.route[0], np.array(track.cells[0]) * PART_SIZE)
    # Consecutive route points are one piece apart: 4 grid units * PART_SIZE.
    spacing = np.linalg.norm(np.diff(track.route, axis=0), axis=1)
    assert np.allclose(spacing, STEP * PART_SIZE)


def test_one_start_one_finish_and_checkpoints_in_order():
    track = generate(5, pieces=50, checkpoints=3)
    parts = track.code.parts
    assert sum(1 for p in parts if p.part_id == START) == 1
    assert sum(1 for p in parts if p.part_id == FINISH) == 1
    assert sum(1 for p in parts if p.start_order is not None) == 1

    orders = sorted(
        p.checkpoint_order for p in parts if p.checkpoint_order is not None
    )
    assert orders == list(range(len(orders)))
    assert all(p.part_id == CHECKPOINT
               for p in parts if p.checkpoint_order is not None)


def test_the_start_faces_the_second_piece():
    """The car has to spawn pointing along the route, not across it."""
    for seed in range(6):
        track = generate(seed, pieces=20)
        start = next(p for p in track.code.parts if p.part_id == START)
        heading = _HEADINGS[start.rotation]
        expected = tuple(
            track.cells[0][i] + STEP * heading[i] for i in range(3)
        )
        assert expected == track.cells[1], seed


def test_a_generated_track_survives_the_codec():
    """It has to become a string the game will load, or none of this matters."""
    track = generate(7, pieces=45)
    again = decode(encode(track.code))
    assert len(again.parts) == len(track.code.parts)
    assert again.environment == track.code.environment
    assert again.sun_direction == track.code.sun_direction


def test_asking_for_an_impossible_walk_raises_rather_than_looping():
    """A self-avoiding walk can trap itself; failing to find one is not a hang."""
    try:
        generate(0, pieces=100_000, max_attempts=2)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for an unroutable length")


# -- geometry, against the real module ----------------------------------------

def test_the_road_surface_is_open_where_the_walk_expects_it():
    """Ask the collision mesh where road actually is, not how far vertices reach.

    Only upward-facing triangles count. This is the check that replaced one
    which passed on an undrivable track; see the module docstring.
    """
    assets = _assets()
    if assets is None:
        return

    def footprint(part_id):
        vertices = np.asarray(assets[part_id].vertices, np.float64).reshape(-1, 3, 3)
        a, b, c = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        normals = np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        good = lengths > 1e-9
        upward = np.zeros(len(vertices), bool)
        upward[good] = (normals[good, 1] / lengths[good]) > 0.7
        return vertices[upward]

    straight = footprint(0)
    assert len(straight) > 0
    # A straight is open along local z and walled along local x.
    assert straight[:, :, 2].max() >= 9.5
    assert straight[:, :, 2].min() <= -9.5
    assert straight[:, :, 0].max() < 9.5

    # TurnSharp is open on local +z and local +x, and closed on -z.
    turn = footprint(1)
    assert turn[:, :, 2].max() >= 9.5
    assert turn[:, :, 0].max() >= 9.5
    assert turn[:, :, 2].min() > -9.5


def test_corners_are_never_back_to_back():
    """At least one straight between corners: no hairpin inside two pieces.

    Two same-handed right angles back to back reverse the car inside 40 m with
    the return leg in the next lane. The scripted driver stopped dead in exactly
    that on the one generated track out of ten it could not finish.
    """
    for seed in range(40):
        track = generate(seed, pieces=30)
        order = {cell: index for index, cell in enumerate(track.cells)}
        corners = sorted(
            order[(p.x, p.y, p.z)] for p in track.code.parts if p.part_id == 1
        )
        assert all(b - a >= 2 for a, b in zip(corners, corners[1:])), (seed, corners)


def _backend(code, assets):
    from polytrack.physics.wasm import WasmPhysics, WasmUnavailable

    try:
        return WasmPhysics(code, assets=assets)
    except WasmUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None


def test_a_corner_connects_only_at_the_rotation_the_model_predicts():
    """The bug this module shipped with, pinned in miniature.

    A car heading +x that must leave the corner toward +z needs the turn at
    rotation 3. The original model, which read the part's axes off the car's
    start transform and so carried a half turn it should not have, chose
    rotation 1 -- and the car went nowhere.
    """
    assets = _assets()
    if assets is None:
        return
    from polytrack.pursuit import PurePursuit
    from polytrack.trackcode import Environment, Part, TrackCode

    def corner(rotation):
        parts = [Part(0, 0, 0, START, 3, 0, 0, None, 0)]
        parts += [Part(4 * i, 0, 0, 0, 3, 0, 0, None, None) for i in (1, 2, 3)]
        parts.append(Part(16, 0, 0, 1, rotation, 0, 0, None, None))
        cells = [(4 * i, 0, 0) for i in range(5)]
        for k in range(1, 6):
            cell = (16, 0, 4 * k)
            cells.append(cell)
            piece = FINISH if k == 5 else 0
            parts.append(Part(*cell, piece, 2, 0, 0, None, None))
        code = TrackCode(
            parts=tuple(parts), environment=Environment.SUMMER,
            sun_direction=90, name=None, author=None, last_modified=None,
            payload=b"",
        )
        return code, np.array(cells, dtype=np.float64) * PART_SIZE

    reached = {}
    for rotation in (3, 1):
        code, route = corner(rotation)
        backend = _backend(code, assets)
        if backend is None:
            return
        result = PurePursuit(route).drive(backend, limit_ms=25_000)
        reached[rotation] = (result.finished, result.reached)

    assert reached[3][0], f"the predicted rotation should finish: {reached}"
    assert not reached[1][0] and reached[1][1] <= 5, (
        f"the old model's rotation should not get past the corner: {reached}"
    )


def test_generated_tracks_with_corners_drive_to_the_finish():
    """End to end on the real module: walk, codec, geometry, checkpoints, driver."""
    assets = _assets()
    if assets is None:
        return
    from polytrack.pursuit import PurePursuit

    for seed in (0, 4):
        track = generate(seed, pieces=20, turn_chance=0.35, checkpoints=2)
        assert any(p.part_id == 1 for p in track.code.parts), "no corners"
        backend = _backend(track.code, assets)
        if backend is None:
            return
        result = PurePursuit(track.route).drive(backend, limit_ms=60_000)
        assert result.finished, (seed, result)
        assert result.checkpoints == 2, (seed, result)


def test_a_straight_track_drives_to_the_finish():
    """The simplest end-to-end case, with no steering at all."""
    assets = _assets()
    if assets is None:
        return
    from polytrack.physics.wasm import WasmPhysics, WasmUnavailable

    track = generate(0, pieces=12, turn_chance=0.0, checkpoints=1)
    try:
        backend = WasmPhysics(track.code, assets=assets)
    except WasmUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return

    backend.reset()
    result = None
    for _ in range(200):
        result = backend.step_many(ACTION_TO_MASK[int(Action.THROTTLE)], 50)
        if result.finished:
            break
    assert result is not None and result.finished, "a straight track must finish"
    assert result.state.checkpoint >= 1


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
