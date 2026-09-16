"""Procedural layout: the invariants that hold, and an honest line round what does not.

The generator is **not finished**. Straight-only tracks are laid out correctly
and drive to the finish on the real module; tracks containing turns are laid out
and encode fine, but the car does not get through the first corner. These tests
pin the half that works so the half that does not cannot quietly break it.

One earlier check is worth recording as a warning. An adjacency test that
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


def test_a_straight_track_drives_to_the_finish():
    """End to end on the real module: layout, codec, geometry, checkpoints.

    Straight-only, because that is what currently works. A track containing
    turns is laid out and encodes correctly but the car does not get through the
    first corner, and that gap is the generator's open bug rather than something
    this test should assert.
    """
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
