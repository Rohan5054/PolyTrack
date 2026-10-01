"""The road map: where the car can actually be, from the real collision geometry.

The load-bearing test is `test_a_real_human_lap_is_on_ground_for_its_whole_length`.
It passes only if the part rotation, the rasterisation and the height query are
all right at once, and it caught two wrong models that each looked plausible:

* keeping the highest surfaces per cell, which the scenery above Winter 5's
  road evicted the road itself out of -- a real lap read as off-ground for 33%
  of its length;
* counting only upward-facing surfaces, which is the obvious reading of
  "ground" and is wrong on this game, because **the car drives on walls**.
  Winter 5's record wall-rides at 400 km/h on vertical faces, and an
  upward-only map lost 14% of that lap.

A map that called everything solid would also score 100%, so
`test_the_map_distinguishes_road_from_void` is its necessary other half.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from polytrack.corridor import RoadMap

_SKIPPED: list[str] = []


def _winter_5():
    """(track code, assets, road map), or None when the game files are absent."""
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable

    try:
        assets = GeometryAssets.load()
        entry = TrackLibrary.load("tracks.npz").by_name("Winter 5")
    except (GeometryUnavailable, FileNotFoundError, KeyError) as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None
    code = entry.track_code()
    return code, assets, RoadMap.build(code, assets, resolution=1.0)


def _lap_positions(code, assets, frames=19025):
    """Where the car actually went on the human record, sampled every 50 ms."""
    from polytrack.physics.wasm import WasmPhysics, WasmUnavailable
    from polytrack.recording import Recording

    records = json.loads(Path("records.json").read_text())
    run = next((r for r in records if r["frames"] == frames), None)
    if run is None:
        _SKIPPED.append(f"no {frames}-frame record in records.json")
        return None
    try:
        backend = WasmPhysics(code, assets=assets)
    except WasmUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None

    backend.reset()
    positions = []
    for mask, ticks in Recording.deserialize(run["recording"]).runs(frames):
        remaining = ticks
        while remaining > 0:
            step = min(50, remaining)
            result = backend.step_many(mask, step)
            remaining -= step
            positions.append(np.asarray(result.state.position))
            if result.finished:
                break
    return np.array(positions)


def test_a_real_human_lap_is_on_ground_for_its_whole_length():
    """The one test that pins rotation, rasterisation and query together."""
    built = _winter_5()
    if built is None:
        return
    code, assets, road = built
    positions = _lap_positions(code, assets)
    if positions is None:
        return

    on = road.ground(positions, positions[:, 1], below=3.0, above=3.0)
    assert on.all(), (
        f"{(~on).sum()} of {len(on)} samples of a real lap had no ground under "
        f"them; first at {np.round(positions[np.flatnonzero(~on)[0]], 1)}"
    )


def test_the_map_distinguishes_road_from_void():
    """Otherwise 'always solid' would pass the test above and say nothing."""
    built = _winter_5()
    if built is None:
        return
    code, assets, road = built

    occupied = (road.occupancy != 0).mean()
    assert 0.1 < occupied < 0.9, f"{occupied:.1%} of cells solid is not a map"

    positions = _lap_positions(code, assets)
    if positions is None:
        return
    # Well above the track there is nothing. Displacing sideways instead would
    # be the obvious test and is unsound: Winter 5 doubles back on itself, so a
    # 150 m shift along x lands back on the track often enough to fail.
    above = positions.copy()
    above[:, 1] += 400.0
    assert not road.ground(above, above[:, 1], below=5.0, above=5.0).any()

    # And outside the map's footprint entirely.
    outside = positions[:5].copy()
    outside[:, 0] += 50_000.0
    assert not road.ground(outside, outside[:, 1]).any()


def test_the_height_band_is_asymmetric():
    """Below the car is somewhere to land; above it is a ceiling."""
    from polytrack.trackcode import Environment, Part, TrackCode

    built = _winter_5()
    if built is None:
        return
    _, assets, _ = built
    code = TrackCode(
        parts=(Part(0, 0, 0, 5, 3, 0, 0, None, 0),),
        environment=Environment.SUMMER, sun_direction=90,
        name=None, author=None, last_modified=None, payload=b"",
    )
    road = RoadMap.build(code, assets, resolution=1.0)
    point = np.array([[0.0, 0.0, 0.0]])

    assert road.ground(point, 20.0, below=40.0, above=2.0).all(), "should reach down"
    assert not road.ground(point, -20.0, below=2.0, above=2.0).any(), (
        "should not reach 20 m up"
    )


def test_a_ray_stops_where_the_ground_does():
    built = _winter_5()
    if built is None:
        return
    code, assets, road = built
    positions = _lap_positions(code, assets)
    if positions is None:
        return

    origin = positions[len(positions) // 2]
    # Straight up-track and straight out sideways, from the same point.
    sideways = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                         [0.0, 0.0, -1.0]])
    hits = road.cast(origin, sideways, max_distance=60.0, step=1.0)
    assert hits.shape == (4,)
    assert (hits > 0).all() and (hits <= 60.0).all()
    # A point in the middle of a track cannot have solid ground for 60 m in
    # every direction, or the map is not measuring an edge at all.
    assert hits.min() < 60.0


def test_rays_fall_back_to_the_reference_line_without_a_map():
    """The stand-in and the synthetic demo tracks have no collision geometry."""
    from polytrack.observation import ObservationBuilder
    from polytrack.track import chicane_track

    builder = ObservationBuilder(chicane_track())
    assert builder.road is None
    assert builder.size > 0


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
        print(f"\ngeometry-backed tests skipped: {_SKIPPED[0]}")
    sys.exit(1 if failures else 0)
