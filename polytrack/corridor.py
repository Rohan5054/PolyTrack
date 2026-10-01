"""Where the ground is, from the collision geometry the module actually uses.

This replaces the last approximation in the world model. `ObservationBuilder`
used to probe the corridor by asking how far a point was from the *reference
line* and calling anything past `Track.width` a wall. That measures distance
from a lap somebody drove, not distance from an edge: where the driven line
hugged one side, it reported a wall on open road and open road where a wall
was. On a generated track, whose reference line is its centreline, it is closer
to meaningless.

What the car can actually be held up by is **any collision surface**, at any
orientation, whatever part it belongs to. PolyTrack has no notion of "off
track" -- the module never reports leaving the road, because leaving the road is
just running out of things to touch. So this asks the only question that
matters: *is there solid geometry here, near the height the car is at?*

Orientation deliberately does not filter. Restricting to upward-facing surfaces
is the obvious reading of "ground" and it is wrong on this game: **the car
drives on walls.** Winter 5's human record wall-rides at 400 km/h along
`WallTrackBottom` pieces, whose supporting faces are vertical, and an
upward-only map reported that real lap as off-ground for 14% of its length.

That it does not care which part a surface came from is the point. In the
shipped tracks the most common pieces by far are `Block`, `Plane` and
`PillarMiddle` -- structure, not road -- and cars drive over plenty of them.
Anything that enumerated road pieces would get those wrong.

## Why a height map rather than a triangle query

A ray cast against the raw mesh would be exact and far too slow: a 19-second
lap is 19,000 ticks and an observation is built every control interval. Parts
sit on a regular grid and their surfaces are overwhelmingly horizontal, so a
rasterised height map answers "is there ground at (x, z) near y" in one array
lookup, and a ray becomes a march over cells.

Several surfaces sit above one another -- bridges, stacked blocks, and on the
shipped tracks a great deal of scenery directly over the road. Each cell
therefore stores a **vertical occupancy bitmask**: one bit per height bin, set
if any surface fell in it. A query tests the bits spanning the car's height
plus a tolerance.

Keeping the N highest surfaces per cell instead, which is the obvious design,
is wrong and measurably so: on Winter 5 the scenery above the track evicted the
road itself from a third of its cells, and a real human lap read as off-ground
for 33% of its length. A bitmask costs the same memory and cannot lose a
surface.

## Rotation

Parts are placed with `ROTATION_QUATERNIONS[axis][rotation]` and **no half
turn**. The half turn in `start_transform` orients the car, not the part, and
assuming otherwise put every corner in the procedural generator 180 degrees out
-- a bug that straight-only tracks could not reveal, because a straight is
symmetric under it. `tests/test_corridor.py` pins the transform by checking that
a real human lap stays on ground for its whole length, which it cannot do if
this is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics.assets import GeometryAssets
from .physics.world import PART_SIZE, ROTATION_QUATERNIONS
from .trackcode import TrackCode

#: Cached per part id: collision triangles in the part's local frame.
_LOCAL_SURFACES: dict[int, np.ndarray] = {}


def _rotation_matrix(quaternion) -> np.ndarray:
    """3x3 rotation from (x, y, z, w), matching `world.rotate_vector`."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _local_surfaces(assets: GeometryAssets, part_id: int) -> np.ndarray:
    """Every collision triangle of a part, (T, 3, 3), in its own frame."""
    cached = _LOCAL_SURFACES.get(part_id)
    if cached is None:
        cached = np.asarray(
            assets[part_id].vertices, dtype=np.float64
        ).reshape(-1, 3, 3)
        _LOCAL_SURFACES[part_id] = cached
    return cached


#: Height bins per cell. One uint64 of bits, so this is the hard ceiling.
_BINS = 64


@dataclass(frozen=True, slots=True)
class RoadMap:
    """Vertical occupancy of ground on a grid over the track's footprint."""

    #: World (x, z) of the centre of cell [0, 0].
    origin: tuple[float, float]
    resolution: float
    #: World y of the bottom of bin 0, and the height each bin spans.
    base_y: float
    bin_size: float
    #: (nz, nx) uint64; bit b set when some surface lies in height bin b.
    occupancy: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupancy.shape

    def _bits(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """A mask of every bin overlapping each [low, high] height span."""
        first = np.clip(
            np.floor((low - self.base_y) / self.bin_size), 0, _BINS - 1
        ).astype(np.int64)
        last = np.clip(
            np.floor((high - self.base_y) / self.bin_size), 0, _BINS - 1
        ).astype(np.int64)
        # Bits [first, last] inclusive, built without a loop over cells.
        all_above = (~np.uint64(0)) << first.astype(np.uint64)
        all_above_last = np.where(
            last >= _BINS - 1,
            np.uint64(0),
            (~np.uint64(0)) << (last + 1).astype(np.uint64),
        )
        return all_above & ~all_above_last

    @classmethod
    def build(
        cls,
        track: TrackCode,
        assets: GeometryAssets,
        *,
        resolution: float = 1.0,
        margin: float = 12.0,
    ) -> RoadMap:
        """Rasterise every upward-facing surface in the track."""
        positions = np.array(
            [[p.x, p.y, p.z] for p in track.parts], dtype=np.float64
        ) * PART_SIZE
        low = positions.min(axis=0) - margin
        high = positions.max(axis=0) + margin
        nx = int(np.ceil((high[0] - low[0]) / resolution)) + 1
        nz = int(np.ceil((high[2] - low[2]) / resolution)) + 1
        origin = (float(low[0]), float(low[2]))

        # Parts carry surfaces above their own origin, so the vertical span has
        # to allow for a part's own height as well as the span of placements.
        base_y = float(low[1]) - margin
        bin_size = max(0.5, (float(high[1]) + 4 * margin - base_y) / (_BINS - 1))
        occupancy = np.zeros((nz, nx), dtype=np.uint64)

        for part in track.parts:
            local = _local_surfaces(assets, part.part_id)
            if len(local) == 0:
                continue
            rotation = _rotation_matrix(
                ROTATION_QUATERNIONS[part.rotation_axis][part.rotation]
            )
            offset = np.array([part.x, part.y, part.z], dtype=np.float64) * PART_SIZE
            world = local.reshape(-1, 3) @ rotation.T + offset
            _stamp(
                occupancy, origin, resolution, base_y, bin_size,
                world.reshape(-1, 3, 3),
            )

        return cls(
            origin=origin, resolution=resolution, base_y=base_y,
            bin_size=bin_size, occupancy=occupancy,
        )

    # -- queries ------------------------------------------------------------

    def ground(
        self, points: np.ndarray, reference_y: np.ndarray | float, *,
        below: float = 12.0, above: float = 4.0,
    ) -> np.ndarray:
        """Is there ground near each (x, y, z) point, relative to ``reference_y``?

        The band is **asymmetric**, and that is the whole point. "Is there road
        ahead" means "is there something down there I would land on", so the
        search reaches well below the car and only a little above it. A
        symmetric band gets this wrong in a way that looks like a geometry bug:
        on Winter 5 the car leaves the ground often enough that a +/-3 m test
        reported a real human lap as off-ground for 15% of its length.

        ``above`` is small because a surface above the car is a ceiling or a
        bridge deck, not somewhere it can drive.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        ix = np.rint((points[:, 0] - self.origin[0]) / self.resolution).astype(int)
        iz = np.rint((points[:, 2] - self.origin[1]) / self.resolution).astype(int)
        nz, nx = self.shape
        inside = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        out = np.zeros(len(points), dtype=bool)
        if not inside.any():
            return out
        target = np.asarray(reference_y, dtype=np.float64)
        target = target[inside] if target.ndim else np.full(int(inside.sum()), target)
        wanted = self._bits(target - below, target + above)
        out[inside] = (self.occupancy[iz[inside], ix[inside]] & wanted) != 0
        return out

    def cast(
        self,
        origin: np.ndarray,
        directions: np.ndarray,
        *,
        max_distance: float,
        step: float = 1.0,
        below: float = 12.0,
        above: float = 4.0,
    ) -> np.ndarray:
        """Distance along each direction until the ground runs out.

        Marches every ray in lockstep as one batched query rather than looping
        per ray -- the observation is built every control interval, and a Python
        loop here was previously the single biggest cost in the step function.
        """
        directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))
        distances = np.arange(step, max_distance + step, step)
        probes = (
            np.asarray(origin, dtype=np.float64)[None, None, :]
            + directions[:, None, :] * distances[None, :, None]
        )
        solid = self.ground(
            probes.reshape(-1, 3), float(origin[1]), below=below, above=above
        ).reshape(len(directions), len(distances))

        gap = ~solid
        first = np.argmax(gap, axis=1)
        any_gap = gap.any(axis=1)
        return np.where(any_gap, distances[first], max_distance)


def _stamp(
    occupancy: np.ndarray, origin: tuple[float, float], resolution: float,
    base_y: float, bin_size: float, triangles: np.ndarray,
) -> None:
    """Mark the height span of each triangle in every cell it covers.

    A triangle's whole vertical extent is marked across its footprint rather
    than its surface height interpolated per cell. That is deliberately
    conservative, and it is what lets a vertical face register at all: a wall's
    projection onto the ground plane is a line with zero area, so anything
    interpolating a height over that projection divides by zero and drops it.
    Collision triangles here span a couple of metres, so the over-marking is on
    that order.
    """
    nz, nx = occupancy.shape
    for tri in triangles:
        xs = (tri[:, 0] - origin[0]) / resolution
        zs = (tri[:, 2] - origin[1]) / resolution
        x0, x1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        z0, z1 = int(np.floor(zs.min())), int(np.ceil(zs.max()))
        x0, z0 = max(x0, 0), max(z0, 0)
        x1, z1 = min(x1, nx - 1), min(z1, nz - 1)
        if x1 < x0 or z1 < z0:
            continue

        low = int(np.clip(np.floor((tri[:, 1].min() - base_y) / bin_size), 0, _BINS - 1))
        high = int(np.clip(np.floor((tri[:, 1].max() - base_y) / bin_size), 0, _BINS - 1))
        mask = np.uint64(0)
        for bit in range(low, high + 1):
            mask |= np.uint64(1) << np.uint64(bit)
        occupancy[z0:z1 + 1, x0:x1 + 1] |= mask
