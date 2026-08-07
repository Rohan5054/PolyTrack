"""Turning a decoded track into the world the physics module is given.

``createCarModel`` takes three things the track code does not directly contain:
the car's **start transform**, the **mountain terrain** ring, and the packed
part buffer. The first two are computed by the client, so they are computed the
same way here -- including the parts that look like they should not matter.

Two of those deserve justification, because both are tempting to skip:

**The mountains are collision geometry, not scenery.** The client generates a
ring of terrain from the track's bounding box and passes its vertices straight
into ``createCarModel``. The ring sits at least 160 m outside the track, so a
car will almost never touch it -- but "almost never" is not a property to bet a
determinism gate on, and the vertices also change how the module builds its
static broadphase.

**The generator is reproducible.** What looks like a PRNG is a fixed table of
128 pre-generated values, cycled -- so the terrain is identical in every client
and identical here, with no seed to guess.

Quaternion arithmetic is transcribed from the three.js the game bundles rather
than written from the maths, so operand order and float rounding match.
"""

from __future__ import annotations

import math

import numpy as np

from .assets import GeometryAssets
from ..trackcode import PART_SIZE, ROTATION_QUATERNIONS, TrackCode

Quaternion = tuple[float, float, float, float]  # (x, y, z, w)
Vector3 = tuple[float, float, float]

#: The start tile faces the car back down the track, so the client composes the
#: part's placement rotation with a half turn about Y. Written as ``Euler(0, pi,
#: 0)`` there, which is *not* exactly ``(0, 1, 0, 0)``: ``cos(pi/2)`` is 6.1e-17,
#: not zero, and that residue propagates into the start position.
_HALF_TURN_Y: Quaternion = (0.0, math.sin(math.pi / 2), 0.0, math.cos(math.pi / 2))


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """``a * b``, in three.js's ``multiplyQuaternions`` term order."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        ax * bw + aw * bx + ay * bz - az * by,
        ay * bw + aw * by + az * bx - ax * bz,
        az * bw + aw * bz + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    x, y, z, w = q
    return (-x, -y, -z, w)


def rotate_vector(v: Vector3, q: Quaternion) -> Vector3:
    """``Vector3.applyQuaternion``, term for term."""
    vx, vy, vz = v
    qx, qy, qz, qw = q
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def forward_vector(q: Quaternion) -> np.ndarray:
    """The car's facing direction: local +Z, rotated.

    Established from the chase camera's ``reset``, which anchors itself at the
    car's local -Z and then looks along the opposite direction.
    """
    return np.array(rotate_vector((0.0, 0.0, 1.0), q), dtype=np.float64)


def up_vector(q: Quaternion) -> np.ndarray:
    """The car's roof direction: local +Y, rotated."""
    return np.array(rotate_vector((0.0, 1.0, 0.0), q), dtype=np.float64)


def angular_velocity(previous: Quaternion, current: Quaternion, rate: float) -> np.ndarray:
    """World-frame angular velocity between two orientations, in rad/s.

    Exact for any rotation magnitude rather than the usual small-angle
    ``2 * dq.xyz / dt``. Over a single 1 ms tick the difference is nil, but a
    control interval spans many ticks, and across one of those a car spinning
    out can turn far enough for the approximation to visibly under-read.
    """
    dx, dy, dz, dw = quaternion_multiply(current, quaternion_conjugate(previous))
    if dw < 0.0:  # take the short way round
        dx, dy, dz, dw = -dx, -dy, -dz, -dw
    sin_half = math.sqrt(dx * dx + dy * dy + dz * dz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, dw)
    scale = angle * rate / sin_half
    return np.array([dx * scale, dy * scale, dz * scale], dtype=np.float64)


class NoStartingPoint(ValueError):
    """The track places no start part, so there is nowhere to spawn the car.

    The client refuses to simulate such a track at all -- ``getStartTransform``
    returns null and both the Verify and CreateCar paths throw.
    """


def start_transform(
    track: TrackCode, assets: GeometryAssets
) -> tuple[Vector3, Quaternion]:
    """Where the car spawns, matching ``TrackData.getStartTransform``.

    When a track places several start parts the client uses the one with the
    **highest** ``startOrder`` -- the most recently placed, not the first. That
    is not a detail to guess at: the shipped ``4_seasons`` community track has
    387 of them.
    """
    chosen = None
    best_order = None
    for part in track.parts:
        offset = assets[part.part_id].start_offset
        if offset is None or part.start_order is None:
            continue
        # `>=` so the last one wins ties, as the client's loop does.
        if best_order is None or part.start_order >= best_order:
            best_order = part.start_order
            chosen = (part, offset)

    if chosen is None:
        raise NoStartingPoint(
            f"track {track.name or '<unnamed>'!s} has no starting point"
        )

    part, offset = chosen
    quaternion = quaternion_multiply(
        ROTATION_QUATERNIONS[part.rotation_axis][part.rotation], _HALF_TURN_Y
    )
    ox, oy, oz = rotate_vector(offset, quaternion)
    position = (
        part.x * PART_SIZE + ox,
        part.y * PART_SIZE + oy,
        part.z * PART_SIZE + oz,
    )
    return position, quaternion


# ------------------------------------------------------------------ terrain --

#: The client's "random" source: 128 fixed values, cycled. Copied verbatim so
#: the terrain matches; see the module docstring.
_TERRAIN_TABLE: tuple[float, ...] = (
    0.12047764760664692, 0.19645762332790628, 0.5525629082262744,
    0.41272626379209965, 0.7795036003541387, 0.13367266027110114,
    0.7999601557377349, 0.9519714253374205, 0.1735048382917752,
    0.7513367084489158, 0.6531386724839523, 0.9026427867068505,
    0.8543272738216994, 0.11176849958868162, 0.6705698284858437,
    0.26628732081296946, 0.31140322993719605, 0.45170300835470933,
    0.12615515120247944, 0.0610638094525735, 0.291990923385425,
    0.4613983868623317, 0.6615759832726253, 0.4373182881232056,
    0.7432890501246443, 0.39316710322388837, 0.49444122821563297,
    0.5994296685114344, 0.060050119050233386, 0.4165885432422003,
    0.43974364800990084, 0.1628314496954224, 0.05787972729968116,
    0.225388541259955, 0.6075775236386991, 0.8908354370882479,
    0.47072983115144584, 0.7662003453186828, 0.20651036895645647,
    0.03724062137286044, 0.17110277274376795, 0.7626426077793496,
    0.8372112804261309, 0.8761690804447455, 0.13887024930406633,
    0.8287513367412203, 0.9794446290917873, 0.807658524448803,
    0.8465629116398186, 0.5187285629536083, 0.33962953580139277,
    0.9798419666114342, 0.6777071959103609, 0.5388899884934379,
    0.7863389168762325, 0.4274591420924474, 0.25631366937500566,
    0.5695289062505289, 0.026841382754547727, 0.18267938207996903,
    0.9853642975717878, 0.24428485895234409, 0.5322028747608949,
    0.9655065842019517, 0.043810183244384016, 0.541216190236913,
    0.05897981610006209, 0.2849168541804703, 0.5349823008832073,
    0.9655676144971486, 0.22831812764497283, 0.7698701658704175,
    0.4103995069939841, 0.25782763124411856, 0.8490222628872495,
    0.39280879489916987, 0.31999467883347554, 0.2860820872456349,
    0.9684928577493004, 0.9973831481899462, 0.2930912094664657,
    0.4847128131859766, 0.7218400909709828, 0.40407009594106236,
    0.7059298060123587, 0.45362146566562744, 0.4640974655488792,
    0.16076769483252273, 0.5989453525750241, 0.585759299589679,
    0.9417035568973537, 0.20117930667657413, 0.5777873180244959,
    0.1991854396549344, 0.8743781441651348, 0.624666386634513,
    0.38720573630932886, 0.9967931526923675, 0.49817894572849486,
    0.24585267823751833, 0.8639168275132305, 0.2865624029759799,
    0.6163605496913385, 0.5864748073339972, 0.8781049154377354,
    0.7497547608938613, 0.7864098057445887, 0.0334170452332867,
    0.4875588105294657, 0.6737395339380896, 0.21851121231639659,
    0.2923739650597854, 0.6073797612662293, 0.41823228947229896,
    0.8531029420136382, 0.3260916332061783, 0.6306262204574675,
    0.5268576689601923, 0.3516570914484707, 0.8659366375222706,
    0.8447448461834428, 0.3794548980890986, 0.9832775904115916,
    0.8442256760399809, 0.3006550591973338, 0.9718660619781394,
    0.5103245035851833, 0.794319831388071,
)

_RINGS = 8  # samples outward per angular slice
_RING_SPACING = 100.0  # metres between successive rings
_HEIGHT_SCALE = 100.0
_MAX_RADIUS = 4500.0  # beyond this the client draws no terrain at all


class _TableCycle:
    """The client's ``next()``: pre-increment, wrap, return.

    Note it returns ``table[1]`` first, not ``table[0]``. Getting that backwards
    shifts the whole terrain by one sample.
    """

    __slots__ = ("_index",)

    def __init__(self) -> None:
        self._index = 0

    def next(self) -> float:
        self._index += 1
        if self._index >= len(_TERRAIN_TABLE):
            self._index = 0
        return _TERRAIN_TABLE[self._index]


def track_bounds(track: TrackCode) -> tuple[int, int, int, int]:
    """``(min_x, min_z, max_x, max_z)`` in grid cells; zeros for an empty track."""
    if not track.parts:
        return 0, 0, 0, 0
    xs = [p.x for p in track.parts]
    zs = [p.z for p in track.parts]
    return min(xs), min(zs), max(xs), max(zs)


def mountain_vertices(track: TrackCode) -> tuple[np.ndarray, Vector3]:
    """Reproduce ``Mountains.createMountainVertices`` for a track.

    Returns the flat float32 vertex array and the offset the client passes
    alongside it. Both go straight into ``createCarModel``.
    """
    min_x, min_z, max_x, max_z = track_bounds(track)
    radius = max(
        200.0,
        160.0
        + max(
            abs(max_x - min_x) * PART_SIZE / 2 * math.sqrt(2),
            abs(max_z - min_z) * PART_SIZE / 2 * math.sqrt(2),
        ),
    )
    offset: Vector3 = (
        (min_x + (max_x - min_x) / 2) * PART_SIZE,
        0.0,
        (min_z + (max_z - min_z) / 2) * PART_SIZE,
    )
    if radius > _MAX_RADIUS:
        return np.zeros(0, dtype=np.float32), (0.0, 0.0, 0.0)

    random = _TableCycle()
    slices = int(radius // 10)
    heights: list[list[float]] = []
    for _ in range(slices):
        row: list[float] = []
        for ring in range(_RINGS):
            # Short-circuiting matters: the table is only advanced where the
            # client advances it. The rim rings are pinned flat at zero, and
            # ring 1 draws once to decide whether it is flat and again for a
            # height only if it is not.
            if ring == 0 or ring == _RINGS - 1:
                row.append(0.0)
            elif ring == 1 and random.next() < 0.5:
                row.append(0.0)
            else:
                row.append(random.next())
        heights.append(row)

    out: list[float] = []
    for index in range(slices):
        theta = index / slices * math.pi * 2
        next_theta = (index + 1) / slices * math.pi * 2
        here = heights[index]
        there = heights[index + 1] if index + 1 < slices else heights[0]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cos_n, sin_n = math.cos(next_theta), math.sin(next_theta)
        for ring in range(_RINGS - 1):
            inner = radius + _RING_SPACING * ring
            outer = radius + _RING_SPACING * (ring + 1)
            out += [cos_t * inner, here[ring] * _HEIGHT_SCALE, sin_t * inner]
            out += [cos_n * inner, there[ring] * _HEIGHT_SCALE, sin_n * inner]
            out += [cos_n * outer, there[ring + 1] * _HEIGHT_SCALE, sin_n * outer]
            out += [cos_t * inner, here[ring] * _HEIGHT_SCALE, sin_t * inner]
            out += [cos_n * outer, there[ring + 1] * _HEIGHT_SCALE, sin_n * outer]
            out += [cos_t * outer, here[ring + 1] * _HEIGHT_SCALE, sin_t * outer]

    return np.asarray(out, dtype=np.float32), offset
