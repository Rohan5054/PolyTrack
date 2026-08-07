"""Track geometry: centerline, width, checkpoints.

The centerline is the backbone of both the reward (progress along it) and the
observation (curvature lookahead). Getting it exactly right is one of PolyTrack's
structural advantages over Trackmania: the Trackmania projects had to *estimate*
track geometry -- TMRL by casting rays against a screenshot, Neinders by fitting
Bezier curves to LIDAR returns -- because the map data was locked inside a closed
engine. PolyTrack exports every track as a portable ``PolyTrack1...`` base62
code, so the geometry can be computed exactly and for free.

Right now tracks are built from explicit waypoints. Decoding real PolyTrack
track codes is a separate task (see ``tools/`` and the README); when that lands
it produces a :class:`Track` and everything downstream is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Track:
    """A racing line and the corridor around it.

    Attributes:
        centerline: (N, 3) ordered points along the middle of the track.
        width: Half-width of the drivable corridor in metres. Leaving the
            corridor by more than this is treated as off-track.
        checkpoint_indices: Indices into ``centerline`` where checkpoints sit.
            The final index is always the finish line.
        name: Human-readable label, used in logs and TAS file headers.
    """

    centerline: np.ndarray
    width: float = 8.0
    checkpoint_indices: tuple[int, ...] = ()
    name: str = "unnamed"

    # Cached derived geometry, built once in __post_init__.
    _segment_vectors: np.ndarray = None  # type: ignore[assignment]
    _segment_lengths: np.ndarray = None  # type: ignore[assignment]
    _cumulative: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.centerline = np.asarray(self.centerline, dtype=np.float64)
        if self.centerline.ndim != 2 or self.centerline.shape[1] != 3:
            raise ValueError(
                f"centerline must be (N, 3), got {self.centerline.shape}"
            )
        if len(self.centerline) < 2:
            raise ValueError("centerline needs at least 2 points")

        self._segment_vectors = np.diff(self.centerline, axis=0)
        self._segment_lengths = np.linalg.norm(self._segment_vectors, axis=1)
        if np.any(self._segment_lengths <= 0):
            raise ValueError("centerline contains duplicate consecutive points")
        # Arc length at each centerline point; _cumulative[i] is the distance
        # from the start to point i.
        self._cumulative = np.concatenate(
            [[0.0], np.cumsum(self._segment_lengths)]
        )

        if not self.checkpoint_indices:
            self.checkpoint_indices = (len(self.centerline) - 1,)

    @property
    def length(self) -> float:
        """Total centerline arc length in metres."""
        return float(self._cumulative[-1])

    @property
    def n_checkpoints(self) -> int:
        return len(self.checkpoint_indices)

    @property
    def start_position(self) -> np.ndarray:
        return self.centerline[0].copy()

    @property
    def start_forward(self) -> np.ndarray:
        """Unit heading the car starts facing."""
        direction = self.centerline[1] - self.centerline[0]
        return direction / np.linalg.norm(direction)

    def project(
        self,
        position: np.ndarray,
        near_arc: float | None = None,
        radius: float = 60.0,
    ) -> tuple[float, float]:
        """Project a world position onto the centerline.

        Args:
            position: World position to project.
            near_arc: If given, only segments within ``radius`` of this arc
                length are considered. **Pass this whenever you have it.**
            radius: Half-width of the search window in metres.

        Returns:
            ``(arc_length, lateral_offset)`` -- how far along the track the car
            is, and how far it sits from the centerline.

        Without ``near_arc`` this returns the *globally* nearest point, which is
        wrong on any track that passes near itself. On a closed loop it reports
        the finish line as arc 0 rather than arc ``length``, so the finish
        checkpoint never triggers and a completed lap reads as unfinished. Real
        PolyTrack tracks are full of loops, bridges and crossovers, so treat the
        global path as a fallback for the first frame only -- see
        :class:`ProgressTracker`.

        The projection is exact rather than nearest-vertex: each segment is
        treated as a line segment and the closest point on it is found by
        clamped projection. Nearest-vertex would make the reward jump in
        discrete steps, showing up as spurious noise on long straights.
        """
        point = np.asarray(position, dtype=np.float64)
        if near_arc is None:
            lo, hi = 0, len(self._segment_lengths)
        else:
            lo, hi = self.segment_window(near_arc, radius)

        starts = self.centerline[lo:hi]
        vectors = self._segment_vectors[lo:hi]
        lengths = self._segment_lengths[lo:hi]

        to_point = point[None, :] - starts
        t = np.einsum("ij,ij->i", to_point, vectors) / (lengths**2)
        np.clip(t, 0.0, 1.0, out=t)
        delta = to_point - t[:, None] * vectors
        sq_distances = np.einsum("ij,ij->i", delta, delta)
        best = int(np.argmin(sq_distances))
        arc = float(self._cumulative[lo + best] + t[best] * lengths[best])
        return arc, float(np.sqrt(sq_distances[best]))

    def segment_window(self, arc_length: float, radius: float) -> tuple[int, int]:
        """Segment index range covering ``arc_length +/- radius``.

        Lets callers skip the parts of the track the car cannot possibly be
        near. Without this, projection cost grows with total track length, which
        is exactly wrong: a 2 km community track would step slower than a 200 m
        official one for no good reason.
        """
        lo = int(np.searchsorted(self._cumulative, arc_length - radius) - 1)
        hi = int(np.searchsorted(self._cumulative, arc_length + radius) + 1)
        n_segments = len(self._segment_lengths)
        return max(lo, 0), min(max(hi, 1), n_segments)

    def project_many(
        self,
        points: np.ndarray,
        near_arc: float | None = None,
        radius: float = 80.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project many world positions onto the centerline at once.

        Returns ``(arc_lengths, lateral_offsets)``, both shape ``(M,)``.

        Args:
            points: (M, 3) world positions.
            near_arc: If given, only segments within ``radius`` of this arc
                length are considered. This is the difference between an O(track
                length) and an O(1) step, and the observation builder always
                knows roughly where the car is, so it always passes it.
            radius: Half-width of the search window in metres.

        Same maths as :meth:`project`, broadcast over an (M, S) grid, with two
        optimisations that matter because this runs ~50x per frame: the window
        above, and comparing *squared* distances so the square root is taken
        once per point instead of once per point-segment pair.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if near_arc is None:
            lo, hi = 0, len(self._segment_lengths)
        else:
            lo, hi = self.segment_window(near_arc, radius)

        starts = self.centerline[lo:hi]
        vectors = self._segment_vectors[lo:hi]
        lengths = self._segment_lengths[lo:hi]

        # (M, S, 3) offsets from each segment start to each query point.
        to_point = pts[:, None, :] - starts[None, :, :]
        t = np.einsum("msj,sj->ms", to_point, vectors) / (lengths[None, :] ** 2)
        np.clip(t, 0.0, 1.0, out=t)
        # Vector from the closest point on each segment to the query point.
        delta = to_point - t[:, :, None] * vectors[None, :, :]
        sq_distances = np.einsum("msj,msj->ms", delta, delta)

        best = np.argmin(sq_distances, axis=1)
        rows = np.arange(len(pts))
        arcs = (
            self._cumulative[lo + best] + t[rows, best] * lengths[best]
        )
        return arcs, np.sqrt(sq_distances[rows, best])

    def points_at(self, arc_lengths: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`point_at` for an array of arc lengths."""
        arcs = np.clip(np.asarray(arc_lengths, dtype=np.float64), 0.0, self.length)
        idx = np.searchsorted(self._cumulative, arcs, side="right") - 1
        idx = np.clip(idx, 0, len(self._segment_lengths) - 1)
        local = (arcs - self._cumulative[idx]) / self._segment_lengths[idx]
        return self.centerline[idx] + local[:, None] * self._segment_vectors[idx]

    def point_at(self, arc_length: float) -> np.ndarray:
        """World position of the centerline at a given arc length.

        Clamps to the track ends, so lookahead near the finish line degrades
        gracefully instead of raising.
        """
        arc = float(np.clip(arc_length, 0.0, self.length))
        idx = int(np.searchsorted(self._cumulative, arc, side="right") - 1)
        idx = min(idx, len(self._segment_lengths) - 1)
        local = (arc - self._cumulative[idx]) / self._segment_lengths[idx]
        return self.centerline[idx] + local * self._segment_vectors[idx]

    def lookahead(
        self, arc_length: float, distances: np.ndarray
    ) -> np.ndarray:
        """Sample centerline points ahead of a given arc length.

        Args:
            arc_length: Where the car currently is along the track.
            distances: How far ahead to sample, in metres.

        Returns:
            (len(distances), 3) world-space points.

        This is the GT Sophy observation, which the Trackmania literature
        repeatedly identifies as the single highest-leverage design choice:
        Neinders' 2023 thesis traced TMRL's inability to plan more than one
        corner ahead directly to its lack of curvature lookahead.
        """
        return self.points_at(arc_length + np.asarray(distances, dtype=np.float64))

    def checkpoint_arc(self, index: int) -> float:
        """Arc length at which checkpoint ``index`` sits."""
        return float(self._cumulative[self.checkpoint_indices[index]])


class ProgressTracker:
    """Follows a car's position along the centerline without losing the thread.

    Holds the last known arc length and projects only within a window around it,
    so the reported progress advances continuously through loops, crossovers and
    start/finish lines instead of teleporting to whichever part of the track
    happens to be nearest in space.

    This is stateful by necessity: "how far along is the car" is genuinely not a
    function of position alone on a track that visits the same place twice. Each
    thing that needs progress -- the environment, the physics backend, every
    branch of a beam search -- keeps its own tracker or carries its own last-arc
    value forward.
    """

    def __init__(self, track: Track, window: float = 60.0) -> None:
        self.track = track
        self.window = window
        self.arc = 0.0

    def reset(self, arc: float = 0.0) -> None:
        """Anchor the tracker at a known arc length, by default the start line.

        Anchoring matters more than it looks. If the first projection runs
        unwindowed, a car sitting a centimetre *behind* the start line on a
        closed loop projects onto the far end of the centerline and reports the
        full track length as its progress -- which reads as a completed lap. A
        beam search will find that in seconds and hand back a "winning run" of
        one frame of reverse.
        """
        self.arc = arc

    def update(self, position: np.ndarray) -> tuple[float, float]:
        """Advance the tracker to a new position, returning ``(arc, offset)``.

        Always windowed around the previous arc, including the first call. The
        window is what makes progress physically meaningful: it is not possible
        to gain more than ``window`` metres of arc in one step, so no amount of
        clipping through geometry can fabricate progress.
        """
        arc, offset = self.track.project(
            position, near_arc=self.arc, radius=self.window
        )
        self.arc = arc
        return arc, offset


def straight_track(length: float = 200.0, spacing: float = 5.0) -> Track:
    """A flat straight line. The simplest possible smoke test."""
    n = int(length / spacing) + 1
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, 0] = np.arange(n) * spacing
    return Track(points, name="straight", checkpoint_indices=(n // 2, n - 1))


def oval_track(radius: float = 60.0, n_points: int = 120) -> Track:
    """A closed oval. Exercises sustained cornering in both directions."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    points = np.stack(
        [radius * np.cos(theta), radius * 1.6 * np.sin(theta), np.zeros_like(theta)],
        axis=1,
    )
    # Start at theta=0 heading +y, and close the loop back to the start.
    points = np.vstack([points, points[:1]])
    quarter = n_points // 4
    return Track(
        points,
        name="oval",
        checkpoint_indices=(quarter, 2 * quarter, 3 * quarter, len(points) - 1),
    )


def chicane_track() -> Track:
    """Straight, then alternating corners. Rewards planning two turns ahead.

    Deliberately built so a purely reactive policy -- one that only sees the
    next corner -- takes a visibly worse line than one with lookahead. Useful as
    a diagnostic for whether the observation design is doing its job.
    """
    segments = []
    x, y = 0.0, 0.0
    heading = 0.0
    # (arc turn in radians, length in metres)
    plan = [(0.0, 60.0), (1.2, 40.0), (-1.6, 45.0), (1.4, 40.0), (0.0, 70.0)]
    for turn, seg_length in plan:
        n_steps = max(int(seg_length / 3.0), 2)
        for _ in range(n_steps):
            heading += turn / n_steps
            x += 3.0 * np.cos(heading)
            y += 3.0 * np.sin(heading)
            segments.append((x, y, 0.0))
    points = np.array(segments, dtype=np.float64)
    n = len(points)
    return Track(
        points,
        name="chicane",
        checkpoint_indices=(n // 3, 2 * n // 3, n - 1),
    )
