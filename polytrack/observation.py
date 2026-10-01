"""Observation construction.

The design here follows GT Sophy (Wurman et al., Nature 2022) and the correction
Neinders made to TMRL in 2023, because the Trackmania literature is unusually
clear on this point: what separates an agent that merely avoids walls from one
that drives a fast line is **speed-scaled lookahead of upcoming track geometry**.

Sophy divides the track ahead into 60 equally spaced segments whose length is
computed dynamically from the car's speed, covering roughly the next 6 seconds
of travel. TMRL, which only had 19 LIDAR beams, could plan for the next corner
but not the one after it -- and plateaued at "safe driving" as a direct result.

Every component below is expressed in the **car's local frame**. That is a
deliberate generalisation choice: world-frame coordinates would let the network
memorise "at position (412, 88), turn left", which produces an agent that
collapses on an unseen track. Local-frame features force it to learn geometry
instead of location.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np

from .physics.base import CarState
from .track import Track


if TYPE_CHECKING:  # pragma: no cover
    from .corridor import RoadMap


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Knobs for the observation vector.

    Attributes:
        n_lookahead: Number of centerline points sampled ahead. Sophy uses 60;
            fewer is cheaper and often sufficient for PolyTrack's shorter tracks.
        lookahead_seconds: How far ahead, in seconds of travel, the furthest
            sample sits. Scaling by speed rather than fixed distance is the
            important part -- at 300 km/h the car needs to see much further.
        min_lookahead_metres: Floor on lookahead distance so the observation
            stays informative when the car is stopped or crawling.
        n_rays: Raycasts probing the track corridor edges. Complements the
            centerline samples: lookahead says where the track goes, rays say
            how much room there is right now.
        include_speed_scaling: Normalise velocities into roughly [-1, 1].
    """

    n_lookahead: int = 20
    lookahead_seconds: float = 6.0
    min_lookahead_metres: float = 30.0
    n_rays: int = 9
    ray_max_distance: float = 40.0
    include_speed_scaling: bool = True
    reference_speed: float = 60.0  # m/s used as the normalisation scale


class ObservationBuilder:
    """Turns a :class:`CarState` into a flat float32 vector for the network."""

    def __init__(
        self,
        track: Track,
        config: ObservationConfig | None = None,
        road: "RoadMap | None" = None,
    ) -> None:
        self.track = track
        self.config = config or ObservationConfig()
        #: Real collision geometry to cast against. Without one the rays fall
        #: back to the reference-line approximation, which is what the stand-in
        #: and the synthetic demo tracks have to use -- they have no geometry.
        self.road = road
        # Sample fractions are fixed; only the total distance varies with speed.
        # Squaring biases samples toward the near field, where precision matters
        # most, while still reaching far enough to see the corner after next.
        self._fractions = (
            np.linspace(0.0, 1.0, self.config.n_lookahead + 1)[1:] ** 2
        )
        self._ray_angles = np.linspace(
            -np.pi / 2, np.pi / 2, self.config.n_rays, dtype=np.float64
        )
        # Fixed march distances along each ray, precomputed once.
        self._n_probes = 12
        self._ray_distances = np.linspace(
            2.0, self.config.ray_max_distance, self._n_probes, dtype=np.float64
        )

    @property
    def size(self) -> int:
        """Length of the observation vector."""
        return (
            3  # local velocity
            + 1  # speed magnitude
            + 3  # angular velocity
            + 1  # airborne flag
            + 4  # per-wheel contact
            + 1  # lateral offset from centerline
            + 2 * self.config.n_lookahead  # lookahead points, local x/y
            + self.config.n_rays  # corridor rays
        )

    def build(
        self, state: CarState, arc: float, lateral_offset: float
    ) -> np.ndarray:
        """Build the observation vector for one frame.

        Args:
            state: Car state from the physics backend.
            arc: How far along the centerline the car is, **from the caller's
                :class:`~polytrack.track.ProgressTracker`**.
            lateral_offset: Distance from the centerline, from the same source.

        ``arc`` is a required argument rather than something computed here, and
        that is deliberate. This builder used to call ``track.project(position)``
        with no window, which is the globally-nearest-point projection the rest
        of the codebase exists to avoid: on any track that passes near itself it
        silently reports the wrong part of the track, and it costs O(track
        length) per frame on top. Worse, the environment already holds a correct
        stateful arc, so the observation and the reward were reading progress
        from two different and occasionally disagreeing sources.

        There is no default. Progress is genuinely not a function of position
        alone, so there is no safe value to fall back to.
        """
        cfg = self.config

        scale = cfg.reference_speed if cfg.include_speed_scaling else 1.0
        local_velocity = state.local_velocity() / scale
        speed = state.speed

        # Lookahead distance grows with speed, floored so a stationary car still
        # sees the track. This is Sophy's dynamic segment length.
        horizon = max(speed * cfg.lookahead_seconds, cfg.min_lookahead_metres)
        distances = self._fractions * horizon
        world_points = self.track.lookahead(arc, distances)

        local_points = self._to_local(world_points, state)
        # Normalise by the horizon so the network sees consistent magnitudes
        # regardless of how fast the car is going.
        local_points = local_points[:, :2] / max(horizon, 1e-6)

        rays = self._cast_rays(state, arc)


        parts = [
            local_velocity,
            np.array([speed / scale]),
            state.angular_velocity / 10.0,
            np.array([1.0 if state.airborne else 0.0]),
            state.wheel_contact.astype(np.float64),
            np.array([lateral_offset / max(self.track.width, 1e-6)]),
            local_points.reshape(-1),
            rays,
        ]
        return np.concatenate(parts).astype(np.float32)

    def _to_local(self, points: np.ndarray, state: CarState) -> np.ndarray:
        """Express world points in the car's frame: +x forward, +y left, +z up."""
        right = np.cross(state.forward, state.up)
        basis = np.stack([state.forward, -right, state.up])  # (3, 3)
        relative = points - state.position[None, :]
        return relative @ basis.T

    def _ray_directions(self, state: CarState) -> np.ndarray:
        """(R, 3) unit directions fanned around the car's heading."""
        right = np.cross(state.forward, state.up)
        return (
            np.cos(self._ray_angles)[:, None] * state.forward[None, :]
            - np.sin(self._ray_angles)[:, None] * right[None, :]
        )

    def _cast_rays(self, state: CarState, arc: float) -> np.ndarray:
        """Distance to the edge of the drivable surface, normalised to [0, 1].

        With a :class:`~polytrack.corridor.RoadMap` this asks the real collision
        geometry where the solid ends. Without one it falls back to the
        reference-line approximation below, which answers a different question
        -- how far until a probe is more than ``Track.width`` from *the lap
        somebody drove* -- and gets it wrong wherever that lap did not run down
        the middle. On a procedurally generated track, whose reference line is
        its own centreline, the fallback is close to meaningless.
        """
        cfg = self.config
        directions = self._ray_directions(state)
        if self.road is not None:
            hits = self.road.cast(
                state.position, directions,
                max_distance=cfg.ray_max_distance,
                step=cfg.ray_max_distance / self._n_probes,
            )
            return hits / cfg.ray_max_distance

        # (R, P, 3) probe points, then flattened to one batched projection.
        # Marching each ray in its own Python loop was the single biggest cost
        # in the whole step function; batching it is worth the reshaping.
        probes = (
            state.position[None, None, :]
            + directions[:, None, :] * self._ray_distances[None, :, None]
        )
        _, offsets = self.track.project_many(
            probes.reshape(-1, 3),
            near_arc=arc,
            radius=cfg.ray_max_distance * 2.0,
        )
        outside = offsets.reshape(cfg.n_rays, self._n_probes) > self.track.width

        # First probe along each ray that left the corridor; no hit -> max range.
        first_hit = np.argmax(outside, axis=1)
        any_hit = outside.any(axis=1)
        hits = np.where(any_hit, self._ray_distances[first_hit], cfg.ray_max_distance)
        return hits / cfg.ray_max_distance
