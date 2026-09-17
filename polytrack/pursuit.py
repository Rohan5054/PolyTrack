"""A scripted driver: pure pursuit along a known route.

Two jobs, neither of which needs learning. It is the **drivability filter** for
generated tracks -- a random walk will happily lay down geometry nothing can
drive, and a track the simplest possible controller finishes is at least a
track. And it is the **floor** any learned policy has to beat to mean anything,
which is the cheapest baseline an evaluation can have.

It is deliberately dumb: aim at the next point on the route, steer toward it
with three-way bang-bang, brake when a corner is coming. No racing line, no
lookahead tuning beyond "is there a corner in the next few pieces". A time from
this controller says what the track allows at a crawl, not what it allows.

## The steering sign was measured, not reasoned

Facing +x with the car at speed, ``THROTTLE_RIGHT`` rotated the forward vector
from ``(1, 0, 0)`` towards ``(0.96, 0, 0.28)`` -- toward +z. With the cross
product used below, ``fwd.z * to.x - fwd.x * to.z``, a target toward +z reads
**negative**. So negative means steer right. Every controller written before
this was measured had the sign the other way round, and every one of them
crashed into the first corner for that reason as much as any other.

## Real module only

PolyTrack is y-up and the road lies in x-z, which is what the flattening below
assumes. ``ReferencePhysics`` and its demo tracks lie in **x-y** with z always
zero, so on the stand-in this driver's lateral axis vanishes and it never steers
at all -- and it still "finishes" the demo chicane, because the stand-in is
lenient. That pass means nothing. Test steering against the real module or
against synthetic states, never against the stand-in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import ACTION_TO_MASK, Action
from .physics.base import PhysicsBackend

_ACTIONS = {
    (False, 0): Action.THROTTLE,
    (False, 1): Action.THROTTLE_RIGHT,
    (False, -1): Action.THROTTLE_LEFT,
    (True, 0): Action.BRAKE,
    (True, 1): Action.BRAKE_RIGHT,
    (True, -1): Action.BRAKE_LEFT,
}


def _flat(vector) -> np.ndarray:
    out = np.array([vector[0], 0.0, vector[2]], dtype=np.float64)
    norm = np.linalg.norm(out)
    return out / norm if norm > 1e-9 else out


@dataclass(frozen=True, slots=True)
class PursuitResult:
    finished: bool
    #: Frame the run ended on: the finish frame if it finished.
    frames: int
    #: Index of the furthest route point the driver reached.
    reached: int
    #: Route length in points, for reading ``reached`` as a fraction.
    of: int
    checkpoints: int

    @property
    def fraction(self) -> float:
        return self.reached / max(self.of - 1, 1)


class PurePursuit:
    """Follow ``route`` -- world-space points in driving order -- at a crawl.

    Args:
        cruise_speed: m/s on straights. The brake is applied above it.
        corner_speed: m/s when a change of direction is within
            ``corner_lookahead`` route points. A ``TurnSharp`` is a right angle
            inside one 20 m piece; 11 m/s is known to get round one.
        reach: how close, in metres, a route point has to be before the driver
            aims at the next one.
        deadband: below this cross product the driver holds straight rather than
            sawing at the wheel.
    """

    def __init__(
        self,
        route: np.ndarray,
        *,
        cruise_speed: float = 24.0,
        corner_speed: float = 11.0,
        corner_lookahead: int = 3,
        reach: float = 9.0,
        deadband: float = 0.06,
    ) -> None:
        self.route = np.asarray(route, dtype=np.float64)
        if len(self.route) < 2:
            raise ValueError("a route needs at least two points")
        self.cruise_speed = cruise_speed
        self.corner_speed = corner_speed
        self.corner_lookahead = corner_lookahead
        self.reach = reach
        self.deadband = deadband
        self._bends = self._find_bends()
        self.target = 1

    def _find_bends(self) -> np.ndarray:
        """Which route points sit where the direction changes."""
        bends = np.zeros(len(self.route), dtype=bool)
        for i in range(1, len(self.route) - 1):
            before = _flat(self.route[i] - self.route[i - 1])
            after = _flat(self.route[i + 1] - self.route[i])
            bends[i] = float(np.dot(before, after)) < 0.9
        return bends

    def reset(self) -> None:
        self.target = 1

    def act(self, state) -> Action:
        position = np.asarray(state.position, dtype=np.float64)
        while (
            self.target < len(self.route) - 1
            and np.linalg.norm((self.route[self.target] - position)[[0, 2]])
            < self.reach
        ):
            self.target += 1

        to = _flat(self.route[self.target] - position)
        forward = _flat(state.forward)
        cross = forward[2] * to[0] - forward[0] * to[2]
        if abs(cross) < self.deadband:
            steer = 0
        else:
            steer = 1 if cross < 0 else -1   # negative is right; see docstring

        ahead = self._bends[self.target: self.target + self.corner_lookahead]
        limit = self.corner_speed if ahead.any() else self.cruise_speed
        speed = float(np.linalg.norm(state.velocity))
        return _ACTIONS[(speed > limit, steer)]

    def drive(
        self,
        backend: PhysicsBackend,
        *,
        limit_ms: int = 120_000,
        interval_ms: int = 50,
    ) -> PursuitResult:
        """Drive from a fresh reset until the finish or ``limit_ms``."""
        self.reset()
        state = backend.reset()
        result = None
        # Frames are milliseconds on the real module; on the stand-in they are
        # not, so convert through the backend's own rate.
        ticks = max(1, round(interval_ms * backend.rate / 1000.0))
        limit = round(limit_ms * backend.rate / 1000.0)
        while backend.frame < limit:
            result = backend.step_many(ACTION_TO_MASK[int(self.act(state))], ticks)
            state = result.state
            if result.finished:
                break
        return PursuitResult(
            finished=bool(result is not None and result.finished),
            frames=int(state.frame),
            reached=int(self.target),
            of=len(self.route),
            checkpoints=int(getattr(state, "checkpoint", 0)),
        )
