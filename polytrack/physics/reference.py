"""A deterministic stand-in simulator.

**This is not PolyTrack physics.** It is a simple bicycle model that satisfies
the :class:`~polytrack.physics.base.PhysicsBackend` contract so the rest of the
stack -- observations, rewards, the Gymnasium environment, the search layer, the
training loop -- can be built, tested and profiled today, before the real WASM
binding exists.

Times produced here mean nothing. Do not compare them to leaderboard times, and
do not tune reward coefficients against them beyond checking that learning
happens at all. Its jobs are narrow:

* prove the pipeline runs end to end,
* let the test suite assert determinism and save/load correctness,
* give the search layer something to search while it is being written.

When :mod:`polytrack.physics.wasm` is verified against a known ghost, swap the
backend and nothing above this file changes.
"""

from __future__ import annotations

import pickle

import numpy as np

from ..actions import DOWN, LEFT, RIGHT, UP
from ..track import ProgressTracker, Track
from .base import CarState, StepResult

DT = 1.0 / 60.0  # fixed physics step, matching the game's 60 Hz


class ReferencePhysics:
    """Kinematic bicycle model on a flat plane.

    Deterministic by construction: fixed timestep, float64 throughout, no RNG,
    no wall-clock. Same inputs from the same state give the same trajectory.
    """

    # Vehicle parameters, chosen so that the demo tracks in polytrack.track are
    # completable but not trivially so -- corners must be braked for. Not fitted
    # to PolyTrack; see the module docstring.
    #
    # STEER_FALLOFF is the parameter to watch. It sets how much steering
    # authority is lost with speed, and it is easy to pick a value that makes
    # tracks silently impossible: at 0.014 the car retained about 3 deg/s of
    # steering at 210 km/h, so every corner was a wall regardless of policy.
    # test_demo_tracks_are_completable pins this down.
    MAX_SPEED = 55.0  # m/s, ~200 km/h
    ENGINE_ACCEL = 25.0  # m/s^2 while throttle held
    BRAKE_DECEL = 45.0  # m/s^2 while brake held
    DRAG = 0.5  # passive deceleration coefficient
    STEER_RATE = 2.2  # rad/s at low speed
    STEER_FALLOFF = 0.0007  # steering authority lost per (m/s)^2
    GRIP = 0.86  # fraction of lateral velocity killed per second

    def __init__(self, track: Track, off_track_tolerance: float = 4.0) -> None:
        self.track = track
        self.off_track_tolerance = off_track_tolerance
        self._progress = ProgressTracker(track)
        self._frame = 0
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._heading = 0.0
        self._checkpoint = 0
        self._finished = False
        self.reset()

    @property
    def frame(self) -> int:
        return self._frame

    def reset(self) -> CarState:
        self._progress.reset()
        self._frame = 0
        self._position = self.track.start_position.astype(np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        forward = self.track.start_forward
        self._heading = float(np.arctan2(forward[1], forward[0]))
        self._checkpoint = 0
        self._finished = False
        return self._state()

    # -- core integration -------------------------------------------------

    def step(self, input_mask: int) -> StepResult:
        if self._finished:
            return StepResult(self._state(), finished=True)

        forward = np.array(
            [np.cos(self._heading), np.sin(self._heading), 0.0], dtype=np.float64
        )
        right = np.array(
            [np.sin(self._heading), -np.cos(self._heading), 0.0], dtype=np.float64
        )

        longitudinal = float(np.dot(self._velocity, forward))
        lateral = float(np.dot(self._velocity, right))

        # Longitudinal forces.
        if input_mask & UP:
            longitudinal += self.ENGINE_ACCEL * DT
        if input_mask & DOWN:
            longitudinal -= self.BRAKE_DECEL * DT
        longitudinal -= longitudinal * self.DRAG * DT
        longitudinal = float(np.clip(longitudinal, -self.MAX_SPEED / 3, self.MAX_SPEED))

        # Steering authority falls off with speed, so the car cannot pivot on
        # the spot at 300 km/h. Crude, but it makes braking-for-corners matter,
        # which is the behaviour we need the reward to be able to express.
        speed = abs(longitudinal)
        authority = self.STEER_RATE / (1.0 + self.STEER_FALLOFF * speed * speed)
        steer = 0
        if input_mask & LEFT:
            steer += 1
        if input_mask & RIGHT:
            steer -= 1
        if steer and speed > 0.5:
            self._heading += steer * authority * DT * np.sign(longitudinal)
            # Steering scrubs a little lateral velocity into existence: the
            # beginnings of a drift, which gives the agent something to exploit.
            lateral += steer * authority * longitudinal * DT * 0.35

        # Grip bleeds lateral velocity away.
        lateral -= lateral * self.GRIP
        lateral = float(lateral)

        forward = np.array(
            [np.cos(self._heading), np.sin(self._heading), 0.0], dtype=np.float64
        )
        right = np.array(
            [np.sin(self._heading), -np.cos(self._heading), 0.0], dtype=np.float64
        )
        self._velocity = longitudinal * forward + lateral * right
        self._position = self._position + self._velocity * DT
        self._frame += 1

        return self._evaluate_progress()

    def _evaluate_progress(self) -> StepResult:
        arc, offset = self._progress.update(self._position)
        terminated = offset > self.track.width + self.off_track_tolerance

        passed = False
        if self._checkpoint < self.track.n_checkpoints:
            target = self.track.checkpoint_arc(self._checkpoint)
            if arc >= target:
                self._checkpoint += 1
                passed = True
                if self._checkpoint >= self.track.n_checkpoints:
                    self._finished = True

        return StepResult(
            state=self._state(),
            passed_checkpoint=passed,
            finished=self._finished,
            terminated=terminated,
            info={"arc": arc, "lateral_offset": offset},
        )

    def _state(self) -> CarState:
        forward = np.array(
            [np.cos(self._heading), np.sin(self._heading), 0.0], dtype=np.float64
        )
        # Flat-plane model: always upright, always in contact.
        return CarState(
            frame=self._frame,
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            forward=forward,
            up=np.array([0.0, 0.0, 1.0]),
            angular_velocity=np.zeros(3),
            wheel_contact=np.ones(4, dtype=bool),
            checkpoint=self._checkpoint,
            finished=self._finished,
        )

    # -- save / restore ---------------------------------------------------

    def save_state(self) -> bytes:
        """Snapshot every field ``step`` reads. Nothing hidden, nothing derived."""
        return pickle.dumps(
            (
                self._frame,
                self._position.copy(),
                self._velocity.copy(),
                self._heading,
                self._checkpoint,
                self._finished,
                self._progress.arc,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def load_state(self, blob: bytes) -> CarState:
        (
            self._frame,
            position,
            velocity,
            self._heading,
            self._checkpoint,
            self._finished,
            self._progress.arc,
        ) = pickle.loads(blob)
        self._position = position.copy()
        self._velocity = velocity.copy()
        return self._state()
