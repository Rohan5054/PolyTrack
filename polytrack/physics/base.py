"""The physics backend interface.

This is the most important abstraction in the project. Everything above it --
observations, rewards, the Gymnasium environment, the search layer -- is written
against :class:`PhysicsBackend` and never touches the simulator directly.

That matters because the real backend is the one piece that cannot be written
blind: it has to be built against PolyTrack's actual WebAssembly physics module
(WASM since v0.4.0, majority rewritten in C++ for v0.6.0). Until that binding
exists and is verified, :class:`~polytrack.physics.reference.ReferencePhysics`
stands in so the rest of the pipeline can be built and tested. When the real
backend lands, nothing above this file changes.

Three capabilities define a usable backend, and they are exactly the ones
TMInterface gives Trackmania researchers:

1. **Deterministic frame stepping** -- the same inputs from the same state
   always produce the same trajectory, bit for bit.
2. **Save/restore state** -- required for trajectory search. Without it, every
   candidate line has to be replayed from frame zero, and search costs go up by
   orders of magnitude.
3. **Full car-state readout** -- position, velocity, orientation, wheel contact,
   checkpoint progress. Not pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class CarState:
    """Complete dynamic state of the car at one physics frame.

    All vectors are in world coordinates; converting to the car-local frame is
    the observation builder's job, not the backend's. Keeping the backend's
    output raw means the same backend serves observation designs we haven't
    thought of yet.
    """

    frame: int
    position: np.ndarray  # (3,) world position, metres
    velocity: np.ndarray  # (3,) world linear velocity, m/s
    forward: np.ndarray  # (3,) unit vector, car's facing direction
    up: np.ndarray  # (3,) unit vector, car's roof direction
    angular_velocity: np.ndarray  # (3,) rad/s, world frame
    wheel_contact: np.ndarray  # (4,) bool, per-wheel ground contact
    checkpoint: int  # count of checkpoints passed so far
    finished: bool

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return float(np.linalg.norm(self.velocity))

    @property
    def airborne(self) -> bool:
        """True when no wheel is touching a surface.

        PolyTrack tracks are full of jumps, loops and wall rides, so a policy
        that cannot tell it is airborne will steer into nothing and waste the
        landing. This flag earns its place in the observation vector.
        """
        return not bool(self.wheel_contact.any())

    def local_velocity(self) -> np.ndarray:
        """Velocity in the car's own frame: (longitudinal, lateral, vertical).

        This is the representation that generalises across tracks -- world-frame
        velocity implicitly encodes *where on the map* the car is, which invites
        per-track memorisation.
        """
        right = np.cross(self.forward, self.up)
        return np.array(
            [
                float(np.dot(self.velocity, self.forward)),
                float(np.dot(self.velocity, right)),
                float(np.dot(self.velocity, self.up)),
            ],
            dtype=np.float64,
        )


@dataclass(slots=True)
class StepResult:
    """What one physics frame produced."""

    state: CarState
    passed_checkpoint: bool = False
    finished: bool = False
    #: Set when the car left the playable area or otherwise cannot continue.
    terminated: bool = False
    info: dict = field(default_factory=dict)


@runtime_checkable
class PhysicsBackend(Protocol):
    """Contract every simulator must satisfy.

    Implementations must be deterministic: ``reset(seed)`` followed by an
    identical input sequence must produce an identical trajectory, on any
    machine. PolyTrack itself enforces this -- the client runs a determinism
    self-test and refuses to submit runs that fail it -- so a backend that
    cannot reproduce a known ghost frame-for-frame is simply wrong, and every
    result built on it is untrustworthy.
    """

    @property
    def frame(self) -> int:
        """Current physics frame index, starting at 0 after reset."""
        ...

    def reset(self) -> CarState:
        """Return the car to the starting point and zero the frame counter."""
        ...

    def step(self, input_mask: int) -> StepResult:
        """Advance exactly one physics frame with the given held-input mask."""
        ...

    def save_state(self) -> bytes:
        """Serialise complete simulator state for later restoration.

        Must capture *everything* the next step depends on. A save that misses
        hidden state (suspension compression, air-brake timers, the landing
        grace period v0.6.0 added) will appear to work and then desync under
        search, which is far worse than failing outright.
        """
        ...

    def load_state(self, blob: bytes) -> CarState:
        """Restore state previously produced by :meth:`save_state`."""
        ...


def replay(backend: PhysicsBackend, masks, max_frames: int = 100_000) -> StepResult:
    """Drive a backend through a sequence of per-frame input masks.

    Returns the final :class:`StepResult`. This is the primitive that the
    verification harness, the search layer and the TAS importer all share.
    """
    result: StepResult | None = None
    for i, mask in enumerate(masks):
        if i >= max_frames:
            break
        result = backend.step(mask)
        if result.finished or result.terminated:
            break
    if result is None:
        raise ValueError("empty input sequence: nothing to replay")
    return result
