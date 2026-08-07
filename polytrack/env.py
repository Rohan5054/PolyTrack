"""The Gymnasium environment.

Wraps a :class:`~polytrack.physics.base.PhysicsBackend` into the standard
interface so any RL library -- CleanRL, Tianshou, SB3, RLlib -- can train against
it without knowing anything about PolyTrack.

**Reward design.** Dense progress along the centerline, plus a per-frame time
penalty. This is not an arbitrary choice: it is what every successful racing
agent in the literature converged on. Fuchs et al. (2021) call it a
"course-progress proxy reward"; GT Sophy uses the same with off-course
penalties; Linesight rewards distance travelled along a reference trajectory;
OfflineMania uses a track-centerline reward. Sparse finish-time reward alone
does not train -- the agent never completes a track by chance, so it never sees
any signal at all.

The per-frame time penalty is what converts "drive safely" into "drive fast".
Without it you get TMRL's failure mode: a policy that reliably completes tracks
several seconds off the pace, because nothing ever punished it for dawdling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import ACTION_TO_MASK, N_ACTIONS
from .observation import ObservationBuilder, ObservationConfig
from .physics.base import PhysicsBackend
from .physics.reference import ReferencePhysics
from .tas import TAS, format_time
from .track import ProgressTracker, Track, straight_track


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients.

    Defaults are a starting point, not tuned values -- they cannot be tuned
    meaningfully until the real physics backend is in place, since the scale of
    ``progress`` depends entirely on how fast the car actually moves.
    """

    progress: float = 1.0  # per metre of centerline progress
    time_penalty: float = 0.02  # per physics frame elapsed
    checkpoint_bonus: float = 5.0  # each checkpoint passed
    finish_bonus: float = 100.0  # on completing the track
    finish_time_weight: float = 0.5  # scales the speed component of the bonus
    off_track_penalty: float = 20.0  # on leaving the corridor
    reference_frames: float = 1800.0  # 30 s: the "par time" the bonus scales against


class PolyTrackEnv(gym.Env):
    """Single-agent time-trial environment.

    Args:
        track: Track to drive. Randomise this across episodes to train a policy
            that generalises rather than memorises.
        backend: Physics implementation. Defaults to the reference stand-in.
        max_frames: Episode truncation limit, in physics frames.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: Track | None = None,
        backend: PhysicsBackend | None = None,
        obs_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        max_frames: int = 3600,  # 60 s at 60 Hz
    ) -> None:
        super().__init__()
        self.track = track or straight_track()
        self.backend = backend or ReferencePhysics(self.track)
        self.observer = ObservationBuilder(self.track, obs_config)
        self.reward_config = reward_config or RewardConfig()
        self.max_frames = max_frames

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observer.size,), dtype=np.float32
        )

        self._progress = ProgressTracker(self.track)
        self._previous_arc = 0.0
        self._input_log: list[int] = []

    # -- gym API ----------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.backend.reset()
        self._progress.reset()
        self._previous_arc, _ = self._progress.update(state.position)
        self._input_log = []
        return self.observer.build(state), {"frame": 0}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        mask = ACTION_TO_MASK[int(action)]
        self._input_log.append(mask)

        result = self.backend.step(mask)
        state = result.state
        arc, lateral_offset = self._progress.update(state.position)

        cfg = self.reward_config
        # Progress since the last frame. Clipped at zero so reversing to farm
        # the same stretch of track repeatedly cannot generate reward -- a
        # classic exploit in progress-shaped racing rewards.
        delta = max(arc - self._previous_arc, 0.0)
        reward = cfg.progress * delta - cfg.time_penalty
        self._previous_arc = max(arc, self._previous_arc)

        if result.passed_checkpoint:
            reward += cfg.checkpoint_bonus

        terminated = False
        truncated = False

        if result.finished:
            terminated = True
            # Scale the completion bonus by how quickly it was achieved, so the
            # agent keeps improving after it can merely finish.
            speed_factor = cfg.reference_frames / max(state.frame, 1)
            reward += cfg.finish_bonus * (
                1.0 + cfg.finish_time_weight * min(speed_factor, 3.0)
            )
        elif result.terminated:
            terminated = True
            reward -= cfg.off_track_penalty
        elif state.frame >= self.max_frames:
            truncated = True

        info: dict[str, Any] = {
            "frame": state.frame,
            "arc": arc,
            "lateral_offset": lateral_offset,
            "speed": state.speed,
            "checkpoint": state.checkpoint,
        }
        if terminated or truncated:
            info["finished"] = result.finished
            info["frames"] = state.frame
            if result.finished:
                info["time"] = format_time(state.frame)

        return self.observer.build(state), float(reward), terminated, truncated, info

    # -- TAS interop ------------------------------------------------------

    def export_tas(self) -> TAS:
        """Export the episode just played as a TAS.

        This is the bridge to verification: the resulting file can be encoded
        into a ghost and replayed in the real game. An agent's run is only
        trustworthy once it reproduces there frame for frame.
        """
        return TAS.from_frame_masks(self._input_log)

    def replay_tas(self, tas: TAS, n_frames: int | None = None) -> dict[str, Any]:
        """Replay a TAS through the backend and report the outcome.

        The core of the verification harness: feed in a known-good human ghost,
        confirm the simulator produces the same finish frame. If it does not,
        the backend is wrong and every downstream result is worthless.
        """
        self.reset()
        frames = tas.n_frames if n_frames is None else n_frames
        result = None
        for mask in tas.expand(min(frames, self.max_frames)):
            self._input_log.append(mask)
            result = self.backend.step(mask)
            if result.finished or result.terminated:
                break
        if result is None:
            return {"finished": False, "frames": 0}
        return {
            "finished": result.finished,
            "terminated": result.terminated,
            "frames": result.state.frame,
            "time": format_time(result.state.frame),
            "checkpoint": result.state.checkpoint,
        }
