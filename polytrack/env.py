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

**One step is a control interval, not a physics tick.** PolyTrack steps physics
at 1000 Hz, and 1000 decisions a second is not an action space -- the game does
not work that way either, since it records inputs as changes and a held key
spans hundreds of ticks. A step here holds one action for
``control_interval_ms`` of simulated time. Everything the reward is denominated
in is a duration for the same reason: the stand-in backend runs at 60 Hz and the
real one at 1000, so a coefficient "per frame" would silently mean two different
things depending on which one is attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import ACTION_TO_MASK, N_ACTIONS
from .observation import ObservationBuilder, ObservationConfig
from .physics.base import PhysicsBackend, ticks_for
from .physics.reference import ReferencePhysics
from .tas import TAS, format_time
from .track import ProgressTracker, Track, straight_track


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients.

    Defaults are a starting point, not tuned values. They still cannot be tuned
    meaningfully against the stand-in backend, whose speeds are invented; with
    the real module attached they can, and that is the next job.

    Everything time-denominated is per *second* of simulated time rather than
    per frame, so the same configuration means the same thing at 60 Hz and at
    1000 Hz, and changing the control interval does not quietly rescale the
    reward.
    """

    progress: float = 1.0  # per metre of reference-line progress
    time_penalty: float = 1.2  # per second elapsed
    checkpoint_bonus: float = 5.0  # each checkpoint passed
    finish_bonus: float = 100.0  # on completing the track
    finish_time_weight: float = 0.5  # scales the speed component of the bonus
    off_track_penalty: float = 20.0  # on leaving the corridor
    reference_seconds: float = 30.0  # the "par time" the finish bonus scales against


class PolyTrackEnv(gym.Env):
    """Single-agent time-trial environment.

    Args:
        track: Track to drive -- for a real PolyTrack track, a reference line
            from :mod:`polytrack.reference`. Randomise this across episodes to
            train a policy that generalises rather than memorises.
        backend: Physics implementation. Defaults to the reference stand-in,
            which is the only thing that can drive the synthetic demo tracks;
            for a real track pass a
            :class:`~polytrack.physics.wasm.WasmPhysics`, or build the pair with
            :func:`polytrack.reference.env_from_recording`.
        control_interval_ms: How long one action is held, in simulated
            milliseconds. Converted to backend ticks via the backend's own rate,
            so it means the same thing at 60 Hz and at 1000 Hz.
        time_limit_ms: Episode truncation limit, in simulated milliseconds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        track: Track | None = None,
        backend: PhysicsBackend | None = None,
        obs_config: ObservationConfig | None = None,
        reward_config: RewardConfig | None = None,
        # 50 ms is 4.2 m at 300 km/h, slightly under PolyTrack's 5 m part grid:
        # roughly one decision per piece of track at top speed. Linesight uses
        # 100 ms in Trackmania, but PolyTrack's grid is tighter and its corners
        # come faster.
        control_interval_ms: float = 50.0,
        time_limit_ms: float = 60_000.0,
    ) -> None:
        super().__init__()
        self.track = track or straight_track()
        self.backend = backend or ReferencePhysics(self.track)
        self.observer = ObservationBuilder(self.track, obs_config)
        self.reward_config = reward_config or RewardConfig()
        self.control_interval_ms = float(control_interval_ms)
        self.time_limit_ms = float(time_limit_ms)

        self.ticks_per_step = ticks_for(self.backend, self.control_interval_ms)
        self._ms_per_tick = 1000.0 / self.backend.rate

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observer.size,), dtype=np.float32
        )

        self._progress = ProgressTracker(self.track)
        self._previous_arc = 0.0
        # Held-input runs rather than one entry per tick: at 1000 Hz a minute of
        # driving is 60,000 ticks and a few hundred input changes, and the TAS
        # format this exports to is a change list anyway.
        self._input_runs: list[tuple[int, int]] = []

    @property
    def elapsed_frames(self) -> int:
        """Backend frames driven this episode."""
        return sum(ticks for _, ticks in self._input_runs)

    @property
    def input_masks(self) -> list[int]:
        """The episode's inputs expanded to one mask per backend frame."""
        return [mask for mask, ticks in self._input_runs for _ in range(ticks)]

    # -- gym API ----------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        state = self.backend.reset()
        self._progress.reset()
        arc, lateral_offset = self._progress.update(state.position)
        self._previous_arc = arc
        self._input_runs = []
        return self.observer.build(state, arc, lateral_offset), {"frame": 0}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        mask = ACTION_TO_MASK[int(action)]

        before = self.backend.frame
        result = self.backend.step_many(mask, self.ticks_per_step)
        state = result.state
        # However far it actually got: a step that finishes mid-interval stops
        # there, and the exported TAS has to say so or it will not replay.
        elapsed = state.frame - before
        if elapsed > 0:
            self._input_runs.append((mask, elapsed))
        interval_seconds = elapsed * self._ms_per_tick / 1000.0
        elapsed_ms = state.frame * self._ms_per_tick

        arc, lateral_offset = self._progress.update(state.position)

        cfg = self.reward_config
        # Progress since the last decision. Clipped at zero so reversing to farm
        # the same stretch of track repeatedly cannot generate reward -- a
        # classic exploit in progress-shaped racing rewards.
        delta = max(arc - self._previous_arc, 0.0)
        reward = cfg.progress * delta - cfg.time_penalty * interval_seconds
        self._previous_arc = max(arc, self._previous_arc)

        if result.passed_checkpoint:
            reward += cfg.checkpoint_bonus

        terminated = False
        truncated = False

        if result.finished:
            terminated = True
            # Scale the completion bonus by how quickly it was achieved, so the
            # agent keeps improving after it can merely finish.
            speed_factor = cfg.reference_seconds / max(elapsed_ms / 1000.0, 1e-6)
            reward += cfg.finish_bonus * (
                1.0 + cfg.finish_time_weight * min(speed_factor, 3.0)
            )
        elif result.terminated:
            terminated = True
            reward -= cfg.off_track_penalty
        elif elapsed_ms >= self.time_limit_ms:
            truncated = True

        info: dict[str, Any] = {
            "frame": state.frame,
            "elapsed_ms": elapsed_ms,
            "arc": arc,
            "lateral_offset": lateral_offset,
            "speed": state.speed,
            "checkpoint": state.checkpoint,
        }
        if terminated or truncated:
            info["finished"] = result.finished
            info["frames"] = state.frame
            if result.finished:
                # format_time counts in PolyTrack frames, so convert first --
                # the stand-in's frames are not milliseconds.
                info["time"] = format_time(round(elapsed_ms))

        observation = self.observer.build(state, arc, lateral_offset)
        return observation, float(reward), terminated, truncated, info

    # -- TAS interop ------------------------------------------------------

    def export_tas(self) -> TAS:
        """Export the episode just played as a TAS.

        This is the bridge to verification: the resulting file can be encoded
        into a ghost and replayed in the real game. An agent's run is only
        trustworthy once it reproduces there frame for frame -- which is why
        this is per backend frame, not per decision.
        """
        return TAS.from_frame_masks(self.input_masks)

    def replay_tas(self, tas: TAS, n_frames: int | None = None) -> dict[str, Any]:
        """Replay a TAS through the backend and report the outcome.

        The core of the verification harness: feed in a known-good human ghost,
        confirm the simulator produces the same finish frame. If it does not,
        the backend is wrong and every downstream result is worthless.

        A TAS is per-frame, so this ignores the control interval and steps the
        backend one frame at a time -- an agent's decisions are quantised, a
        replay of them must not be re-quantised.
        """
        self.reset()
        limit = round(self.time_limit_ms / self._ms_per_tick)
        frames = tas.n_frames if n_frames is None else n_frames
        result = None
        for mask in tas.expand(min(frames, limit)):
            self._input_runs.append((mask, 1))
            result = self.backend.step(mask)
            if result.finished or result.terminated:
                break
        if result is None:
            return {"finished": False, "frames": 0}
        return {
            "finished": result.finished,
            "terminated": result.terminated,
            "frames": result.state.frame,
            "time": format_time(round(result.state.frame * self._ms_per_tick)),
            "checkpoint": result.state.checkpoint,
        }
