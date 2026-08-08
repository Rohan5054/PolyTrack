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
    """Reward coefficients. **Every term is non-negative, and that is a rule.**

    Scales are relative rather than absolute: a full lap is worth ``lap_reward``
    on every track, whether it is 1300 m or 3980 m. Without that, one shared
    Q-function has to represent returns three times larger on Summer 5 than on
    Summer 1 for reasons that have nothing to do with driving well -- and all the
    collectors feed one buffer.

    **Why nothing here is a penalty.** The first version charged a per-second
    time penalty and a proportional forfeit on failure. Twelve minutes of
    training found the exploit: with the episode over, a failure costs a fixed
    amount whenever it happens, while staying alive keeps paying the clock -- so
    the fastest possible failure wins. The agent learned to drive sideways off
    the start line, the failure histogram flipped from 74% ``stalled`` to 88%
    ``off_line``, mean return improved, and progress went to zero. It was
    optimising correctly; the objective was wrong.

    Charging the full time budget on failure, so that bailing out early saves
    nothing, inverts the exploit rather than fixing it: discounting shrinks any
    large terminal penalty, so stalling for eight seconds before failing is worth
    -46 against -200 for failing at once. The agent would learn to sit still.

    Both failures are the same failure. A negative step reward makes ending the
    episode attractive; a large terminal penalty makes delaying it attractive.
    The robust answer is that ending an episode must never pay, which means no
    reward may be negative -- then a terminal state is worth zero and continuing
    is worth at least zero, so quitting is never strictly better.

    Time pressure comes from the finish bonus instead, which is what the agent
    is really being asked to maximise, and secondarily from the discount: reward
    collected sooner is worth more. That leaves the early curriculum honest --
    learn to complete a lap, then learn to complete it quickly.
    """

    lap_reward: float = 100.0  # for covering the whole reference line
    checkpoint_bonus: float = 5.0  # each checkpoint passed
    #: Paid on finishing, scaled by ``par_time / elapsed``: the par time pays it
    #: in full, twice the par time pays half, and beating par pays more. This is
    #: the entire source of time pressure, so it is deliberately large relative
    #: to a lap.
    finish_bonus: float = 100.0
    #: Cap on that ratio, so the first freakishly quick run cannot dominate the
    #: replay buffer.
    finish_speed_cap: float = 3.0
    #: Par pace assumed when no reference time is supplied, m/s. 60 is 216 km/h,
    #: at the slow end of what the human runs average.
    nominal_speed: float = 60.0

    #: Per second of simulated time. **Leave at zero unless you know why.**
    #: Any positive value re-creates the suicide exploit above, in proportion to
    #: its size: it becomes worth ending an episode early to stop paying it. Only
    #: consider it once the agent finishes reliably, and re-read the failure
    #: histogram afterwards -- a jump in ``off_line`` at zero progress is the
    #: signature.
    time_penalty: float = 0.0


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
        par_frames: int | None = None,
        # Cut an episode that has stopped getting anywhere. The real backend
        # never reports `terminated` -- the module has no notion of "off track",
        # a car in a ditch is simply a car in a ditch -- so without this a run
        # that fails on the first corner still costs the whole time limit. At
        # ~0.5x realtime that is a collector doing nothing for a minute, and it
        # is the difference between a usable throughput figure and an unusable
        # one. Linesight cuts episodes the same way and for the same reason.
        stall_seconds: float = 3.0,
        stall_metres: float = 5.0,
        off_line_limit: float = 30.0,
        # Grace before the stall cut arms. Without it the cut fights the very
        # thing it is meant to enable: a car on the line has not moved yet, so
        # an untrained policy is killed at ~3 m every episode and never collects
        # enough throttle-then-move transitions to discover the throttle. Long
        # enough to get rolling, short enough that a wedged car is still cheap.
        stall_grace_seconds: float = 5.0,
    ) -> None:
        super().__init__()
        self.track = track or straight_track()
        self.backend = backend or ReferencePhysics(self.track)
        self.observer = ObservationBuilder(self.track, obs_config)
        self.reward_config = reward_config or RewardConfig()
        self.control_interval_ms = float(control_interval_ms)
        self.time_limit_ms = float(time_limit_ms)
        self.stall_seconds = float(stall_seconds)
        self.stall_metres = float(stall_metres)
        self.off_line_limit = float(off_line_limit)
        self.stall_grace_seconds = float(stall_grace_seconds)

        # The reward is denominated in laps and par times, so both scales have
        # to be known up front. Without a reference time, assume a nominal pace
        # over this line's length -- which keeps the synthetic demo tracks
        # working without pretending we know what a good time on them is.
        self.par_seconds = (
            par_frames / 1000.0 if par_frames
            else self.track.length / self.reward_config.nominal_speed
        )
        self._per_metre = self.reward_config.lap_reward / max(self.track.length, 1e-6)
        self._per_second = self.reward_config.time_penalty

        self.ticks_per_step = ticks_for(self.backend, self.control_interval_ms)
        self._ms_per_tick = 1000.0 / self.backend.rate

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observer.size,), dtype=np.float32
        )

        self._progress = ProgressTracker(self.track)
        self._previous_arc = 0.0
        #: Simulated seconds at which the high-water arc last improved.
        self._last_gain_seconds = 0.0
        self._gain_marker = 0.0
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
        self._last_gain_seconds = 0.0
        self._gain_marker = arc
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
        reward = self._per_metre * delta - self._per_second * interval_seconds
        self._previous_arc = max(arc, self._previous_arc)

        if result.passed_checkpoint:
            reward += cfg.checkpoint_bonus

        # Has the run got anywhere lately? Measured against a marker that only
        # moves when real ground is gained, so creeping forward a centimetre at
        # a time does not reset the clock.
        elapsed_seconds = elapsed_ms / 1000.0
        if self._previous_arc - self._gain_marker >= self.stall_metres:
            self._gain_marker = self._previous_arc
            self._last_gain_seconds = elapsed_seconds
        stalled = (
            elapsed_seconds >= self.stall_grace_seconds
            and elapsed_seconds - self._last_gain_seconds >= self.stall_seconds
        )
        lost = lateral_offset > self.off_line_limit

        terminated = False
        truncated = False

        if result.finished:
            terminated = True
            # The only time pressure in the whole reward. Scaled by how quickly
            # the lap was done, capped so one freak run cannot dominate replay.
            speed_factor = self.par_seconds / max(elapsed_ms / 1000.0, 1e-6)
            reward += cfg.finish_bonus * min(speed_factor, cfg.finish_speed_cap)
        elif result.terminated or stalled or lost:
            # No penalty. A failed episode simply stops earning, and the lap
            # reward it never collected is the whole of its cost -- see
            # RewardConfig for what happened when this was a penalty instead.
            terminated = True
            failure = "off_line" if lost else "stalled" if stalled else "terminated"
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
            if terminated and not result.finished:
                info["failure"] = failure
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
