"""Reference lines: turning a real run into geometry the agent can be scored on.

A PolyTrack track code is a pile of placed parts. It says exactly where every
piece of road is, and nothing whatsoever about the route through them -- which
part you enter first, which way round the loop goes, whether the shortcut over
the jump is a shortcut. The reward and the observation both need that route.

So take it from a lap that was actually driven. Replay a recording through the
verified backend, sample where the car went, and resample that path at fixed arc
spacing: the result is a :class:`~polytrack.track.Track` whose "centerline" is
one real racing line. This is Linesight's design for Trackmania, and the reasons
carry over intact:

* It is a *route*, so it is unambiguous where the track crosses itself.
* It is drivable by construction, which a geometric centerline through a series
  of jumps and wall-rides is not.
* Progress along it is a proper dense reward, and beating the run that produced
  it is a legible goal.

The cost is that a reference line is only as good as the lap behind it, and the
agent is being pointed at that lap's route. That is the right trade for now --
the lap is a human personal best, and it is the same signal every successful
racing agent in the literature trained against. It stops being a constraint at
Stage 2: a procedurally generated track knows its own route by construction, so
generated training tracks get a reference line for free and never need a lap
driven for them.

Sampling does not perturb the replay. The recording is walked as runs of held
input exactly as :mod:`polytrack.verify` walks it, and those runs are merely cut
into shorter pieces at the sample interval -- the module sees the same input on
the same tick either way, so a sampled replay lands on the same finish frame as
the gate. :func:`sample_replay` checks that rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PolyTrackEnv
from .physics.assets import GeometryAssets
from .physics.wasm import WasmPhysics
from .recording import Recording
from .track import Track
from .trackcode import TrackCode, decode


class ReferenceLineError(RuntimeError):
    """A replay could not be turned into a usable reference line."""


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Where the car was, sampled along a replay.

    Attributes:
        positions: (M, 3) world positions, in driving order.
        frames: (M,) the physics frame each sample was taken at.
        checkpoints: (M,) checkpoints passed as of each sample.
        speed_kmh: (M,) the module's own speed reading at each sample.
        finished: Whether the run reached the finish line.
        final_frame: The frame the replay stopped on.
    """

    positions: np.ndarray
    frames: np.ndarray
    checkpoints: np.ndarray
    speed_kmh: np.ndarray
    finished: bool
    final_frame: int

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def checkpoint_samples(self) -> tuple[int, ...]:
        """Sample indices at which the checkpoint count went up."""
        increased = np.flatnonzero(np.diff(self.checkpoints) > 0) + 1
        return tuple(int(i) for i in increased)

    def to_track(
        self,
        name: str = "reference",
        *,
        spacing: float = 2.0,
        smoothing: int = 3,
        width: float = 5.0,
    ) -> Track:
        """Resample into a :class:`~polytrack.track.Track`.

        See :meth:`~polytrack.track.Track.from_trajectory` for what the knobs do.
        """
        return Track.from_trajectory(
            self.positions,
            spacing=spacing,
            smoothing=smoothing,
            width=width,
            checkpoint_samples=self.checkpoint_samples,
            name=name,
        )


def sample_replay(
    backend: WasmPhysics,
    recording: Recording,
    target_frames: int,
    *,
    sample_ms: int = 20,
    require_finish: bool = True,
) -> Trajectory:
    """Replay a recording, recording where the car was every ``sample_ms``.

    Args:
        backend: A backend on the recording's track, at its start. Driven to the
            finish by this call.
        recording: The input sequence.
        target_frames: The frame the run finished on, from the record alongside
            it. A recording carries no length of its own.
        sample_ms: Sample interval in milliseconds. 20 ms is ~1.7 m at 300 km/h,
            comfortably finer than the 5 m part grid.
        require_finish: Raise if the run does not reach the finish. A line built
            from a partial lap stops in the middle of the track, and every
            episode run against it would truncate there.

    Raises:
        ReferenceLineError: on a respawn, an implausible jump, or -- under
            ``require_finish`` -- a run that did not finish on the frame claimed.
    """
    if recording.reset:
        raise ReferenceLineError(
            f"this recording presses respawn {len(recording.reset)} time(s), at "
            f"frame(s) {recording.reset[:4]}. The car teleports back to a "
            "checkpoint, so the path it traces is not a single lap and cannot be "
            "resampled into one. Use a clean run."
        )
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    ticks = max(1, int(round(sample_ms * backend.rate / 1000.0)))
    # Nothing in PolyTrack goes this fast; the check is here to catch a teleport,
    # not to police the physics.
    max_step = 400.0 * sample_ms / 1000.0

    state = backend.reset()
    positions = [np.asarray(state.position, dtype=np.float64)]
    frames = [backend.frame]
    checkpoints = [state.checkpoint]
    speeds = [0.0]

    result = None
    for mask, run_ticks in recording.runs(target_frames):
        remaining_run = min(run_ticks, target_frames - backend.frame)
        while remaining_run > 0:
            # Cutting a run of held input into sample-sized pieces does not
            # change what the module sees, so this stays an exact replay.
            chunk = min(ticks, remaining_run)
            previous = positions[-1]
            result = backend.step_many(mask, chunk)
            remaining_run -= chunk

            position = np.asarray(result.state.position, dtype=np.float64)
            jump = float(np.linalg.norm(position - previous))
            if jump > max_step:
                raise ReferenceLineError(
                    f"the car moved {jump:.1f} m in {chunk} ms at frame "
                    f"{backend.frame}, which is not driving. A respawn or a "
                    "desync -- either way the path is not a lap."
                )
            positions.append(position)
            frames.append(backend.frame)
            checkpoints.append(result.state.checkpoint)
            speeds.append(result.info.get("speed_kmh", 0.0))
            if result.finished:
                break
        if result is not None and result.finished:
            break

    finished = bool(result is not None and result.finished)
    if require_finish and not (finished and backend.frame == target_frames):
        raise ReferenceLineError(
            f"replay did not reproduce the recorded run: finished={finished} on "
            f"frame {backend.frame}, expected {target_frames}. Sampling cannot "
            "cause this -- run the acceptance gate on this record."
        )

    return Trajectory(
        positions=np.array(positions, dtype=np.float64),
        frames=np.array(frames, dtype=np.int64),
        checkpoints=np.array(checkpoints, dtype=np.int32),
        speed_kmh=np.array(speeds, dtype=np.float64),
        finished=finished,
        final_frame=backend.frame,
    )


def reference_line(
    track: TrackCode | str,
    recording: Recording,
    target_frames: int,
    *,
    assets: GeometryAssets | None = None,
    sample_ms: int = 20,
    spacing: float = 2.0,
    smoothing: int = 3,
    width: float = 5.0,
) -> tuple[Track, WasmPhysics, Trajectory]:
    """Build a reference line for a track from one run on it.

    Returns the line, the backend that produced it (reset and ready to drive),
    and the raw trajectory. The backend comes back because registering 186 parts
    of collision geometry is most of the cost of building one, and the caller is
    about to want one on this exact track.
    """
    backend = WasmPhysics(track, assets=assets)
    trajectory = sample_replay(
        backend, recording, target_frames, sample_ms=sample_ms
    )
    name = backend.track.name or "reference"
    line = trajectory.to_track(
        name, spacing=spacing, smoothing=smoothing, width=width
    )
    backend.reset()
    return line, backend, trajectory


def env_from_recording(
    track: TrackCode | str,
    recording: Recording,
    target_frames: int,
    *,
    assets: GeometryAssets | None = None,
    sample_ms: int = 20,
    spacing: float = 2.0,
    smoothing: int = 3,
    width: float = 5.0,
    **env_kwargs,
) -> PolyTrackEnv:
    """A :class:`~polytrack.env.PolyTrackEnv` on real physics and a real line.

    The one-call version of the above: decode the track, derive its reference
    line from the supplied run, and hand back an environment driving PolyTrack's
    own physics module. ``env_kwargs`` go to the environment -- ``time_limit_ms``
    is the one worth setting, since it defaults to a minute and a long community
    track needs more.
    """
    line, backend, _ = reference_line(
        track, recording, target_frames,
        assets=assets, sample_ms=sample_ms, spacing=spacing,
        smoothing=smoothing, width=width,
    )
    from .corridor import RoadMap

    # Rays want the real collision geometry. `assets` being None means the
    # caller has none, and the observation falls back to the reference line.
    road = env_kwargs.pop("road", None)
    if road is None and assets is not None:
        code = decode(track) if isinstance(track, str) else track
        road = RoadMap.build(code, assets)
    return PolyTrackEnv(track=line, backend=backend, road=road, **env_kwargs)
