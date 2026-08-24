"""Collector processes: several copies of the physics module, one learner.

The module runs at roughly half realtime per instance and cannot be made faster
-- ~700 us of its 1 ms tick is its own internal work, and the Python host is
already down to a raw wasmtime call. One collector therefore yields about eleven
transitions a second at a 50 ms control interval. That is the entire reason this
file exists: parallelism across processes is not an optimisation here, it is the
only source of throughput.

Three things about this are Windows-specific and all three bite:

* **Processes start by ``spawn``, not ``fork``**, so nothing is inherited. Every
  worker rebuilds its own geometry, its own physics instances and its own network
  from picklable configuration. A :class:`~polytrack.physics.wasm.WasmPhysics`
  cannot cross a process boundary at all -- it holds ctypes pointers into a
  wasmtime store.
* **Torch grabs every core by default.** Six workers each starting twelve
  compute threads on a twelve-core machine spends its life in contention; the
  measured cost of forgetting ``set_num_threads(1)`` is worse than running one
  worker. Set in :func:`collector_process`, before any tensor exists.
* **The reference lines must be cached.** Deriving one costs a full replay, ~45 s
  per track. Six workers rebuilding four lines each at every launch is eighteen
  minutes of nothing; :class:`~polytrack.library.TrackLibrary` makes it instant.

Exploration follows Ape-X: each worker gets its own fixed epsilon on a geometric
spread, so the buffer holds both near-greedy trajectories that refine the current
line and noisy ones that find new ones, without any single actor having to
compromise between the two.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from ..env import PolyTrackEnv, RewardConfig
from ..observation import ObservationConfig
from .iqn import IQNAgent, IQNConfig


@dataclass(frozen=True)
class CollectorConfig:
    """Everything a worker needs, in a form that survives pickling."""

    library_path: str = "tracks.npz"
    #: Restrict training to these tracks. Empty means all of them.
    track_names: tuple[str, ...] = ()
    geometry_path: str | None = None
    wasm_path: str | None = None
    control_interval_ms: float = 50.0
    #: Episode budget as a multiple of the track's par time. 3x leaves room to
    #: be bad without letting a stuck car burn a collector for a minute.
    time_limit_factor: float = 3.0
    n_step: int = 5
    gamma: float = 0.99
    #: Ape-X epsilon spread. Worker i gets base ** (1 + i/(n-1) * spread).
    epsilon_base: float = 0.4
    epsilon_spread: float = 7.0
    #: Weights are pulled at most this often, to keep a worker from spending its
    #: time deserialising instead of driving.
    weight_poll_seconds: float = 2.0
    obs_config: ObservationConfig = field(default_factory=ObservationConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    iqn_config: IQNConfig = field(default_factory=IQNConfig)


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """What one episode achieved. Sent to the learner for logging only."""

    worker: int
    track: str
    steps: int
    frames: int
    reward: float
    arc: float
    track_length: float
    finished: bool
    epsilon: float
    #: ``stalled``, ``off_line``, ``terminated``, or ``timeout``. Which one
    #: dominates says a great deal: all ``off_line`` means the car cannot hold
    #: the track, all ``stalled`` means it cannot get off the line.
    failure: str
    #: The inputs, in PolyTrack's own recording format, for episodes that
    #: finished. Without this a record lap is a number in a log and nothing
    #: else -- the run that produced it is gone the moment the episode ends,
    #: and no ghost of it can ever be built. Encoded in the worker because the
    #: format is run-length coded: a 19-second lap is a couple of hundred bytes
    #: rather than 19,000 integers down a queue.
    recording: str | None = None

    @property
    def fraction(self) -> float:
        return self.arc / max(self.track_length, 1e-6)


def worker_epsilon(index: int, n_workers: int, config: CollectorConfig) -> float:
    """Ape-X's geometric epsilon spread across actors."""
    if n_workers <= 1:
        return config.epsilon_base
    exponent = 1.0 + (index / (n_workers - 1)) * config.epsilon_spread
    return float(config.epsilon_base**exponent)


def collector_process(
    index: int,
    n_workers: int,
    config: CollectorConfig,
    transitions: mp.Queue,
    weights: mp.Queue,
    stats: mp.Queue,
    stop,
) -> None:
    """Drive episodes forever, pushing n-step transitions to the learner."""
    try:
        import torch

        from ..priority import lift_background_qos

        # Before anything allocates a tensor. See the module docstring.
        torch.set_num_threads(1)
        # Inherited from the learner in the normal case; not free to
        # assume, and the cost of being wrong is an order of magnitude.
        lift_background_qos()

        from ..library import TrackLibrary
        from ..physics.assets import GeometryAssets
        from ..physics.wasm import WasmPhysics
        from ..recording import Recording
        from .replay import NStepAccumulator

        np.random.seed(1000 + index)
        torch.manual_seed(1000 + index)

        assets = GeometryAssets.load(config.geometry_path)
        library = TrackLibrary.load(config.library_path)
        if config.track_names:
            library = library.select(config.track_names)
        agent = IQNAgent(config.iqn_config)
        epsilon = worker_epsilon(index, n_workers, config)

        # Built lazily and kept: creating one registers 186 collision meshes,
        # which is cheap (~0.16 s) but not free, and reset() is the right way to
        # start a new episode anyway.
        envs: dict[str, PolyTrackEnv] = {}

        def env_for(entry) -> PolyTrackEnv:
            existing = envs.get(entry.name)
            if existing is None:
                backend = WasmPhysics(
                    entry.track_code(), assets=assets, wasm_path=config.wasm_path
                )
                existing = PolyTrackEnv(
                    track=entry.line,
                    backend=backend,
                    obs_config=config.obs_config,
                    reward_config=config.reward_config,
                    control_interval_ms=config.control_interval_ms,
                    time_limit_ms=entry.par_frames * config.time_limit_factor,
                    par_frames=entry.par_frames,
                )
                envs[entry.name] = existing
            return existing

        accumulator = NStepAccumulator(config.n_step, config.gamma)
        last_poll = 0.0
        episode = 0

        while not stop.is_set():
            # Round-robin the tracks rather than sampling, so a short run still
            # covers all of them and no track is starved by luck.
            entry = library[(index + episode) % len(library)]
            env = env_for(entry)
            observation, _ = env.reset()
            accumulator.clear()

            total_reward = 0.0
            steps = 0
            info: dict = {}
            while not stop.is_set():
                action = agent.act(observation, epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                for transition in accumulator.push(
                    observation, action, reward, next_observation, terminated
                ):
                    try:
                        transitions.put(transition, timeout=1.0)
                    except queue.Full:
                        pass  # learner is behind; dropping is better than stalling

                observation = next_observation
                if terminated or truncated:
                    break

                now = time.monotonic()
                if now - last_poll >= config.weight_poll_seconds:
                    last_poll = now
                    latest = None
                    while True:  # drain to the newest available
                        try:
                            latest = weights.get_nowait()
                        except queue.Empty:
                            break
                    if latest is not None:
                        agent.load_weights(latest)

            finished = bool(info.get("finished", False))
            stats.put(
                EpisodeStats(
                    worker=index,
                    track=entry.name,
                    steps=steps,
                    frames=env.elapsed_frames,
                    reward=total_reward,
                    arc=float(info.get("arc", 0.0)),
                    track_length=entry.line.length,
                    finished=finished,
                    epsilon=epsilon,
                    failure=str(info.get("failure", "timeout")),
                    # Only for laps that completed: everything else is not a
                    # run anyone would want to watch, and encoding costs time
                    # the collector should spend driving.
                    recording=(
                        Recording.from_masks(env.input_masks).serialize()
                        if finished else None
                    ),
                )
            )
            episode += 1
    except Exception:
        # A worker dying silently looks exactly like a slow worker, and costs an
        # afternoon to notice. Report and let the learner decide.
        stats.put(f"worker {index} died:\n{traceback.format_exc()}")


class CollectorPool:
    """Starts, feeds and stops the worker processes."""

    def __init__(self, n_workers: int, config: CollectorConfig) -> None:
        self.n_workers = n_workers
        self.config = config
        context = mp.get_context("spawn")
        # Bounded: if the learner falls behind, workers should block briefly and
        # then drop, not consume memory until the machine swaps.
        self.transitions: mp.Queue = context.Queue(maxsize=20_000)
        self.stats: mp.Queue = context.Queue()
        self.stop = context.Event()
        self._weight_queues = [context.Queue(maxsize=2) for _ in range(n_workers)]
        self._processes = [
            context.Process(
                target=collector_process,
                args=(
                    i, n_workers, config, self.transitions,
                    self._weight_queues[i], self.stats, self.stop,
                ),
                daemon=True,
                name=f"collector-{i}",
            )
            for i in range(n_workers)
        ]

    def start(self) -> None:
        for process in self._processes:
            process.start()

    def broadcast(self, weights: dict) -> None:
        """Offer new weights to every worker, without ever blocking on one."""
        for channel in self._weight_queues:
            try:
                channel.put_nowait(weights)
            except queue.Full:
                pass  # that worker has not picked up the previous set yet

    def drain_transitions(self, limit: int = 4096) -> list:
        out = []
        for _ in range(limit):
            try:
                out.append(self.transitions.get_nowait())
            except queue.Empty:
                break
        return out

    def drain_stats(self) -> list:
        out = []
        while True:
            try:
                out.append(self.stats.get_nowait())
            except queue.Empty:
                return out

    def close(self, timeout: float = 10.0) -> None:
        self.stop.set()
        for process in self._processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
