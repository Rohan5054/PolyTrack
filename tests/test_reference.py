"""Reference lines, and the environment's control interval.

Split the same way as ``test_wasm.py``. The first half is pure geometry and
arithmetic and runs anywhere. The second half derives a reference line from one
of the real recordings in ``records.json`` and needs the same two uncommitted
game-derived files the backend does, plus a track to look the recording's id up
in::

    POLYTRACK_TRACKS=path/to/tracks python tests/test_reference.py

Those tests skip with a note when anything is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from polytrack.env import PolyTrackEnv, RewardConfig
from polytrack.physics.base import ticks_for
from polytrack.physics.reference import ReferencePhysics
from polytrack.track import (
    ProgressTracker,
    Track,
    resample_polyline,
    straight_track,
)

_SKIPPED: list[str] = []
_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- geometry --


def _lemniscate(n: int = 400, scale: float = 60.0) -> np.ndarray:
    """A figure-eight: the smallest path that crosses itself."""
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.stack(
        [scale * np.sin(t), scale * np.sin(t) * np.cos(t), np.zeros_like(t)], axis=1
    )


def test_resample_polyline_spaces_points_evenly():
    """Evenly *along the path*, which is not the same as evenly apart.

    Output points sit at exact arc multiples of the spacing, so the straight-line
    distance between two of them is a chord and comes out slightly short wherever
    the path curves. Asserting the chords are exactly the spacing would be
    asserting the path is straight.
    """
    path = _lemniscate()
    line = resample_polyline(path, 2.0)

    arcs, _ = Track(path, name="source").project_many(line)
    expected = np.minimum(np.arange(len(line)) * 2.0, arcs[-1])
    assert np.allclose(arcs, expected, atol=1e-6)

    chords = np.linalg.norm(np.diff(line, axis=0), axis=1)
    assert chords.max() <= 2.0 + 1e-9
    assert chords[:-1].min() > 1.99  # curvature loss, not a spacing bug


def test_resample_polyline_keeps_both_endpoints():
    path = _lemniscate()
    line = resample_polyline(path, 2.0)
    assert np.allclose(line[0], path[0])
    assert np.allclose(line[-1], path[-1])


def test_resample_polyline_stays_on_the_path():
    """Interpolation, not fitting.

    A spline through a driven line cuts the inside of every corner the car
    actually took, which would make the reference line undrivable exactly where
    it matters most.
    """
    path = _lemniscate()
    line = resample_polyline(path, 1.0)
    source = Track(path, name="source")
    _, offsets = source.project_many(line)
    assert offsets.max() < 1e-9


def test_from_trajectory_survives_a_stationary_start():
    """A real run begins with the car parked on the line.

    Every one of those samples is coincident with the last, and Track rejects
    zero-length segments -- so a raw recording would refuse to build a line at
    all until the duplicates are dropped.
    """
    driving = _lemniscate()
    parked = np.repeat(driving[:1], 250, axis=0)
    track = Track.from_trajectory(np.vstack([parked, driving]), spacing=2.0)
    assert len(track.centerline) > 100
    assert np.allclose(track.centerline[0], driving[0])


def test_from_trajectory_places_checkpoints_where_they_were_passed():
    straight = np.stack(
        [np.linspace(0.0, 200.0, 201), np.zeros(201), np.zeros(201)], axis=1
    )
    track = Track.from_trajectory(
        straight, spacing=2.0, smoothing=0, checkpoint_samples=(50, 150)
    )
    assert track.n_checkpoints == 3  # two checkpoints plus the finish
    assert abs(track.checkpoint_arc(0) - 50.0) < 2.0
    assert abs(track.checkpoint_arc(1) - 150.0) < 2.0
    assert abs(track.checkpoint_arc(2) - track.length) < 1e-9


def test_smoothing_leaves_the_endpoints_alone():
    """The start line and the finish must not creep.

    A moving average pulls the ends of a path inward, and on a reference line
    that means the car spawns behind its own arc-zero -- which is the closed-loop
    projection bug all over again.
    """
    path = _lemniscate()
    rough = Track.from_trajectory(path, spacing=2.0, smoothing=0)
    smooth = Track.from_trajectory(path, spacing=2.0, smoothing=9)
    assert np.allclose(rough.centerline[0], smooth.centerline[0])
    assert np.allclose(rough.centerline[-1], smooth.centerline[-1])


def _out_and_back(length: float = 200.0, gap: float = 3.0) -> np.ndarray:
    """Out along y=0, tight U-turn, back along y=gap. A hairpin.

    Nothing exotic: PolyTrack tracks double back on themselves constantly, and
    3 m is well inside one part's width, so the two legs are genuinely close
    enough to be confused for each other.
    """
    forward = np.arange(0.0, length, 1.0)
    out = np.stack([forward, np.zeros_like(forward), np.zeros_like(forward)], axis=1)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 24)
    turn = np.stack(
        [length + (gap / 2) * np.cos(theta),
         gap / 2 + (gap / 2) * np.sin(theta),
         np.zeros_like(theta)],
        axis=1,
    )
    reverse = forward[::-1]
    back = np.stack(
        [reverse, np.full_like(reverse, gap), np.zeros_like(reverse)], axis=1
    )
    return np.vstack([out, turn, back])


def test_a_driven_line_stays_unambiguous_where_it_runs_alongside_itself():
    """The whole reason a reference line is a route and not a point cloud.

    A car on the return leg of a hairpin, drifting a little toward the outbound
    leg, is nearer in space to where it was twenty seconds ago than to where the
    line says it is. Projected globally it reads as barely started. Followed
    statefully -- the only correct way -- it reads as nearly done.
    """
    path = _out_and_back()
    track = Track.from_trajectory(path, spacing=1.0, smoothing=0, width=5.0)

    # On the return leg, drift toward the outbound one, reaching 1.6 m over the
    # first 40 m. The car ends up 1.4 m from the leg it is not on and 1.6 m from
    # the leg it is -- still well inside the corridor, and enough to fool a
    # nearest-point projection. The drift is gradual because a car's is: right
    # at the apex the two legs are 3 m apart and nothing can tell them apart, so
    # a test that teleports across there is testing the wrong thing.
    queries = path.copy()
    queries[-200:, 1] -= np.minimum(np.arange(200) / 40.0, 1.0) * 1.6

    tracker = ProgressTracker(track)
    tracker.reset()
    arcs, globals_ = [], []
    for query in queries:
        arcs.append(tracker.update(query)[0])
        globals_.append(track.project(query)[0])

    assert all(b >= a - 1e-9 for a, b in zip(arcs, arcs[1:])), "arc went backwards"
    assert arcs[-1] > track.length * 0.98, (
        f"followed the line to arc {arcs[-1]:.1f} of {track.length:.1f} m"
    )
    divergence = np.abs(np.array(arcs) - np.array(globals_)).max()
    assert divergence > 100.0, (
        "global projection never disagreed, so the two legs are not actually "
        "confusable and this test proves nothing"
    )


# ------------------------------------------------------- control interval --


def test_control_interval_is_a_duration_not_a_tick_count():
    """50 ms is 3 frames of the 60 Hz stand-in and 50 of the real module."""
    env = PolyTrackEnv(track=straight_track(), control_interval_ms=50.0)
    assert env.backend.rate == 60.0
    assert env.ticks_per_step == 3

    env.reset()
    env.step(3)
    assert env.backend.frame == 3
    assert env.elapsed_frames == 3

    # And the conversion itself, without needing a 1000 Hz backend to hand.
    class _Fake:
        rate = 1000.0

    assert ticks_for(_Fake(), 50.0) == 50
    assert ticks_for(_Fake(), 0.4) == 1  # never rounds down to nothing


def test_reward_does_not_depend_on_how_finely_it_is_sliced():
    """Holding one action for a fixed duration must pay the same either way.

    The coefficients are per second, so cutting the same drive into twice as
    many decisions must not double the time penalty -- otherwise every change to
    the control interval silently retunes the reward.
    """
    total = 0.0
    for interval, steps in ((50.0, 12), (100.0, 6)):
        env = PolyTrackEnv(
            track=straight_track(length=400.0),
            control_interval_ms=interval,
            reward_config=RewardConfig(),
        )
        env.reset()
        earned = sum(env.step(3)[1] for _ in range(steps))
        assert env.elapsed_frames == 36
        if total:
            assert abs(earned - total) < 1e-9, (
                f"600 ms of throttle paid {earned:.6f} in {steps} decisions "
                f"but {total:.6f} in the other split"
            )
        total = earned


def test_a_faster_lap_scores_higher_than_a_slow_one():
    """The property the whole reward exists to express.

    Two runs covering identical ground, one taking twice as long. If the slower
    one is not clearly worse, the time penalty is decoration -- which is what it
    was, at 1.3% of the progress signal, before the scales were normalised.
    """
    line = Track.from_trajectory(_out_and_back(), spacing=1.0, smoothing=0)
    tracker = ProgressTracker(line)

    def score(seconds: float) -> float:
        env = PolyTrackEnv(track=line, par_frames=20_000)
        tracker.reset()
        # Score the same arc profile by hand, independent of any backend: the
        # question is about the reward function, not about driving.
        per_metre = env._per_metre
        per_second = env._per_second
        return per_metre * line.length - per_second * seconds

    assert score(20.0) > score(40.0)
    assert score(18.0) > score(20.0), "beating par must pay"
    # A lap at half the reference pace is worth roughly nothing.
    assert abs(score(40.0)) < 1.0


def test_the_reward_scale_is_the_same_on_every_track():
    """A lap is a lap, whether the track is 1300 m or 3980 m.

    Without this, one shared Q-function has to represent returns three times
    larger on Summer 5 than on Summer 1 for reasons that have nothing to do with
    driving well -- and the collectors all feed one buffer.
    """
    short = Track.from_trajectory(_out_and_back(length=100.0), spacing=1.0)
    long = Track.from_trajectory(_out_and_back(length=400.0), spacing=1.0)
    a = PolyTrackEnv(track=short, par_frames=10_000)
    b = PolyTrackEnv(track=long, par_frames=40_000)

    # A full lap is worth lap_reward on both, and par time costs the same.
    assert abs(a._per_metre * short.length - b._per_metre * long.length) < 1e-9
    assert abs(a._per_second * 10.0 - b._per_second * 40.0) < 1e-9


def test_the_crash_exploit_does_not_depend_on_where_you_are():
    """The suicide exploit, and why the failure penalty is proportional.

    A failed episode ends, so its future is worth zero, while continuing to
    drive badly keeps paying the time penalty. For a poor policy that makes
    crashing the better move -- it is the cheapest way to stop the clock, and an
    agent will find it.

    Scaling the forfeit by the fraction of track *not* driven makes both sides
    of that comparison scale identically with position, so the whole thing
    cancels: whether crashing beats driving on stops being a question about
    where the car is and becomes a single question about pace. A flat penalty
    does not cancel, and leaves a band near the start where bailing out always
    wins.
    """
    line = Track.from_trajectory(_out_and_back(), spacing=1.0)
    cfg = PolyTrackEnv(track=line, par_frames=20_000).reward_config

    def crash(done: float) -> float:
        return -cfg.failure_penalty * cfg.lap_reward * (1.0 - done)

    def carry_on(done: float, pace: float) -> float:
        """Value of driving the rest of the lap at ``pace`` x the par speed."""
        remaining = 1.0 - done
        return remaining * cfg.lap_reward - (remaining / pace) * cfg.par_time_penalty

    # Both sides scale with the track left, so the margin between them does too
    # -- what has to be constant is the margin *per unit of remaining track*,
    # because that is what fixes the sign, and the sign is the decision.
    places = (0.1, 0.3, 0.5, 0.7, 0.9)
    rates = [(carry_on(d, 0.6) - crash(d)) / (1.0 - d) for d in places]
    assert max(rates) - min(rates) < 1e-9, (
        f"whether crashing pays depends on position: {rates}"
    )
    assert min(rates) > 0.0, "crashing beats driving on at 60% of par pace"

    # And the pace below which crashing wins, which is what the configuration
    # actually chooses. A third of the human's pace is a car barely moving.
    break_even = cfg.par_time_penalty / (
        cfg.lap_reward * (1.0 + cfg.failure_penalty)
    )
    assert break_even < 0.4, (
        f"an agent managing {break_even:.0%} of par pace would rather crash; "
        "raise failure_penalty or lower par_time_penalty"
    )
    assert carry_on(0.0, break_even * 1.2) > crash(0.0)
    assert carry_on(0.0, break_even * 0.8) < crash(0.0)


def test_the_input_log_is_per_frame_so_a_tas_replays():
    """An exported TAS is per millisecond; a decision is not."""
    env = PolyTrackEnv(track=straight_track(), control_interval_ms=50.0)
    env.reset()
    for action in (3, 4, 3):
        env.step(action)
    tas = env.export_tas()
    assert tas.n_frames == 9  # three decisions, three frames each
    assert len(tas) == 3  # collapsed back to three input changes


def test_step_many_matches_stepping_one_at_a_time():
    """The interval must be a grouping, not a different simulation."""
    masks = (1, 9, 5, 1, 3)
    one_at_a_time = ReferencePhysics(straight_track(length=400.0))
    one_at_a_time.reset()
    for mask in masks:
        for _ in range(7):
            single = one_at_a_time.step(mask)

    batched = ReferencePhysics(straight_track(length=400.0))
    batched.reset()
    for mask in masks:
        grouped = batched.step_many(mask, 7)

    assert single.state.frame == grouped.state.frame
    assert np.allclose(single.state.position, grouped.state.position)
    assert np.allclose(single.state.velocity, grouped.state.velocity)


# ------------------------------------------------------------ the real run --


def _shortest_record():
    """The quickest real run we have, with its track. ``None`` if unavailable."""
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.trackcode import PREFIX, decode

    records_path = _ROOT / "records.json"
    if not records_path.exists():
        _SKIPPED.append("no records.json: run tools/extract_records.py")
        return None
    root = os.environ.get("POLYTRACK_TRACKS")
    if not root:
        _SKIPPED.append("POLYTRACK_TRACKS is not set, so no track geometry")
        return None
    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None

    tracks = {}
    for path in Path(root).rglob("*.track"):
        text = path.read_text(encoding="utf-8").strip()
        if text.startswith(PREFIX):
            try:
                track = decode(text)
            except Exception:
                continue
            tracks[track.track_id] = track

    candidates = [
        (r, tracks[r["track_id"]])
        for r in json.loads(records_path.read_text())
        if r["track_id"] in tracks
    ]
    if not candidates:
        _SKIPPED.append("no record in records.json matches an available track")
        return None
    record, track = min(candidates, key=lambda pair: pair[0]["frames"])
    return record, track, assets


_CACHE: list = []


def _reference():
    """Build the line once; three tests want it and it costs a real replay."""
    if not _CACHE:
        from polytrack.recording import Recording
        from polytrack.reference import reference_line

        found = _shortest_record()
        if found is None:
            _CACHE.append(None)
        else:
            record, track, assets = found
            line, backend, trajectory = reference_line(
                track,
                Recording.deserialize(record["recording"]),
                record["frames"],
                assets=assets,
            )
            _CACHE.append((line, backend, trajectory, record, track))
    return _CACHE[0]


def test_sampling_a_replay_does_not_perturb_it():
    """The strongest check here: the sampled replay is still the gate's replay.

    Cutting runs of held input into 20 ms pieces feeds the module the same input
    on the same tick, so the run must still land on its recorded finish frame.
    If it does not, the trajectory a reference line is built from is not the
    trajectory the run actually took.
    """
    built = _reference()
    if built is None:
        return
    _, _, trajectory, record, _ = built
    assert trajectory.finished
    assert trajectory.final_frame == record["frames"], (
        f"sampled replay finished on {trajectory.final_frame}, recorded "
        f"{record['frames']}"
    )


def test_the_reference_line_covers_the_whole_run():
    built = _reference()
    if built is None:
        return
    line, _, trajectory, record, track = built

    # Every checkpoint on the track has to appear on the line, plus the finish.
    expected = sum(1 for p in track.parts if p.checkpoint_order is not None)
    assert line.n_checkpoints == expected + 1, (
        f"{line.n_checkpoints} checkpoints on the line, track has {expected}"
    )

    # A sane average speed for a run of this length: the line is metres and the
    # frame count is milliseconds, so this catches a units error immediately.
    average = line.length / (record["frames"] / 1000.0)
    assert 10.0 < average < 120.0, f"{average:.1f} m/s average is not driving"

    # The line has to pass near every sampled position, or it is not this run.
    _, offsets = line.project_many(trajectory.positions[::10])
    assert offsets.max() < 2.0, f"line strays {offsets.max():.2f} m from the run"


def test_an_environment_on_real_physics_makes_progress():
    """The end of the wiring: real module, real track, real reference line."""
    built = _reference()
    if built is None:
        return
    line, backend, _, _, _ = built

    env = PolyTrackEnv(track=line, backend=backend, control_interval_ms=50.0)
    assert env.ticks_per_step == 50

    observation, _ = env.reset()
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()

    info = {}
    for _ in range(40):  # two seconds of throttle
        observation, reward, terminated, truncated, info = env.step(3)
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            break

    assert info["frame"] == env.elapsed_frames
    assert info["arc"] > 10.0, f"went nowhere: arc {info['arc']:.2f} m"
    assert info["speed"] > 5.0


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if _SKIPPED:
        print(f"\nrun-backed tests skipped: {_SKIPPED[0]}")
    sys.exit(1 if failures else 0)
