"""Input recordings and the acceptance gate.

The codec tests run anywhere. The gate tests need the physics module and
`geometry.npz`, and skip cleanly without them -- see `tests/test_wasm.py`.

Note what the gate tests here do and do not establish. They prove the *harness*
is exact: that a run verifying on frame N fails on N-1 and N+1, so a simulator
one millisecond out cannot slip through. They say nothing about whether the
physics matches PolyTrack, because the target frame comes from our own backend.
Only a recording made by the real game can settle that, and it has to be paired
with the finish frame the real game recorded for it --
`tools/extract_records.py` pulls both out of a locally played run.
"""

from __future__ import annotations

import zlib

from polytrack.actions import DOWN, LEFT, RIGHT, UP
from polytrack.physics.wasm import RESET
from polytrack.recording import MAX_FRAMES, Recording, RecordingError, _b64url_encode

_SKIPPED: list[str] = []


def _sprint_track():
    """Start, a run of straights, finish. Or ``None`` if the game files are absent."""
    from polytrack.physics.assets import GeometryAssets, GeometryUnavailable
    from polytrack.trackcode import Part, TrackCode

    try:
        assets = GeometryAssets.load()
    except GeometryUnavailable as exc:
        _SKIPPED.append(str(exc).splitlines()[0])
        return None, None
    ids = {part.name: part.id for part in assets.parts}
    parts = [Part(0, 0, 0, ids["Start"], 0, 0, 0, None, 0)]
    parts += [Part(0, 0, -z, ids["Straight"], 0, 0, 0, None, None) for z in range(1, 14)]
    parts += [Part(0, 0, -14, ids["Finish"], 0, 0, 0, None, None)]
    return TrackCode(parts=tuple(parts), name="Sprint"), assets


# ------------------------------------------------------------------- codec --


def test_round_trips_through_the_wire_format():
    recording = Recording(
        up=(0, 1500, 1600, 40000),
        right=(220, 260),
        down=(),
        left=(90, 130, 5000),
        reset=(),
    )
    assert Recording.deserialize(recording.serialize()) == recording


def test_an_empty_recording_is_still_valid():
    """Five empty sequences: fifteen zero bytes. The client writes this."""
    empty = Recording()
    assert Recording.deserialize(empty.serialize()) == empty
    assert empty.last_frame == 0
    assert empty.mask_at(0) == 0


def test_inputs_are_held_on_odd_parity():
    """Toggles, not presses: a frame in a sequence flips that input's state."""
    recording = Recording(up=(10, 20), right=(15,))
    assert recording.mask_at(0) == 0
    assert recording.mask_at(9) == 0
    assert recording.mask_at(10) == UP  # the toggle applies on its own frame
    assert recording.mask_at(14) == UP
    assert recording.mask_at(15) == UP | RIGHT
    assert recording.mask_at(19) == UP | RIGHT
    assert recording.mask_at(20) == RIGHT  # up toggles back off
    assert recording.mask_at(10_000) == RIGHT  # and stays off


def test_the_walking_and_bisecting_readers_agree():
    """``masks`` walks cursors, ``mask_at`` bisects. They must not diverge."""
    recording = Recording(
        up=(0, 300), right=(5, 6, 7, 400), down=(1,), left=(), reset=(299,)
    )
    walked = list(recording.masks(500))
    assert walked == [recording.mask_at(frame) for frame in range(500)]


def test_runs_cover_every_frame_exactly_once():
    """The run decomposition is what actually gets replayed; gaps lose inputs."""
    recording = Recording(up=(0, 300), right=(5, 400), down=(1,), left=(250,))
    expanded = []
    for mask, ticks in recording.runs(500):
        expanded += [mask] * ticks
    assert len(expanded) == 500
    assert expanded == list(recording.masks(500))


def test_from_masks_inverts_masks():
    original = Recording(up=(0, 1500), right=(220, 260), left=(90, 130), reset=(700,))
    assert Recording.from_masks(original.masks(2000)) == original


def test_every_input_including_reset_survives_a_round_trip():
    """Reset is not in the action space but is in the format, and must not be lost."""
    masks = [0, UP, UP | RIGHT, UP | DOWN | LEFT, RESET, RESET | UP, 0]
    recording = Recording.from_masks(masks)
    assert [recording.mask_at(f) for f in range(len(masks))] == masks
    assert Recording.deserialize(recording.serialize()) == recording


def test_malformed_recordings_raise_rather_than_decoding_partially():
    """A half-decoded recording replays as a *different run*.

    That would fail the gate for the wrong reason and send someone hunting
    through the physics for a bug in the parser.
    """
    for bad in ("", "!!!!", "not base64url at all!"):
        try:
            Recording.deserialize(bad)
        except RecordingError:
            continue
        raise AssertionError(f"expected {bad!r} to be rejected")


def test_a_truncated_sequence_is_rejected():
    body = bytearray()
    body += (2).to_bytes(3, "little")  # up: claims two toggles
    body += (5).to_bytes(3, "little")  # ...but supplies one
    truncated = _b64url_encode(zlib.compress(bytes(body), 9))
    try:
        Recording.deserialize(truncated)
    except RecordingError as exc:
        assert "up" in str(exc)
        return
    raise AssertionError("a truncated sequence decoded without complaint")


def test_the_pre_060_ghost_format_is_rejected_not_misread():
    """The community's TAS ghosts use the older encoding, and must not parse.

    That format is a flat list of u24 frames followed by packed control
    nibbles. Read as five length-prefixed sequences it yields absurd counts,
    and the point of this test is that we notice rather than silently
    replaying nonsense. It is also a version marker: those ghosts predate the
    v0.6.0 physics rewrite, so they are not valid oracles for 0.6.2 anyway.
    """
    frames = [0, 633, 666, 1766, 1833]
    body = bytearray()
    for frame in frames:
        body += frame.to_bytes(3, "little")
    body += bytes((len(frames) + 1) // 2)  # packed control nibbles
    legacy = _b64url_encode(zlib.compress(bytes(body), 9))
    try:
        Recording.deserialize(legacy)
    except RecordingError:
        return
    raise AssertionError("a pre-0.6.0 ghost was accepted as a 0.6.2 recording")


def test_frames_past_the_cap_are_rejected():
    try:
        Recording.from_masks([UP] * 2)  # fine
    except RecordingError:
        raise AssertionError("a short recording was rejected")
    recording = Recording(up=(MAX_FRAMES + 1,))
    try:
        Recording.deserialize(recording.serialize())
    except RecordingError:
        return
    raise AssertionError("a toggle past maxFrames was accepted")


# -------------------------------------------------------------------- gate --


def test_the_gate_accepts_the_exact_frame_and_nothing_else():
    """One millisecond either way must fail.

    This is the property the whole gate rests on. A simulator that is 0.01% off
    finishes a 40-second run four frames early, so anything that tolerates
    "close" would pass a backend that is quietly wrong.
    """
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.verify import verify

    track, assets = _sprint_track()
    if track is None:
        return

    # Establish the true finish frame by driving it once.
    backend = WasmPhysics(track, assets=assets)
    result = backend.step_many(UP, 8000)
    assert result.finished, "the sprint track should be completable on throttle"
    target = backend.frame
    assert result.info["record"].finish_frames == target

    recording = Recording(up=(0,))
    for offset, expected in ((0, True), (-1, False), (+1, False)):
        fresh = WasmPhysics(track, assets=assets)
        outcome = verify(fresh, recording, target + offset)
        assert outcome.passed is expected, (
            f"target {target + offset:+} relative to {target}: "
            f"{outcome.describe()}"
        )
        if offset != 0:
            assert outcome.drift == -offset or not outcome.finished


def test_the_gate_reports_a_run_that_never_finishes():
    """Rather than hanging, or claiming success because nothing went wrong."""
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.verify import verify

    track, assets = _sprint_track()
    if track is None:
        return

    backend = WasmPhysics(track, assets=assets)
    coasting = Recording()  # no inputs at all; the car never moves
    outcome = verify(backend, coasting, 3000)
    assert not outcome.passed
    assert not outcome.finished
    assert outcome.frames == 3000
    assert "did not finish" in outcome.describe()


def test_replaying_a_recording_matches_driving_it_directly():
    """The run decomposition must not change the trajectory it produces."""
    from polytrack.physics.wasm import WasmPhysics

    track, assets = _sprint_track()
    if track is None:
        return

    masks = [UP] * 400 + [UP | RIGHT] * 200 + [UP] * 400
    recording = Recording.from_masks(masks)

    direct = WasmPhysics(track, assets=assets)
    for mask in masks:
        direct.step(mask)

    replayed = WasmPhysics(track, assets=assets)
    for mask, ticks in recording.runs(len(masks)):
        replayed.step_many(mask, ticks)

    assert replayed.frame == direct.frame == len(masks)
    assert replayed.record == direct.record


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
        print(f"\ngate tests skipped: {_SKIPPED[0]}")
    sys.exit(1 if failures else 0)
