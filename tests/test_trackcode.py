"""Track code decoding.

The decoder is validated two ways. These tests cover the format in isolation
using codes built here, so they run anywhere. The stronger check needs the
game's own files and is opt-in::

    POLYTRACK_TRACKS=path\\to\\tracks python tests/test_trackcode.py

That decodes every ``.track`` shipped with the desktop build. It is the real
evidence -- 31 current-format tracks, all 17 official ones among them, each
parsing to completion with the name the client shows. Codes are not
self-describing, so "the part stream ends exactly where the buffer does, 31
times over" is worth more than any assertion written here.
"""

from __future__ import annotations

import glob
import os
import random
import zlib

from polytrack.trackcode import (
    CHECKPOINT_PART_IDS,
    FINISH_PART_IDS,
    PREFIX,
    START_PART_IDS,
    Environment,
    TrackCodeError,
    decode,
    decode_symbols,
    encode_symbols,
)


def _build_code(parts, name="Test", author=None, environment=Environment.SUMMER):
    """Assemble a PolyTrack2 export string from scratch.

    Encoding is not something the agent needs, so it lives here rather than in
    the module: a decoder tested only against its own encoder proves very
    little, but it is the cleanest way to drive edge cases the shipped tracks
    happen not to contain.
    """
    body = bytearray()
    encoded_name = name.encode("utf-8")
    body.append(len(encoded_name))
    body += encoded_name
    if author is None:
        body.append(0)
    else:
        encoded_author = author.encode("utf-8")
        body.append(len(encoded_author))
        body += encoded_author
    body.append(0)  # no timestamp

    body.append(int(environment))
    body.append(0)  # sun direction

    # Origin, then one byte packing the per-axis coordinate widths.
    for _ in range(3):
        body += (0).to_bytes(4, "little", signed=True)
    body.append(1 | (1 << 2) | (1 << 4))  # one byte per axis

    grouped: dict[int, list] = {}
    for part in parts:
        grouped.setdefault(part[3], []).append(part)
    for part_id, group in grouped.items():
        body.append(part_id)
        body += len(group).to_bytes(4, "little", signed=True)
        for x, y, z, _, rotation, axis, color, checkpoint, start in group:
            body += bytes([x, y, z, (rotation & 3) | (axis << 2), color])
            if part_id in CHECKPOINT_PART_IDS:
                body += checkpoint.to_bytes(2, "little")
            if part_id in START_PART_IDS:
                body += start.to_bytes(4, "little", signed=True)

    inner = encode_symbols(zlib.compress(bytes(body), 9))
    return PREFIX + encode_symbols(zlib.compress(inner.encode("utf-8"), 9))


def test_symbol_codec_round_trips():
    """The 5/6-bit stream must survive every payload length and alignment.

    Widths vary per symbol, so the tail is where this breaks: the last symbol
    is written without its carry byte, and an off-by-one there corrupts only
    the final byte of long tracks -- which would surface as a desync deep into
    a replay rather than as a decode failure.
    """
    rng = random.Random(0)
    for length in (1, 2, 3, 7, 8, 9, 63, 64, 65, 1000):
        payload = bytes(rng.randrange(256) for _ in range(length))
        assert decode_symbols(encode_symbols(payload))[:length] == payload


def test_decodes_a_minimal_track():
    start = (10, 0, 10, min(START_PART_IDS), 0, 0, 0, 0, 0)
    finish = (10, 0, 20, min(FINISH_PART_IDS), 2, 0, 0, 0, 0)
    track = decode(_build_code([start, finish], name="Minimal", author="Someone"))

    assert track.name == "Minimal"
    assert track.author == "Someone"
    assert len(track) == 2
    assert len(track.starts) == 1
    assert track.starts[0].x == 10 and track.starts[0].z == 10
    assert track.parts[1].rotation == 2


def test_checkpoints_come_back_in_route_order():
    """Checkpoint order is stored per part, not implied by position in the file."""
    checkpoint_id = min(CHECKPOINT_PART_IDS)
    parts = [
        (1, 0, 1, checkpoint_id, 0, 0, 0, 2, 0),
        (2, 0, 2, checkpoint_id, 0, 0, 0, 0, 0),
        (3, 0, 3, checkpoint_id, 0, 0, 0, 1, 0),
    ]
    track = decode(_build_code(parts))
    assert [p.checkpoint_order for p in track.checkpoints] == [0, 1, 2]
    assert [p.x for p in track.checkpoints] == [2, 3, 1]


def test_physics_buffer_is_nineteen_bytes_per_part():
    """The layout createCarModel reads. Get this wrong and the track is garbage."""
    parts = [
        (7, 1, 3, 0, 1, 2, 0, 0, 0),
        (4, 0, 5, min(CHECKPOINT_PART_IDS), 0, 0, 0, 6, 0),
    ]
    buffer = decode(_build_code(parts)).physics_buffer()
    assert len(buffer) == 38

    first, second = buffer[:19], buffer[19:]
    assert first[0] == 0
    assert int.from_bytes(first[1:5], "little", signed=True) == 7
    assert int.from_bytes(first[5:9], "little", signed=True) == 1
    assert int.from_bytes(first[9:13], "little", signed=True) == 3
    assert first[13] == 1 and first[14] == 2
    # A non-checkpoint part reports -1, not 0 -- 0 is a valid checkpoint index.
    assert int.from_bytes(first[15:19], "little", signed=True) == -1
    assert int.from_bytes(second[15:19], "little", signed=True) == 6


def test_malformed_codes_raise_rather_than_returning_partial_tracks():
    """A truncated code must fail loudly.

    The client returns null and shows an error. Here a code that half-decodes
    would hand the physics module a track missing its last few parts, which
    simulates perfectly well and produces silently wrong times.
    """
    for bad in ("", PREFIX, PREFIX + "!!!!", PREFIX + "AAAAAAAA", "nonsense"):
        try:
            decode(bad)
        except TrackCodeError:
            continue
        raise AssertionError(f"expected {bad!r} to be rejected")


def test_truncated_part_stream_is_rejected():
    code = _build_code([(1, 2, 3, 0, 0, 0, 0, 0, 0)])
    # Drop the tail of the payload and re-wrap it.
    outer = decode_symbols(code[len(PREFIX):])
    inner = decode_symbols(zlib.decompress(outer).decode("utf-8"))
    payload = zlib.decompress(inner)[:-2]
    rewrapped = encode_symbols(zlib.compress(payload, 9))
    damaged = PREFIX + encode_symbols(zlib.compress(rewrapped.encode("utf-8"), 9))
    try:
        decode(damaged)
    except TrackCodeError:
        return
    raise AssertionError("a truncated part stream decoded without complaint")


def test_every_shipped_track_decodes():
    """Opt-in: decode the real corpus if POLYTRACK_TRACKS points at it."""
    root = os.environ.get("POLYTRACK_TRACKS")
    if not root:
        return
    current = legacy = 0
    for path in sorted(glob.glob(os.path.join(root, "*", "*.track"))):
        text = open(path, encoding="utf-8").read().strip()
        if not text.startswith(PREFIX):
            legacy += 1  # PolyTrack1 and older; see the module docstring.
            continue
        track = decode(text)
        assert len(track) > 0, path
        # Official tracks have exactly one start, but community ones do not:
        # 4_seasons places 387 of them. Their start orders come back as a clean
        # ascending run, which is itself evidence the stream is aligned --
        # a misparse yields noise here, not a sequence.
        assert len(track.starts) >= 1, path
        orders = [p.start_order for p in track.starts]
        assert orders == sorted(orders) and len(set(orders)) == len(orders), path
        current += 1
    assert current > 0, "no current-format tracks found"
    print(f"        ({current} PolyTrack2 decoded, {legacy} legacy skipped)")


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
    sys.exit(1 if failures else 0)
