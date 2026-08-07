"""Decoder for PolyTrack's shareable track codes.

A track code is what the physics module actually needs: ``createCarModel``
takes a packed list of placed parts, and every part's grid position, rotation
and checkpoint order comes from here. That makes this a Stage 0 dependency
rather than the later "nice to have exact geometry" step it looks like.

The format, as implemented by ``Track.fromSaveString`` in the 0.6.2 client:

``PolyTrack2`` + symbols( deflate( symbols( deflate( header + parts ) ) ) )

Two rounds, because the inner stage produces text in a 62-symbol alphabet and
that text compresses again usefully. ``symbols`` is *not* base62: it is a
variable-width bitstream that emits six bits per character normally and five
when the low bits would overflow the alphabet -- see :func:`decode_symbols`.

Bare save strings (no ``PolyTrack2`` prefix, no name/author header) use the
same pipeline and are also accepted. Five older formats exist and are not
supported; every current track, official or community, is ``PolyTrack2``.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789"
)
_REVERSE = {char: index for index, char in enumerate(_ALPHABET)}

#: Symbols whose low bits collide with the alphabet's 62-value limit are
#: emitted five bits wide instead of six. 0b11110.
_ESCAPE = 30

#: Grid spacing in metres. Part coordinates are integer grid cells.
PART_SIZE = 5

PREFIX = "PolyTrack2"


class Environment(IntEnum):
    SUMMER = 0
    WINTER = 1
    DESERT = 2


class RotationAxis(IntEnum):
    Y_POSITIVE = 0
    Y_NEGATIVE = 1
    X_POSITIVE = 2
    X_NEGATIVE = 3
    Z_POSITIVE = 4
    Z_NEGATIVE = 5


#: ``ROTATION_QUATERNIONS[axis][rotation]`` as (x, y, z, w). Straight from the
#: client's lookup table rather than derived, so the values match bit for bit.
_S = 0.7071067811865475
_T = 0.7071067811865476
_U = 0.7071067811865477
ROTATION_QUATERNIONS: tuple[tuple[tuple[float, float, float, float], ...], ...] = (
    ((0, 0, 0, 1), (0, _S, 0, _T), (0, 1, 0, 0), (0, _T, 0, -_S)),
    ((0, 0, 1, 0), (_S, 0, _T, 0), (1, 0, 0, 0), (_T, 0, -_S, 0)),
    ((0, 0, -_U, _S), (0.5, 0.5, -0.5, 0.5), (_S, _U, 0, 0), (0.5, 0.5, 0.5, -0.5)),
    ((0, 0, _S, _T), (0.5, -0.5, 0.5, 0.5), (_T, -_S, 0, 0), (0.5, -0.5, -0.5, -0.5)),
    ((_S, 0, 0, _T), (0.5, 0.5, 0.5, 0.5), (0, _T, _S, 0), (-0.5, 0.5, 0.5, -0.5)),
    ((-_U, 0, 0, _S), (-0.5, -0.5, 0.5, 0.5), (0, -_S, _U, 0), (0.5, -0.5, 0.5, -0.5)),
)


class TrackCodeError(ValueError):
    """A code could not be decoded. Raised rather than returning None.

    The client returns null and shows "Failed to load track". Here a bad code
    almost always means the decoder is wrong, not the code, so it should be
    loud.
    """


# --------------------------------------------------------------- bitstream --


def decode_symbols(text: str) -> bytes:
    """Decode the 62-symbol variable-width bitstream into bytes.

    Each character carries six bits, except that a symbol whose low bits match
    ``_ESCAPE`` carries only five -- which is how a 6-bit-per-symbol stream fits
    into a 62-character alphabet without needing padding or an escape symbol.
    """
    out = bytearray()
    offset = 0
    last = len(text) - 1
    for index, char in enumerate(text):
        value = _REVERSE.get(char)
        if value is None:
            raise TrackCodeError(f"invalid character {char!r} at position {index}")
        width = 5 if (value & _ESCAPE) == _ESCAPE else 6
        _write_bits(out, offset, width, value, index == last)
        offset += width
    return bytes(out)


def encode_symbols(data: bytes) -> str:
    """Inverse of :func:`decode_symbols`. Used to prove round-trips in tests."""
    out: list[str] = []
    offset = 0
    total = 8 * len(data)
    while offset < total:
        value = _read_six(data, offset)
        if (value & _ESCAPE) == _ESCAPE:
            out.append(_ALPHABET[value & 31])
            offset += 5
        else:
            out.append(_ALPHABET[value])
            offset += 6
    return "".join(out)


def _read_six(data: bytes, offset: int) -> int:
    index = offset // 8
    shift = offset - 8 * index
    current = data[index]
    if shift <= 2 or index >= len(data) - 1:
        return (current & (63 << shift)) >> shift
    return ((current & (63 << shift)) >> shift) | (
        (data[index + 1] & (63 >> (8 - shift))) << (8 - shift)
    )


def _write_bits(
    out: bytearray, offset: int, width: int, value: int, is_last: bool
) -> None:
    index = offset // 8
    while index >= len(out):
        out.append(0)
    shift = offset - 8 * index
    out[index] |= (value << shift) & 255
    if shift > 8 - width and not is_last:
        if index + 1 >= len(out):
            out.append(0)
        out[index + 1] |= value >> (8 - shift)


# ------------------------------------------------------------------ record --


@dataclass(frozen=True, slots=True)
class Part:
    """One placed track piece."""

    x: int
    y: int
    z: int
    part_id: int
    rotation: int  # 0-3, quarter turns
    rotation_axis: int  # :class:`RotationAxis`
    color: int
    #: Order along the route, for parts that are checkpoints. ``None`` otherwise.
    checkpoint_order: int | None = None
    #: Order among starting points, for start parts. ``None`` otherwise.
    start_order: int | None = None

    @property
    def quaternion(self) -> tuple[float, float, float, float]:
        """Placement rotation as (x, y, z, w)."""
        return ROTATION_QUATERNIONS[self.rotation_axis][self.rotation]


@dataclass(frozen=True, slots=True)
class TrackCode:
    """A decoded track: metadata plus every placed part."""

    parts: tuple[Part, ...]
    environment: int = Environment.SUMMER
    sun_direction: int = 0
    name: str | None = None
    author: str | None = None
    #: Unix timestamp from the code's header, if it carried one.
    last_modified: int | None = None

    def __len__(self) -> int:
        return len(self.parts)

    @property
    def checkpoints(self) -> tuple[Part, ...]:
        """Checkpoint parts in route order."""
        return tuple(
            sorted(
                (p for p in self.parts if p.checkpoint_order is not None),
                key=lambda p: p.checkpoint_order,
            )
        )

    @property
    def starts(self) -> tuple[Part, ...]:
        """Starting-point parts, in start order."""
        return tuple(
            sorted(
                (p for p in self.parts if p.start_order is not None),
                key=lambda p: p.start_order,
            )
        )

    def physics_buffer(self) -> bytes:
        """Pack the parts exactly as ``createCarModel`` expects them.

        Nineteen bytes per part, little-endian::

            u8 partId  i32 x  i32 y  i32 z  u8 rotation  u8 rotationAxis
            i32 checkpointOrder   (-1 when the part is not a checkpoint)

        Order matters: the client walks its parts in storage order and the
        checkpoint indices reported back by the simulation refer to positions
        in this buffer.
        """
        out = bytearray()
        for part in self.parts:
            out.append(part.part_id & 0xFF)
            out += part.x.to_bytes(4, "little", signed=True)
            out += part.y.to_bytes(4, "little", signed=True)
            out += part.z.to_bytes(4, "little", signed=True)
            out.append(part.rotation & 0xFF)
            out.append(part.rotation_axis & 0xFF)
            order = -1 if part.checkpoint_order is None else part.checkpoint_order
            out += order.to_bytes(4, "little", signed=True)
        return bytes(out)


# ----------------------------------------------------------------- decoding --

#: Part ids that carry a 16-bit checkpoint order, and those that carry a 32-bit
#: start order. The client derives both from the part table -- checkpoints are
#: the parts with a Checkpoint detector, starts are the parts with a start
#: offset -- so these are recomputed by ``tools/extract_geometry.mjs`` and can
#: be cross-checked against ``geometry.npz`` rather than trusted blindly.
#: Checkpoint, CheckpointWide, PlaneCheckpoint, PlaneCheckpointWide.
CHECKPOINT_PART_IDS: frozenset[int] = frozenset({52, 65, 75, 77})
#: Start, StartWide, PlaneStart, PlaneStartWide.
START_PART_IDS: frozenset[int] = frozenset({5, 91, 92, 93})
#: Finish, FinishWide, PlaneFinish, PlaneFinishWide. These carry no extra
#: field -- listed because "which part ends the track" is not otherwise
#: obvious from the code, and the finish detector is what stops the clock.
FINISH_PART_IDS: frozenset[int] = frozenset({6, 74, 76, 78})


class _Reader:
    """Bounds-checked little-endian cursor. Every read can fail; none may
    silently run off the end, because a truncated code that decodes to
    plausible-looking garbage is far worse than one that raises."""

    __slots__ = ("data", "offset")

    def __init__(self, data: bytes, offset: int = 0) -> None:
        self.data = data
        self.offset = offset

    def __len__(self) -> int:
        return len(self.data) - self.offset

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self.take(2), "little")

    def i32(self) -> int:
        return int.from_bytes(self.take(4), "little", signed=True)

    def uint(self, width: int) -> int:
        return int.from_bytes(self.take(width), "little")

    def take(self, count: int) -> bytes:
        if len(self) < count:
            raise TrackCodeError(
                f"truncated at byte {self.offset}: wanted {count}, "
                f"{len(self)} remain"
            )
        chunk = self.data[self.offset : self.offset + count]
        self.offset += count
        return chunk


def _inflate_stages(body: str) -> bytes:
    """Undo the two compression rounds, returning the raw record."""
    try:
        inner = zlib.decompress(decode_symbols(body)).decode("utf-8")
    except (zlib.error, UnicodeDecodeError) as exc:
        raise TrackCodeError(f"outer stage failed: {exc}") from None
    try:
        return zlib.decompress(decode_symbols(inner))
    except zlib.error as exc:
        raise TrackCodeError(f"inner stage failed: {exc}") from None


def _read_header(reader: _Reader) -> tuple[str | None, str | None, int | None]:
    """Read the name/author/timestamp block that export strings carry."""
    name = reader.take(reader.u8()).decode("utf-8")
    author_length = reader.u8()
    author = reader.take(author_length).decode("utf-8") if author_length else None
    flag = reader.u8()
    if flag == 0:
        return name, author, None
    if flag != 1:
        raise TrackCodeError(f"bad timestamp flag {flag}")
    return name, author, reader.i32() & 0xFFFFFFFF


def _read_parts(reader: _Reader) -> Iterator[Part]:
    """Read the part table.

    Coordinates are delta-encoded against a per-track origin and stored in the
    narrowest width that fits, which is why a 3,000-part track still fits in a
    few kilobytes of code.
    """
    origin_x, origin_y, origin_z = reader.i32(), reader.i32(), reader.i32()

    widths = reader.u8()
    width_x = widths & 3
    width_y = (widths >> 2) & 3
    width_z = (widths >> 4) & 3
    if not (1 <= width_x <= 3 and 1 <= width_y <= 3 and 1 <= width_z <= 3):
        raise TrackCodeError(f"bad coordinate widths byte 0x{widths:02x}")

    while len(reader):
        part_id = reader.u8()
        count = reader.i32()
        if count < 0:
            raise TrackCodeError(f"negative run length {count} for part {part_id}")
        for _ in range(count):
            x = reader.uint(width_x) + origin_x
            y = reader.uint(width_y) + origin_y
            z = reader.uint(width_z) + origin_z
            packed = reader.u8()
            rotation = packed & 3
            rotation_axis = (packed >> 2) & 7
            if rotation_axis > RotationAxis.Z_NEGATIVE:
                raise TrackCodeError(f"bad rotation axis {rotation_axis}")
            color = reader.u8()
            checkpoint_order = (
                reader.u16() if part_id in CHECKPOINT_PART_IDS else None
            )
            start_order = reader.i32() if part_id in START_PART_IDS else None
            yield Part(
                x, y, z, part_id, rotation, rotation_axis, color,
                checkpoint_order, start_order,
            )


def decode(text: str) -> TrackCode:
    """Decode a ``PolyTrack2`` export string or a bare save string.

    Raises:
        TrackCodeError: on anything malformed.
    """
    stripped = "".join(text.split())
    has_header = stripped.startswith(PREFIX)
    body = stripped[len(PREFIX):] if has_header else stripped
    if not body:
        raise TrackCodeError("empty track code")

    reader = _Reader(_inflate_stages(body))
    name = author = None
    last_modified = None
    if has_header:
        name, author, last_modified = _read_header(reader)

    environment = reader.u8()
    if environment not in tuple(Environment):
        raise TrackCodeError(f"unknown environment {environment}")
    sun_direction = reader.u8()
    if not 0 <= sun_direction < 180:
        raise TrackCodeError(f"sun direction {sun_direction} out of range")

    return TrackCode(
        parts=tuple(_read_parts(reader)),
        environment=environment,
        sun_direction=sun_direction,
        name=name,
        author=author,
        last_modified=last_modified,
    )
