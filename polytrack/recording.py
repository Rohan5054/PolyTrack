"""PolyTrack's input recording format -- ghosts, replays, leaderboard runs.

This is what the acceptance gate needs. A recording is the input side of a run:
no positions, no times, just when each key went down and up. Replaying one
through the physics module and landing on the same finish frame is the only
evidence that this project's simulator is the game's simulator.

The format, from ``CarRecording`` in the 0.6.2 client:

    base64url( deflate( five sequences of toggle frames ) )

Each of the five inputs -- up, right, down, left, reset -- gets its own
sequence, written in that order as::

    u24 count,  then count x u24 delta   (first entry absolute, rest deltas)

A frame appears in a sequence every time that input *changes*, so the input is
held whenever an odd number of its toggles have happened at or before the
current frame. That is the whole encoding: a run holding throttle for thirty
seconds costs three bytes.

Two things worth stating plainly, because both are easy to get wrong:

* **Frames are milliseconds.** PolyTrack steps physics at 1000 Hz, so a
  40-second run spans 40,000 frames. See :mod:`polytrack.physics.wasm`.
* **A recording carries no length.** It ends when the last key is released,
  which is not when the run ends. The finish frame comes from the leaderboard
  entry alongside it, and the gate needs both.
"""

from __future__ import annotations

import base64
import zlib
from bisect import bisect_right
from dataclasses import dataclass, field

from .actions import DOWN, LEFT, RIGHT, UP
from .physics.wasm import RESET

#: The client's cap, and the reason we know frames are milliseconds: it is
#: 99:59.999 expressed in them.
MAX_FRAMES = 5_999_999

#: Sequence order in the serialised stream. Also the bit order of the masks
#: this module produces.
_INPUTS = (("up", UP), ("right", RIGHT), ("down", DOWN), ("left", LEFT),
           ("reset", RESET))


class RecordingError(ValueError):
    """A recording could not be decoded, or is inconsistent."""


@dataclass(frozen=True, slots=True)
class Recording:
    """Five ascending sequences of frames at which an input toggles."""

    up: tuple[int, ...] = ()
    right: tuple[int, ...] = ()
    down: tuple[int, ...] = ()
    left: tuple[int, ...] = ()
    reset: tuple[int, ...] = ()

    @property
    def sequences(self) -> tuple[tuple[int, ...], ...]:
        return (self.up, self.right, self.down, self.left, self.reset)

    @property
    def last_frame(self) -> int:
        """The last frame at which anything changed. **Not the run's length.**"""
        return max((seq[-1] for seq in self.sequences if seq), default=0)

    def mask_at(self, frame: int) -> int:
        """The input mask held at ``frame``.

        An input is held when an odd number of its toggles have occurred at or
        before this frame -- which is what the client's own ``getFrame`` says,
        via a linear scan for the last entry ``<= frame``. Bisection here, since
        replaying a run calls this once per millisecond.
        """
        mask = 0
        for (_, flag), sequence in zip(_INPUTS, self.sequences):
            if bisect_right(sequence, frame) & 1:
                mask |= flag
        return mask

    def masks(self, frames: int):
        """Yield the mask for frames ``0 .. frames - 1``.

        Walks the sequences once rather than bisecting per frame. Replaying a
        40-second run means 40,000 lookups, and this turns them into five
        cursors.
        """
        cursors = [0] * len(_INPUTS)
        held = [False] * len(_INPUTS)
        sequences = self.sequences
        for frame in range(frames):
            mask = 0
            for index, (_, flag) in enumerate(_INPUTS):
                sequence = sequences[index]
                cursor = cursors[index]
                while cursor < len(sequence) and sequence[cursor] <= frame:
                    held[index] = not held[index]
                    cursor += 1
                cursors[index] = cursor
                if held[index]:
                    mask |= flag
            yield mask

    def runs(self, frames: int):
        """The same masks, collapsed into ``(mask, ticks)`` runs.

        This is the shape the backend actually wants: inputs change a few
        hundred times in a run, not 40,000 times, so this turns a replay into a
        few hundred :meth:`~polytrack.physics.wasm.WasmPhysics.step_many` calls.
        """
        boundaries = sorted(
            {0, frames}
            | {f for sequence in self.sequences for f in sequence if 0 < f < frames}
        )
        for start, end in zip(boundaries, boundaries[1:]):
            yield self.mask_at(start), end - start

    def serialize(self) -> str:
        """Encode back to the client's string form."""
        body = bytearray()
        for sequence in self.sequences:
            body += _write_sequence(sequence)
        # Level 9 zlib, matching the client's pako Deflate({level: 9}).
        return _b64url_encode(zlib.compress(bytes(body), 9))

    @classmethod
    def deserialize(cls, text: str) -> Recording:
        """Decode the client's string form.

        Raises:
            RecordingError: on anything malformed. The client returns null and
                the caller reports a broken ghost; here a half-decoded
                recording would replay as a *different run* and quietly fail
                the gate for the wrong reason.
        """
        raw = _b64url_decode(text)
        try:
            body = zlib.decompress(raw)
        except zlib.error as exc:
            raise RecordingError(f"inflate failed: {exc}") from None

        sequences = []
        offset = 0
        for name, _ in _INPUTS:
            sequence, offset = _read_sequence(body, offset, name)
            sequences.append(sequence)
        if offset != len(body):
            raise RecordingError(
                f"{len(body) - offset} trailing bytes after the five sequences"
            )
        return cls(*sequences)

    @classmethod
    def from_masks(cls, masks) -> Recording:
        """Build a recording from a per-frame mask sequence.

        The inverse of :meth:`masks`, and how a search result becomes something
        the real game can replay.
        """
        sequences: list[list[int]] = [[] for _ in _INPUTS]
        held = [False] * len(_INPUTS)
        frame = -1
        for frame, mask in enumerate(masks):
            for index, (_, flag) in enumerate(_INPUTS):
                if bool(mask & flag) != held[index]:
                    held[index] = not held[index]
                    sequences[index].append(frame)
        if frame >= MAX_FRAMES:
            raise RecordingError(f"{frame + 1} frames exceeds the cap of {MAX_FRAMES}")
        return cls(*(tuple(s) for s in sequences))


def _write_sequence(sequence: tuple[int, ...]) -> bytes:
    out = bytearray(len(sequence).to_bytes(3, "little"))
    previous = 0
    for index, frame in enumerate(sequence):
        delta = frame if index == 0 else frame - previous
        out += (delta & 0xFFFFFF).to_bytes(3, "little")
        previous = frame
    return bytes(out)


def _read_sequence(body: bytes, offset: int, name: str) -> tuple[tuple[int, ...], int]:
    if len(body) - offset < 3:
        raise RecordingError(f"truncated before the {name} sequence header")
    count = int.from_bytes(body[offset : offset + 3], "little")
    offset += 3
    if len(body) - offset < 3 * count:
        raise RecordingError(
            f"{name} sequence claims {count} toggles but only "
            f"{(len(body) - offset) // 3} fit"
        )
    frames: list[int] = []
    current = 0
    for _ in range(count):
        current += int.from_bytes(body[offset : offset + 3], "little")
        offset += 3
        frames.append(current)
    if current > MAX_FRAMES:
        raise RecordingError(f"{name} toggles at frame {current}, past the cap")
    return tuple(frames), offset


def _b64url_encode(data: bytes) -> str:
    """The client's flavour: standard base64, +/ swapped for -_, padding cut."""
    return base64.b64encode(data).decode("ascii").translate(
        str.maketrans("+/", "-_")
    ).rstrip("=")


def _b64url_decode(text: str) -> bytes:
    cleaned = "".join(text.split()).translate(str.maketrans("-_", "+/"))
    try:
        return base64.b64decode(cleaned + "=" * (-len(cleaned) % 4))
    except Exception as exc:
        raise RecordingError(f"not valid base64url: {exc}") from None
