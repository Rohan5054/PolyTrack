"""Read and write .tas input files.

Format (from beatrixwashere/polytracktas)::

    0,w
    143,wd
    287,w
    350,

Each line is ``frame,keys``: at that physics frame the *held key set becomes*
``keys``. It is a change-list, not a per-frame dump -- frames between two
entries inherit the earlier entry's key set. An empty key field means all keys
released.

This is the project's interchange format on purpose. It is what the existing
PolyTrack TAS community already uses, it round-trips through their
``tas_to_ghost.py`` encoder into a ghost the real game can replay, and it is
plain text so a desync can be diffed by eye. Every component here -- the
environment, the search layer, the verifier -- speaks this format.

Frames are physics frames, matching the game's own recording format -- which
means **milliseconds**. PolyTrack steps its physics at 1000 Hz, not 60; see
``polytrack.physics.wasm`` for the evidence. A ``.tas`` line per frame is
therefore a line per millisecond, which is why the format is written and read
as runs of held inputs rather than one row per tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .actions import keys_to_mask, mask_to_keys

#: The game's fixed physics rate. Everything in this project is frame-indexed
#: rather than time-indexed; wall-clock never enters the simulation.
PHYSICS_HZ = 60

#: Upper bound on run length taken from the game client (Lv.maxFrames).
MAX_FRAMES = 5_999_999


@dataclass(frozen=True, slots=True)
class InputChange:
    """A single 'held keys become X at frame N' event."""

    frame: int
    mask: int

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError(f"frame must be non-negative, got {self.frame}")


class TAS:
    """An ordered list of input changes, plus expansion to per-frame masks.

    Carries an explicit ``n_frames`` run length. The bare change-list format
    cannot express this on its own: the last entry says when inputs last
    *changed*, which is not when the run *ended* -- a run whose final action is
    "hold throttle to the finish" has its last change hundreds of frames before
    it stops. Without a length, replaying an exported TAS silently continues
    past the original episode. It is written to file as a ``# frames:`` comment,
    so files stay compatible with tools that ignore comments.
    """

    def __init__(
        self, changes: Iterable[InputChange] = (), n_frames: int | None = None
    ) -> None:
        self._changes: list[InputChange] = sorted(changes, key=lambda c: c.frame)
        self._validate()
        implied = self._changes[-1].frame + 1 if self._changes else 0
        if n_frames is not None and n_frames < implied:
            raise ValueError(
                f"n_frames={n_frames} is shorter than the last input change "
                f"at frame {implied - 1}"
            )
        self._n_frames = implied if n_frames is None else n_frames

    def _validate(self) -> None:
        seen: set[int] = set()
        for change in self._changes:
            if change.frame in seen:
                raise ValueError(
                    f"duplicate entry for frame {change.frame}; a frame can only "
                    "have one resulting key state"
                )
            seen.add(change.frame)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_frame_masks(cls, masks: Sequence[int]) -> "TAS":
        """Build from a dense per-frame mask sequence, collapsing runs.

        This is how an episode rollout becomes a TAS file: the environment
        records one mask per frame, and we keep only the frames where the held
        key set actually changed.
        """
        changes: list[InputChange] = []
        previous: int | None = None
        for frame, mask in enumerate(masks):
            if mask != previous:
                changes.append(InputChange(frame, mask))
                previous = mask
        return cls(changes, n_frames=len(masks))

    @classmethod
    def parse(cls, text: str) -> "TAS":
        """Parse .tas text. Blank lines and ``#`` comments are ignored."""
        changes: list[InputChange] = []
        n_frames: int | None = None
        for lineno, raw in enumerate(text.splitlines(), start=1):
            body, _, comment = raw.partition("#")
            if comment.strip().startswith("frames:"):
                n_frames = int(comment.split(":", 1)[1].strip())
            line = body.strip()
            if not line:
                continue
            frame_str, _, keys = line.partition(",")
            try:
                frame = int(frame_str.strip())
            except ValueError:
                raise ValueError(
                    f"line {lineno}: expected 'frame,keys', got {raw!r}"
                ) from None
            try:
                mask = keys_to_mask(keys)
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {exc}") from None
            changes.append(InputChange(frame, mask))
        return cls(changes, n_frames=n_frames)

    @classmethod
    def load(cls, path: str | Path) -> "TAS":
        return cls.parse(Path(path).read_text(encoding="utf-8"))

    # -- output ----------------------------------------------------------

    def to_text(self) -> str:
        lines = [f"# frames: {self._n_frames}"]
        lines += [f"{c.frame},{mask_to_keys(c.mask)}" for c in self._changes]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_text(), encoding="utf-8")

    # -- access ----------------------------------------------------------

    @property
    def changes(self) -> tuple[InputChange, ...]:
        return tuple(self._changes)

    @property
    def n_frames(self) -> int:
        """Total run length in physics frames."""
        return self._n_frames

    @property
    def duration(self) -> float:
        """Run length in seconds."""
        return frames_to_seconds(self._n_frames)

    @property
    def last_frame(self) -> int:
        """Frame of the final input change (not the end of the run)."""
        return self._changes[-1].frame if self._changes else 0

    def mask_at(self, frame: int) -> int:
        """Held key mask at ``frame``. Linear scan; use :meth:`expand` in loops."""
        mask = 0
        for change in self._changes:
            if change.frame > frame:
                break
            mask = change.mask
        return mask

    def expand(self, n_frames: int | None = None) -> Iterator[int]:
        """Yield the held mask for frames ``0..n_frames-1``.

        Defaults to the recorded run length, so ``expand()`` reproduces exactly
        the episode that was captured.

        This is the hot path when replaying a TAS through the simulator, so it
        walks the change list once instead of re-scanning per frame.
        """
        n_frames = self._n_frames if n_frames is None else n_frames
        if n_frames > MAX_FRAMES:
            raise ValueError(f"{n_frames} frames exceeds game limit {MAX_FRAMES}")
        idx = 0
        mask = 0
        for frame in range(n_frames):
            while idx < len(self._changes) and self._changes[idx].frame == frame:
                mask = self._changes[idx].mask
                idx += 1
            yield mask

    def __len__(self) -> int:
        return len(self._changes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TAS):
            return NotImplemented
        return self._changes == other._changes and self._n_frames == other._n_frames

    def __repr__(self) -> str:
        return f"TAS({len(self._changes)} changes, {self._n_frames} frames)"


def frames_to_seconds(frames: int) -> float:
    """Convert a frame count to the wall-clock time the game would display."""
    return frames / PHYSICS_HZ


def format_time(frames: int) -> str:
    """Format a frame count as the game does, ``M:SS.mmm``."""
    total = frames_to_seconds(frames)
    minutes, seconds = divmod(total, 60)
    return f"{int(minutes)}:{seconds:06.3f}"
