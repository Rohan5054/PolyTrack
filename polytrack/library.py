"""A set of tracks ready to train on, and a file to keep them in.

Building a reference line costs a full replay -- 46 seconds of wall clock for a
19-second run, because the physics module runs at roughly half realtime. That is
fine once. It is not fine once per collector process per launch, and a training
run wants six or eight collectors.

So build them once and write them to disk. A :class:`TrackLibrary` holds, per
track, everything a collector needs to stand up an environment: the track code to
rebuild the physics from, the reference line to measure progress along, and the
time the run that produced it took. Loading one is instant.

The file is a ``.npz`` with a JSON manifest inside, the same shape as
``geometry.npz``. **It embeds the game's track codes, so it is derived from
Kodub's content -- keep it local and do not commit it.**
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .physics.assets import GeometryAssets
from .physics.wasm import WasmPhysics
from .track import Track
from .trackcode import TrackCode, decode

#: Bumped whenever the on-disk layout changes, so a stale file fails loudly
#: rather than loading as something subtly different.
FORMAT_VERSION = 1


class LibraryError(RuntimeError):
    """A library file could not be read, or does not contain what was asked for."""


@dataclass(frozen=True, slots=True)
class TrackEntry:
    """One track: how to simulate it, and how to score progress on it."""

    #: The game's own export string. Kept rather than the decoded parts because
    #: it is the thing that round-trips, and because ``track_id`` is a hash of
    #: bytes that only survive a real decode.
    code: str
    line: Track
    #: Frames the run that produced the line took. A frame is a millisecond, so
    #: this is the human time to beat, and the natural scale for a finish bonus.
    par_frames: int

    @property
    def name(self) -> str:
        return self.line.name

    def track_code(self) -> TrackCode:
        return decode(self.code)

    def backend(self, assets: GeometryAssets | None = None) -> WasmPhysics:
        """A fresh physics backend on this track."""
        return WasmPhysics(self.track_code(), assets=assets)

    @property
    def par_speed(self) -> float:
        """Average speed of the reference run, m/s. A useful reward scale."""
        return self.line.length / (self.par_frames / 1000.0)


class TrackLibrary:
    """An ordered collection of :class:`TrackEntry`, saved and loaded as one file."""

    def __init__(self, entries: list[TrackEntry] | None = None) -> None:
        self.entries: list[TrackEntry] = list(entries or [])

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, index: int) -> TrackEntry:
        return self.entries[index]

    def select(self, names) -> "TrackLibrary":
        """A sub-library of the named tracks, in the order given.

        Training on one track rather than four gives each four times the
        episodes, which is the right trade while the question is still "can it
        get round at all". It is the wrong trade for generalisation, but four
        tracks does not buy that either -- only the procedural generator does.

        Matching is case-insensitive and ignores spaces, so ``winter5`` finds
        ``Winter 5``.
        """
        def key(text: str) -> str:
            return "".join(text.split()).lower()

        index = {key(entry.name): entry for entry in self.entries}
        chosen = []
        for name in names:
            entry = index.get(key(name))
            if entry is None:
                available = ", ".join(e.name for e in self.entries)
                raise LibraryError(f"no track named {name!r} ({available})")
            chosen.append(entry)
        return TrackLibrary(chosen)

    def by_name(self, name: str) -> TrackEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        available = ", ".join(e.name for e in self.entries) or "none"
        raise LibraryError(f"no track named {name!r} in this library ({available})")

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        manifest = {
            "format": FORMAT_VERSION,
            "tracks": [
                {
                    "name": entry.name,
                    "code": entry.code,
                    "par_frames": entry.par_frames,
                    "width": entry.line.width,
                    "checkpoint_indices": list(entry.line.checkpoint_indices),
                }
                for entry in self.entries
            ],
        }
        arrays = {
            f"line_{i}": np.ascontiguousarray(entry.line.centerline, dtype=np.float64)
            for i, entry in enumerate(self.entries)
        }
        np.savez_compressed(
            Path(path),
            manifest=np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrackLibrary":
        resolved = Path(path)
        if not resolved.exists():
            raise LibraryError(
                f"no track library at {resolved}. Build one with "
                "scripts/build_library.py -- it needs records.json and the "
                "game's .track files, and takes a full replay per track."
            )
        with np.load(resolved) as data:
            manifest = json.loads(bytes(data["manifest"]).decode("utf-8"))
            if manifest.get("format") != FORMAT_VERSION:
                raise LibraryError(
                    f"{resolved} is format {manifest.get('format')}, this build "
                    f"reads {FORMAT_VERSION}. Rebuild it."
                )
            entries = []
            for index, record in enumerate(manifest["tracks"]):
                line = Track(
                    data[f"line_{index}"],
                    width=record["width"],
                    checkpoint_indices=tuple(record["checkpoint_indices"]),
                    name=record["name"],
                )
                entries.append(
                    TrackEntry(
                        code=record["code"],
                        line=line,
                        par_frames=int(record["par_frames"]),
                    )
                )
        return cls(entries)

    def describe(self) -> str:
        rows = [f"{len(self.entries)} track(s)"]
        for entry in self.entries:
            rows.append(
                f"  {entry.name:<14} {entry.line.length:7.1f} m  "
                f"{entry.line.n_checkpoints} cp  "
                f"par {entry.par_frames / 1000:6.3f} s  "
                f"({entry.par_speed * 3.6:.0f} km/h avg)"
            )
        return "\n".join(rows)
