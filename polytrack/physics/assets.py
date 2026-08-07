"""The collision geometry the physics module has to be fed.

``initializeCarCollisionShape`` and ``addTrackPartConfiguration`` want raw
float32 vertex arrays, and those are the one input the game's JavaScript does
not spell out -- they come from Draco-compressed ``.glb`` models.
``tools/extract_geometry.mjs`` recovers them by replaying the game's own model
pipeline in real three.js and checking each result against the SHA-256 the game
carries for that part; ``tools/build_geometry.py`` packs the result into
``geometry.npz``. This module is the read side.

The file is derived from Kodub's models, so it is deliberately **not** in the
repo and not redistributable. Build it locally from a copy of the game you own.
Everything here fails with an explanatory error rather than a ``KeyError`` when
it is missing, because "the backend silently used no geometry" is a failure mode
that would only show up as an inexplicable desync.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Detector type as ``addTrackPartConfiguration`` numbers it. -1 means the part
#: has no detector, which is all but six of them.
DETECTOR_NONE = -1
DETECTOR_CHECKPOINT = 0
DETECTOR_FINISH = 1

_DETECTOR_TYPES = {"Checkpoint": DETECTOR_CHECKPOINT, "Finish": DETECTOR_FINISH}

#: Where to look when no path is given. The environment variable wins so the
#: file can live outside the working tree.
_ENV_VAR = "POLYTRACK_GEOMETRY"
_DEFAULT_NAME = "geometry.npz"


class GeometryUnavailable(RuntimeError):
    """``geometry.npz`` is missing or does not match the pinned game version."""


@dataclass(frozen=True, slots=True)
class PartAsset:
    """One track-part configuration, as the module wants to receive it."""

    id: int
    name: str
    #: Flat float32 xyz triples -- the game's ``physicsShapeVertices`` verbatim.
    vertices: np.ndarray
    detector_type: int
    detector_center: tuple[float, float, float]
    detector_size: tuple[float, float, float]
    #: Where the car sits on this part when it is the starting point, in the
    #: part's local frame. Only the four Start variants have one.
    start_offset: tuple[float, float, float] | None
    checksum: str

    @property
    def is_start(self) -> bool:
        return self.start_offset is not None


class GeometryAssets:
    """Everything ``geometry.npz`` holds, in registration order.

    Order matters. The game builds its part table once, in the order the
    configuration array in ``main.bundle.js`` declares, and registers every part
    -- not just the ones a given track uses. We do the same: a physics engine is
    entitled to key its broadphase off insertion order or allocation address,
    and matching the game costs nothing.
    """

    __slots__ = ("version", "mass_offset", "car_collision_vertices", "parts", "_by_id")

    def __init__(
        self,
        version: str,
        mass_offset: float,
        car_collision_vertices: np.ndarray,
        parts: tuple[PartAsset, ...],
    ) -> None:
        self.version = version
        self.mass_offset = mass_offset
        self.car_collision_vertices = car_collision_vertices
        self.parts = parts
        self._by_id = {part.id: part for part in parts}

    def __len__(self) -> int:
        return len(self.parts)

    def __getitem__(self, part_id: int) -> PartAsset:
        try:
            return self._by_id[part_id]
        except KeyError:
            raise GeometryUnavailable(
                f"track uses part id {part_id}, which is not in geometry.npz "
                f"({len(self.parts)} parts, version {self.version}). Either the "
                "file was built from a different game version or the track code "
                "decoded wrongly."
            ) from None

    def __contains__(self, part_id: int) -> bool:
        return part_id in self._by_id

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> GeometryAssets:
        """Load the pack, searching the usual places when no path is given."""
        resolved = _resolve(path)
        with np.load(resolved) as data:
            manifest = json.loads(bytes(data["manifest"]).decode("utf-8"))
            car = np.ascontiguousarray(data["car_collision_vertices"], dtype=np.float32)
            parts = tuple(
                _read_part(entry, data) for entry in manifest["parts"]
            )
        return cls(manifest["version"], float(manifest["mass_offset"]), car, parts)


def _resolve(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise GeometryUnavailable(f"no geometry pack at {candidate}")
        return candidate

    candidates: list[Path] = []
    from_env = os.environ.get(_ENV_VAR)
    if from_env:
        candidates.append(Path(from_env))
    # The repo root, two levels up from polytrack/physics/.
    candidates.append(Path(__file__).resolve().parents[2] / _DEFAULT_NAME)
    candidates.append(Path.cwd() / _DEFAULT_NAME)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise GeometryUnavailable(
        f"no {_DEFAULT_NAME} found (looked in "
        + ", ".join(str(c) for c in candidates)
        + "). It is not in the repo because it is derived from the game's "
        "models. Build it from a copy of the game you own:\n"
        "  node tools/extract_geometry.mjs main.bundle.js models/ geometry.json\n"
        "  python tools/build_geometry.py geometry.json geometry.npz"
    )


def _read_part(entry: dict, data) -> PartAsset:
    detector = entry.get("detector")
    if detector is None:
        detector_type = DETECTOR_NONE
        center = size = (0.0, 0.0, 0.0)
    else:
        # The extractor stubs the game's enum with a name proxy, so the type
        # arrives as a string and is mapped back here rather than there.
        detector_type = _DETECTOR_TYPES[detector["type"]]
        center = tuple(float(v) for v in detector["center"])
        size = tuple(float(v) for v in detector["size"])

    start = entry.get("start_offset")
    start_offset = (
        None if start is None else (float(start["x"]), float(start["y"]), float(start["z"]))
    )
    return PartAsset(
        id=int(entry["id"]),
        name=entry["name"],
        vertices=np.ascontiguousarray(data[entry["key"]], dtype=np.float32),
        detector_type=detector_type,
        detector_center=center,  # type: ignore[arg-type]
        detector_size=size,  # type: ignore[arg-type]
        start_offset=start_offset,
        checksum=entry["checksum"],
    )
