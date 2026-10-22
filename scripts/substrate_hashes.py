#!/usr/bin/env python3
"""Hash the four files every result in this repo depends on.

    python scripts/substrate_hashes.py           # a table
    python scripts/substrate_hashes.py --json    # the same, machine-readable

For polytrack_physics.wasm, geometry.npz, tracks.npz and records.json this
prints the SHA-256 of the file. For the two .npz files it also prints a
*content digest*. An .npz is a zip archive, and two archives holding identical
arrays can differ byte for byte -- member timestamps, compression, member order
-- depending on the numpy version and platform that wrote them. The content
digest ignores the packing: it is one SHA-256 over every array, in sorted key
order, each fed in as

    name NUL dtype NUL shape NUL nbytes NUL raw-bytes

where dtype is numpy's ``dtype.str`` (byte order included, e.g. ``<f4``), shape
is the dimensions joined by commas, nbytes is the length of raw-bytes in
decimal, and raw-bytes is the array in C order. Two .npz files with the same
content digest hold the same arrays, whatever wrote them.

Run it on each machine that produced results and compare the output. Matching
file hashes mean the same files. Differing file hashes with matching content
digests mean the same data, packed differently. `scripts/verify_baselines.py`
checks against the values frozen on 21 October 2026.

The wasm and geometry files are found exactly as the physics backend finds
them -- POLYTRACK_WASM / POLYTRACK_GEOMETRY first, then the repo root, then the
working directory -- by calling the backend's own resolvers. tracks.npz and
records.json are read from the repo root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The order every report uses.
NAMES = ("polytrack_physics.wasm", "geometry.npz", "tracks.npz", "records.json")


def locate(name: str) -> Path | None:
    """Where the code would read ``name`` from, or None if it is absent."""
    if name == "polytrack_physics.wasm":
        from polytrack.physics.wasm import WasmUnavailable, _resolve_wasm

        try:
            return _resolve_wasm(None)
        except WasmUnavailable:
            return None
    if name == "geometry.npz":
        from polytrack.physics.assets import GeometryUnavailable, _resolve

        try:
            return _resolve(None)
        except GeometryUnavailable:
            return None
    path = ROOT / name
    return path if path.exists() else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(path: Path) -> str:
    """SHA-256 of an .npz's arrays, independent of how the archive was packed."""
    import numpy as np

    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as archive:
        for name in sorted(archive.files):
            array = archive[name]
            raw = np.ascontiguousarray(array).tobytes()
            shape = ",".join(str(n) for n in array.shape)
            digest.update(
                f"{name}\0{array.dtype.str}\0{shape}\0{len(raw)}\0".encode("utf-8")
            )
            digest.update(raw)
    return digest.hexdigest()


def substrate() -> dict[str, dict | None]:
    """{name: {"path", "sha256", "content_sha256" for .npz}}, None where absent."""
    report: dict[str, dict | None] = {}
    for name in NAMES:
        path = locate(name)
        if path is None:
            report[name] = None
            continue
        entry = {"path": str(path), "sha256": file_sha256(path)}
        if name.endswith(".npz"):
            entry["content_sha256"] = content_sha256(path)
        report[name] = entry
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="print JSON instead of a table")
    args = parser.parse_args(argv)

    report = substrate()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, entry in report.items():
            if entry is None:
                print(f"\n  {name}: MISSING")
                continue
            print(f"\n  {name}  ({entry['path']})")
            print(f"    sha256          {entry['sha256']}")
            if "content_sha256" in entry:
                print(f"    content sha256  {entry['content_sha256']}")
        print()
    return 0 if all(report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
