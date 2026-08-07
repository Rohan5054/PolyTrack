#!/usr/bin/env python3
"""Convert the extractor's JSON into the compact ``geometry.npz`` the backend loads.

``tools/extract_geometry.mjs`` writes JSON because it is inspectable and because
Node has no good ``.npz`` writer. That file is ~14 MB of decimal floats for
~3.2 MB of actual data, so it is not what should sit on the hot path. This
converts it once.

Usage::

    node tools/extract_geometry.mjs main.bundle.js models/ geometry.json
    python tools/build_geometry.py geometry.json geometry.npz

The result holds, per track part, the exact ``physicsShapeVertices`` float32
array the game feeds to ``addTrackPartConfiguration`` -- byte-identical to what
the browser build computes, verified against the SHA-256 checksums the game
carries for that purpose. Keep it local: it is derived from Kodub's models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="JSON from extract_geometry.mjs")
    parser.add_argument("out", type=Path, help="destination .npz")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"no such file: {args.source}")
        return 1

    data = json.loads(args.source.read_text())

    arrays: dict[str, np.ndarray] = {
        "car_collision_vertices": np.asarray(
            data["carCollisionShapeVertices"], dtype=np.float32
        )
    }
    # Metadata travels as JSON rather than as arrays: detectors and start
    # offsets are sparse and irregularly shaped, and keeping them structured
    # beats flattening them into padded arrays nobody can read.
    manifest = {
        "version": data["version"],
        "mass_offset": data["massOffset"],
        "parts": [],
    }
    for part in data["parts"]:
        key = f"part_{part['id']}"
        arrays[key] = np.asarray(part["vertices"], dtype=np.float32)
        manifest["parts"].append(
            {
                "id": part["id"],
                "name": part["name"],
                "key": key,
                "n_vertices": len(part["vertices"]) // 3,
                "detector": part["detector"],
                "start_offset": part["startOffset"],
                "checksum": part["checksum"],
            }
        )

    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest).encode("utf-8"), dtype=np.uint8
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    total = sum(a.size for k, a in arrays.items() if k != "manifest")
    print(f"  {len(manifest['parts'])} track parts, {total:,} floats")
    print(f"  car hull {arrays['car_collision_vertices'].size // 3} vertices, "
          f"mass offset {manifest['mass_offset']}")
    print(f"  wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
