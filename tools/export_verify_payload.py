#!/usr/bin/env python3
"""Export the payloads the game's own simulation worker needs to verify a lap.

    python tools/export_verify_payload.py -o runs/client_check.json

`polytrack/verify.py` mirrors the game's `Verify` loop, and the acceptance gate
says that mirror reproduces eight real recordings frame-exactly. That validates
the *simulator*. It does not, on its own, validate a lap this project generated:
for those the target frame is what our own search claimed, so a PASS means
"these inputs reproduce that frame **here**", and the step to "the game agrees"
is an inference from the gate rather than a measurement.

This closes that gap by running the client's own
`simulation_worker.bundle.js` -- its code, not ours -- over our recording, via
`tools/verify_in_client.mjs`. The worker decodes the track code itself
(`fromSaveString`) and deserialises the recording itself (`deserialize`), so a
PASS there independently exercises `polytrack/trackcode.py` and
`polytrack/recording.py` as well as the physics.

Nothing from the game is written into this repo. The payload is our own
geometry, our own track code and our own recordings; the client's files are read
from wherever they were extracted and stay there.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

#: Laps to check: (label, artifact, frames to pick when the file holds several).
LAPS = [
    ("human PB", "records.json", 19025),
    ("Arm 1  policy alone", "arms/arm1-policy/winter_5.json", None),
    ("Arm 2  policy + guided beam", "arms/arm2-search/winter_5.json", None),
]


def _f32(array) -> str:
    return base64.b64encode(
        np.ascontiguousarray(array, dtype=np.float32).tobytes()
    ).decode("ascii")


def _bare_save_string(code: str) -> str:
    """Re-emit an exported ``.track`` string without its name/author header.

    The client's `fromSaveString` takes an editor save string, not an export.
    The two differ by a header that lives *inside* the compressed payload, so it
    cannot be sliced off the outside -- the record has to be inflated, the
    header skipped, and the remainder recompressed the way the format nests it:
    ``symbols(deflate(symbols(deflate(payload))))``.

    Only the container is rebuilt. The payload bytes are the game's own, passed
    through untouched, and the result is checked back through our decoder before
    it is used.
    """
    import zlib

    from polytrack.trackcode import (
        PREFIX, _Reader, _inflate_stages, _read_header, decode, encode_symbols,
    )

    body = code[len(PREFIX):] if code.startswith(PREFIX) else code
    raw = _inflate_stages(body)
    reader = _Reader(raw)
    if code.startswith(PREFIX):
        _read_header(reader)
    payload = raw[reader.offset:]

    inner = encode_symbols(zlib.compress(payload))
    bare = encode_symbols(zlib.compress(inner.encode("utf-8")))

    # Cheap guard against re-encoding something subtly different.
    before, after = decode(code), decode(bare)
    if len(before.parts) != len(after.parts):
        raise SystemExit(
            f"  bare re-encode changed the track: {len(before.parts)} parts "
            f"-> {len(after.parts)}"
        )
    return bare


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="Winter 5")
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument("-o", "--out", type=Path,
                        default=Path("runs/client_check.json"))
    args = parser.parse_args(argv)

    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.world import mountain_vertices

    entry = TrackLibrary.load(args.library).by_name(args.track)
    assets = GeometryAssets.load()
    code = _bare_save_string(entry.code)
    vertices, offset = mountain_vertices(entry.track_code())

    parts = []
    for part in assets.parts:
        parts.append({
            "id": int(part.id),
            "vertices": _f32(part.vertices),
            # The worker passes -1 for "no detector", which is how the absence
            # is encoded on our side too.
            "detector": None if part.detector_type < 0 else {
                "type": int(part.detector_type),
                "center": [float(x) for x in part.detector_center],
                "size": [float(x) for x in part.detector_size],
            },
            "startOffset": (
                None if part.start_offset is None
                else [float(x) for x in part.start_offset]
            ),
        })

    laps = []
    for label, path, pick in LAPS:
        blob = Path(path)
        if not blob.exists():
            print(f"  skipping {label}: no {path}")
            continue
        data = json.loads(blob.read_text())
        if isinstance(data, list):  # records.json holds every track
            chosen = next(r for r in data if r["frames"] == pick)
            recording, frames = chosen["recording"], int(pick)
        else:
            recording, frames = data["recording"], int(data["frames"])
        laps.append({
            "label": label,
            "artifact": path,
            "trackData": code,
            "carRecording": recording,
            "targetFrames": frames,
            "mountainVertices": _f32(vertices),
            "mountainOffset": {
                "x": float(offset[0]), "y": float(offset[1]), "z": float(offset[2])
            },
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "init": {
            "version": assets.version,
            "isRealtime": False,
            "carMassOffset": float(assets.mass_offset),
            "carCollisionShapeVertices": _f32(assets.car_collision_vertices),
            "trackParts": parts,
        },
        "laps": laps,
    }))
    print(f"  wrote {args.out}: {len(parts)} parts, {len(laps)} laps, "
          f"version {assets.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
