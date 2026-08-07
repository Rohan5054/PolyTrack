#!/usr/bin/env python3
"""Pull personal-best runs out of the desktop game's local storage.

This is how the acceptance gate gets its oracle. The gate needs a pair -- an
input recording and the exact frame the run finished on -- and the client
already keeps both. Every time you set a personal best, ``saveRecord`` writes::

    polytrack_v5_prod_record_<slot>_default_<trackId>
      -> {"uploadId":…, "tokenHash":…, "frames":…, "recording":"…"}

``frames`` is the finish frame in milliseconds and ``recording`` is the
serialised input recording. That is exactly what ``polytrack.verify`` wants, it
comes off your own machine, and it needs no network and no leaderboard.

Usage::

    python tools/extract_records.py                 # find and list runs
    python tools/extract_records.py -o records.json # save them

Then::

    python scripts/run_gate.py records.json

**Play a track first.** With no personal best there is nothing stored, and this
will tell you so rather than failing obscurely.

On reading the store directly: PolyTrack is an Electron app, so localStorage is
a Chromium LevelDB. Rather than take a LevelDB dependency to read a handful of
values, this scans the raw ``.log``/``.ldb`` files for the record JSON. That
works because these particular values are plain ASCII, and because we only need
to read them -- never write. Chromium keeps superseded versions until it
compacts, so an old slower run may surface alongside the current best; both are
equally valid for the gate, which only cares that inputs and finish frame agree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# The key the client writes under, and the shape of the value.
_KEY = re.compile(rb"polytrack_v5_prod_record_(\d+)_(default|undeterministic)_([0-9a-f]{64})")
_VALUE = re.compile(rb'\{"uploadId":.{0,64000}?\}', re.DOTALL)


def storage_directories() -> list[Path]:
    """Where an Electron app called PolyTrack would keep its local storage."""
    roots: list[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "PolyTrack")
    home = Path.home()
    roots += [
        home / "AppData" / "Roaming" / "PolyTrack",
        home / "Library" / "Application Support" / "PolyTrack",
        home / ".config" / "PolyTrack",
    ]
    found = []
    for root in roots:
        leveldb = root / "Local Storage" / "leveldb"
        if leveldb.is_dir() and leveldb not in found:
            found.append(leveldb)
    return found


def scan(directory: Path) -> list[dict]:
    """Recover every record JSON blob in a leveldb directory."""
    records: list[dict] = []
    for path in sorted(directory.iterdir()):
        if path.suffix not in (".log", ".ldb"):
            continue
        blob = path.read_bytes()
        # Keys and values are stored separately, and a value does not name its
        # own track. Pair them by position: the key precedes its value.
        keys = [(m.start(), m) for m in _KEY.finditer(blob)]
        for match in _VALUE.finditer(blob):
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if "frames" not in payload or "recording" not in payload:
                continue
            preceding = [k for pos, k in keys if pos < match.start()]
            key = preceding[-1] if preceding else None
            records.append(
                {
                    "track_id": key.group(3).decode() if key else None,
                    "slot": int(key.group(1)) if key else None,
                    "deterministic": key.group(2) == b"default" if key else None,
                    "frames": payload["frames"],
                    "recording": payload["recording"],
                    "source": path.name,
                }
            )
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for record in records:
        key = (record["track_id"], record["frames"], record["recording"])
        seen.setdefault(key, record)
    return sorted(
        seen.values(), key=lambda r: (r["track_id"] or "", r["frames"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", type=Path, help="write the records as JSON")
    parser.add_argument(
        "--dir", type=Path, help="a leveldb directory to scan, if autodetection fails"
    )
    args = parser.parse_args()

    directories = [args.dir] if args.dir else storage_directories()
    if not directories:
        print("No PolyTrack local storage found. Looked for a 'Local Storage/leveldb'")
        print("directory under, among others:")
        for root in ("%APPDATA%\\PolyTrack", "~/.config/PolyTrack"):
            print(f"  {root}")
        print("\nIf the game is installed elsewhere, pass --dir.")
        return 1

    records: list[dict] = []
    for directory in directories:
        print(f"scanning {directory}")
        records += scan(directory)
    records = deduplicate(records)

    if not records:
        print("\nNo saved runs found. Play a track to the finish line once -- the")
        print("client stores the recording and the finish frame on a personal best.")
        return 1

    # Name the tracks where we can, so the output is readable.
    names = _official_track_names()
    print(f"\n{len(records)} run(s):\n")
    for record in records:
        name = names.get(record["track_id"], "")
        flag = "" if record["deterministic"] is not False else "  [non-deterministic]"
        seconds = record["frames"] / 1000
        print(
            f"  {record['frames']:>8} frames  {seconds:7.3f} s  "
            f"{(name or record['track_id'] or '?')[:28]:28s}"
            f"  {len(record['recording']):>5} B{flag}"
        )

    if args.out:
        args.out.write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.out}")
        print(f"run the gate:  python scripts/run_gate.py {args.out}")
    else:
        print("\nre-run with -o records.json to save them")
    return 0


def _official_track_names() -> dict[str, str]:
    """Map track ids to names, if the shipped .track files are reachable."""
    root = os.environ.get("POLYTRACK_TRACKS")
    if not root:
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from polytrack.trackcode import PREFIX, decode
    except ImportError:
        return {}
    names = {}
    for path in Path(root).rglob("*.track"):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text.startswith(PREFIX):
                continue
            track = decode(text)
            names[track.track_id] = track.name or path.stem
        except Exception:
            continue
    return names


if __name__ == "__main__":
    sys.exit(main())
