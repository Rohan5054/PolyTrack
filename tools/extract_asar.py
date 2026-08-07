#!/usr/bin/env python3
"""Extract files from an Electron ``.asar`` archive.

PolyTrack's desktop build is Electron -- Chromium wrapped around the same web
build the browser runs. That means the WebAssembly physics module is not loose
on disk; it lives inside ``resources/app.asar``. This pulls it out.

Usage::

    python tools/extract_asar.py path/to/app.asar --list
    python tools/extract_asar.py path/to/app.asar --wasm
    python tools/extract_asar.py path/to/app.asar --out extracted/

``--wasm`` is the one you want: it extracts only ``.wasm`` files into the
current folder and tells you what to run next.

Nothing is downloaded. This reads a file you already have and writes copies of
its contents. Keep the results local; do not redistribute them.

Format notes (asar uses Chromium's Pickle):

    UInt32LE  4                 always 4 -- size of the next field
    UInt32LE  header pickle size
    UInt32LE  header string size
    UInt32LE  JSON byte length
    <JSON header>
    <padding to 4-byte alignment>
    <file contents, concatenated>

Offsets in the JSON are relative to the end of that padded header, not to the
start of the file -- the single most common thing to get wrong here.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

WASM_MAGIC = b"\x00asm"


def read_header(data: bytes) -> tuple[dict, int]:
    """Parse the asar JSON header. Returns ``(tree, data_start_offset)``."""
    if len(data) < 16:
        raise ValueError("file is too small to be an asar archive")

    _, _, _, json_len = struct.unpack("<4I", data[:16])
    json_bytes = data[16 : 16 + json_len]
    try:
        tree = json.loads(json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"header is not valid JSON: {exc}") from None

    # Contents begin after the header, padded up to a 4-byte boundary.
    data_start = 16 + json_len
    if data_start % 4:
        data_start += 4 - (data_start % 4)
    return tree, data_start


def walk(node: dict, prefix: str = "") -> list[tuple[str, dict]]:
    """Flatten the archive tree into ``(path, entry)`` pairs."""
    found: list[tuple[str, dict]] = []
    for name, entry in node.get("files", {}).items():
        path = f"{prefix}/{name}" if prefix else name
        if "files" in entry:
            found.extend(walk(entry, path))
        else:
            found.append((path, entry))
    return found


def extract(data: bytes, entry: dict, data_start: int) -> bytes:
    offset = int(entry["offset"])
    size = int(entry["size"])
    start = data_start + offset
    return data[start : start + size]


def scan_for_wasm(data: bytes) -> list[int]:
    """Fallback: locate WASM modules by magic bytes if the header won't parse."""
    positions: list[int] = []
    start = 0
    while True:
        idx = data.find(WASM_MAGIC + b"\x01\x00\x00\x00", start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 8
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asar", type=Path, help="path to app.asar")
    parser.add_argument("--list", action="store_true", help="list contents only")
    parser.add_argument("--wasm", action="store_true", help="extract .wasm files only")
    parser.add_argument("--out", type=Path, default=Path("."), help="output folder")
    args = parser.parse_args()

    if not args.asar.exists():
        print(f"no such file: {args.asar}")
        return 1

    data = args.asar.read_bytes()
    print(f"\n{args.asar}  ({len(data) / 1_000_000:.2f} MB)")

    try:
        tree, data_start = read_header(data)
    except ValueError as exc:
        print(f"\nCould not parse the asar header: {exc}")
        hits = scan_for_wasm(data)
        if hits:
            print(f"But found {len(hits)} WASM module(s) by magic bytes, at:")
            for pos in hits:
                print(f"  offset {pos:,}")
            print("\nThe archive may be a newer format. Try the Node tool instead:")
            print("  npx @electron/asar extract app.asar extracted")
        return 1

    entries = walk(tree)
    print(f"{len(entries)} files\n")

    if args.list:
        for path, entry in sorted(entries, key=lambda e: -int(e[1].get("size", 0))):
            size = int(entry.get("size", 0))
            print(f"  {size / 1000:10.1f} KB  {path}")
        return 0

    targets = entries
    if args.wasm:
        targets = [(p, e) for p, e in entries if p.lower().endswith(".wasm")]
        if not targets:
            print("No .wasm files inside the archive.")
            print("\nRun with --list to see what is in there. If the physics is")
            print("bundled into a .js file, the module may be base64-embedded.")
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path, entry in targets:
        if "offset" not in entry:
            continue  # directories and symlinks
        blob = extract(data, entry, data_start)
        destination = args.out / (Path(path).name if args.wasm else path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        written.append(destination)
        ok = blob[:4] == WASM_MAGIC if path.lower().endswith(".wasm") else True
        flag = "" if ok else "   <-- WARNING: not valid WASM"
        print(f"  wrote {destination}  ({len(blob) / 1000:.1f} KB){flag}")

    if args.wasm and written:
        best = max(written, key=lambda p: p.stat().st_size)
        print(f"\nNext step:\n\n  python tools\\inspect_wasm.py {best} --json report.json\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
