#!/usr/bin/env python3
"""Find PolyTrack's WebAssembly physics module on this machine.

Searches the usual install locations for ``.wasm`` files and ranks them by how
likely each is to be the physics module. Written because "find the wasm file" is
the first practical obstacle in Stage 0 and the answer differs on every OS and
every install method.

Usage::

    python tools/find_wasm.py                 # search common locations
    python tools/find_wasm.py ~/Downloads     # search somewhere specific
    python tools/find_wasm.py --all           # search your whole home folder (slow)

Nothing is downloaded and nothing is modified. This only reads directory listings
on your own machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Filename fragments that suggest a physics module rather than, say, a codec or
# a font shaper that happens to ship as WASM.
STRONG = ("physic", "sim", "engine", "car", "vehicle", "core")
WEAK = ("polytrack", "game", "main", "app")

#: Where the itch.io app and common manual extractions put things, per OS.
def candidate_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home / "Downloads",
        home / "Desktop",
        home / "Documents",
        home / "Games",
    ]
    if sys.platform == "win32":
        roots += [
            home / "AppData" / "Roaming" / "itch" / "apps",
            Path("C:/Program Files/PolyTrack"),
            Path("C:/Program Files (x86)/PolyTrack"),
        ]
    elif sys.platform == "darwin":
        roots += [
            home / "Library" / "Application Support" / "itch" / "apps",
            Path("/Applications"),
        ]
    else:
        roots += [
            home / ".config" / "itch" / "apps",
            home / ".local" / "share" / "itch" / "apps",
            Path("/opt"),
        ]
    return [r for r in roots if r.exists()]


def score(path: Path) -> int:
    """Rank a candidate. Higher is more likely to be the physics module."""
    name = path.name.lower()
    parts = " ".join(p.lower() for p in path.parts)
    points = 0
    if any(fragment in name for fragment in STRONG):
        points += 10
    if any(fragment in name for fragment in WEAK):
        points += 3
    if "polytrack" in parts:
        points += 8
    # Physics modules are typically a few hundred KB to a few MB. Very small
    # files are usually shims; very large ones are usually whole game bundles.
    try:
        size = path.stat().st_size
    except OSError:
        return points
    if 50_000 <= size <= 20_000_000:
        points += 5
    return points


def search(roots: list[Path], max_depth: int = 8) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        print(f"  searching {root} ...", flush=True)
        try:
            for path in root.rglob("*.wasm"):
                # Skip absurdly deep paths; those are usually node_modules.
                if len(path.relative_to(root).parts) > max_depth:
                    continue
                found.append(path)
        except (PermissionError, OSError):
            continue
    return sorted(set(found), key=score, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="folders to search")
    parser.add_argument("--all", action="store_true", help="search your whole home folder")
    args = parser.parse_args()

    if args.all:
        roots = [Path.home()]
    elif args.paths:
        roots = [p.expanduser() for p in args.paths]
    else:
        roots = candidate_roots()

    if not roots:
        print("No searchable folders found. Pass one explicitly:")
        print("  python tools/find_wasm.py /path/to/polytrack")
        return 1

    print("\nLooking for .wasm files")
    print("-" * 60)
    results = search(roots)

    if not results:
        print("\nNo .wasm files found.")
        print("\nIf the game is installed somewhere else, point at it directly:")
        print("  python tools/find_wasm.py \"C:/path/to/PolyTrack\"")
        print("\nIf you only have the web version, open the game in your browser,")
        print("press F12 -> Network tab -> reload -> filter by 'wasm', and save")
        print("the file from there.")
        return 1

    print(f"\nFound {len(results)} file(s), best guess first:\n")
    for path in results[:15]:
        size_mb = path.stat().st_size / 1_000_000
        marker = "  <-- most likely" if path is results[0] else ""
        print(f"  [{score(path):3d}]  {size_mb:7.2f} MB  {path}{marker}")

    print("\nNext step -- inspect the best candidate:\n")
    print(f'  python tools/inspect_wasm.py "{results[0]}" --json report.json\n')
    return 0


if __name__ == "__main__":
    sys.exit(main())
