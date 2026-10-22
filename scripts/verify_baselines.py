#!/usr/bin/env python3
"""Re-verify every baseline lap and the frozen substrate.

    python scripts/verify_baselines.py

The enforceable half of EXPERIMENT.md. A number in a table is worth nothing on
its own -- this project already lost a 19.105 lap that way, recorded in a log and
discarded before anything could replay it. Every figure quoted in the write-up
must survive this script, and the script must be runnable by anyone holding the
same artifacts.

Two checks, and they answer different questions:

* **Laps.** Replay each recording on a fresh backend, on its own track, and
  confirm it reproduces its own finish frame with every checkpoint. For a human
  record the target frame comes from the game's own storage, so this is the
  Stage 0 acceptance gate. For a generated lap the target is what produced it,
  so a PASS means deterministic reproduction -- the run is not an artifact of
  save/restore, splicing or a search bug. Neither case is the game accepting a
  submission.
* **Substrate.** Hash the inputs all three arms share, in full, with
  `scripts/substrate_hashes.py`. If `tracks.npz` changes, the Winter 5 reference
  line changed, and results before and after are not comparable -- Arm 2's
  ranking is measured against that line. An .npz whose file hash differs but
  whose content digest matches holds the same arrays packed differently, and
  passes with a note.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import substrate_hashes  # noqa: E402  -- the one definition of both digests

#: (label, artifact relative to the repo root, frames to pick when the file
#: holds several runs). A lap file names its track; a records.json entry is
#: matched to its track by id.
LAPS = [
    ("human PB", "records.json", 19025),
    ("human best", "records.json", 18809),
    ("Arm 1  policy alone", "arms/arm1-policy/winter_5.json", None),
    ("  lineage: training harvest",
     "arms/arm2-search/lineage/01_training_harvest_19249.json", None),
    ("  lineage: beam from harvest",
     "arms/arm2-search/lineage/02_beam_19245.json", None),
    ("Arm 2  policy + guided beam", "arms/arm2-search/winter_5.json", None),
    ("Arm 3  independent", "arms/arm3-independent/winter_5.json", None),
    # The best lap of each 400k Summer 1 run, 21 October 2026 (arms/separator).
    ("scratch 1, gamma 0.99",
     "arms/separator/sep-g099-1/best_summer_1.json", None),
    ("scratch 1, gamma 0.999",
     "arms/separator/sep-g0999-1/best_summer_1.json", None),
    ("pretrained 1", "arms/separator/xlong-pre-1/best_summer_1.json", None),
    ("pretrained 2", "arms/separator/xlong-pre-2/best_summer_1.json", None),
    ("pretrained 3", "arms/separator/xlong-pre-3/best_summer_1.json", None),
    ("scratch 2", "arms/separator/xlong-scratch-2/best_summer_1.json", None),
    ("scratch 3", "arms/separator/xlong-scratch-3/best_summer_1.json", None),
]

#: Arms that have not produced a lap yet are reported, not counted as failures.
OPTIONAL = {"arms/arm3-independent/winter_5.json"}

#: Full SHA-256 of each file, and for each .npz the content digest that
#: `scripts/substrate_hashes.py` defines, as they stand on the M5 MacBook Air
#: that ran the separator and transfer experiments on 21 October 2026 and the
#: ten-lap acceptance gate.
#:
#: Until then the frozen values were 16-character prefixes:
#:   polytrack_physics.wasm  d4ef02676973d41a
#:   geometry.npz            6fb32e3b719811dc
#:   tracks.npz              5cdc6a530cf527c7
#:   records.json            f87af733f13d7b0f
#: The generalisation and Winter 5 results were produced on a Windows laptop,
#: before 21 October 2026, against those files. The 20 and 21 October Summer 1
#: runs (arms/transfer, arms/separator) and the ten-lap gate were produced on
#: the M5, against the files below. The wasm's
#: old prefix is the first 16 characters of its full hash, so it is the same
#: module. The old records.json prefix is the eight-lap file as checked out
#: with Windows (CRLF) line endings, which .gitattributes now prevents; the
#: file has since gained two laps. Run `scripts/substrate_hashes.py` on the
#: Windows laptop to see whether geometry.npz and tracks.npz differ in content
#: or only in packing.
SUBSTRATE = {
    "polytrack_physics.wasm": {
        "sha256": "d4ef02676973d41afc34b23b5248f6950b35dc4cc7e3047e3a9c6bd88e4c180e",
    },
    "geometry.npz": {
        "sha256": "ccbd8f1adf7315cd9ea25c18e41b2264cf9c16ad103be8f1cd64643003cae0e5",
        "content_sha256": "eda58e383d6d6199529c880d7dbfdcc1568b76889bac74b45cbbe978c00378dc",
    },
    "tracks.npz": {
        "sha256": "710d3113592b093b87fadd3cda021de90875b5245f63ac145fed6ff714945bcc",
        "content_sha256": "400311c6ce6dfc5c374de3b46c35af9d905ec4784d9ca5cf8ace4cf38a819dd5",
    },
    "records.json": {
        "sha256": "8594b39c32fd540e38ab333e289bf9440b432dc8b2c6ecf6502e570418ed1555",
    },
}


def main() -> int:
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.verify import verify

    # Everything from the repo root, as the substrate check reads it, so the
    # files replayed are the files hashed wherever this is run from.
    root = substrate_hashes.ROOT
    library = TrackLibrary.load(root / "tracks.npz")
    by_id = {entry.track_code().track_id: entry for entry in library}
    assets = GeometryAssets.load()

    print(f"\n  {'artifact':<34}{'track':<10}{'frames':>8}   gate")
    failures = 0
    for label, path, pick in LAPS:
        blob = root / path
        if not blob.exists():
            state = (
                "attempted, did not finish -- see the arm's README"
                if path in OPTIONAL else f"MISSING {path}"
            )
            print(f"  {label:<34}{'':<10}{'--':>8}   {state}")
            failures += path not in OPTIONAL
            continue
        data = json.loads(blob.read_text())
        if isinstance(data, list):  # records.json holds every track
            chosen = next(r for r in data if r["frames"] == pick)
            serialised, frames = chosen["recording"], pick
            entry = by_id.get(chosen["track_id"])
        else:
            serialised, frames = data["recording"], int(data["frames"])
            entry = next((e for e in library if e.name == data["track"]), None)
        if entry is None:
            print(f"  {label:<34}{'':<10}{frames:>8}   MISSING its track in tracks.npz")
            failures += 1
            continue

        result = verify(
            WasmPhysics(entry.track_code(), assets=assets),
            Recording.deserialize(serialised),
            frames,
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"  {label:<34}{entry.name:<10}{frames:>8}   {verdict} "
              f"(reached {result.frames}, cp {result.checkpoints}"
              f"/{result.expected_checkpoints})")
        failures += not result.passed

    print(f"\n  {'substrate':<24}{'sha256':<66}status")
    repacked = []
    actual = substrate_hashes.substrate()
    for name, expected in SUBSTRATE.items():
        found = actual.get(name)
        if found is None:
            print(f"  {name:<24}{'--':<66}MISSING")
            failures += 1
            continue
        if found["sha256"] == expected["sha256"]:
            status = "frozen"
        elif (
            "content_sha256" in expected
            and found.get("content_sha256") == expected["content_sha256"]
        ):
            status = "content matches, file differs"
            repacked.append(name)
        else:
            status = f"CHANGED, expected {expected['sha256']}"
            if "content_sha256" in expected:
                status += (f"\n  {'':<24}content {found['content_sha256']}"
                           f"  expected {expected['content_sha256']}")
            failures += 1
        print(f"  {name:<24}{found['sha256']:<66}{status}")

    if failures:
        print(f"\n  {failures} check(s) failed. Numbers quoted from this repo are "
              "not currently\n  reproducible; do not report them until this is "
              "green.\n")
        return 1
    if repacked:
        hold, were = ("holds", "was") if len(repacked) == 1 else ("hold", "were")
        print(f"\n  Note: {' and '.join(repacked)} {hold} the frozen arrays but "
              f"{were} written differently\n  (an .npz is a zip archive: "
              "timestamps, compression and member order vary\n  with whatever "
              "wrote it). The data is the same, so this passes.")
    print("\n  All laps reproduce their finish frame. Substrate unchanged.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
