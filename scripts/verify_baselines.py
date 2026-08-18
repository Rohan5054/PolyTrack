#!/usr/bin/env python3
"""Re-verify every baseline lap and the frozen substrate.

    python scripts/verify_baselines.py

The enforceable half of EXPERIMENT.md. A number in a table is worth nothing on
its own -- this project already lost a 19.105 lap that way, recorded in a log and
discarded before anything could replay it. Every figure quoted in the write-up
must survive this script, and the script must be runnable by anyone holding the
same artifacts.

Two checks, and they answer different questions:

* **Laps.** Replay each recording on a fresh backend and confirm it reproduces
  its own finish frame with every checkpoint. For a human record the target frame
  comes from the game's own storage, so this is the Stage 0 acceptance gate. For
  a generated lap the target is what produced it, so a PASS means deterministic
  reproduction -- the run is not an artifact of save/restore, splicing or a
  search bug. Neither case is the game accepting a submission.
* **Substrate.** Hash the inputs all three arms share. If `tracks.npz` changes,
  the Winter 5 reference line changed, and results before and after are not
  comparable -- Arm 2's ranking is measured against that line.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: (label, artifact, frames to pick when the file holds several runs)
LAPS = [
    ("human PB", "records.json", 19025),
    ("Arm 1  policy alone", "arms/arm1-policy/winter_5.json", None),
    ("  lineage: training harvest",
     "arms/arm2-search/lineage/01_training_harvest_19249.json", None),
    ("  lineage: beam from harvest",
     "arms/arm2-search/lineage/02_beam_19245.json", None),
    ("Arm 2  policy + guided beam", "arms/arm2-search/winter_5.json", None),
    ("Arm 3  independent", "arms/arm3-independent/winter_5.json", None),
]

#: Arms that have not produced a lap yet are reported, not counted as failures.
OPTIONAL = {"arms/arm3-independent/winter_5.json"}

#: Frozen for the duration of Phase 2. See EXPERIMENT.md.
SUBSTRATE = {
    "polytrack_physics.wasm": "d4ef02676973d41a",
    "geometry.npz": "6fb32e3b719811dc",
    "tracks.npz": "5cdc6a530cf527c7",
    "records.json": "f87af733f13d7b0f",
}


def main() -> int:
    from polytrack.library import TrackLibrary
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.verify import verify

    entry = TrackLibrary.load("tracks.npz").by_name("Winter 5")
    assets = GeometryAssets.load()

    print(f"\n  {'artifact':<34}{'frames':>8}   gate")
    failures = 0
    for label, path, pick in LAPS:
        blob = Path(path)
        if not blob.exists():
            state = "not attempted" if path in OPTIONAL else f"MISSING {path}"
            print(f"  {label:<34}{'--':>8}   {state}")
            failures += path not in OPTIONAL
            continue
        data = json.loads(blob.read_text())
        if isinstance(data, list):  # records.json holds every track
            chosen = next(r for r in data if r["frames"] == pick)
            serialised, frames = chosen["recording"], pick
        else:
            serialised, frames = data["recording"], int(data["frames"])

        result = verify(
            WasmPhysics(entry.track_code(), assets=assets),
            Recording.deserialize(serialised),
            frames,
        )
        verdict = "PASS" if result.passed else "FAIL"
        print(f"  {label:<34}{frames:>8}   {verdict} "
              f"(reached {result.frames}, cp {result.checkpoints}"
              f"/{result.expected_checkpoints})")
        failures += not result.passed

    print(f"\n  {'substrate':<26}{'sha256':<20}status")
    for name, expected in SUBSTRATE.items():
        path = Path(name)
        if not path.exists():
            print(f"  {name:<26}{'--':<20}MISSING")
            failures += 1
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        ok = digest == expected
        print(f"  {name:<26}{digest:<20}"
              + ("frozen" if ok else f"CHANGED, expected {expected}"))
        failures += not ok

    if failures:
        print(f"\n  {failures} check(s) failed. Numbers quoted from this repo are "
              "not currently\n  reproducible; do not report them until this is "
              "green.\n")
        return 1
    print("\n  All laps reproduce their finish frame. Substrate unchanged.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
