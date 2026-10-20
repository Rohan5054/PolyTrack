"""Drive one checkpoint greedily on Summer 1; print where each attempt ended.

    python arms/transfer/where_attempts_end.py . runs/xfer-pre-3/agent.pt xfer-pre-3 10

The script that produced ``where_attempts_end.jsonl``, kept as it was run.

Mirrors scripts/evaluate.py (RoadMap observation, greedy acting) with the
training run's episode budget of 3x par, and reports the furthest arc of each
attempt as a fraction of the reference line plus the failure reason.
"""

import json
import sys
from pathlib import Path

repo, checkpoint, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
attempts = int(sys.argv[4]) if len(sys.argv) > 4 else 10
sys.path.insert(0, str(repo))

import numpy as np
import torch

torch.set_num_threads(1)

from polytrack.corridor import RoadMap
from polytrack.env import PolyTrackEnv
from polytrack.learning.iqn import IQNAgent
from polytrack.library import TrackLibrary
from polytrack.physics.assets import GeometryAssets
from polytrack.physics.wasm import WasmPhysics

agent = IQNAgent.load(checkpoint)
entry = TrackLibrary.load(repo / "tracks.npz").by_name("Summer 1")
assets = GeometryAssets.load(repo / "geometry.npz")
code = entry.track_code()
road = RoadMap.build(code, assets)
centre = entry.line.centerline
length = float(np.sum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))

ends = []
for _ in range(attempts):
    backend = WasmPhysics(code, assets=assets)
    env = PolyTrackEnv(track=entry.line, backend=backend, road=road,
                       par_frames=entry.par_frames,
                       time_limit_ms=entry.par_frames * 3.0)
    observation, _ = env.reset()
    furthest, info = 0.0, {}
    while True:
        observation, _, terminated, truncated, info = env.step(
            agent.act(observation, epsilon=0.0))
        furthest = max(furthest, float(info.get("arc", 0.0)))
        if terminated or truncated:
            break
    ends.append({"fraction": round(furthest / length, 3),
                 "failure": info.get("failure") or ("finished" if info.get("finished") else "?")})
    backend.close()

print(json.dumps({"run": name, "steps": agent.steps, "ends": ends}))
