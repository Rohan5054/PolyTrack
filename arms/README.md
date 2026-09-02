# Arms

One folder per experimental approach. These are the **official artifacts** --
version controlled, unlike `runs/`, which is scratch and gitignored. A number
that is not backed by a file in here is not a result.

| folder | approach | Winter 5 |
|---|---|---|
| `arm1-policy/` | learned policy alone, no test-time search | 19422 |
| `arm2-search/` | the same policy plus guided beam search | **18786** |
| `arm3-independent/` | continuous optimisation, no policy | did not finish, 1089 m of 1747 |

Arm 3 was attempted and stalled at 62% of the track for 17.8M simulated
frames, against the 914,944 that produced Arm 2's finish. A run that never
crosses the line has no finish frame, so it has no gate result and no row in the
verified table -- `arms/arm3-independent/partial_1089m.json` is kept to resume
from, not as a result. It is also not clean-room: same author as Arm 2. Its
README says both plainly.

Human reference for the track: **19025**, in `records.json` at the repo root.
That lap is also where the Winter 5 reference line comes from, which matters for
Arm 2 and is spelled out in `EXPERIMENT.md`.

Verify every one of them:

```bash
python scripts/verify_baselines.py
```

Rules, budgets and what the gate does and does not prove: `EXPERIMENT.md`.

Each lap file is the same shape and is enough on its own to replay a run:

```json
{"track": "Winter 5", "frames": 18786, "recording": "<base64url>"}
```

`recording` is a sequence of input toggles in PolyTrack's own format. It carries
no game content, which is why these are safe to commit while
`polytrack_physics.wasm`, `geometry.npz` and `tracks.npz` are not.

Watch any of them:

```bash
python tools/watch_run.py --from-best arms/arm2-search/winter_5.json \
    --track "Winter 5" --game path/to/extracted/app
```
