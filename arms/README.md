# Arms

One folder per experimental approach. These are the **official artifacts** --
version controlled, unlike `runs/`, which is scratch and gitignored. A number
that is not backed by a file in here is not a result.

| folder | approach | Winter 5 |
|---|---|---|
| `arm1-policy/` | learned policy alone, no test-time search | 19422 |
| `arm2-search/` | the same policy plus guided beam search | **18786** |
| `arm3-independent/` | a separate approach, developed in isolation | -- |

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
