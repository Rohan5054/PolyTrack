# Arm 1 -- policy alone

**Winter 5: 19422** (`winter_5.json`, gate PASS, both checkpoints)

The learned policy driving with no test-time search: the fastest of twenty
greedy laps from `runs/w5-fwd/agent.pt`, produced by

```bash
python tools/watch_run.py runs/w5-fwd/agent.pt --track "Winter 5" --attempts 20
```

Twenty attempts rather than one because greedy is **not deterministic** -- IQN
samples fresh quantile fractions on every forward pass, so twenty greedy laps are
twenty different laps. The twenty spanned 19422 to 21073.

## Why this is the baseline and 19105 is not

`runs/w5-fwd/history.json` records a 19105 episode. It is excluded. Record laps
were written to disk only when they beat the all-time best, so that episode's
inputs were discarded and no recording of it survives. It cannot be replayed,
gate-checked or watched.

19422 is slower, which makes the Arm 2 improvement look larger. Any write-up has
to say so: a 19105 lap was observed once, during training, and lost.

## Configuration

* IQN, 62-float observation, 9 actions restricted to the three throttle actions
* 50 ms control interval
* checkpoint at 717,248 gradient steps

## The guiding checkpoint no longer exists

`runs/w5-fwd/agent.pt` is one mutable file and training continued past this
point: by the time Arm 2's searches ran, the same path held **801,472** gradient
steps, and `runs/` is gitignored, so the 717,248 state is gone.

That does not weaken this result -- the lap itself is banked here and reproduces
its finish frame on a fresh backend, which is the whole claim. It does mean the
twenty greedy laps cannot be re-drawn from the same network. If a later arm needs
a like-for-like policy baseline, take it from the checkpoint that arm actually
used and say which one.

The forward-only restriction is a per-track claim with per-track evidence: the
19025 human lap holds throttle for 100.0% of its frames and never brakes, while
every other track in `records.json` brakes for 8-21% of the lap.
