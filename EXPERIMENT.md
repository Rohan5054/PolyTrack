# Phase 2 protocol: three arms on Winter 5

How the three approaches are compared, and what counts as a result. Written
against numbers this repo has generated and re-verified; nothing here is quoted
from memory or from outside the artifacts in `runs/`.

## Verified baselines

Official artifacts live in `arms/`, which is version controlled; `runs/` is
scratch and gitignored. Every figure below was replayed through
`polytrack/verify.py` on a fresh backend
and reproduced its own finish frame exactly, passing both checkpoints. Re-run the
whole table at any time with `scripts/verify_baselines.py`.

| lap | frames | artifact | what it is |
|---|---|---|---|
| human PB | 19025 | `records.json` | the run the Winter 5 reference line is derived from |
| **Arm 1** | **19422** | `arms/arm1-policy/winter_5.json` | policy alone: best of 20 greedy laps, no search |
| training harvest | 19249 | `arms/arm2-search/lineage/01_training_harvest_19249.json` | fastest collector episode of a sitting |
| searched, arc ranking | 19405 | `arms/arm2-search/lineage/superseded_beam_19405.json` | guided beam, ranked on arc |
| searched, better seed | 19245 | `arms/arm2-search/lineage/02_beam_19245.json` | guided beam from the 19249 harvest |
| **Arm 2** | **18786** | `arms/arm2-search/winter_5.json` | guided beam ranked on distance from the reference line |

Arm 2 stands at **18.786 s**: 239 ms under the human PB and 636 ms under Arm 1.

### Why Arm 1 is 19.422 and not 19.105

`history.json` records a 19.105 episode, and it is excluded from this protocol.
Record laps were only written to disk when they beat the all-time best, so that
episode's inputs were discarded and **no recording of it exists**. It cannot be
replayed, gate-checked, or watched. A baseline that cannot be reproduced is not a
baseline.

19.422 is the honest "policy without search" figure: the fastest of twenty greedy
laps, saved, and gate-verified. It is slower than 19.105, which makes the Arm 2
improvement look larger -- so state plainly in the write-up that 19.105 was
observed once and lost, or the comparison flatters itself.

## Frozen substrate

All three arms run against identical inputs. Hashes as frozen:

| file | sha256 (first 16) | bytes |
|---|---|---|
| `polytrack_physics.wasm` | `d4ef02676973d41a` | 396005 |
| `geometry.npz` | `6fb32e3b719811dc` | 771818 |
| `tracks.npz` | `5cdc6a530cf527c7` | 110982 |
| `records.json` | `f87af733f13d7b0f` | 4378 |

Game version **0.6.2** on every machine. v0.6.0 rewrote the physics; times and
replays do not cross versions.

**`tracks.npz` must not be rebuilt during Phase 2.** It carries the Winter 5
reference line, and that line is the human 19.025 lap. Arm 2's result comes from
ranking candidates by distance from it, so regenerating the line from Arm 2's own
18.786 output would put Arm 2 output into the shared substrate and contaminate
Arm 3. Improved lines live in separate files used by one arm only.

Neither `polytrack_physics.wasm` nor `geometry.npz` may be sent to a
collaborator. They are derived from Kodub's game. Each person extracts them from
a copy they own, with `tools/capture_geometry.py` and `tools/build_geometry.py`.

## Budget, not hardware

Lap time here is a count of simulated milliseconds from a deterministic module.
**18.786 is 18.786 on any CPU.** Running every arm on one machine standardises a
variable that does not vary, and it standardises the wrong thing: what hardware
changes is how much search or training fits in an hour.

On the development machine that budget is not stable either, and the reason
turned out not to be the one recorded here for weeks. The learner ran at 40
gradient steps/s in one sitting and 4/s in another, and this document called the
slow case thermal throttling. It was not. Windows tags a process it considers
background work -- anything launched detached, which is how every long run here
was started -- and confines it to the efficiency cores at their lowest clock. On
a fixed single-threaded benchmark that is **52.6 s against 3.6 s, 14.7x**;
measured on the training loop, replay ratio 0.11 against 0.46. Clocks were
boosting at 110% of base throughout, with no thermal or power cap, which is
precisely why it read as thermal for so long. `polytrack/priority.py` opts out,
and every long-running entry point calls it.

That correction strengthens the argument rather than weakening it. A 14.7x
difference in effective budget sat undetected on one machine for weeks, between
runs, with nothing in any log to show for it. A "same laptop" rule would have
certified those runs as comparable. One training block also lost 87 minutes to
the machine sleeping.

So the controlled quantity is hardware-independent and already logged:

* **search** -- `SearchResult.n_frames_simulated`, printed by every run
* **training** -- `agent.steps`, carried in the checkpoint and in `history.json`

Both survived the throttle untouched: a gradient step is a gradient step and a
simulated frame is a simulated frame whichever core ran it, so Arm 2's reported
budget needs no revision even though it was earned at a fraction of the speed.
Declare a budget per arm, report actual usage, and the comparison reproduces on
anyone's machine. Running the official numbers on one device is fine as a
secondary note; it is not the control.

Verify once that the module is deterministic across machines -- replay one
recording on both and compare finish frames -- and the single-device rule becomes
unnecessary.

## What the gate proves

`polytrack/verify.py` mirrors the game's own `Verify` path: step the recording and
assert `hasFinished && frames == targetFrames`.

* For a **human record**, the target frame comes from the game's own local
  storage, so a PASS means the simulator reproduces the real game. That is the
  Stage 0 acceptance gate, and `scripts/run_gate.py` is the tool for it -- it
  needs `records.json`, because it needs an externally recorded finish frame.
* For a **generated lap** there is no external oracle; the target is what the
  search claims. A PASS therefore means **deterministic reproduction**: the inputs
  really do produce that frame, so the run is not an artifact of save/restore,
  splicing, or a search bug. `search_run.py` and `tools/watch_run.py` do this
  automatically.

Describe generated laps as reproduced, not as "validated by the game". For the
stronger claim, install the ghost with `tools/watch_run.py --game ...` and let the
real client replay it.

**The gate does not check track geometry.** The module has no notion of off-track
-- a car in a ditch is just a car in a ditch, and it never reports termination.
Checkpoints are verified, which catches gross corner-cutting, but do not claim
clipping was ruled out.

## The arms

**Arm 1 -- policy alone.** IQN, forward-only action space, best of N greedy laps,
no test-time search. 19422.

**Arm 2 -- policy plus guided beam search.** The same checkpoint used as an action
prior, beam anchored in a known lap, carrying that lap as a guide so the result is
bounded below by its seed, ranked on `arc + lookahead*speed - line_weight*|lateral|`.
18786.

**Arm 3 -- independent.** Must not consume Arm 2 trajectories, tuned coefficients,
or checkpoints. Different in kind, not a variation: a different RL algorithm is a
variation and a reviewer will say so. Genuinely distinct options include geometric
racing-line optimisation with a tracking controller, continuous trajectory
optimisation such as CMA-ES over a spline, or behaviour cloning from the human lap
followed by fine-tuning.

### Independence has a leak worth declaring

Data isolation is enforceable; insight isolation is not. Arm 2 established by
measurement that the whole Winter 5 deficit is *route* -- 0.3 degrees of slip on
both laps, zero collision impulse on both, identical throttle, steering effort
within 0.002, and the agent 8.3 m off the racing line driving 6.2 m further. That
finding cost seven failed searches. Anyone who reads it starts Arm 3 with the
answer.

Record what was shared and when. Claiming clean-room independence after the fact
is the kind of thing peer review is designed to catch.

## Reporting a result

A run counts when all of these hold:

1. A JSON artifact exists in `runs/` with `track`, `frames` and `recording`.
2. `scripts/verify_baselines.py` reports PASS for it -- finish frame reproduced,
   both checkpoints.
3. The budget it consumed is reported: simulated frames for search, gradient
   steps for training.
4. The substrate hashes above are unchanged.

Bot and TAS runs stay off the human leaderboard. Check Kodub's Terms of Service,
added in v0.6.0 with server-side replay verification, before submitting anything.
