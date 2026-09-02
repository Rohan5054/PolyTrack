# Arm 3 -- independent approach

**Winter 5: did not finish. 1089.3 m of 1747 m (62%).**

`partial_1089m.json` and `partial_1089m.knots.npy` hold the furthest run. It is
**not a result** under `EXPERIMENT.md`: a run that never crosses the line has no
finish frame, so there is nothing for the gate to check. It reproduces its 1089.3 m
exactly on a fresh backend, which is the most that can be said for it, and it is
kept so a later attempt can resume rather than restart.

## What it is

Continuous trajectory optimisation, the second of the three decompositions this
file used to propose. A lap is a scalar steering signal sampled per block and
thresholded into three actions; a **separable cross-entropy method** -- one
independent Gaussian per parameter, refit from the elite fraction -- optimises a
window of it at a time behind a saved anchor. Scored on arc reached at a fixed
horizon, which is the "equal frames, so furthest is fastest" argument and needs
no coefficients.

`polytrack/spline.py`, `scripts/arm3_run.py`, six tests in `tests/test_spline.py`.

```bash
python scripts/arm3_run.py --minutes 165 --resume <knots>.npy
```

**No policy, no checkpoint, and no Arm 2 artifact is read anywhere in it.** The
seed is the human 19.025 record, which is shared substrate: it is in
`records.json`, and it is already the Winter 5 reference line in `tracks.npz`
that every arm may use.

## Budget consumed

| run | rollouts | simulated frames | outcome |
|---|---|---|---|
| first full run | 7,456 | 10,433,640 | 992.4 m, stalled |
| second (resumed) | 2,308 | 6,758,760 | 1089.3 m, stalled |
| smoke tests | 347 | 601,680 | -- |
| **total** | **10,111** | **17,794,080** | did not finish |

For scale, Arm 2 reached a verified 18.786 s finish in **914,944** frames on its
winning run. Arm 3 spent nineteen times that and did not complete a lap. That
comparison is the point of the table, and it is the honest headline.

A further ~4 hours of wall clock was lost when the machine slept mid-run -- see
the jump from 77.3 to 323.4 minutes in `runs/arm3/run3.log`. Wall clock is not
the budget unit here for exactly this reason; the frame counts above are unaffected.

## Two measurements killed the premise

The reason to prefer this decomposition, argued in this file before any of it
was run, was that a spline makes steering **smooth by construction** and sets the
parameter count by knot spacing rather than lap length. Both halves failed on
this track, and they failed for the same underlying reason.

**Winter 5 does not tolerate input-timing error at all.** Projecting the 19.025
record onto a fixed block grid and replaying it reaches:

| grid | reached |
|---|---|
| 50 ms | 26 m |
| 25 ms | 349 m |
| 10 ms | 477 m |
| 5 ms | 468 m |

of 1747. Zero timing error replays exactly -- `reference.sample_replay` cuts at
input boundaries and `test_sampling_a_replay_does_not_perturb_it` pins it. Five
milliseconds does not. This is the same sensitivity that let part 159's
1.8e-4 collision-mesh error throw a car off this track thirteen seconds later.

**So the smoothing cannot be used.** At a two-block knot spacing the projection
reproduces **97.8%** of the record's actions and reaches 72 m against 477 -- a
2.2% difference in actions costing 400 m. The working setting is one knot per
block, where the parameterisation is a continuous *relaxation* of the input
sequence and not a smooth curve through it. The module docstring says so.

The consequence for the arm is worse than losing a selling point: the seed is a
**diverged** lap to be repaired, not a good one to be polished, so the optimiser
never had the head start the design assumed.

## How it stalled, twice

Both stalls were the failure `CLAUDE.md` already records for beam search, and
not recognising it cost most of the compute above.

**The block where progress stops is not the block that caused it.** A car that
clips a wall keeps crawling forward for seconds, so the frontier sits well past
the damage. The crash was near knot 1250; the frontier read block 1583. Six
consecutive passes anchored at 1540, never reached the cause, and returned the
same 992.4 m -- 67 of the first run's 150 minutes.

Fixed by backing off progressively when a pass gains nothing, by bounding the
sweep to the frontier (windows past it are unreachable, so each one costs a full
evaluation to score identically), and by raising the horizon to 3 s so a corner
is visible before it is entered. Then the fix was defeated by floating point: a
pass that gained a hundredth of a millimetre counted as progress and reset the
back-off, so the next run re-swept the same six unreachable windows. Progress now
needs a metre.

With the back-off working, the ceiling moved 992.4 -> 1089.3 m. It did not break.

## What a further attempt should change

* **The greedy horizon is the ceiling.** Scoring arc at 3 s still picks the line
  carrying most speed into a corner that punishes it at 4 s. Arm 2 hit this
  exactly and needed a *guide* -- a known-good sequence kept in the population
  every generation, so the search is bounded below by its seed. This has no
  equivalent, and it is the first thing to add.
* **Repairing a diverged lap may be the wrong problem.** Optimising from the
  human projection means fighting to reconstruct a line the parameterisation
  cannot hold. Starting from nothing is harder but honest, and the finding above
  says the head start was mostly illusory.
* Budget it in frames, not hours, and expect ~20M to be the entry price rather
  than the whole cost.

## Independence: declared, not claimed

**This arm was written by the same author as Arm 2 and is not clean-room.** The
design rule in this file was that Arm 3 be developed without reference to Arm 2's
outputs, and the data rule held -- no trajectory, coefficient or checkpoint of
Arm 2's is read. The insight rule did not and could not: Arm 2 established by
measurement that the entire Winter 5 deficit is *route*, and that finding shaped
what was built here. Any write-up must say so. An arm that reached 62% of the
track is not the place where this matters most, but the declaration would have
been owed either way.

A genuinely independent third arm is still open, and the two remaining
decompositions -- a geometric racing line with a tracking controller, or
behaviour cloning from the human lap followed by fine-tuning -- are untouched.
