# Generalisation: trained on generated tracks, measured on unseen ones

**A policy trained on 48 procedurally generated tracks finishes 94% of attempts
on 16 it has never seen, and beats the scripted driver on all 16.**

The experiment the whole project was built for, and the first result here that
is about driving rather than about one lap.

## Result

| | tracks | finished at least once | mean finish rate | median best/par | beat the driver |
|---|---|---|---|---|---|
| **held-out** (unseen seeds) | 16 | 16/16 | **94%** | 1.35 | **16/16** |
| training (first 16 of 48) | 16 | 16/16 | 92% | 1.36 | 16/16 |
| zero-shot, official tracks | 4 | **0/4** | **0%** | -- | -- |

**There is no measurable generalisation gap.** Held-out finish rate is 94%
against 92% on training tracks, and the median time-to-par is 1.35 against
1.36. The held-out number being marginally *higher* is noise, not an effect;
what it rules out is memorisation.

Against the floor, the policy is **1.55x faster than the scripted driver** on
held-out tracks (median; range 1.37-1.73) and beat it on every single one.

## What the zero-shot row means, and does not

0 of 4 official tracks is a real failure and belongs in any write-up, but it is
**out-of-distribution transfer, not a held-out test**. The generator uses five
part types out of the game's 186: a start, a straight, one right-angle corner,
a checkpoint and a finish. It builds flat tracks with no jumps, no banking, no
walls to ride and no elevation. Winter 5's human record wall-rides at 400 km/h
along vertical faces; nothing in the training distribution contains a wall, let
alone rewards driving on one.

So this measures the distance between the generated distribution and
human-designed tracks, which is large, rather than a failure to generalise
within a distribution, which did not happen. Closing it is a generator problem
-- more of the part vocabulary -- not a learning one.

## Setup

* **Policy**: IQN, 1,220,000 gradient steps over 28.5 hours, six collectors,
  full nine-action space. No `--forward-only`: generated corners need braking.
* **Observation**: 62 floats, rays cast at the real collision geometry via
  `corridor.RoadMap`.
* **Libraries**: `tracks_generated.npz` (48 tracks, seeds 0+) and
  `tracks_heldout.npz` (16, seeds 1000+). Disjoint seeds, matched
  distributions: 375-1145 m and 374-1124 m, 3-19 and 3-16 corners.
* **Par**: the scripted driver's time halved. A guess, and it is why
  `best/par` is a relative figure rather than an absolute standard -- see
  `scripts/build_generated_library.py`.
* **Evaluation**: five greedy attempts per track, because greedy is not
  deterministic here. `scripts/evaluate.py`.

## Learning curve

`learning_curve.csv`, 121 points. Most of the learning is early:

| steps | hours | finish rate | greedy lap fraction |
|---|---|---|---|
| 8k | 0.2 | 0% | 0.08 |
| 61k | 1.7 | 6% | 0.37 |
| 130k | 3.3 | 37% | 0.83 |
| 257k | 5.3 | 82% | 0.98 |
| 735k | 19.8 | 91% | 1.00 |
| 987k | 23.6 | **93%** | 1.00 |
| 1,220k | 28.5 | 87% | 0.92 |

82% by 257k steps and roughly flat after. The last 1M gradient steps bought
very little, which is worth knowing before budgeting a repeat.

## What this still does not show

* **One seed.** No repeats, so no error bars on any of it.
* **No learned baseline.** The scripted driver is a floor, not a competitor.
  Nothing here is measured against another algorithm.
* **Par is a convention**, not a standard, so `best/par` compares runs within
  this project and nothing outside it.
* **The distribution is narrow** -- five part types, flat ground -- which is
  exactly what the zero-shot row measures the cost of.
