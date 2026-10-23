# Generalisation: trained on generated tracks, measured on unseen ones

**Three independently seeded policies each finish every one of 16 unseen
generated tracks, at 97.9% ± 0.6% of attempts.**

The experiment the whole project was built for, and the first result here that
is about driving rather than about one lap.

## Result, three seeds

Each seed: a separate IQN policy, 300,000 gradient steps (~3.2 h, six
collectors, full nine-action space), evaluated with five greedy attempts on
each of 16 tracks generated from seeds disjoint from the 48 it trained on.
240 held-out episodes in total.

| seed | tracks finished >= once | mean finish rate | median best/par |
|---|---|---|---|
| 1 | 16/16 | 99% | 1.39 |
| 2 | 16/16 | 98% | 1.46 |
| 3 | 16/16 | 98% | 1.44 |
| **across seeds** | **100% ± 0%** | **97.9% ± 0.6%** | **1.43 ± 0.03** |

Spread is half the range. `seeds_heldout.log` is the raw output;
`scripts/aggregate_seeds.py` pools it.

**One track accounts for every miss.** `Generated 1011` was finished on 4 of 5
attempts by all three seeds -- identically 80%, three times. Every other track
was finished every time by every seed. That is a property of that track rather
than of training, and it is the thing a pooled average hides.

## The long run, and what the extra compute bought

A single policy trained to **1,220,000 steps over 28.5 hours** was measured
first, and against the same held-out library:

| | tracks | finished >= once | mean finish rate | median best/par | beat the driver |
|---|---|---|---|---|---|
| held-out | 16 | 16/16 | 94% | 1.35 | 16/16 |
| training (16 of 48) | 16 | 16/16 | 92% | 1.36 | 16/16 |
| zero-shot, official | 4 | **0/4** | **0%** | -- | -- |

**There is no measurable generalisation gap**: 94% held-out against 92% on the
tracks it trained on, 1.35 against 1.36 on time. What that rules out is
memorising 48 layouts.

Four times the compute did **not** buy reliability -- 94% against the seeds'
97.9% -- and the long run's own curve agrees, dipping from 93% at 987k steps to
87% at 1.22M. What it bought was pace: 1.35x par against 1.43x. If finishing
matters, 300k steps is enough and cheaper; if lap time matters, the extra
million steps is worth about 6%.

Against the floor, the long run is **1.55x faster than the scripted driver** on
held-out tracks (median; range 1.37-1.73) and beat it on every one.

## What the zero-shot row means, and does not

0 of 4 official tracks is a real failure and belongs in any write-up. It is
also not a near miss -- the policy does not run out of time, it stops working
almost immediately:

| track | length | best reached | % of lap | outcome |
|---|---|---|---|---|
| Winter 5 | 1747 m | 54 m | **3.1 %** | stalled |
| Desert 3 | 1581 m | 46 m | **2.9 %** | stalled |
| Summer 1 | 1301 m | 306 m | **23.5 %** | stalled |
| Summer 5 | 3980 m | 59 m | **1.5 %** | stalled |

Three of the four end around 50 m, two or three pieces in. That matters for
how the row is read: a policy finishing 90% of a lap would suggest a
distribution close enough to polish, and this is not that.

It is **out-of-distribution transfer, not a held-out test**. The generator uses five
part types out of the game's 186: a start, a straight, one right-angle corner,
a checkpoint and a finish. It builds flat tracks with no jumps, no banking, no
walls to ride and no elevation. Winter 5's human record wall-rides at 400 km/h
along vertical faces; nothing in the training distribution contains a wall, let
alone rewards driving on one.

So this measures the distance between the generated distribution and
human-designed tracks, which is large, rather than a failure to generalise
within a distribution, which did not happen. Closing it is a generator problem
-- more of the part vocabulary -- not a learning one.

That paragraph used to end there, as an inference. It is now measured.

## The distance, measured

`scripts/measure_support.py` takes the band between the 0.1st and 99.9th
percentile of every observation feature the policy saw while driving its 48
training tracks -- 24,545 frames -- and counts how far outside that band the
observation sits on other tracks.

| | frames with any feature outside | features outside, of 62 |
|---|---|---|
| **control**: 16 unseen *generated* tracks, 16/16 finished | **1 %** | **0.0 -- 0.1** |
| **test**: 4 official tracks, 0/4 finished | **57 -- 62 %** | **7.0 -- 9.2** |

**The control is the point.** An out-of-support count means nothing on its own,
because a band estimated too tightly reports one for every track including the
ones the policy drives perfectly. Unseen tracks from the training distribution
sit inside the band; the official tracks are **147-154x further outside it**.
Ranges are two passes of the same measurement: greedy is not deterministic
here, IQN resamples its quantile fractions every forward pass, so the
measurement itself carries about +/-15%. Both passes are in `support/`.

**The gap opens during the attempt, not at step 0.** This file previously
claimed the latter; at step 0 the count is 1 of 62 on every track, which is
noise. Per-attempt halves, from `support/violations_*.csv`:

| track | step 0 | first half | second half | whole |
|---|---|---|---|---|
| Winter 5 | 1 | 7.3 | 14.4 | 10.9 |
| Desert 3 | 1 | 10.0 | 14.3 | 12.1 |
| Summer 5 | 1 | 0.2 | 24.5 | 12.5 |
| Summer 1 | 1 | 0.1 | 2.8 | 1.4 |
| *control, pooled* | | | | *0.05* |

Winter 5 and Desert 3 are far outside support while the car is still driving.
Summer 5 and Summer 1 leave support as the attempt deteriorates, which is
**compatible with the distance being a consequence of the failure rather than
its cause** -- a car against a wall reads unlike anything in training. The
magnitude and the ordering across tracks both stand: Summer 1, which got
furthest at 23.5% of its lap, is the least out of distribution.

## What the measurement cannot see, and what nearly broke it

**12 of the 62 features cannot register a violation at all.** `airborne` and
the four wheel-contact flags are 0/1 and training contains both values; the
nine rays are distances normalised into `[0.0833, 1]` and training spans the
whole interval. A bounded feature saturated in training lies outside no band by
construction, which is why `rays` reads 0% on every track in the table. The
test has eyes only for the unbounded features.

**95% of the violations are `lookahead`** -- and the lookahead points and
`lateral_offset` are measured *against the reference line*, which the two
libraries get from different places. A generated track's line is its own
centreline; an official track's is a human lap, and a racing line cuts corners.
So most of that 147x was, on the face of it, dismissible as
racing-line-against-centreline rather than road-against-road.

`scripts/measure_line_confound.py` settles it by changing one variable. Take 12
tracks the policy **trained on**, keep the road exactly as it is, and swap only
the line -- from the generator's centreline to a line derived from a lap the
policy itself drove, built the same way every official track's line was.

| line on the same 12 roads | frames with any OOS | features per frame |
|---|---|---|
| centreline (as trained) | 1 % | **0.03** |
| derived from a driven lap | 20 % | **0.77** |
| *official tracks, for scale* | *62 %* | *9.2* |

Line provenance is **real but small: it accounts for at most 8%** of the
official-track distance. The remaining ~92% is the road. Two unconfoundable
features agree, since nothing about where a line runs makes a car roll or
fall: `vel_up` reaches **4.4x** beyond the training band on official tracks and
`angvel_z` **10x** -- elevation and roll, which a flat grid of right-angle
corners cannot produce.

A two-track pilot of this said 1%, and was wrong, because relining does nothing
on some roads (Generated 0, 1) and a great deal on others (Generated 7 and 11
reach 4.4 and 3.2 features per frame). Twelve tracks is the smallest honest
version of it.

## What the measurement licenses

* The 0/4 is **distribution distance, quantified** -- not a learning failure
  and no longer an inference.
* It is **not a near miss to polish**. The policy is outside its training
  support before it moves.
* Closing it is a **generator** problem, and the measurement names the
  features to aim at: elevation (`vel_up`) and roll (`angvel_z`) first.
* It says nothing about whether the *representation* transfers. That needs a
  fine-tune against a from-scratch control, which has not been run.

Artifacts in `support/`: `pass1.log`, `pass2.log`, `support_band.csv` (the band
itself, reusable via `--band` so re-analysis costs minutes rather than the 30
the 48 training laps take), `groups_*.csv`, `line_confound.log` and
`line_confound.csv`.

## The policy, and the laps it drove

`agent.pt` is the 1.22M-step checkpoint behind every figure in the long-run
table and all of the support measurement. `laps/` holds the 48 best laps it
drove on its training tracks, which is what `measure_line_confound.py` relines
to build each road's twin.

Both are committed, unlike everything in `runs/`, because a checkpoint is 28.5
hours of compute on one machine and without it none of this reproduces. They
pass the test the lap recordings pass -- **no game content**: 20 weight tensors
for a 62->256->256->9 MLP, Adam state, an `IQNConfig` of hyperparameters and a
step count, with no track codes, geometry or reference lines anywhere in it.

Both measurement scripts default to these paths, so from a clone:

```bash
python scripts/measure_support.py          # needs tracks*.npz + geometry.npz
python scripts/measure_line_confound.py    # reuses the committed band
```

Still missing for a stranger: `geometry.npz` and the three `tracks*.npz`
libraries, which embed the game's content and are rebuildable with
`tools/capture_geometry.py`, `scripts/build_library.py` and
`scripts/build_generated_library.py`.

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

* **The long run is one seed.** The 300k result has three and carries error
  bars; every figure in the 1.22M table above is a single sample, including the
  0/4 -- though the support measurement that explains it was run twice.
* **No transfer result.** Zero-shot failing says nothing either way about
  whether the learned representation helps when fine-tuned on an official
  track. Nobody has run that against a from-scratch control.
* **No learned baseline.** The scripted driver is a floor, not a competitor.
  Nothing here is measured against another algorithm.
* **Par is a convention**, not a standard, so `best/par` compares runs within
  this project and nothing outside it.
* **The distribution is narrow** -- five part types, flat ground -- which is
  exactly what the zero-shot row measures the cost of.
