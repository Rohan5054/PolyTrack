# A Frame-Exact Reinforcement Learning Environment from a Shipped Game Binary, with Procedural Track Generation

**Draft. Not submitted anywhere.**

> **How to read this file.** Every section is marked with its status. Sections
> marked **WRITTEN** are backed by measurements in this repository and can be
> read as a draft. Sections marked **GAP** describe experiments that have not
> been run; they contain no results, only what would have to be done. Nothing in
> this document reports a number that was not produced by code in this repo.
>
> Citations come from `Literature_Review.md` and **have not been re-verified
> against the primary sources**. Check every one before submission.

---

## Abstract

> **STATUS: WRITTEN. Every figure in it is measured and in the repo; no claim
> here depends on a GAP section any more.**

Reinforcement learning environments for racing are usually built by
reimplementing a game's dynamics, which makes fidelity to the original
unverifiable and typically unverified. We take the opposite approach for
PolyTrack, a commercial browser racing game: we host the game's own compiled
WebAssembly physics module directly, and recover its collision geometry by
intercepting the initialisation message the shipped client sends to its own
simulation worker, rather than reconstructing it. The result is validated to
**frame-exactness** against an external oracle — eight human runs recorded by
the game itself replay in our host and land on their recorded finish frame to
the millisecond, the longest a 61,428-frame run — and separately against the
game's own replay verifier, which we execute unmodified and which agrees with
every lap our agents produce. We further contribute a procedural track
generator which emits valid track codes the game loads, and which returns each
track's route by construction, removing the need for a human lap before a new
track can be trained on. Three independently seeded policies trained on 48
generated tracks each finish all 16 of a held-out set, at **97.9 ± 0.6 %** of
attempts, and a longer run shows **no measurable generalisation gap** — 94 % on
unseen tracks against 92 % on its own training tracks. The same policies
transfer to none of the game's four official tracks, and we quantify why rather
than asserting it: measured against the 0.1st–99.9th percentile band of every
observation feature the policy saw in training, unseen *generated* tracks fall
inside the band while the official tracks sit **147–154× further outside it**,
already out of support at the first frame before the car moves. We isolate the
obvious confound — that a generated track's reference line is its own
centreline while an official track's comes from a human lap — by re-driving
training roads with only the line's provenance swapped, which accounts for at
most 8 % of that distance. We also report a learned policy and a policy-guided
beam search on one official track, where the combination is 239 ms faster than
the human reference lap.

---

## 1. Introduction

> **STATUS: WRITTEN.**

Simulated racing is a well-studied reinforcement learning domain, and the
strongest results — GT Sophy's superhuman wheel-to-wheel racing [Wurman et al.,
2022], its time-trial predecessor [Fuchs et al., 2021], and Linesight's
world-record Trackmania laps — were all obtained inside the real game, with
access to its true dynamics. Research environments that do not have that access
reimplement the dynamics instead, and a reimplementation's fidelity is rarely
measured at all.

This is a real methodological gap rather than a cosmetic one. A racing policy
optimises a trajectory to the limit of the dynamics it was trained against; if
those dynamics differ from the game's, the resulting lap is not a lap of that
game. We give a concrete instance from our own work in §3.4: a collision mesh
wrong by 1.8 × 10⁻⁴ was invisible, drivable at 308 km/h, and still threw a
400 km/h wall-ride clean off the track thirteen seconds later — on one of four
tracks and not the other three.

We make four contributions.

**(1) A method for building a faithful environment from a shipped game.** We
host the game's compiled physics module from Python, capture rather than
reimplement its collision geometry, and validate the result to frame-exactness
against two independent oracles (§3).

**(2) A procedural track generator** that writes track codes the game itself
loads and that yields each track's route by construction, removing the
reference-lap bootstrap that otherwise gates every new track (§4).

**(3) An agent and search stack** built on the above: policies that generalise
across procedurally generated tracks with error bars over three training seeds,
a policy-guided search that beats the human reference lap on an official track,
and an explicit account of what each does and does not establish (§5, §6).

**(4) A protocol for telling out-of-distribution transfer failure apart from
learning failure**, and for separating it from the confounds of the
environment's own construction (§6.2). Our policies finish 100 % of unseen
tracks from their training distribution and 0 % of the game's official ones.
Attributing that to distributional distance is a claim about the observation
space, so we measure it there — against a percentile band of what the policy
actually saw — and make it interpretable with a control arm of unseen
in-distribution tracks, without which any such count is unfalsifiable. We then
isolate the one asymmetry in how the two track sets are built. The protocol is
not specific to racing; it applies wherever an agent is evaluated on tasks a
generator did not produce.

---

## 2. Related work

> **STATUS: WRITTEN. Citations need verification.**

**Racing RL.** GT Sophy [Wurman et al., *Nature* 602:223–228, 2022] reached
superhuman wheel-to-wheel racing in Gran Turismo Sport; its time-trial
predecessor [Fuchs et al., *IEEE RA-L* 6(3):4257–4264, 2021; arXiv:2008.07971]
established the template this work follows most closely: a dense
course-progress proxy reward rather than sparse finish time, observations in the
car-local frame, and explicit lookahead of upcoming geometry. Vision-based
successors [arXiv:2406.12563; arXiv:2504.09021] show pixels are possible but
harder, which argues for structured state when the simulator can provide it.

**Trackmania.** Linesight (pb4git & Agade09) is the strongest public agent,
using IQN over discrete keyboard inputs with a reference-line reward, and became
the first to beat world records on official campaign tracks. TMRL is the most
reusable open framework (SAC/REDQ, LIDAR observations). Both operate the real
game; neither offers a reusable, fidelity-validated environment to third
parties, which is the gap this work addresses.

**Racing benchmarks.** OfflineMania [Macaluso et al., arXiv:2407.09415] is a
Trackmania-inspired Unity environment with a Gymnasium interface, built for
offline RL. F1TENTH [O'Kelly et al., NeurIPS 2019] is the main scaled-autonomy
benchmark; its survey [Evans et al., arXiv:2402.18558] finds that
optimisation-and-tracking pipelines achieve the fastest laps, ahead of
end-to-end RL — a result our §6 is consistent with. These environments are
purpose-built simulators; none claims frame-exact agreement with a shipped
commercial title, because none is one.

**Algorithms.** We use IQN [Dabney et al., arXiv:1806.06923] with prioritised
replay and n-step returns in a multi-collector, single-learner arrangement,
following Ape-X and the designs used by Linesight and Fuchs et al.

**Generalisation.** Procedural content generation substantially improves
generalisation in RL [Justesen et al., arXiv:1806.10729]. PolyTrack's portable
track codes make an effectively unbounded track distribution available, which
§4 exploits.

**Learned policy plus test-time search.** Pairing a fast learned policy with
test-time search is the pattern behind AlphaZero and, recently, superhuman
Stratego [Sokota et al., arXiv:2511.07312]. §5.2 applies it here.

---

## 3. The environment

> **STATUS: WRITTEN. This is the contribution that is finished.**

### 3.1 Hosting the game's own physics

PolyTrack moved its physics to WebAssembly in v0.4.0 and rewrote most of it in
C++ in v0.6.0. We load the shipped `polytrack_physics.wasm` module in Python
through `wasmtime`. All nine of its imports are ordinary Emscripten hooks; only
heap resizing needs a real implementation. There is no browser, no Node process
and no inter-process communication in the loop.

Exported functions are called through the raw C API rather than the Python
wrapper, which boxes every argument: approximately 4 µs against 100 µs for a
seven-argument call. At 1000 calls per simulated second this is the difference
between a usable environment and an unusable one.

**The physics runs at 1000 Hz: one frame is one millisecond.** Three independent
confirmations appear in the shipped client — the worker drains its accumulator
in 0.001 s steps, `maxFrames` is 5,999,999 (99:59.999 in milliseconds), and the
time formatter divides by 60000 and 1000. A frame count is therefore a count of
milliseconds, and 1000 decisions per second is not a usable action space: agents
hold one action across a control interval, 50 ms by default.

### 3.2 Capturing rather than reimplementing geometry

The one thing not described in the client's JavaScript is the collision
geometry, which is compressed inside the game's 3D models. We serve a local copy
of the client over loopback with a hook ahead of its deferred bundles and
intercept the initialisation message it sends to its own simulation worker,
which carries every part's collision vertices verbatim.

We also reimplemented this pipeline in the same 3D library the game uses, as a
cross-check, and got 185 of 186 parts bit-identical. The one part it missed cost
more engineering time than everything else in this stage combined, and its
consequence is §3.4. **Capture is the primary path; the reimplementation is
retained only for its checksum verification**, which is what localised the
failure.

### 3.3 Validation

We validate against two oracles that fail in different ways.

**The acceptance gate (external oracle).** The game stores, for each personal
best, both the input recording and the finish frame it recorded. Replaying such
a recording and reproducing its finish frame *exactly* is a test the game
implements itself. We mirror that loop.

> **The gate passes on 8 of 8 runs**, across four tracks (Summer 1, Summer 5,
> Desert 3, Winter 5), each landing on its recorded finish frame to the
> millisecond, the longest a 61,428-frame run.

Exactness matters: a simulator 0.01 % off finishes a 40-second run four
milliseconds early, which is larger than the margins these agents compete over.

**The client's own verifier (code oracle).** The gate validates *our* host. For
a lap our search produced there is no external finish frame to check against —
the target is our own claim. We therefore execute the shipped simulation worker
unmodified under Node and send it the same initialisation and verification
messages the game sends itself. It decodes the track with its own decoder and
deserialises the recording with its own codec.

> **The client's verifier agrees with all three banked laps** (19025, 19422,
> 18786).

This independently exercises our track-code decoder, our recording codec and our
call sequence against the game's own code. It does **not** independently confirm
the physics: both sides run the same module, hash-identical. The two oracles are
complementary — the gate says the module reproduces reality; the verifier says
we drive the module the way the game does.

### 3.4 A measurement worth isolating

The one collision mesh our reimplementation got wrong was wrong by
1.8 × 10⁻⁴. It was invisible, and the car drove over it at 308 km/h without
incident. It still threw a 400 km/h wall-ride off the track thirteen seconds
later, and it broke exactly one of the four tracks we had oracles for.

**The size of a geometry error is no guide to the size of its consequences**, and
no amount of inspection substitutes for an exact replay test. We consider this
the strongest single argument for the methodology in §3.1–3.3.

### 3.5 What the agent is allowed to perceive

An environment can be faithful in its dynamics and still lie in its
observations. Ours did. The observation includes rangefinder-style probes, after
Fuchs et al., and these originally measured distance from the *reference line*,
treating anything beyond a nominal track width as a wall. That answers a
different question from the one it appears to: where a driven lap hugged one
side of the road it reports a wall on open tarmac, and on a procedurally
generated track, whose reference line is its own centreline, it is close to
uninformative.

We now rasterise the collision geometry into a grid in which each cell holds a
vertical occupancy bitmask, and march rays through it. Validation is that a real
human lap must be on ground for its entire length — a test that passes only if
the part transform, the rasterisation and the height query are simultaneously
correct, and which rejected three plausible designs:

| model | fraction of a real lap reported off-ground |
|---|---|
| keep the N highest surfaces per cell | 33 % |
| count only upward-facing surfaces | 14 % |
| symmetric height tolerance | 15 % |

The second is worth stating as a finding about the game rather than a bug in our
code: **the car drives on walls.** The human record wall-rides at 400 km/h along
pieces whose supporting faces are vertical, so "drivable surface" cannot be
defined as "upward-facing surface". A benchmark that filtered on surface normals
— the obvious choice — would have silently misreported a seventh of that lap.

Cost is 1.9 s to build a track and 109 µs for nine rays, against a 50 ms control
interval.

### 3.6 Cost

The module costs **650–800 µs per 1 ms tick** on an unloaded machine, nearly
independent of track size, so it is the module's own fixed work. That is roughly
real time, making parallel collectors mandatory: one collector yields about
**11 transitions per second** at a 50 ms control interval, and six give 50–80.

Whole-heap state capture costs **13.9 ms**, against 14.4 s to replay a
19-second lap from frame zero — a factor of a thousand, and what makes anchored
search (§5.2) affordable. This figure had been recorded in our own notes as
"far too slow for search" for months without ever being measured.

---

## 4. Procedural track generation

> **STATUS: WRITTEN. The generator works and policies have been trained on it
> (§6.1); its narrowness is what §6.2 measures the cost of.**

A track code is a compressed list of placed parts. We implement an **encoder**
as well as a decoder, validated three ways: every shipped track that decodes at
all (31 of 31) re-encodes to identical parts, environment and sun direction; the
client's own worker loads a track rebuilt from parsed parts and all three banked
laps still verify frame-exactly on it; and unit tests cover the edge cases.

The generator lays out self-avoiding walks on the game's grid, placing a start,
straights, right-angle corners, checkpoints and a finish. Because the walk *is*
the route, a generated track carries its own reference line by construction.
**This removes the reference-lap bootstrap**: previously, deriving a route meant
replaying a lap a human had driven, so no new track could be trained on until
somebody drove it.

> **24 of 24 generated tracks with corners are drivable.** A scripted
> pure-pursuit driver finished 23 of 24 at its default corner speed, in
> 36–47 s, passing every checkpoint; the 24th finished at a lower corner speed.
> Each track had 30 pieces and 4–8 right-angle corners.

### 4.1 Two findings from building it

**Part orientation is not what the start transform implies.** The transform that
places the car composes the part rotation with a half turn about the vertical
axis; parts themselves are placed without it. A straight piece is symmetric
under a half turn, so **every check built from straight-only tracks passed under
the wrong model** — including full end-to-end drives to the finish. Only corners
expose it. We resolved it by building one corner at all four rotations and
driving each: exactly the two the corrected model predicts were drivable.

**A check can measure the wrong thing and look like proof.** An adjacency test
reported "0 broken connections" on a track no car could drive. It measured how
far each piece's geometry *reached* toward its neighbour — and a barrier reaches
a boundary exactly as well as an opening does. Its replacement keeps only
upward-facing surfaces and asks where road actually exists.

### 4.2 Limitations of the generator

The vocabulary is narrow: flat ground, one corner type, and at least one
straight between corners by default (two same-handed corners in succession form
a 180° hairpin inside 40 m that the scripted driver cannot take). Generated
tracks are *drivable*, which says the geometry is sound; it says nothing about
whether they are interesting to race.

---

## 5. Agent and search

> **STATUS: WRITTEN for method. Results are §6.**

### 5.1 Policy

IQN [Dabney et al.] over nine discrete actions, with prioritised replay, 5-step
returns, and several collector processes feeding one learner. The observation is
a 62-float car-local vector with speed-scaled lookahead of upcoming geometry,
following Fuchs et al. and GT Sophy. Distributional value estimation is
well-matched to racing: committing to a corner at speed either works or puts the
car into a wall, and the mean of those two futures describes neither.

### 5.2 Search

A beam search over action blocks, using the policy as an **action prior** (only
its top-*k* actions are expanded, reducing 9⁵ = 59,049 futures five blocks ahead
to 243 the policy considers drivable), anchored part-way through a known lap, and
carrying that lap as a **guide** at every depth so the result is bounded below by
its seed.

Candidates are scored on **progress at equal depth, never on shaped reward**.
Every candidate at a given depth has burned the same number of frames, so
"furthest along the track" *is* "fastest to here", which needs no coefficients
and admits none of the exploits a shaped reward invites.

### 5.3 A negative result on reward design

Our first reward charged a per-second time penalty and forfeited part of the
un-driven lap on failure. Twelve minutes of training broke it: with the episode
over, a failure costs a fixed amount whenever it arrives while staying alive
keeps paying the clock, so **the fastest possible failure wins**. The agent
learned to drive sideways off the start line — the failure histogram moved from
74 % stalled to 88 % off-track while mean return improved and progress fell to
zero.

The obvious repair, charging the full time budget on failure, inverts the exploit
rather than removing it: discounting shrinks any large terminal penalty, so
stalling before failing beats failing at once, and the agent learns to sit still.

With every reward term non-negative, a terminal state is worth zero and
continuing is worth at least zero, so quitting is never strictly better for any
policy, any discount, anywhere on the track. Time pressure then has to come from
the finish bonus, scaled by par over elapsed. We state this as a design rule:
**a negative step reward makes ending the episode attractive; a large terminal
penalty makes delaying it attractive.** This is a known hazard but we have not
seen it stated in this form for racing, where it is particularly sharp because
failure is always one steering input away.

---

## 6. Results

> **STATUS: WRITTEN. Three seeds on the headline result, one seed on the long
> run and on the single-track search. Read §7 alongside.**

### 6.1 Generalisation to unseen tracks

Three IQN policies were trained from independent seeds, 300,000 gradient steps
each (~3.2 h, six collectors, full nine-action space), on 48 procedurally
generated tracks. Each was evaluated with five greedy attempts — greedy is not
deterministic, since IQN samples fresh quantile fractions each forward pass —
on 16 tracks generated from disjoint seeds. 240 held-out episodes in total.

| seed | tracks finished ≥ once | mean finish rate | median best/par |
|---|---|---|---|
| 1 | 16/16 | 99 % | 1.39 |
| 2 | 16/16 | 98 % | 1.46 |
| 3 | 16/16 | 98 % | 1.44 |
| **across seeds** | **100 ± 0 %** | **97.9 ± 0.6 %** | **1.43 ± 0.03** |

Spread is half the range. **One track accounts for every miss**: `Generated
1011` was finished on four attempts of five by all three seeds — identically
80 %, three times — and every other track was finished every time by every
seed. That is a property of that track, not of training, and it is what a
pooled average conceals.

A fourth policy trained to **1,220,000 steps over 28.5 hours** was measured
first, on the same held-out library, and also against 16 of its training
tracks:

| | tracks | finished ≥ once | mean finish rate | median best/par | beat the driver |
|---|---|---|---|---|---|
| held-out | 16 | 16/16 | 94 % | 1.35 | 16/16 |
| training | 16 | 16/16 | 92 % | 1.36 | 16/16 |
| zero-shot, official | 4 | **0/4** | **0 %** | — | — |

**There is no measurable generalisation gap**: 94 % held-out against 92 % on
the tracks it trained on, and 1.35 against 1.36 on time. What that rules out is
memorisation of 48 layouts.

Four times the compute did not buy reliability — 94 % against the seeds'
97.9 % — and that run's own curve agrees, falling from 93 % at 987,000 steps to
87 % at 1.22 M. It bought pace: 1.35× par against 1.43×. Against the scripted
baseline it is **1.55× faster** on held-out tracks (median, range 1.37–1.73).

Most of the learning is early: 82 % finish rate by 257,000 steps and roughly
flat thereafter.

**The zero-shot row is the honest limitation**, and it is not a near miss. The
policy reaches 3.1 %, 2.9 %, 23.5 % and 1.5 % of the four official tracks
before stalling — three of them inside 60 m, two or three track pieces from the
line. A policy completing most of a lap would imply a distribution close enough
to close by polishing; this does not.

The generator uses five of the game's 186 part types — a start, a straight, one right-angle corner, a
checkpoint, a finish — on flat ground, with no jumps, banking, elevation or
walls. The official tracks have all of those; the human record on one of them
wall-rides at 400 km/h along vertical faces. So this row measures the distance
between our generated distribution and human-designed levels, which is large,
and not a failure to generalise within a distribution, which did not occur.
Closing it is a generator problem rather than a learning one. §6.2 replaces
that sentence, which was an inference, with a measurement.

### 6.2 The zero-shot gap, measured

A claim of "distribution distance" is cheap, and distinguishing it from a
learning failure requires measuring the distance in the space the policy
actually consumes: its observation vector. We take the band between the 0.1st
and 99.9th percentile of each of the 62 features, over the 24,545 frames the
1.22 M-step policy generated while greedily driving its 48 training tracks, and
count features outside that band elsewhere. Percentiles rather than extrema,
because a single outlying frame in twenty-five thousand would otherwise widen a
feature's band enough to swallow anything.

| | frames with ≥ 1 feature outside | features outside, of 62 |
|---|---|---|
| **control**: 16 unseen *generated* tracks (16/16 finished) | **1 %** | **0.0–0.1** |
| **test**: 4 official tracks (0/4 finished) | **57–62 %** | **7.0–9.2** |

**The control carries the result.** An out-of-support count is uninterpretable
alone: a band estimated too tightly produces one for every track, including
those the policy drives perfectly. Tracks the policy has never seen but which
come from the training distribution sit inside the band, so the band is not too
tight, and the official tracks are **147–154× further outside it**. Ranges span
two passes of the same measurement; greedy action selection is not
deterministic under IQN, which resamples its quantile fractions each forward
pass, so the measurement itself carries roughly ±15 %.

**The observation is out of support at the first frame of all four official
tracks** — arc 0.0 m, car stationary on the start line, before any action is
taken. The failure is not a policy driving competently into progressively
stranger geometry; it is a policy handed inputs unlike anything it trained on
from the outset. Consistently, the track it got furthest on (23.5 % of the lap)
is also the least out of distribution, at 1.4–1.7 features per frame against
7.4–12.5 for the other three.

**Isolating the reference-line confound.** 95 % of the violations fall on the
lookahead features, and the lookahead points and lateral offset are measured
*against the reference line* — which the two libraries obtain differently: a
generated track's line is its own centreline, an official track's is derived
from a human lap, and a racing line cuts corners. On its face, then, most of
the effect is dismissible as racing-line-versus-centreline rather than
road-versus-road. We test it by changing one variable: 12 tracks the policy
trained on, driven again on the identical road with only the line swapped for
one derived from a lap the policy itself drove, built exactly as every official
track's line is.

| line, same 12 roads | frames with ≥ 1 outside | features outside |
|---|---|---|
| centreline (as trained) | 1 % | **0.03** |
| derived from a driven lap | 20 % | **0.77** |
| *official tracks, for scale* | *62 %* | *9.2* |

Line provenance is a real effect and a small one: **at most 8 %** of the
official-track distance, leaving the road to account for the rest. Two features
that cannot be confounded agree, since nothing about where a line runs makes a
car roll or fall — vertical velocity reaches **4.4×** beyond the training band
on official tracks and roll rate **10×**. Those are elevation and banking,
which a flat grid of right-angle corners cannot express, and they name the
features a wider generator should target first.

A two-track pilot of this test reported 1 % and was wrong, because relining
changes nothing on some roads and a great deal on others — two of the twelve
reach 3.2 and 4.4 features per frame. We report the twelve-track figure.

**What the measurement cannot see.** 12 of the 62 features are bounded and
saturated in training — an airborne flag, four wheel-contact flags, and nine
ray distances normalised into [0.083, 1] whose full range occurs in training —
so they lie outside no band by construction, and the test is blind to them. Any
distance it reports is therefore a lower bound on the true distributional
distance.

### 6.3 Search on a single official track

All figures are gate-verified and confirmed by the client's own verifier.

| | Winter 5 | verified by |
|---|---|---|
| Human reference lap | 19.025 s | the game's own storage (external oracle) |
| Policy alone, best of 20 greedy laps | 19.422 s | gate + client verifier |
| **Policy + guided beam search** | **18.786 s** | gate + client verifier |
| Continuous trajectory optimisation | did not finish, 1089 m of 1747 | — |

The policy was trained on this track with a forward-only action space (justified
per-track: the human reference lap holds throttle for 100.0 % of its frames and
never brakes). Two details belong in the record. The 19.422 laps were taken at
717,248 gradient steps, from a checkpoint that **no longer exists** — training
continued in the same file, which now holds 801,472 steps, so those twenty
greedy laps cannot be re-drawn from the same network, though the banked lap
itself still reproduces. And 860,000 steps is the furthest any run reached, on a
*different* reference line; it produced no banked lap and is not the checkpoint
behind any figure above.

Budgets, in hardware-independent units:

| | rollouts | simulated frames |
|---|---|---|
| The run that produced 18.786 | 18,309 | 914,944 |
| All logged searches in the project | 186,387 | 9,897,112 |
| Continuous optimisation, all runs | 10,111 | 17,794,080 |

Three observations we consider more informative than the lap time:

**Search contributes everything.** The policy alone is *slower* than the human
reference. The entire margin comes from test-time search, consistent with the
F1TENTH survey's finding that optimisation-and-tracking beats end-to-end RL on
pure lap time.

**The search was steered by a human racing line.** The reference line on this
track *is* the human lap, and the ranking term that charges candidates for
distance from it found 459 ms in a single run where seven runs ranking on arc
alone had found 14 ms in total. The control sequence is the agent's own and
drives that route 239 ms faster than the human did, but the *route* was not
discovered independently.

**Continuous optimisation failed on this track, and the failure is informative.**
A separable cross-entropy method over a per-block steering relaxation, seeded
from the human lap, never finished, for nineteen times the budget of the search
that did. Projecting the human lap onto a fixed block grid and replaying it
reaches 26 m at 50 ms resolution, 349 m at 25 ms, 477 m at 10 ms and 468 m at
5 ms, of 1747 m total; zero timing error replays exactly. At a two-block spline
knot spacing, 97.8 % of the actions are unchanged and the car still reaches only
72 m. **This track is chaotically sensitive to millisecond input timing**, which
constrains what any trajectory parameterisation can represent.

---

## 7. What this does not show

> **STATUS: WRITTEN. Keep this section in any submitted version.**

- **Three seeds, one configuration.** §6.1 repeats the training run three
  times, which gives a spread on the headline figure. It does not vary
  anything else — one architecture, one reward, one set of hyperparameters —
  so nothing here is an ablation.
- **The single-track search results are still n = 1.** §6.3 was not repeated.
- **The support measurement is descriptive, not causal.** §6.2 establishes that
  the official-track observation lies far outside the training band and that
  reference-line provenance does not explain it. It does not demonstrate that
  the distance *causes* the failure: a policy could in principle be robust to
  inputs outside its training support. The experiment that would close this is
  a wider generator (§8.1), where reducing the measured distance should
  recover performance if the mechanism is what we claim.
- **12 of the 62 observation features cannot register a violation**, being
  bounded and saturated in training, so every distance in §6.2 is a lower
  bound.
- **No transfer result.** That zero-shot fails says nothing either way about
  whether the learned representation *helps* when fine-tuned on an official
  track. That requires a matched from-scratch control and has not been run
  (§8.5).
- **No learned baseline.** The scripted driver is a floor, not a competitor.
  Nothing here is measured against Linesight, TMRL, GT Sophy or any standard
  RL algorithm on a common task.
- **Par is a convention, not a standard.** On generated tracks it is the
  scripted driver's time halved, so `best/par` compares runs inside this paper
  and means nothing outside it.
- **The generated distribution is narrow.** Five part types on flat ground;
  §6.1's zero-shot row measures what that costs and §6.2 quantifies it.
- **The three approaches on the official track** differ along several axes at
  once and are not a controlled comparison; two share an author, and the third
  was informed by the second's findings.
- **The policy does not beat a human.** It is 397 ms slower than the reference
  lap; only policy-plus-search is faster.
- **The environment is validated on four tracks.** Eight runs is a strong gate
  but a small corpus.

---

## 8. Experiments still required

> **STATUS: GAP, except where a subsection says otherwise. §8.2 is partly
> done; nothing in §8.3–§8.6 has been run and no results exist for them.**

### 8.1 Widen the generated distribution

The generator uses five of 186 part types on flat ground, and the official
tracks need jumps, banking, elevation and wall-rides. Each addition is a fresh
geometry problem of the kind §4.1 describes — and elevation in particular turns
the self-avoiding walk from a 2-D problem into a 3-D one.

§6.2 makes this both targeted and falsifiable, which it was not before. It
names the two features furthest outside the training band that no confound
explains — vertical velocity at 4.4× and roll rate at 10× — so elevation and
banking come first. And it supplies the intermediate measurement: adding a part
family should *reduce* the measured out-of-support distance on the official
tracks, and that can be read off without training a policy to convergence
first. If distance falls and zero-shot performance does not follow, the
mechanism claimed in §6.2 is wrong, which is the outcome worth being able to
detect.

### 8.2 Seeds and variance

**Partly done.** §6.1's headline has three independent training seeds and
carries a spread; §6.2's support measurement was run twice. The long 1.22 M-step
run, every figure in §6.3, and the input-timing sensitivity results remain
single samples. What is still wanted is 3–5 seeds on *each* configuration,
including the search.

### 8.3 Baselines

At minimum: the scripted pure-pursuit driver as a floor (implemented, §4), and
one standard algorithm besides our IQN. A comparison against Linesight's method
on a common track would be stronger still.

### 8.4 Ablations

The observation's lookahead horizon; the action-space restriction; the search's
ranking terms; control interval.

### 8.5 Does the representation transfer?

The cheapest experiment that would convert §6.1's negative row into a positive
claim, and it is not run. Fine-tune the generated-track policy on one official
track against a from-scratch control matched on every other axis, and compare
**gradient steps to the first completed lap**. "Zero-shot fails, but
pretraining cuts time-to-first-lap by *k*×" is a more useful statement than a
clean zero-shot success would have been, because it makes the generated
distribution a usable curriculum rather than a dead end. A null result is
informative and a negative one more so.

**Summer 1 is the right track**, not Winter 5. §6.2 measures it as the least
out of distribution of the four, 1.4–1.7 features per frame against 7.4–12.5;
the policy already reaches 23.5 % of its lap zero-shot, so there is a gradient
to climb rather than a cliff; and at 1301 m it is among the shortest. Winter 5,
where all the single-track infrastructure lives, is the worst available choice
on the first two counts, and its existing 801 k-step policy cannot serve as the
control in any case, having been trained with a restricted action space.

Two things are matched without any work, which is worth stating because both
would otherwise be confounds. The training budget is relative to the resumed
checkpoint's step count, so both arms take the same number of new gradient
steps. And exploration follows Ape-X — each collector holds a fixed epsilon
rather than decaying one over the run — so a resumed arm explores exactly as a
fresh one does, where a step-indexed schedule would have handed the pretrained
arm a quieter policy at the moment of measurement.

One thing is not matched by default: a checkpoint carries Adam's moment
estimates, so a resumed arm and a fresh one differ in weights *and* in
effective step size, over precisely the early steps the metric is made of.
The primary comparison should therefore load weights only
(`train.py --reset-optimiser`), with carrying the optimiser state as a separate
arm, since that is what a practitioner would do.

Three seeds per arm. At the throughput in §8.6 a 150,000-step budget is about
3.5 h, so the result is six runs and roughly 21 h; a single-seed pilot at
50,000 steps costs 2.5 h and establishes whether the effect is large enough to
be worth the rest. Secondary measures: steps to a sustained 50 % finish rate,
final best lap time, and the shape of the failure histogram over time, where a
`stalled` → `off_line` transition marks a policy that has learned to leave the
line before it has learned to hold the track.

### 8.6 Compute

The binding constraint. At 50–80 transitions per second from six collectors on a
laptop, tens of millions of transitions is days to weeks of wall-clock. The
workload is CPU-bound and parallel, so it scales nearly linearly with cores; a
GPU does not help, as the network is a small MLP.

---

## 9. Reproducibility and licensing

> **STATUS: WRITTEN. Section 9.2 is an open problem.**

### 9.1 What is released

The environment, agent, search, generator and scripted driver, with **159
passing tests**, and every reported lap as an input recording that replays
through the gate. `scripts/verify_baselines.py` re-verifies every lap time in
§6.3 and hashes the inputs they were produced against.

The §6.2 measurement ships as `scripts/measure_support.py` (the band, the
control arm and the test) and `scripts/measure_line_confound.py` (the
reference-line arm), with both passes' raw logs, the percentile band itself and
the per-frame violation series under `arms/generalisation/support/`. The band
is a reusable artifact: re-analysis against a saved band takes minutes, where
re-deriving it costs 48 training laps.

Two caveats on reproducing §6 exactly. Greedy evaluation is not deterministic
under IQN, so finish rates and support distances reproduce to within a few per
cent rather than exactly — which is why §6.2 quotes two independent passes.
And the libraries the policies were trained and evaluated on embed the game's
content, so they are rebuildable from a seed with the scripts here but are not
themselves distributable (§9.2).

### 9.2 What cannot currently be released, and why it matters

The environment depends on the game's compiled physics module and on collision
geometry derived from it. Neither is redistributable. **A benchmark nobody else
can run is not a benchmark**, and we state this as an unresolved limitation
rather than a detail.

The game's author has confirmed that *using* the physics module directly is
acceptable, subject to the game's terms, and asked that the tooling not permit
cheating and not submit runs to the leaderboard — conditions the released code
satisfies by construction, not by promise (the replay viewer serves the client on
a throwaway origin and blocks requests to the game's servers at two levels).
**Redistribution has not been granted and is not addressed by the terms**, so
reproduction currently requires a copy of the game and an extraction step, which
is a weaker reproducibility story than a benchmark venue expects.

### 9.3 Ethics

All tool-assisted runs are kept off the game's human leaderboard, following the
convention Linesight states explicitly for Trackmania. No part of this work
interacts with the game's online services.

---

## 10. An engineering note

> **STATUS: WRITTEN. Probably an appendix, or cut.**

Throughput measurements on hybrid-core Windows machines can be wrong by a large
factor in a way that does not look like a throughput problem. A process the
operating system classifies as background work — anything launched detached — is
confined to the efficiency cores at their lowest clock. On our machine this cost
**52.6 s against 3.6 s** on a fixed single-threaded benchmark, and on the
training loop moved the achieved gradient-step-to-transition ratio from 0.11 to
0.46 and throughput from 38 to 75 transitions per second.

The symptom is deceptive: total CPU reads low, clocks read normal, and nothing
is thermally limited, which reads as a lock rather than a throttle. The tell is
that *everything* slows by the same factor at once. We mention it because
published throughput figures from laptop experiments may be affected by it
silently.

---

## References

> **STATUS: GAP — transcribed from `Literature_Review.md` and NOT verified
> against primary sources. Check every one, and fill in the ones marked
> incomplete, before this goes anywhere.**

- Wurman, P. R. et al. Outracing champion Gran Turismo drivers with deep
  reinforcement learning. *Nature* **602**, 223–228 (2022).
- Fuchs, F. et al. Super-Human Performance in Gran Turismo Sport Using Deep
  Reinforcement Learning. *IEEE RA-L* **6**(3), 4257–4264 (2021).
  arXiv:2008.07971.
- Dabney, W., Ostrovski, G., Silver, D., Munos, R. Implicit Quantile Networks
  for Distributional Reinforcement Learning. arXiv:1806.06923.
- Justesen, N. et al. Illuminating Generalization in Deep Reinforcement Learning
  through Procedural Level Generation. arXiv:1806.10729.
- Macaluso, G., Sestini, A., Bagdanov, A. OfflineMania: A Benchmark Environment
  for Offline RL. arXiv:2407.09415 (2024).
- O'Kelly, M. et al. F1TENTH: An Open-source Evaluation Environment for
  Continuous Control and Reinforcement Learning. NeurIPS 2019 competition track.
- Evans, B. et al. A Survey of Autonomous Racing. arXiv:2402.18558 (2024).
  *(title needs checking)*
- Sokota, S. et al. Superhuman AI for Stratego Using Self-Play RL and Test-Time
  Search. arXiv:2511.07312 (2025).
- Remonda, A. et al. Formula RL. arXiv:2104.11106 (2021).
- Horgan, D. et al. Distributed Prioritized Experience Replay (Ape-X). ICLR 2018.
  *(reference incomplete)*
- Schaul, T. et al. Prioritized Experience Replay. ICLR 2016.
  *(reference incomplete)*
- Linesight. pb4git & Agade09. github.com/Linesight-RL/linesight. *(software)*
- TMRL. *(software; reference incomplete)*
- PolyTrack. Kodub. kodub.com. *(game; version 0.6.2)*

---

## Appendix A: Figures and tables this draft would need

> **STATUS: GAP.**

1. Environment architecture: client → captured geometry → wasm host → Gymnasium
   interface, marking where each oracle attaches.
2. Gate results table: all eight runs, track, recorded frame, reproduced frame,
   drift (currently zero everywhere).
3. Generated track examples: a handful of layouts with their routes.
4. **Generalisation curve** — finish rate on held-out tracks against training
   transitions. The data exists (`arms/generalisation/learning_curve.csv`, 121
   points); the figure does not.
5. **Out-of-support distance** (§6.2) — features outside the training band per
   frame, against distance along the lap, one line per official track with the
   control's 16 held-out tracks as a shaded band at the bottom. The per-frame
   data is written per track by `scripts/measure_support.py`. This is the
   figure that makes the zero-shot row a finding rather than a disclosure.
6. **Violations by feature group**, control against official, with the
   reference-line confound arm beside it (§6.2).
7. Search ranking ablation on Winter 5.
8. Input-timing sensitivity: distance reached against grid resolution (§6.3).
