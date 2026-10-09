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

> **STATUS: WRITTEN, but the second half is contingent on the GAP sections.**

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
track can be trained on. We report a learned policy and a policy-guided beam
search on one track, where the combination is 239 ms faster than the human
reference lap. We do **not** yet report generalisation results.

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

We make three contributions.

**(1) A method for building a faithful environment from a shipped game.** We
host the game's compiled physics module from Python, capture rather than
reimplement its collision geometry, and validate the result to frame-exactness
against two independent oracles (§3).

**(2) A procedural track generator** that writes track codes the game itself
loads and that yields each track's route by construction, removing the
reference-lap bootstrap that otherwise gates every new track (§4).

**(3) An agent and search stack** built on the above, with results on one track
and an explicit account of what they do and do not establish (§5, §6).

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

> **STATUS: WRITTEN. The generator works; nothing has been trained on it (§7).**

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

> **STATUS: WRITTEN. One training seed throughout. Read §7 alongside.**

### 6.1 Generalisation to unseen tracks

A single IQN policy was trained for 1,220,000 gradient steps (28.5 hours, six
collectors) on 48 procedurally generated tracks, with the full nine-action
space. It was evaluated with five greedy attempts per track — greedy is not
deterministic, since IQN samples fresh quantile fractions each forward pass —
on 16 tracks generated from disjoint seeds, and for comparison on 16 of the
training tracks.

| | tracks | finished ≥ once | mean finish rate | median best/par | beat the driver |
|---|---|---|---|---|---|
| **held-out** | 16 | 16/16 | **94 %** | 1.35 | **16/16** |
| training | 16 | 16/16 | 92 % | 1.36 | 16/16 |
| zero-shot, official | 4 | **0/4** | **0 %** | — | — |

**There is no measurable generalisation gap.** 94 % against 92 %, and 1.35
against 1.36 on time. The held-out figure being marginally higher is noise; what
it rules out is memorisation of 48 layouts. Against the scripted baseline the
policy is **1.55× faster** on held-out tracks (median, range 1.37–1.73).

Most of the learning is early: 82 % finish rate by 257,000 steps and roughly
flat thereafter, with the greedy policy completing every track in the windows
around 735,000–987,000 steps. The final million gradient steps bought very
little.

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
Closing it is a generator problem rather than a learning one.

### 6.2 Search on a single official track

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

- **One training seed.** The generalisation result in §6.1 is a single run.
  No repeats, so no error bars on any figure in this paper.
- **No learned baseline.** The scripted driver is a floor, not a competitor.
  Nothing here is measured against Linesight, TMRL, GT Sophy or any standard
  RL algorithm on a common task.
- **Par is a convention, not a standard.** On generated tracks it is the
  scripted driver's time halved, so `best/par` compares runs inside this paper
  and means nothing outside it.
- **The generated distribution is narrow.** Five part types on flat ground, and
  §6.1's zero-shot row measures what that costs.
- **The three approaches on the official track** differ along several axes at
  once and are not a controlled comparison; two share an author, and the third
  was informed by the second's findings.
- **The policy does not beat a human.** It is 397 ms slower than the reference
  lap; only policy-plus-search is faster.
- **The environment is validated on four tracks.** Eight runs is a strong gate
  but a small corpus.

---

## 8. Experiments still required

> **STATUS: GAP. Nothing below has been run. No results exist for these.**

### 8.1 Widen the generated distribution

Done for the core claim; see §6.1. What remains is the zero-shot gap. The
generator uses five of 186 part types on flat ground, and the official tracks
need jumps, banking, elevation and wall-rides. Each addition is a fresh
geometry problem of the kind §4.1 describes, and the zero-shot row is the
measurement that says whether it worked.

### 8.2 Seeds and variance

3–5 training seeds per configuration, with error bars on every reported figure.

### 8.3 Baselines

At minimum: the scripted pure-pursuit driver as a floor (implemented, §4), and
one standard algorithm besides our IQN. A comparison against Linesight's method
on a common track would be stronger still.

### 8.4 Ablations

The observation's lookahead horizon; the action-space restriction; the search's
ranking terms; control interval.

### 8.5 Compute

The binding constraint. At 50–80 transitions per second from six collectors on a
laptop, tens of millions of transitions is days to weeks of wall-clock. The
workload is CPU-bound and parallel, so it scales nearly linearly with cores; a
GPU does not help, as the network is a small MLP.

---

## 9. Reproducibility and licensing

> **STATUS: WRITTEN. Section 9.2 is an open problem.**

### 9.1 What is released

The environment, agent, search, generator and scripted driver, with **147
passing tests**, and every reported lap as an input recording that replays
through the gate. `scripts/verify_baselines.py` re-verifies every figure in §6
and hashes the inputs they were produced against.

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
   transitions. *This figure does not exist and is the one the paper turns on.*
5. Search ranking ablation on Winter 5.
6. Input-timing sensitivity: distance reached against grid resolution (§6).
