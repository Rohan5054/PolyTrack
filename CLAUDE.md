# CLAUDE.md

Context for Claude Code working on this repo.

## What this is

An RL + trajectory-search agent for **PolyTrack** (Kodub's browser racing game).
Design: a learned policy for generalisation across tracks, test-time search for
the last tenths on any specific track. Derived from a literature review of
Trackmania and Gran Turismo racing agents — the short version is in
`README.md`, and the reasoning behind each choice is in the module docstrings.

## First thing to do

```bash
git init && git add -A && git commit -m "Initial import from chat session"
```

This repo was built in a chat sandbox with **no version control**, and that
caused a real problem: work from an earlier part of the session reappeared
without provenance, and a duplicate search implementation was written that
silently shadowed the first via Python's package-over-module resolution. Both
cost time to untangle. Under git neither is possible. Commit before you change
anything.

## Run it

```bash
pip install -e ".[learning]"    # torch is training-only; the rest runs without
python tests/test_core.py       # 24 tests
python tests/test_search.py     # 17 tests
python tests/test_trackcode.py  # 12 tests
python tests/test_wasm.py       # 19 tests
python tests/test_recording.py  # 14 tests
python tests/test_reference.py  # 18 tests
python tests/test_learning.py   # 33 tests
python tests/test_spline.py     #  6 tests
python tests/test_generator.py  # 12 tests
python tests/test_pursuit.py    #  8 tests
python tests/test_corridor.py   #  5 tests
python scripts/smoke_test.py
python scripts/search_demo.py
```

All 168 pass. Four of the suites reach for files that are not in the repo and
are stronger when they find them:

* `test_trackcode.py` decodes every `.track` the game ships —
  `POLYTRACK_TRACKS=path\to\tracks python tests/test_trackcode.py`.
* `test_wasm.py` drives the real physics module when `polytrack_physics.wasm`
  and `geometry.npz` are in the repo root (or `POLYTRACK_WASM` /
  `POLYTRACK_GEOMETRY` point at them). Neither file is committed — both are
  derived from Kodub's game. Without them those tests skip and say so, which
  means a green run does not by itself prove the backend was exercised.
* `test_reference.py` additionally wants `POLYTRACK_TRACKS`, since it builds a
  reference line from a real run in `records.json` and a recording identifies
  its track only by id.

* `test_learning.py` skips its torch half without torch. The replay and n-step
  tests run regardless.

`scripts/reference_demo.py` is the end-to-end demonstration: track code →
collision meshes → module → replayed trajectory → reference line → observation →
reward, on every record that matches an available track.

## Training

```bash
POLYTRACK_TRACKS=path\to\tracks python scripts/build_library.py   # once, ~3 min
python scripts/train.py --smoke                                   # 90 s, plumbing
python scripts/train.py --workers 6
python scripts/train.py --workers 6 --track "Winter 5" --forward-only
```

`--gamma` now defaults to `IQNConfig`'s **0.999**, the discount the agent was
designed around. **The banked results were trained at 0.99**, the flag's old
default, which always overrode the config's: the generalisation and Summer 1
checkpoints record it, and for the Winter 5 and 300k-seed runs, whose
checkpoints are gone, it is inferred from that default. Pass `--gamma 0.99` to
reproduce one. A resume keeps the checkpoint's discount unless `--gamma` is
given, and the collectors take theirs from `agent.config`, never the flag:
`learn` reads no gamma of its own, so the collectors' n-step discount is the
one the critic trains at. `CollectorConfig` has a gamma of its own defaulting
to 0.99, so `test_the_collectors_discount_is_the_agents` checks what
`collector_config_for` builds. `--lr` works the same way: unset means
`IQNConfig`'s, a resume keeps the checkpoint's unless it is given, and given,
it moves Adam's rate too.

`tracks.npz` is the cache that makes collectors viable: deriving a reference line
costs a full replay, so six workers rebuilding four lines each at launch would be
eighteen minutes of nothing. Like `geometry.npz` it embeds the game's content and
is gitignored.

Throughput is the whole story. One collector yields **~11 transitions/second** at
a 50 ms control interval, because the module runs at roughly half realtime and
~700 µs of its 1 ms tick is its own internal work. Six collectors give 50–80/s
aggregate. Budget hours for a first driving policy and days for anything that
generalises — and note that a 62-float MLP is nowhere near the bottleneck, which
is why CPU torch is the right build.

Three things are load-bearing on Windows and all of them have cost time
somewhere. Two are in `learning/collector.py`: processes start by **spawn** so
nothing is inherited and `WasmPhysics` can never cross a process boundary; and
torch grabs every core per process unless told otherwise, so
`set_num_threads(1)` in a worker is not a tuning nicety but the difference
between six workers and worse-than-one.

The third is `polytrack/priority.py`, and it is the one that does not look like
what it is. **Windows confines a detached process to the efficiency cores at
their lowest clock.** Anything launched in the background -- `nohup`, a
background shell, a scheduled task -- is tagged with background quality of
service ("EcoQoS"), and on a hybrid CPU that is not a small penalty:
**52.6 s against 3.6 s** on a fixed single-threaded benchmark, measured here.
Every entry point that runs long therefore calls `lift_background_qos()` first
-- `train.py`, `search_run.py`, and the collector worker.

The symptom is deceptive and cost an afternoon. A tagged training run showed
every python process pegging the four E-cores at 100% while the eight P-core
threads sat at 2-24%, **total CPU 29%**, clocks boosting normally at 110% of
base, no thermal cap, no paging. That reads as "the machine is three-quarters
idle, so the learner must be blocked on a lock", and it is not: the learner is
running flat out on a core eight times too slow. The tell is that *everything*
slows by the same factor at once -- collectors and learner together -- which a
lock does not do. Measured on this repo: achieved ratio **0.11 -> 0.46** and
**38 -> 75 transitions/second**, with the gradient step falling from 262 ms to
roughly 30.

Raising the priority class also lifts the tag, but as a side effect of a
different mechanism: it is not durable -- Windows re-tags the process within
minutes, and the ratio oscillates -- and it makes the rest of the machine
sluggish. `SetProcessInformation(ProcessPowerThrottling)` is the documented
way, and it is inherited by spawned children.

## The one thing blocking everything

**`ReferencePhysics` is not PolyTrack.** It is a bicycle model that exists so the
stack could be built and tested. Every time it produces is meaningless. Do not
tune reward coefficients against it beyond checking that learning occurs.

`WasmPhysics` now drives the real module and everything below the acceptance
gate works: the car spawns where the client would put it, drives, collides,
saves and restores, and two fresh instances agree byte for byte. **What is still
unproven is whether it matches the actual game.** Until the gate below passes,
treat every number from it as provisional too.

How the backend was built, and how to rebuild the inputs it needs:

1. Get the module from a build you own. Keep it local; do not redistribute it.
   Neither `polytrack_physics.wasm` nor `geometry.npz` is committed.
2. **Read the glue before probing anything.** `app.asar` ships
   `lib/polytrack_physics.js` (exact export/import mapping),
   `simulation_worker.bundle.js` (the call sequence and the game's own replay
   verifier) and `main.bundle.js` (track-code decoder, the `CarState`
   serialiser, the part table, the start transform and the terrain generator).
   Everything the module needs is described there in plain JS.
3. Host it from **Python via `wasmtime`**. All nine imports are ordinary
   Emscripten hooks; only `emscripten_resize_heap` needs a real implementation.
   No Node process, no IPC.
4. The one thing not in the JS is the collision geometry, which comes from
   Draco-compressed `.glb` models. **Capture it, do not reimplement it:**
   `tools/capture_geometry.py` serves your local copy of the client over
   127.0.0.1 with a hook ahead of its deferred bundles and intercepts the `Init`
   message it sends its simulation worker, which carries every part's
   `physicsShapeVertices`. Then `tools/build_geometry.py` packs the result.

   `tools/extract_geometry.mjs` reimplements the pipeline in real three.js and
   got 185 of 186 parts bit-identical. The one it missed, 159 `StraightTilted`,
   cost more time than everything else in Stage 0 put together — see
   PROJECT_STATUS.md §4.6. Keep it for its checksum verification, which is what
   localised the problem, but capture is the primary path.

Two performance notes, both learned the hard way:

* **Call `wasmtime` exports through the raw C API on the hot path.** The Python
  wrapper boxes every argument: ~100 µs for a seven-argument call against ~4 µs
  unchecked. At 1000 ticks per simulated second that was most of the budget.
  `WasmModule.bind` does this; use it rather than `call` in a loop.
* **The module costs ~650–800 µs per 1 ms tick**, nearly independent of track
  size, so it is the module's own fixed work and not something in this repo.
  That is roughly 1× realtime, which makes parallel collectors mandatory rather
  than an optimisation.

Do **not** try to locate state by stepping and diffing memory. That was the
original plan and it produced confident nonsense — a determinism PASS on a call
sequence that did nothing, and "car position" candidates that were integer
counters reinterpreted as denormal floats. `tools/step_probe.py` is kept only as
a record of that; do not trust its output.

### The acceptance gate

> Replay a known recording through the backend and reproduce its finish frame
> exactly.

Not approximately. A simulator 0.01% off finishes a 40-second run four
milliseconds early. The game implements this check itself — the `Verify` path in
`simulation_worker.bundle.js` steps a recording to `targetFrames` and asserts
`hasFinished && frames == targetFrames` — and `polytrack/verify.py` mirrors it.

The client's own worker also verifies our laps directly. `tools/verify_in_client.mjs`
runs the shipped `simulation_worker.bundle.js` under Node and sends it the
`Init` and `Verify` messages the game sends itself; `tools/export_verify_payload.py`
builds the payload. Winter 5's 19025, 18809, 19422 and **18786** all PASS. That checks
`trackcode.py`, `recording.py` and the `wasm.py` call sequence against the game's
own code -- but **not** the physics, since both sides run the same module. Three
things cost time getting it running and are worth knowing before you touch it:
the bundled js-sha256 takes a Node branch into webpack modules that were stubbed
out empty, so `process` has to be hidden while the bundle is evaluated; the
bundle assigns `onmessage` under `"use strict"`, which is a ReferenceError
unless the property already exists; and `fromSaveString` wants an editor save
string, so an exported `.track` has to be inflated, stripped of its header and
recompressed.

**The gate passes, 10 of 10.** Real runs across five tracks -- Summer 1, Summer
5, Desert 3, Winter 5 and Winter 4 -- played in the game, read out of its local
storage, replay here and land on their recorded finish frame to the millisecond,
the longest 61,428 frames. The two newest, Winter 5 18.809 and Winter 4 19.980,
were played by the author on 21 October 2026. Stage 0 is complete.

An oracle is a recording *paired with the finish frame the real game recorded
for it*. A recording alone will not do: it ends when the last key is released,
not when the run ends.

Get one by playing a single track. On a personal best the client stores both in
local storage under `polytrack_v5_prod_record_<slot>_default_<trackId>`:

```bash
python tools/extract_records.py -o records.json
POLYTRACK_TRACKS=path/to/tracks python scripts/run_gate.py records.json
```

Any track, any time; play more of them and the evidence gets stronger. Do not
reach for the leaderboard API — `vps.kodub.com`
returns 403 on every path for this client, and that is an answer, not an
obstacle to route around. The community TAS repo does not work either: its one
ghost is in the pre-0.6.0 encoding and carries no finish frame, and v0.6.0
rewrote the physics, so those inputs desync in the real game too.

Pin game version **0.6.2** — the worker throws on a mismatch, and replays,
track codes and times are all version-sensitive.

## Architecture

```
polytrack/
  actions.py      9 discrete actions; opposing inputs cancel (per v0.6.0)
  tas.py          .tas read/write; carries explicit run length
  trackcode.py    PolyTrack2 code <-> placed parts -> physics buffer
  track.py        Track geometry, ProgressTracker, reference-line resampling
  observation.py  Sophy-style observation vector (62 floats)
  env.py          Gymnasium env; progress reward, no negative terms
  reference.py    replay -> trajectory -> reference line -> environment
  library.py      tracks.npz: track code + reference line + par time, cached
  search.py       BeamSearch (+ optional policy prior), CEMSearch, RefinementSearch
  spline.py       Arm 3: steering relaxation + separable CEM
  generator.py    procedural tracks: self-avoiding grid walks, route by construction
  pursuit.py      scripted pure-pursuit driver: drivability filter + baseline floor
  corridor.py     RoadMap: where the car can be, from the real collision geometry
  priority.py     Windows efficiency-core opt-out
  learning/
    replay.py     sum-tree prioritised replay + n-step returns
    iqn.py        the quantile critic, its loss, and the agent
    collector.py  worker processes; one physics instance each
  recording.py    ghost/replay codec: five toggle sequences, deflate, base64url
  verify.py       the acceptance gate, mirroring the game's own Verify loop
  physics/
    base.py       PhysicsBackend protocol + CarState   <- the seam
    reference.py  deterministic stand-in (NOT PolyTrack)
    assets.py     geometry.npz: per-part collision meshes, car hull, massOffset
    world.py      start transform, mountain terrain, three.js quaternion maths
    wasm.py       the real backend: wasmtime host + state record + save/restore
```

`PhysicsBackend` is the seam. Everything above it is written against the
protocol and never touches a simulator directly, so swapping the real backend in
changes nothing upstream. Preserve this.

## Invariants that must not regress

These are load-bearing. Each has a test; each was a real bug found the hard way.

**Progress must be tracked statefully, via `ProgressTracker`.** Stateless nearest-point
projection is genuinely ambiguous wherever a track passes near itself — closed
loops, figure-eights, crossovers. On the oval, a car 1 cm *behind* the start line
projects onto the track's end and reads as a completed lap. Beam search found
this in seconds and returned a "winning run" of one frame of reverse.
See `test_reversing_at_the_start_does_not_report_a_full_lap`.

**Arc cannot advance faster than the car physically travels.** This is what makes
progress reward trustworthy. See
`test_progress_cannot_jump_further_than_the_car_can_travel`.

**Progress reward uses a high-water mark.** Otherwise an agent oscillates over one
stretch of track and farms reward forever.

**Search output must replay identically.** A reported time that cannot be
reproduced from the reported inputs is worthless. Tested at every layer, and it
is the same check to run against the real game.

**The size of a geometry error is no guide to the size of its consequences.**
Part 159's collision mesh was wrong by 1.8e-4 — invisible, and drivable at
308 km/h. It still threw a 400 km/h wall-ride clean off the track thirteen
seconds later, and Winter 5 was the only one of four tracks it broke. Do not
wave away small numeric differences in the world model; the gate is the only
thing that can tell you whether one matters.

**Everything is frame-indexed, never time-indexed.** Wall-clock never enters the
simulation. This is what makes runs reproducible — and what made the existing
`Polybot` project fail, since it fought display refresh rate and accumulated
2–3 ms of error per event.

**A frame is one millisecond. The physics runs at 1000 Hz, not 60.** This was
wrong everywhere in this repo until the backend landed, and it is the kind of
error that makes every number plausible and every number wrong. Three
independent confirmations, all in the shipped client:

* `simulation_worker.bundle.js` drains its accumulator in `o -= .001` steps, one
  `updateCarModel` per step;
* `maxFrames = 5999999`, which is 99:59.999 counted in milliseconds;
* `formatTimeString` renders `numberOfFrames` by dividing by 60000 and 1000.

So `frames` in the state record, in recordings, and in every published PolyTrack
time is a count of milliseconds. The community's 0.5.2 API notes claim 60 Hz and
quote `maxFrames = 5,999,999` in the same sentence; those two cannot both be
true, and the second one is.

The consequence: 1000 decisions a second is not an action space. The game
records inputs as *changes*, so a held key spans hundreds of ticks. Agents hold
one action across a control interval — `WasmPhysics.step_many(mask, ticks)` —
which is both the right semantics and much faster than ticking from Python.
`PolyTrackEnv` does this for you: one gym step is `control_interval_ms` of
simulated time, 50 ms by default.

**A backend knows its own rate; nothing above it may assume one.** `format_time`
counts PolyTrack frames, i.e. milliseconds. The stand-in's frames are sixtieths,
and anything that formats one without dividing by `backend.rate` first is
reporting a time 16.67× too long. `tas.py` did exactly that until this was
fixed, which is why `SearchResult` now carries a `rate` field: a bare frame
count is not a time.

**Reward coefficients are per second, not per frame.** Two backends at different
rates and a tunable control interval make "per frame" three different things.
`test_reward_does_not_depend_on_how_finely_it_is_sliced` pins this: the same
drive cut into twice as many decisions must pay exactly the same.

**No reward term may be negative.** This is the hardest-won rule in the file and
it was learned from a real run, not from reasoning.

The first version charged a per-second time penalty and forfeited a share of the
un-driven lap on failure. Twelve minutes of training broke it. With the episode
over, a failure costs a fixed amount whenever it arrives, while staying alive
keeps paying the clock — so the *fastest possible failure wins*. The agent
learned to drive sideways off the start line: the failure histogram went from
74% `stalled` to 88% `off_line`, mean return improved, and progress fell to
zero. It was optimising correctly. The objective was wrong.

The obvious repair — charge the full time budget on failure, so bailing out
early saves nothing — inverts the exploit instead of fixing it. Discounting
shrinks any large terminal penalty, so stalling eight seconds before failing is
worth −46 against −200 for failing at once, and the agent learns to sit still.

Both are the same bug. **A negative step reward makes ending the episode
attractive; a large terminal penalty makes delaying it attractive.** With every
term non-negative, a terminal state is worth zero and continuing is worth at
least zero, so quitting is never strictly better — for any policy, any discount,
anywhere on the track. `test_no_step_reward_is_ever_negative` drives badly on
purpose and checks it.

Time pressure therefore comes from the finish bonus, scaled by `par / elapsed`.
That leaves the curriculum honest: learn to complete a lap, then to complete it
quickly. `RewardConfig.time_penalty` exists and defaults to 0 — every positive
value reopens the exploit in proportion to its size.

**Encode certainty as a constraint, uncertainty as a reward.** The corollary of
the rule above: since no term may be negative, "stop doing X" cannot be
expressed as a penalty at all. When you are confident an action is never right,
take it out of the action space — it costs nothing, shrinks credit assignment,
and cannot distort the objective, whereas paying the agent not to use it is a
tug of war with lap time needing a coefficient nobody can tune.

`IQNConfig.allowed_actions` is that mechanism, and `--forward-only` is the one
use of it so far. It is a *per-track* claim and the evidence is per track.
Measured over `records.json`: the 19.025 Winter 5 lap holds throttle for
**100.0%** of its 19,025 frames and never brakes; the 21.899 lap on the same
track lifts for 12% and is 2.9 s slower. Every other track brakes for 8–21% of
the lap, so forward-only would make them unfinishable. Only set it on a run that
trains one track — `train.py` warns if you do otherwise.

Restricting means masking in **three** places, and the third is the one that
fails silently: the acting argmax, the epsilon-random branch, and the Double-Q
bootstrap argmax inside `learn`. Leave the bootstrap unmasked and every target
still reaches through `max_a Q(s', a)` over all nine actions, so the critic
converges to the value of a policy that brakes while the car never brakes — the
exact overestimation Double Q is in the loop to prevent. See
`test_the_bootstrap_does_not_reach_through_a_forbidden_action`.

**The restriction lives on `IQNConfig`, so it travels.** Into collector
processes via `CollectorConfig.iqn_config`, and into the checkpoint, so
`report.py --eval` and `tools/watch_run.py` inherit it without being told. Two
consequences: `train.py` must build the collectors' config from `agent.config`
and never from the CLI flags — a resumed learner takes its config from the
checkpoint, and an action the collectors can take but the learner masks is a
transition the critic is asked to value for a policy that never plays it. And
because `IQNConfig` is `slots=True`, a field added later unpickles *unset* rather
than defaulted, which is why `IQNAgent.load` runs `_upgrade_config`. These runs
are hundreds of thousands of gradient steps; a checkpoint must keep loading.

**A failed episode must still be cut.** The real backend never reports
`terminated` — the module has no notion of "off track", a car in a ditch is just
a car in a ditch — so without a stall cut a run that fails on the first corner
costs the whole time limit, which at ~0.5× realtime is a collector idling for a
minute. It took the smoke run from 0 completed episodes in 90 s to 17. The cut
needs a grace period or it kills the bootstrap: a car on the line has not moved
yet, and a policy cut at 3 m every episode never collects the
throttle-then-move transitions that teach it the throttle.

**The car drives on walls, so "drivable surface" is not "upward-facing
surface".** Winter 5's human record wall-rides at 400 km/h along
`WallTrackBottom` pieces whose supporting faces are vertical. `corridor.RoadMap`
therefore records **every** collision triangle at any orientation and asks only
whether there is solid geometry near the car's height; filtering to normals
within 45 degrees of up is the obvious reading and it reported that real lap as
off-ground for 14% of its length. Two other models failed the same test before
it: keeping the N highest surfaces per cell (the scenery above the road evicts
the road, 33% off-ground) and a symmetric height band (a car that jumps reads as
airborne, 15%). The band is asymmetric -- far below the car, barely above it --
because below is somewhere to land and above is a ceiling.

`test_a_real_human_lap_is_on_ground_for_its_whole_length` is what caught all
three, and it passes only if the part rotation, the rasterisation and the height
query are simultaneously right. A map that called everything solid would pass it
too, so `test_the_map_distinguishes_road_from_void` is its other half.

**A wasm backend does not free itself.** Its imports are `Func`s over bound
methods, the `Instance` holds the `Func`s and the module holds the `Instance`:
a cycle routed through wasmtime's C types, where the cycle collector cannot
follow. `del` plus `gc.collect()` frees nothing, and build-and-drop measured
**29 MB of private commit each, linearly**. Call `WasmPhysics.close()`, which
deletes the car model and then `dispose()`s the module; with that, 24 rebuilds
cost the same as one. `EnvCache` does this on eviction.

This only bites once something rebuilds backends. Bounding the collector's
environment cache is what started it -- an unbounded cache built 48 and
stopped, a bounded one rebuilds about one per episode -- so the bound turned a
large fixed footprint into an unbounded leak, 1.6 GB a worker growing to 5.8 GB
in seventy minutes, 48.7 GB of commit against a 49.8 GB limit. The fix is
tested by `test_building_and_closing_backends_does_not_grow_memory`.

Compiling is the other half: an `Engine` owns JIT code that is never handed
back, so `_COMPILED` keeps one engine and module per `.wasm` per process. Each
backend still gets its own `Store`, which is what save/restore and determinism
depend on. Build cost fell from 226 ms to 141 ms.

**A reference line is a route, not a centerline.** A track code says where every
piece of road is and nothing about the order they are driven in, so
`polytrack/reference.py` takes the route from a lap that was actually driven.
Sampling that replay must not perturb it — runs of held input are cut into
sample-sized pieces, which feeds the module the same input on the same tick, and
`test_sampling_a_replay_does_not_perturb_it` checks the sampled replay still
lands on the recorded finish frame.

**Projection is windowed.** Without it, step cost grows with track length, so a
2 km community track would run slower than a 200 m one for no reason.

## A note on search

Search is an adversarial consumer of the reward signal. It will find any way to
score highly that does not involve driving fast, and it does so within seconds.
Both progress bugs above were found by beam search, not by unit tests. **When you
change the reward or the progress logic, run a search against it** — anything
that survives that is far more likely to be correct.

## Roadmap after Stage 0

Done since: the track code decoder, and the environment now runs on real
physics — `reference.env_from_recording(track, recording, frames)` returns a
`PolyTrackEnv` driving PolyTrack's own module on a reference line derived from
that run.

Done since: the track code decoder; the environment on real physics
(`reference.env_from_recording`); reward coefficients rescaled to laps and par
times; and the IQN training loop with parallel collectors.

1. ~~**Actually train something**~~ — done, twice. Winter 5 forward-only to
   801k steps, and **48 generated tracks to 1.22M steps over 28.5 h, which
   generalises**. Three independently seeded 300k-step runs each finish every
   one of 16 unseen tracks, at **97.9% +/- 0.6%** of attempts; the 1.22M-step
   run gets 94% held-out against 92% on its training tracks, so there is no
   measurable generalisation gap. All beat the scripted driver on all 16, and
   all get 0/4 on the official tracks, reaching 1.5-23.5% of those laps before
   stalling -- the generator uses five of 186 part types on flat ground, so
   that is distribution distance, not a learning failure, and that is now
   **measured rather than asserted**: `scripts/measure_support.py` puts the
   official tracks 147-154x further outside the policy's own observation
   support than unseen generated tracks, which sit inside it, and finds them
   out of support at step 0 before the car moves. Two things make that number
   mean something and both are easy to get wrong. The **control** -- unseen
   tracks from the training distribution, scored on the same band -- is what
   rules out a band too tight to be informative. And **95% of the violations
   are `lookahead`**, which is measured against the reference line, so
   `scripts/measure_line_confound.py` re-drives training roads with their line
   swapped for one derived from a driven lap: provenance explains at most 8%,
   the road the rest. Reuse the saved band with `--band` or pay 30 minutes to
   re-drive 48 laps. Note that 12 of the 62 features are bounded and saturated
   in training, so they cannot register a violation at all. See
   `arms/generalisation/`. **Four times the compute did not buy reliability**
   (94% against 97.9%) -- it bought pace (1.35x par against 1.43x). Budget
   300k steps unless lap time is what you want. Watch three things: `ratio`
   (achieved gradient steps per transition — under ~0.3 and the learner is
   wasting most of what the collectors produce), `lap`, and the `failures`
   histogram. All `off_line` means the car cannot hold the track; all `stalled`
   means it cannot leave the line; a sudden flip between them at zero progress
   means the reward is being gamed again.
2. ~~**Procedural track generator**~~ — done. `polytrack/generator.py` lays out
   self-avoiding grid walks and returns the route by construction, which
   retires the reference-line bootstrap. 24 of 24 generated tracks with corners
   are drivable by `polytrack/pursuit.py`. **The part orientation is not the
   one `start_transform` implies**: that transform carries a half turn for the
   car, and parts are placed without it, so a `TurnSharp` at rotation r joins
   `_HEADINGS[r+2]` and `_HEADINGS[r+3]`. Straights are symmetric under a half
   turn and cannot reveal this; test corners. Next: a generated-track library,
   par times from the scripted driver, and training on it without
   `--forward-only`.
3. ~~**Policy-guided search**~~ — done. `BeamSearch(prior=..., top_k=...)` and
   `scripts/search_run.py`. See below.

4. **Does the representation transfer?** — designed, not run, and the best
   value per hour left on the list: it can add a claim, where the ablations can
   only defend numbers that already exist. Fine-tune the generated-track policy
   on one official track against a from-scratch control and compare gradient
   steps to the first completed lap.

   ```bash
   python scripts/train.py --library tracks.npz --track "Summer 1" \
       --workers 6 --max-steps 150000 --seed 1 --gamma 0.99 --log-every 5 \
       --name xfer-hand-scratch-1
   python scripts/train.py --library tracks.npz --track "Summer 1" \
       --workers 6 --max-steps 150000 --seed 1 --gamma 0.99 --log-every 5 \
       --name xfer-hand-pre-1 --resume runs/gen1/agent.pt --weights-only
   ```

   The banked Summer 1 numbers came from `scripts/transfer_experiment.py`, and
   that is what reproduces them. The names above differ from its
   `xfer-scratch-1` and `xfer-pre-1` on purpose: `train.py` writes into an
   existing `runs/<name>`, and those directories hold the banked 150k
   checkpoints. `--log-every 5` is the runner's too, since time to first lap is
   read off the log: ~600 gradient steps of resolution, against ~2400 at the
   default 20.

   `tracks.npz` does not ship -- it embeds the game's content, and
   `build_library.py` rebuilds it in three minutes. **The checkpoint does
   ship**, at `arms/generalisation/agent.pt`, and the command above reads the
   same weights from `runs/gen1/` only because that is where training wrote
   them.

   That was a deliberate decision and worth recording, because `runs/` is
   gitignored and a 1.22M-step checkpoint is 28.5 hours on one machine: without
   it neither the transfer experiment's pretrained arm nor the §6.2
   measurement could be reproduced by anyone else. The test it has to pass is
   the one the committed recordings pass -- **carries no game content** -- and
   it does: 20 weight tensors for a 62->256->256->9 MLP, Adam state, an
   `IQNConfig` of hyperparameters, a step count. No track codes, no geometry,
   no reference lines. 2.6 MB. The 48 laps the policy drove on its training
   tracks ship beside it in `arms/generalisation/laps/` for the same reason,
   171 KB, since `measure_line_confound.py` needs them to build a relined
   twin. Git cannot un-ignore a path under an ignored directory, so these are
   copies in `arms/` rather than exceptions to `runs/` -- which is also where
   `arms/README.md` says version-controlled artifacts belong.

   **Summer 1, not Winter 5.** The support measurement puts it least out of
   distribution of the four (1.4-1.7 features per frame against 7.4-12.5) and
   the policy already reaches 23.5% of it zero-shot, so there is a gradient to
   climb. Winter 5 is where the single-track infrastructure lives and is the
   worst choice on both counts; its 801k policy cannot be the control either,
   being `--forward-only`.

   Two confounds are handled: `--max-steps` is **relative** to the resumed step
   count, so both arms get the same number of new steps; and epsilon is
   **Ape-X fixed per worker**, not decayed over steps, so a resume explores
   exactly as a fresh run does.

   **Two are not, and `--reset-optimiser` alone covers only one of them.** A
   checkpoint carries **Adam's moment estimates**, so a plain `--resume`
   differs from a fresh run in weights *and* effective step size over exactly
   the early steps the metric is made of; `--reset-optimiser` drops the
   moments. Do not make it the default -- every interrupted run wants them
   kept, and `test_loading_can_keep_the_weights_and_drop_the_optimiser` pins
   both halves.

   But it **keeps the step counter**, and the counter drives a schedule. The
   prioritised replay anneals its importance-sampling exponent from 0.4 to 1
   over 500k learner steps (`replay.py`, `beta_steps`), keyed on `agent.steps`
   via `replay.sample(batch, agent.steps)` in `train.py`. A run resumed from
   the 1.22M-step checkpoint therefore samples at beta = 1.0 from its first
   batch while a fresh run starts at 0.4 and reaches only 0.58 by 150k. That is
   the same class of early-training difference the flag exists to remove, and
   an earlier version of this file claimed no step-indexed schedule was in
   play. **It was wrong**; checking that epsilon is fixed per worker is not the
   same as checking that nothing is keyed on the step count.

   Weights-only loading therefore has to zero the counter too -- online weights
   into both networks, fresh optimiser, `steps = 0` -- and `--weights-only`
   does exactly that, which is why the command above uses it rather than
   `--reset-optimiser`; `test_loading_weights_only_starts_a_new_run_from_the_online_weights`
   pins it. It starts a new run rather than a later sitting of one, so the
   source run's `history.json`, clock and best laps stay behind, and it refuses
   a `--name` whose directory already holds a run, which it would write over.
   Both arms pass `--gamma 0.99`, the checkpoint's: without it the scratch arm
   would get `IQNConfig`'s 0.999. The lr needs no flag here, since the
   checkpoint's 5e-5 is `IQNConfig`'s, but a checkpoint with another one wants
   `--lr` on both arms. `scripts/transfer_experiment.py` on the
   `transfer-summer1` branch predates the flag and gets the same start by
   writing its own init checkpoint, which is why the Summer 1 runs are
   unaffected. See `PAPER_PLAN.md` §2.2 on that branch.

   Three seeds an arm. 150k steps is ~3.5 h, so the result is six runs and
   ~21 h; a one-seed 50k pilot costs 2.5 h and says whether the effect is
   worth the rest. Read all three outcomes as results: sooner is positive
   transfer; no difference is consistent with the support measurement rather
   than contradicting it, since features that far out of band may simply carry
   no information; and *worse* is negative transfer, the most interesting of
   the three, because it says the generated distribution actively misleads.

The biggest available speed-up is sample efficiency, not throughput: episodes
re-drive the first half of the track constantly. Linesight starts episodes from
points spread along the reference line. That needs `save_state`, and `save_state`
turns out to be affordable — see below — so this is now mostly plumbing.

## Search

```bash
python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5" \
    --beam 16 --top-k 3 --budget-ms 25000
```

**Score on progress at equal depth, never on reward.** Every candidate at a
given depth has burned exactly the same number of frames, so "furthest along the
track" *is* "fastest to here". No coefficients, and none of the exploits that
make search such an adversarial consumer of a shaped reward.

Three things make a beam across a whole lap affordable, and the first was a
wrong assumption in this file for a long time:

* **`save_state` costs 13.9 ms**, not the prohibitive figure the old note here
  implied. Against 14.4 s to replay a 19.4-second lap from frame zero that is a
  factor of a thousand. Measure before designing around a cost.
* **The prior.** Nine actions five blocks ahead is 59,049 futures; the policy's
  top three is 243, and they are 243 the policy thinks are driveable. Restricted
  actions arrive as `-inf` and are dropped, so a `--forward-only` checkpoint
  searches its three throttle actions and cannot propose a braking line — guide
  with an unrestricted checkpoint if you want one considered.
* **Bounded memory, via two passes per depth.** Scoring children with their 24 MB
  states attached holds `beam_width * top_k` of them at once — 3.5 GB at the old
  default width. Pass one keeps only the numbers; pass two re-runs the survivors
  to materialise their states. A third more simulation, and peak memory drops
  from 4× the beam to 2× (measured: 717 MB at beam 16). **This is sound only
  because the backend is exactly deterministic across save and restore** — the
  same property the gate checks.

Take the **earliest** finisher in a depth, not the first one expanded.
Candidates cross the line at different frames *within* a block, so first-found
is an ordering accident worth up to a whole block — 50 ms, more than the margin
these searches are run to find.

### How a beam search dies, and what it cost to learn

The first full run on Winter 5 did not finish. It was *ahead* of the policy
through the first half — 471.4 m at 8.05 s against the policy's 438.3 m — and
then the entire beam reached **891.8 m and froze there for the remaining twelve
seconds** of budget. 1.26 million frames simulated, nothing to show.

Two mechanisms, and the second is the one worth remembering:

* **A restricted action space removes the recovery.** A forward-only car cannot
  back out of a wall, and the prior returns `-inf` for reverse, so no candidate
  could ever escape. The restriction that made training work is what made a
  stuck search state permanent.
* **Arc-greedy pruning actively prefers the doomed.** The trap was a corner
  entered too fast, so every candidate that got there first got there fastest.
  A stuck branch *holds* its arc, while a branch that gives ground to recover
  looks worse — so pruning on arc alone keeps the dead ones and discards the
  slower arrival that would have got through.

`stall_blocks` drops a branch gaining nothing, and `reserve` holds places for
candidates *behind* the leader, one per band. Neither existed for the first run;
both have tests that reproduce the failure in miniature.

### Windowed beam, and the guide

```bash
python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5" \
    --refine runs/searched/seed.json --beam-window 17.5,21.0 --beam 6 --top-k 2
```

Anchor part way through a known lap and beam from there. Two rules, both learned
by breaking them:

**Run the window to the finish; never splice.** Replaying the seed's remaining
inputs after a window that changed the car is *open loop* and unsound — those
inputs were tuned for the seed's trajectory. The first windowed run put the beam
**9 m ahead at 14.0 s** and then reported all sixteen spliced laps as slower,
which was the splice failing, not the lines.

**Always carry a guide.** `BeamSearch(guide=...)` keeps a known-good action
sequence in the beam at every depth however it scores, and repopulates from it
alone if everything else dies. Without it a beam prunes on arc, fills with the
branches carrying most speed, and has nothing left when a corner punishes all of
them at once — 891.8 m on the whole-track run, 1362 m on a windowed one that was
22 m ahead at 16.0 s. Both times the surviving line existed and had been
discarded several depths earlier: the seed's own. With a guide the search is
bounded below by its seed, for one extra expansion per depth.

**Cost is proportional to (finish − anchor), so work backwards.** A late window
is cheap and an early one expensive, and an early anchor subsumes every later
one. Search the end first, bank the gain, re-seed, then move the anchor earlier.
The first gain in this whole effort — 19.419 → **19.405** — came from a
three-minute smoke test on the last two seconds at beam 6.

### Refinement

```bash
python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5" \
    --refine runs/searched/seed.json --iterations 400 --window-blocks 4
```

**Prefer this once a finishing lap exists.** Beam search has to re-solve the
whole track and can lose everything to one corner; refinement starts from a lap
that works and keeps a change only if the lap got faster, so its output is
bounded below by its input. It is also local — it polishes the line the policy
found, and will not discover a different one.

Three things make it affordable on a 19-second lap:

* **Anchors.** The window sweeps in order, so everything before it is frozen and
  its entry state is saved once and restored per attempt. Polishing the end of a
  lap costs the end of a lap, not a whole one. This is the port the quarantined
  package prototyped.
* **Branch and bound.** Every block is exactly `block_frames` long, so block
  index and frame line up between candidate and incumbent. Past
  `abandon_metres` behind at the same frame, nothing later justifies the ten
  seconds to confirm it. Most perturbations are worse, so this is most of the
  throughput.
* **A restricted proposal set.** A lap that never brakes should not spend
  evaluations proposing brakes.

Thin the window positions rather than truncating them when there are fewer
evaluations than positions: a sweep that only polishes the first two seconds is
worse than a coarse one covering the lap.

`scripts/search_demo.py` is unrelated and stays on `ReferencePhysics` and a
synthetic chicane; it demonstrates the algorithms, not the track.

## Known loose ends

- `quarantine/duplicate_search_package/` — a second, function-based search
  implementation (`beam_search()` / `cem_search()` / `refine()`) written before
  the collision was noticed. Its windowed refinement resumes from a saved anchor
  state rather than replaying from frame zero, which is worth porting if you
  build a windowed refiner. Otherwise delete it. It also still calls
  `format_time` on stand-in frames, which now reads 16.67× short.
- Reward coefficients in `RewardConfig` are placeholders. There is finally a
  real backend to tune them against; nobody has.
- `WasmPhysics.save_state` pickles the entire heap, 24 MB a call. This was
  recorded here as "far too slow for beam search" and that was never measured:
  it is **13.9 ms**, against 14.4 s to replay a lap. Narrowing it would cut
  search memory, which is the real constraint, but it is not urgent and it is
  the kind of change that can only be made with the gate to check against.
- The 47 legacy pre-`PolyTrack2` community tracks still do not decode, which
  shrinks the corpus the generator can learn a style from.

## Permission and leaderboard etiquette

**Kodub has confirmed that using the physics WASM directly is fine** (Discord,
8 September 2026), pointing at section 4 of
https://www.kodub.com/terms/polytrack and asking for two things: that people
cannot cheat using the tool, and that it does not upload runs to the
leaderboard. Both hold by construction -- `tools/watch_run.py` serves the client
on a throwaway `http://127.0.0.1` origin and hooks `fetch` and
`XMLHttpRequest.open` to reject `vps.kodub.com`, `leaderboard` and `/api/`.
Do not weaken either guard.

That permission covers **use**, not **redistribution**. Section 4 governs mods
and reuploads and does not address shipping an extracted `.wasm` in a research
repo, so the game-derived files stay out of git until that is asked explicitly.

Attribution is now a requirement rather than a courtesy: credit Kodub, link the
official CrazyGames page, state that this is unofficial and non-commercial.

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted
to official leaderboards. The PolyTrack community draws the same line. Check
Kodub's Terms of Service — added in v0.6.0, with server-side replay verification
— before submitting anything.
