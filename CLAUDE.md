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
pip install -e .[learning]      # torch is training-only; the rest runs without
python tests/test_core.py       # 24 tests
python tests/test_search.py     #  9 tests
python tests/test_trackcode.py  #  7 tests
python tests/test_wasm.py       # 17 tests
python tests/test_recording.py  # 14 tests
python tests/test_reference.py  # 17 tests
python tests/test_learning.py   # 14 tests
python scripts/smoke_test.py
python scripts/search_demo.py
```

All 102 pass. Four of the suites reach for files that are not in the repo and
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
```

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

Two things in `learning/collector.py` are load-bearing on Windows and all of them
have cost time somewhere: processes start by **spawn** so nothing is inherited
and `WasmPhysics` can never cross a process boundary; and torch grabs every core
per process unless told otherwise, so `set_num_threads(1)` in a worker is not a
tuning nicety but the difference between six workers and worse-than-one.

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

**The gate passes, 8 of 8.** Real runs across Summer 1, Summer 5, Desert 3 and
Winter 5 — played in the game, read out of its local storage — replay here and
land on their recorded finish frame to the millisecond, up to a 61,428-tick run.
Stage 0 is complete.

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
  trackcode.py    PolyTrack2 code -> placed parts -> physics buffer
  track.py        Track geometry, ProgressTracker, reference-line resampling
  observation.py  Sophy-style observation vector (62 floats)
  env.py          Gymnasium env; progress reward + time penalty
  reference.py    replay -> trajectory -> reference line -> environment
  library.py      tracks.npz: track code + reference line + par time, cached
  search.py       BeamSearch, CEMSearch, RefinementSearch
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

**A failed episode must still be cut.** The real backend never reports
`terminated` — the module has no notion of "off track", a car in a ditch is just
a car in a ditch — so without a stall cut a run that fails on the first corner
costs the whole time limit, which at ~0.5× realtime is a collector idling for a
minute. It took the smoke run from 0 completed episodes in 90 s to 17. The cut
needs a grace period or it kills the bootstrap: a car on the line has not moved
yet, and a policy cut at 3 m every episode never collects the
throttle-then-move transitions that teach it the throttle.

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

1. **Actually train something.** One 17-minute run has happened; it found the
   reward exploit above and was stopped. The reward has been rebuilt since and
   **nothing has trained against the new one.** Watch three things: `ratio`
   (achieved gradient steps per transition — under ~0.3 and the learner is
   wasting most of what the collectors produce), `lap`, and the `failures`
   histogram. All `off_line` means the car cannot hold the track; all `stalled`
   means it cannot leave the line; a sudden flip between them at zero progress
   means the reward is being gamed again.
2. **Procedural track generator** — the editor plus exportable codes gives an
   infinite training distribution. This is what buys generalisation, and no
   Trackmania project had it cheaply. It also retires the reference-line
   bootstrap: a generated track knows its own route, so it needs no lap driven
   for it. Until it exists, **four tracks is a memorisation problem, not a
   generalisation one.**
3. **Policy-guided search** — feed the trained policy in as an action prior so
   search only expands plausible actions. This is the hybrid the review
   recommended, and where record times come from.

The biggest available speed-up is sample efficiency, not throughput: episodes
re-drive the first half of the track constantly. Linesight starts episodes from
points spread along the reference line, which needs `save_state` to be cheaper
than a 24 MB heap pickle. Narrow it with the gate to check against.

## Known loose ends

- `quarantine/duplicate_search_package/` — a second, function-based search
  implementation (`beam_search()` / `cem_search()` / `refine()`) written before
  the collision was noticed. Its windowed refinement resumes from a saved anchor
  state rather than replaying from frame zero, which is worth porting if you
  build a windowed refiner. Otherwise delete it. It also still calls
  `format_time` on stand-in frames, which now reads 16.67× short.
- Reward coefficients in `RewardConfig` are placeholders. There is finally a
  real backend to tune them against; nobody has.
- `ObservationBuilder._cast_rays` still approximates the corridor from the
  reference line and `Track.width`. With the real backend that measures distance
  from *the driven lap*, not from a wall — the module has the collision geometry
  and this should ask it. Isolated in one method for exactly this reason.
- `WasmPhysics.save_state` pickles the entire heap (~24 MB a call). Correct, and
  far too slow for beam search. Narrow it only with the gate to check against.
- The 47 legacy pre-`PolyTrack2` community tracks still do not decode, which
  shrinks the corpus the generator can learn a style from.

## Leaderboard etiquette

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted
to official leaderboards. The PolyTrack community draws the same line. Check
Kodub's Terms of Service — added in v0.6.0, with server-side replay verification
— before submitting anything.
