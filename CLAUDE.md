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
pip install -e .
python tests/test_core.py       # 24 tests
python tests/test_search.py     #  9 tests
python tests/test_trackcode.py  #  7 tests
python tests/test_wasm.py       # 17 tests
python tests/test_recording.py  # 14 tests
python scripts/smoke_test.py
python scripts/search_demo.py
```

All 71 pass. Two of the suites reach for the game's own files and are stronger
when they find them:

* `test_trackcode.py` decodes every `.track` the game ships —
  `POLYTRACK_TRACKS=path\to\tracks python tests/test_trackcode.py`.
* `test_wasm.py` drives the real physics module when `polytrack_physics.wasm`
  and `geometry.npz` are in the repo root (or `POLYTRACK_WASM` /
  `POLYTRACK_GEOMETRY` point at them). Neither file is committed — both are
  derived from Kodub's game. Without them those tests skip and say so, which
  means a green run does not by itself prove the backend was exercised.

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
   Draco-compressed `.glb` models. `tools/extract_geometry.mjs` replays the
   game's model pipeline in real three.js and checks every part against the
   SHA-256 the game carries for it; `tools/build_geometry.py` packs the result.

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

**The harness is built and tested; what it lacks is an oracle.** It needs a
recording *paired with the finish frame the real game recorded for it*. A
recording alone will not do: it ends when the last key is released, not when the
run ends.

Get one by playing a single track. On a personal best the client stores both in
local storage under `polytrack_v5_prod_record_<slot>_default_<trackId>`:

```bash
python tools/extract_records.py -o records.json
POLYTRACK_TRACKS=path/to/tracks python scripts/run_gate.py records.json
```

Any track, any time. Do not reach for the leaderboard API — `vps.kodub.com`
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
  track.py        Track geometry + ProgressTracker
  observation.py  Sophy-style observation vector (62 floats)
  env.py          Gymnasium env; progress reward + time penalty
  search.py       BeamSearch, CEMSearch, RefinementSearch
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

**Projection is windowed.** Without it, step cost grows with track length, so a
2 km community track would run slower than a 200 m one for no reason.

## A note on search

Search is an adversarial consumer of the reward signal. It will find any way to
score highly that does not involve driving fast, and it does so within seconds.
Both progress bugs above were found by beam search, not by unit tests. **When you
change the reward or the progress logic, run a search against it** — anything
that survives that is far more likely to be correct.

## Roadmap after Stage 0

1. **Track code decoder** — parse `PolyTrack1...` base62 codes into a `Track`.
   Unlocks exact geometry and the procedural generator.
2. **Procedural track generator** — the editor plus exportable codes gives an
   infinite training distribution. This is what buys generalisation, and no
   Trackmania project had it cheaply.
3. **IQN training loop** — distributional Q-learning with PER and n-step,
   multiple collectors feeding one learner (the Linesight architecture). Start
   from CleanRL or Tianshou.
4. **Policy-guided search** — feed the trained policy in as an action prior so
   search only expands plausible actions. This is the hybrid the review
   recommended, and where record times come from.

## Known loose ends

- `quarantine/duplicate_search_package/` — a second, function-based search
  implementation (`beam_search()` / `cem_search()` / `refine()`) written before
  the collision was noticed. Its windowed refinement resumes from a saved anchor
  state rather than replaying from frame zero, which is worth porting if you
  build a windowed refiner. Otherwise delete it.
- Throughput figures in `README.md` were measured before `ProgressTracker`
  landed. Re-measure before relying on them.
- Reward coefficients in `RewardConfig` are placeholders. They cannot be tuned
  meaningfully until the real physics backend exists.

## Leaderboard etiquette

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted
to official leaderboards. The PolyTrack community draws the same line. Check
Kodub's Terms of Service — added in v0.6.0, with server-side replay verification
— before submitting anything.
