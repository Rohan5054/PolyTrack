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
PYTHONPATH=. python tests/test_core.py     # 22 tests
PYTHONPATH=. python tests/test_search.py   #  9 tests
PYTHONPATH=. python scripts/smoke_test.py
PYTHONPATH=. python scripts/search_demo.py
```

All 31 pass as of handoff.

## The one thing blocking everything

**`ReferencePhysics` is not PolyTrack.** It is a bicycle model that exists so the
stack could be built and tested. Every time this repo produces is meaningless.
Do not tune reward coefficients against it beyond checking that learning occurs.

The project becomes real when a physics backend is built against PolyTrack's
actual WebAssembly module and verified. **This could not be done in the chat
sandbox because its network allowlist blocks `kodub.com`.** On a local machine
it is straightforward. That is the main reason this work moved to Claude Code.

Full procedure and the verified C API are in `PROJECT_STATUS.md` §3. Summary:

1. Get the module from a build you own. Keep it local; do not redistribute it.
2. **Read the glue before probing anything.** `app.asar` ships
   `lib/polytrack_physics.js` (exact export/import mapping),
   `simulation_worker.bundle.js` (the call sequence and the game's own replay
   verifier) and `main.bundle.js` (track-code decoder and the `CarState`
   serialiser). Everything the module needs is described there in plain JS.
3. Host it from **Python via `wasmtime`**. All nine imports are ordinary
   Emscripten hooks; only `emscripten_resize_heap` needs a real implementation.
   No Node process, no IPC.
4. The one thing not in the JS is the collision geometry, which comes from
   Draco-compressed `.glb` models. Dump it once from a running game rather than
   reimplementing the decode.

Do **not** try to locate state by stepping and diffing memory. That was the
original plan and it produced confident nonsense — a determinism PASS on a call
sequence that did nothing, and "car position" candidates that were integer
counters reinterpreted as denormal floats. `tools/step_probe.py` is kept only as
a record of that; do not trust its output.

### The acceptance gate

> Replay a known TAS through the backend and reproduce its finish frame exactly.

Not approximately. The game already implements this exact check: the `Verify`
path in `simulation_worker.bundle.js` steps a recording to `targetFrames` and
asserts `hasFinished && frames == targetFrames`. Mirror it. The client also
runs a determinism self-test and refuses to submit runs that fail it, so a
backend that cannot reproduce a known ghost is simply wrong.

Pin game version **0.6.2** — the worker throws on a mismatch, and replays,
track codes and times are all version-sensitive.

## Architecture

```
polytrack/
  actions.py      9 discrete actions; opposing inputs cancel (per v0.6.0)
  tas.py          .tas read/write; carries explicit run length
  track.py        Track geometry + ProgressTracker
  observation.py  Sophy-style observation vector (62 floats)
  env.py          Gymnasium env; progress reward + time penalty
  search.py       BeamSearch, CEMSearch, RefinementSearch
  physics/
    base.py       PhysicsBackend protocol + CarState   <- the seam
    reference.py  deterministic stand-in (NOT PolyTrack)
    wasm.py       skeleton for the real backend
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

**Everything is frame-indexed at 60 Hz, never time-indexed.** Wall-clock never
enters the simulation. This is what makes runs reproducible — and what made the
existing `Polybot` project fail, since it fought display refresh rate and
accumulated 2–3 ms of error per event.

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
