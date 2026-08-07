# PolyTrack Agent

An RL + trajectory-search agent for PolyTrack, built on the design the research
review landed on: **a learned policy for generalisation, test-time search for the
last tenths.**

## Status

| Component | State |
|---|---|
| Discrete action space | Done |
| `.tas` format read/write | Done |
| Track geometry, centerline projection | Done |
| Sophy-style observation builder | Done |
| Reward function | Done |
| Gymnasium environment | Done |
| Reference stand-in physics | Done |
| WASM inspection tool | Done |
| **Real WASM physics backend** | **Not started — this is Stage 0** |
| Track code decoder | Not started |
| Procedural track generator | Not started |
| IQN training loop | Not started |
| Trajectory search layer | Not started |

22 tests passing. Pipeline runs end to end at ~1,165 env steps/s single-process
(~19x realtime) on the stand-in physics.

## Quick start

```bash
pip install gymnasium numpy wasmtime
PYTHONPATH=. python tests/test_core.py     # 22 tests
PYTHONPATH=. python scripts/smoke_test.py  # end-to-end demo
```

## The one thing that matters right now

**Everything currently runs on `ReferencePhysics`, which is not PolyTrack.** It
is a bicycle model that exists so the rest of the stack could be built and
tested. Times it produces are meaningless. Do not tune reward coefficients
against it beyond confirming that learning happens at all.

The project becomes real when `polytrack/physics/wasm.py` is finished and
verified. Until then every other component is unvalidated scaffolding.

## Stage 0: the physics backend

The goal is the PolyTrack equivalent of TMInterface — the instrumentation layer
that made Linesight and every serious Trackmania agent possible. Three
capabilities, no more: **deterministic frame stepping, save/restore state, full
car-state readout.**

### 1. Get the module

PolyTrack moved physics to WebAssembly in v0.4.0 (July 2024) and rewrote the
majority of it in C++ for v0.6.0 (March 2026). It runs on its own thread,
separate from rendering, communicating car state back to the main thread — which
is precisely why a headless harness is feasible.

Work from a copy of the game you have legitimately (the itch.io desktop build is
pay-what-you-want including zero, and is easier to inspect than the web bundle).
Keep the extracted module local. Do not redistribute it.

**Pin a version.** Physics changed materially between 0.4.x, 0.5.x and 0.6.x, and
track codes and replays are version-sensitive. Record which build you are on;
results are not comparable across versions.

### 2. Inspect it

```bash
python tools/inspect_wasm.py path/to/physics.wasm --json report.json
```

The import count drives the single biggest architecture decision:

- **Few or no imports** → host the module from Python via `wasmtime`. One
  process, no IPC, trivially parallel across workers. Best case.
- **Many `env.*` / Emscripten imports** → reimplementing that glue in Python is
  a large job. Host the module in Node.js, which already has the generated glue,
  and talk to Python over stdio or shared memory. **Batch many frames per round
  trip** or IPC will dominate everything.

The tool also flags exports whose names suggest randomness or wall-clock
dependency. Anything seeded from the clock must be pinned before results are
reproducible.

### 3. Find the state offsets

Car state almost certainly lives in linear memory rather than being returned.
Step once, dump memory, step again, diff. Look for float64 triples that move
plausibly with the car. Fill in `WasmLayout` in `polytrack/physics/wasm.py`.

### 4. Verify against a known ghost — the gate

This is the acceptance test for the entire stage:

> Replay a known TAS through your backend and reproduce its finish frame exactly.

Not approximately. Exactly. Sources for ground truth: `beatrixwashere/polytracktas`
ships TAS files and a `tas_to_ghost.py` encoder; the game's own recordings API
serves top human ghosts.

The game itself enforces determinism — the client runs a self-test and refuses to
submit runs that fail it — so a backend that cannot reproduce a known ghost frame
for frame is simply wrong, and every result built on it is worthless.

Run the module **single-threaded**. PolyTrack's car simulation is multithreaded,
and threading is the hardest case for deterministic replay.

### 5. Swap it in

```python
env = PolyTrackEnv(track=track, backend=WasmPhysics(...))
```

Nothing else changes. That is the point of the `PhysicsBackend` protocol.

## Architecture

```
polytrack/
  actions.py         9 discrete actions; opposing inputs cancel, per v0.6.0
  tas.py             .tas read/write; carries explicit run length
  track.py           centerline, windowed projection, speed-scaled lookahead
  observation.py     Sophy-style observation vector (62 floats by default)
  env.py             Gymnasium env; progress reward + time penalty
  physics/
    base.py          PhysicsBackend protocol + CarState  <- the seam
    reference.py     deterministic stand-in (NOT PolyTrack)
    wasm.py          skeleton for the real backend
tools/inspect_wasm.py
scripts/smoke_test.py
tests/test_core.py
```

### Design decisions worth knowing

**Frame-indexed, never time-indexed.** Everything counts physics frames, and a
PolyTrack frame is **one millisecond** — the physics runs at 1000 Hz, not the
60 Hz this project assumed until the real backend was built. Wall-clock never
enters the simulation. This is what makes runs reproducible and what made
Polybot's screen-scraping approach fail — it fought display refresh rate and
accumulated 2–3 ms of error per event.

Because 1000 decisions a second is not an action space, an agent holds one
action across a control interval (`WasmPhysics.step_many`). The game does the
same thing: recordings store input *changes*, so a held key spans hundreds of
ticks.

**`.tas` is the interchange format.** The environment, the search layer and the
verifier all speak it. It round-trips into a ghost the real game can replay,
which is the only trustworthy proof an agent's run is real.

**Observations are car-local.** World-frame coordinates would let the network
memorise "at position (412, 88), turn left" — which collapses on unseen tracks.
Local-frame features force it to learn geometry instead of location.

**Speed-scaled lookahead.** GT Sophy samples the track ahead over roughly the
next 6 seconds of travel, scaling sample distance by speed. Neinders traced
TMRL's plateau at "safe but slow" directly to its lack of this. PolyTrack can
compute it *exactly* from the track code rather than estimating it from LIDAR —
a structural advantage over every Trackmania project.

**Progress reward with a time penalty.** Dense centerline progress is what every
successful racing agent converged on; sparse finish-time reward never trains
because the agent never finishes by chance. The time penalty is what converts
"drive safely" into "drive fast". Progress uses a high-water mark so an agent
cannot oscillate over one stretch of track and farm reward forever.

**Projection is windowed.** Only segments near the car are considered. Without
this, step cost grows with track length — a 2 km community track would run
slower than a 200 m official one for no reason. Worth 2.6x on its own, and
`test_windowed_projection_matches_full_scan_near_the_car` pins the correctness.

## Roadmap after Stage 0

1. **Track code decoder.** Parse `PolyTrack1...` base62 codes into a `Track`.
   Unlocks both exact geometry and the procedural generator.
2. **Procedural track generator.** The editor plus exportable codes gives an
   infinite training distribution. This is the piece that buys generalisation,
   and Trackmania projects never had it cheaply.
3. **IQN training loop.** Distributional Q-learning with PER and n-step, multiple
   collector processes feeding one learner — the Linesight architecture. Start
   from CleanRL or Tianshou; both have readable IQN.
4. **Trajectory search.** Beam search / CEM over input sequences, warm-started
   from the policy, using save/restore. This is where world-record times come
   from: Yosh's RL agent reached human-record level, but going past it needed
   test-time search and exact-state exploitation.

## A note on leaderboards

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted to
official leaderboards. The PolyTrack community draws the same line.

Check Kodub's current Terms of Service before submitting anything — the game
added a ToS in v0.6.0 and enforces server-side replay verification. Label runs
clearly as TAS/AI and you will get help from the Discord rather than a ban.
