# PolyTrack AI Agent — Project Status & Roadmap

**Last updated:** 8 August 2026
**Repo:** `C:\Users\Rohan\Polytrack\polytrack-agent`

---

## 1. The Goal

Build an AI agent that plays **PolyTrack** (Kodub's low-poly racing game) and
completes — then optimises — every track.

Two distinct targets, deliberately separated:

| Target | Meaning | Difficulty |
|---|---|---|
| **Complete** | Finish every official track, including ones never seen in training | Moderate |
| **Optimise** | Match or beat the best known times on specific tracks | Hard |

---

## 2. The Approach

**A learned policy for generalisation, plus test-time search for speed.**

This is not a stylistic choice. The Trackmania literature is consistent on it:

- **Linesight** (IQN, discrete inputs, reference-line reward) beat 10 of 12
  official campaign world records in 2024.
- **GT Sophy** (Nature, 2022) established that *speed-scaled lookahead of
  upcoming track geometry* is the highest-leverage observation design.
- **Yosh's agent** reached human-world-record pace with RL alone — but going
  *past* it required test-time search and exact-state exploitation.

Neither half suffices. A policy generalises to an unseen track; a search finds
the frame-perfect line on a known one.

**PolyTrack's structural advantage over Trackmania:** its physics is a separate,
deterministic, frame-stepped WebAssembly module. That makes a headless simulator
with save/restore genuinely achievable — so search is practical here in a way it
never was for Trackmania, where every project fought a closed real-time engine.

---

## 3. Where We Are Right Now

### Stage 0 — Physics Backend · **COMPLETE. Gate passed, 8 of 8 real runs, exactly**

This is the only stage that matters right now. Everything else is scaffolding
until it lands.

**Done:**

- Full stack built and tested — **85 tests passing** (24 core + 9 search +
  7 trackcode + 17 wasm + 14 recording/gate + 14 reference)
- The environment runs on the real backend. `reference.env_from_recording`
  returns a `PolyTrackEnv` driving PolyTrack's own module on a reference line
  derived from a real run; `scripts/reference_demo.py` shows it on all four
- Discrete action space, `.tas` format, track geometry, Sophy-style observations,
  reward function, Gymnasium environment
- Search layer: `BeamSearch`, `CEMSearch`, `RefinementSearch`
- Desktop build identified as **Electron**; `polytrack_physics.wasm` (396 KB)
  extracted from `resources\app.asar`
- Module **instantiates in Python** with all 9 imports stubbed → no Node host
  needed
- **Whole C API recovered from the shipped Emscripten glue**, not guessed —
  see below
- **Car state layout recovered from the shipped JS deserialiser** — no memory
  diffing required
- **End-to-end call sequence driven from Python with synthetic geometry**: the
  car spawns at the requested transform, accelerates under throttle, and writes
  a well-formed state record every frame
- **Real collision geometry recovered and checksum-verified** — 185/186 track
  parts plus the car hull, bit-identical to the game (`geometry.npz`, 0.77 MB)

- **Track codes decode** — `PolyTrack2` → placed parts → the packed buffer
  `createCarModel` takes. All 17 official tracks and 14 community ones
- **`WasmPhysics` drives the real module.** Registers all 186 part
  configurations and the car hull, generates the client's mountain terrain,
  spawns at the client's start transform, and steps. On Summer 1 the car
  accelerates from the start line under throttle with all four wheels in
  contact, reaching 78 km/h in three seconds
- **The module's own `testDeterminism` returns true**, and two fresh instances
  fed identical inputs produce byte-identical state records
- **Save/restore round-trips exactly** — a 60-step tail replayed after restoring
  a snapshot matches the original record for record

- **The gate harness** — `polytrack/recording.py` (the input-recording codec),
  `polytrack/verify.py` (the game's own Verify loop), `tools/extract_records.py`
  and `scripts/run_gate.py`
- **⭐ THE GATE PASSES, 8 OF 8.** Real runs, played in the game and read back
  out of its local storage, replay here and land on their recorded finish frame
  **to the millisecond**:

  | Track | Recorded | Replayed | |
  |---|---|---|---|
  | Summer 1 | 24778 · 24125 · 23805 | identical | ✅ |
  | Summer 5 | 61428 · 57037 | identical | ✅ |
  | Desert 3 | 21778 | identical | ✅ |
  | Winter 5 | 21899 · 19025 | identical | ✅ |

  Four tracks, eight runs, zero drift — including a **61,428-tick run**, which a
  simulator 0.01% off would miss by six frames, and Winter 5, which sustains
  390–410 km/h through vertical wall-rides.

- **Collision geometry is now captured from the game, not reimplemented.**
  `tools/capture_geometry.py` serves your own local copy of the client over
  127.0.0.1 with a hook injected ahead of its deferred bundles, intercepts the
  `Init` message it sends its simulation worker — which carries every part's
  `physicsShapeVertices` — and writes it in the format `build_geometry.py`
  already consumes. Self-verifying: 185 parts and the car hull came back
  byte-identical to the reimplementation, and part 159 came back matching the
  game's own SHA-256.

**Stage 0 is done.** The simulator is PolyTrack's, and there is no longer any
part of the world model that is a guess.

### Do not extract this by memory diffing

The game ships everything needed to drive the module. Two files inside
`app.asar` matter more than any amount of probing:

| File | What it gives you |
|---|---|
| `lib/polytrack_physics.js` | Emscripten glue — the **exact** export and import mapping |
| `simulation_worker.bundle.js` | The call sequence the game itself uses, and its own replay-verification loop |
| `main.bundle.js` | `Track.fromSaveString` (track codes) and the `CarState` serialiser/deserialiser |

`tools/step_probe.py` was written before these were found. Its heuristics are
**actively misleading on this module** and it should not be trusted: it reports
determinism PASS for a call sequence that does nothing, and its "smoothest
offsets" are integer counters read as denormal floats. Kept only for reference.

### The module's C API (verified)

Export letters come straight from `lib/polytrack_physics.js`:

| Export | Function | Signature |
|---|---|---|
| `j` | linear memory | 258 pages (16.9 MB) |
| `k` | `__wasm_call_ctors` | `() -> ()` — call once; reads the clock once |
| `l` / `m` | `malloc` / `free` | |
| `n` | `initializeCarCollisionShape` | `(massOffset f32, vertsPtr, nFloats)` |
| `o` | `addTrackPartConfiguration` | `(id, vertsPtr, nFloats, detectorType, cx,cy,cz, sx,sy,sz, hasStartOffset, ox,oy,oz)` |
| `p` | `createCarModel` | `(carId, mountainVertsPtr, nFloats, ox,oy,oz, trackDataPtr, nParts, px,py,pz, qx,qy,qz,qw)` |
| `q` | `deleteCarModel` | `(carId)` |
| **`r`** | **`updateCarModel`** | **`(carId, up, right, down, left, reset, stateOutPtr)`** — the physics step |
| `s` | `testDeterminism` | `() -> bool` |
| `t` | `_emscripten_timeout` | `(i32, f64)` — timer callback, never fires for us |
| `u`/`v`/`w` | `stackRestore` / `stackAlloc` / `stackSave` | |

**There is no `dt` argument.** The timestep is fixed inside the module — and it
is **one millisecond, not one 60th of a second**. See §7.

**Imports** `a.a`–`a.i` are ordinary Emscripten runtime hooks:
`a` `__cxa_throw`, `b` `exit`, `c` keepalive-clear, `d` `setTimeout`,
`e` `abort`, `f` `emscripten_resize_heap`, `g` `fd_write`, `h` `Date.now`,
`i` `__assert_fail`. Only `f` needs a real implementation (grow the memory and
return 1). `h` is read **once, during `k`** — never while stepping.

### Call sequence

```
k()                                   once per instance
n(massOffset, hullVerts, n)           once — car collision shape
o(id, verts, n, detector..., ...)     once per distinct track part type
p(carId, mountain..., trackData, nParts, startPos, startQuat)   per car
r(carId, up, right, down, left, reset, statePtr)                per frame
```

`trackData` is a packed array, **19 bytes per placed part**, little-endian:

```
u8  partId   i32 x   i32 y   i32 z   u8 rotation   u8 rotationAxis   i32 checkpointOrder (-1 = none)
```

### Car state record (227 bytes, verified)

`updateCarModel` writes into a caller-owned buffer. It is `i32 carId` followed
by a **variable-length** record — parse it, do not index fixed offsets:

```
u24  frames
f32  speedKmh
u8   flags: 1 hasStarted | 2 hasFinished | 4 hasCheckpointToRespawnAt | 8,16,32,64 wheelContact[0..3]
u24  finishFrames            only present when hasFinished
u16  nextCheckpointIndex
f32x3  position
f32x4  quaternion
u8   nCollisionImpulses (<=4), then that many f32
f32x3 position + f32x3 normal, per wheel whose contact bit is set
f32x4  wheelSuspensionLength / wheelSuspensionVelocity / wheelDeltaRotation / wheelSkidInfo
f32  steering
u8   controls bitfield (up|right|down|left|reset) + bit5 brakeLightEnabled
```

4 + 223 = 227 at maximum extent. `hasFinished` is bit 1 of byte 11 of the whole
buffer — the game's own verifier reads exactly that.

**Note what is absent: there is no velocity vector and no angular velocity.**
Only `speedKmh` and the orientation quaternion. `CarState` in
`polytrack/physics/base.py` currently declares both; see §8.

---

## 4. Remaining Steps — Stage 0

1. ~~**Get the real collision geometry.**~~ **Done.** `tools/extract_geometry.mjs`
   replays the game's own model pipeline in real three.js under Node and
   `tools/build_geometry.py` packs the result into a 0.77 MB `geometry.npz`.
   **185 of 186 track parts are bit-identical to the game's SHA-256 checksums,
   as is the car hull** (`c12d4421…`); `massOffset` is the literal `0.6`.
   No Electron hooking needed, and the result is reproducible rather than an
   opaque dump.

   One outlier: part 159 `StraightTilted`, the only part built through the
   quaternion path. Every ordering of scale/rotate/translate was tried. The
   game logs a checksum mismatch as `console.error` and carries on, so a stale
   entry in its own table fits the evidence as well as a bug here. Decide it by
   launching the game and reading the console. It blocks nothing unless a
   target track actually places that part.
2. ~~**Port `Track.fromSaveString`.**~~ **Done** — `polytrack/trackcode.py`.
   A `PolyTrack2…` code decodes to placed parts and to the 19-byte-per-part
   buffer `createCarModel` wants. **All 31 current-format tracks shipped with
   the game decode, including all 17 official ones**, each parsing to
   completion with the name the client displays, up to a 45,455-part community
   track.

   Gap: the 47 shipped legacy community tracks. Those use one of the five
   older formats `Track.fromSaveString` still accepts — a different symbol
   codec and part record — so it is a separate port, not a flag. It costs
   nothing for the gate, since the official campaign is entirely
   `PolyTrack2`, but it does shrink the Stage 2 training corpus.

   Also here: `TrackCode.track_id`, the `sha256` the leaderboard keys on. It
   is taken from the decoded payload rather than by re-serialising, so it
   needs no encoder and cannot disagree with what was read.
3. ~~**Write `WasmPhysics`.**~~ **Done** — `polytrack/physics/wasm.py`, plus
   `assets.py` (the geometry pack) and `world.py` (start transform, mountain
   terrain, quaternion arithmetic transcribed from the bundled three.js).
   `WasmLayout` is gone; it modelled the module as fixed memory offsets, which
   was never how it worked.

   Two things the module does not provide, and the calls made:

   - **No velocity or angular velocity** — only `speedKmh` and a quaternion.
     Both vectors are finite-differenced in the backend. They are
     observation-only and never feed back into the simulation, so they cannot
     affect a replay, and doing it here keeps everything above the
     `PhysicsBackend` seam unchanged. Cross-checked against the module's own
     `speedKmh`: the two agree. The raw record travels in `StepResult.info`.
   - **`reset()` is delete-then-create**, not the module's `reset` control bit.
     That bit is the in-game respawn key: it returns the car to its last
     checkpoint and leaves the clock running.
4. ~~**Implement save/restore.**~~ **Done**, and deliberately unoptimised: the
   whole of linear memory plus the stack pointer, ~24 MB a call. A 60-step tail
   replayed after a restore matches the original record for record. This will
   dominate beam search and is meant to be narrowed — but only after the gate
   passes, since a lossy save desyncs *only* under search.
5. **⛔ THE GATE — replay a known recording and reproduce its finish frame
   exactly.** The game already implements this check: `simulation_worker`'s
   `Verify` path steps a recording to `targetFrames` and asserts
   `hasFinished && frames == targetFrames`. `polytrack/verify.py` mirrors it.
   If this fails, everything downstream is worthless.

   **Built. Waiting on an oracle — see §4.5.**
6. **Swap the backend in.** One line — `PolyTrackEnv(backend=WasmPhysics(...))`.
   Nothing above the `PhysicsBackend` protocol changes.

### 4.5 Getting an oracle for the gate

The gate needs a pair: an input recording, and the exact frame the run finished
on. A recording alone is not enough — it ends when the last key is released,
which is not when the run ends.

**The way that works: play one track.** On a personal best the client writes
both to local storage:

```
polytrack_v5_prod_record_<slot>_default_<trackId>
  -> {"uploadId":…, "tokenHash":…, "frames":…, "recording":"…"}
```

`frames` is the finish frame in milliseconds. So:

```
python tools/extract_records.py -o records.json
POLYTRACK_TRACKS=path	o	racks python scripts/run_gate.py records.json
```

Any track, any time — a slow clean lap is a *better* first oracle than a fast
one, since it spends longer in ordinary driving and less in edge cases.

**Two other routes were tried and do not work:**

| Route | Outcome |
|---|---|
| `GET vps.kodub.com/v6/leaderboard` — download a verified ghost | **403 Forbidden on every path, including `/`.** Kodub's server refuses this client outright. Not worked around, deliberately |
| `beatrixwashere/polytracktas` — the community TAS repo | Holds exactly one TAS. Its `.ghost` is the **pre-0.6.0** encoding (a flat u24 frame list plus packed control nibbles, not the five toggle sequences 0.6.2 uses), and neither the repo nor the file records a finish frame. v0.6.0 rewrote the physics in C++, so those inputs would desync in the real game too. Confirmed by replaying it: the car leaves the track and ends 44 m below it |

The TAS is still useful as a corroboration of the 1000 Hz finding — its input
frames are 633, 1766, 4999, 14265, 14270, spaced in ways only milliseconds
explain.

### 4.6 How part 159 was resolved

Winter 5 failed the gate until the last step, and the cause was a single
collision mesh: part 159 `StraightTilted`. It is the only part of 186 whose
vertices the reimplementation never got right, and the reason nothing else
caught it is sharp — **`scale` and `quaternion` are each used by exactly one
entry in the entire part table, that one.** No other part exercises either code
path, so 185 passing checksums said nothing about them.

Reading the source never explained it. Ruled out, with part 0 `Straight` — the
same base mesh with no options, matching bit-for-bit — as the control:

| Ruled out | How |
|---|---|
| three.js version skew | Re-extracted against 0.181.0, the pinned version |
| Transform order | 168 orderings, inverses, fused matrices, axis permutations |
| Numeric literals | 288 combinations of angle, scale and offset |
| Maths implementation | The bundled `scale`, `applyQuaternion` and `makeRotationFromQuaternion` are byte-for-byte upstream |
| Base mesh | `Road/Straight` is a 4-child group with a (0, 0, 11) group matrix |
| Fitting to `ToTiltedRight` | Reproduces its y-range exactly, still misses the checksum |

The web build confirmed the table was not stale: with the console filtered to
`checksum` it logged nothing, so the game computes `48992a17…` and the
reimplementation was simply wrong.

**So the reimplementation was abandoned in favour of observation.** The client
hands its simulation worker every part's `physicsShapeVertices` in the `Init`
message, so that message is ground truth. `tools/capture_geometry.py` reads it.

The error turned out to be tiny and pervasive: max 1.8e-4 in y, 6.3e-5 in x,
z untouched — every one of the 396 vertices slightly off, consistent with a
rotation matrix differing in its last bits. Far too small to see, and enough to
throw a 400 km/h wall-ride off the track thirteen seconds later. That is worth
remembering: **the size of a geometry error is no guide to the size of its
consequences.**

The lesson for the extractor: `tools/extract_geometry.mjs` is still useful, and
its checksum verification is what localised this. But capture beats
reimplementation whenever the real thing is reachable, and here it was.

---

## 5. Stages After That

### Stage 1 — Track Code Decoder
Parse track codes into a `Track`. Unlocks exact centerline geometry (no
estimation needed) and makes Stage 2 possible. Largely pulled forward into
Stage 0, since `createCarModel` needs the decoded part list anyway.

The current codes are **`PolyTrack2`** — deflate (two passes, `windowBits` 9
then 15) wrapped in a custom base62 alphabet. `Track.fromSaveString` in
`main.bundle.js` also accepts five older formats; only the newest matters.
All 17 official tracks ship as `.track` files inside `app.asar`, so the decoder
has a ready-made test corpus with known-good contents.

### Stage 1.5 — Reference Lines · **Done**
A track code says where every piece of road is and nothing about the route
through it, which the reward and the observation both need. So the route comes
from a lap that was driven: replay a recording through the verified backend,
sample the trajectory, resample at fixed arc spacing. Linesight's design, and
the same caveat — it is one lap, not the middle of the road, so lateral offset
is distance from that lap.

Sampling is exact. Runs of held input are cut into sample-sized pieces, which
feeds the module the same input on the same tick, so a sampled replay still
lands on the recorded finish frame. `test_sampling_a_replay_does_not_perturb_it`
asserts that rather than assuming it.

Measured, on the four tracks we have runs for: 1300–3980 m of line, 197–331 km/h
average, 482 km/h peak. Building one costs a full replay — 46 s of wall clock
for a 19-second run, which is the ~1× realtime figure showing up again.

### Stage 2 — Procedural Track Generator
Generate thousands of valid track codes spanning straights, curves, jumps and
loops. **This is what buys generalisation.** A policy trained on 17 official
tracks memorises them; one trained on 5,000 generated tracks learns to drive.
No Trackmania project had this cheaply — PolyTrack's built-in editor does.

### Stage 3 — Train the Policy
IQN (distributional Q-learning) with prioritised replay and n-step returns.
Multiple collector processes feeding one learner — the Linesight architecture.
Start from CleanRL or Tianshou. Curriculum: easy → hard tracks.

Then re-tune reward coefficients, which are currently placeholders and cannot
be calibrated until real physics exists.

### Stage 4 — Policy-Guided Search
Feed the trained policy into `BeamSearch` as an action prior so search only
expands plausible actions. Then per-track refinement. **This is where record
times come from.**

### Stage 5 — Verify and Publish
Export runs as `.tas`, encode to ghosts, confirm they replay in the real game.
Label everything as TAS/AI. **Keep bot runs off the human leaderboard** — the
Trackmania and PolyTrack communities both draw this line. Check Kodub's Terms of
Service (added in v0.6.0) before submitting anything.

---

## 6. Tools Built

| Tool | Purpose |
|---|---|
| `find_wasm.py` | Locate `.wasm` files on disk |
| `extract_asar.py` | Pull files out of Electron's `app.asar` |
| `inspect_wasm.py` | Static analysis — exports, imports, signatures |
| `probe_wasm.py` | Instantiate with logging stubs; see which imports fire |
| `map_wasm.py` | Call exports, diff memory ⚠️ *no instance isolation — results after a stack-modifying call are unreliable* |
| `step_probe.py` | ⚠️ **Superseded and misleading.** Its determinism check passes on a no-op call sequence, and its "smoothest offsets" are integer counters read as denormal floats. The glue JS supersedes all of it. |
| `api_probe.py` | Drove the call sequence with synthetic geometry to confirm the export mapping. Superseded by `polytrack/physics/wasm.py`, which is now the reference |
| `extract_geometry.mjs` | Node + three.js. Reimplements the game's model pipeline; 185/186 parts bit-identical. Keep it for its checksum verification, which is what localised part 159 — but it is no longer the source of the geometry |
| `capture_geometry.py` | **The primary path.** Serves your local client over 127.0.0.1 with a hook ahead of its deferred bundles, and intercepts the `Init` message it sends its simulation worker, which carries every part's real collision mesh. Observation instead of reimplementation |
| `build_geometry.py` | Packs that JSON into `geometry.npz` (0.77 MB) for the backend |
| `extract_records.py` | Scrapes personal bests out of the client's Chromium local storage — a recording paired with the finish frame the game recorded for it, which is what an oracle is |
| `run_gate.py` | The acceptance gate over every record. Exit 0 only if all verify |
| `reference_demo.py` | Track code → meshes → module → trajectory → reference line → observation → reward, on every record that matches a track |

---

## 7. Invariants That Must Not Break

Each has a test. Each was a real bug found the hard way.

- **Progress must be tracked statefully** (`ProgressTracker`). Stateless
  nearest-point projection is ambiguous wherever a track passes near itself. On
  the oval, a car 1 cm *behind* the start line projected onto the track's end and
  read as a completed lap — beam search found this in seconds and returned a
  "winning run" of one frame of reverse.
- **Arc cannot advance faster than the car physically travels.**
- **Progress reward uses a high-water mark** — otherwise an agent oscillates over
  one stretch and farms reward forever.
- **Search output must replay identically.** A time that can't be reproduced from
  its own inputs is fiction.
- **Everything is frame-indexed, never time-indexed.** Wall-clock never enters
  the simulation. Confirmed at the source: `updateCarModel` takes no `dt`.
- **A frame is one millisecond — the physics runs at 1000 Hz, not 60.** This
  document and the whole repo said 60 Hz until the backend was written. Three
  independent confirmations in the shipped client: the worker drains its
  accumulator in `o -= .001` steps, one `updateCarModel` per step; `maxFrames`
  is 5999999, which is 99:59.999 in milliseconds; and `formatTimeString`
  divides `numberOfFrames` by 60000 and 1000. The community's 0.5.2 API notes
  assert 60 Hz and quote that same `maxFrames` one clause later — the two are
  not compatible, and the frame count is the one that checks out.
  It follows that an agent must hold an action across a control interval rather
  than choose 1000 times a second; `WasmPhysics.step_many` exists for that, and
  `PolyTrackEnv` holds one action for `control_interval_ms` (50 ms by default,
  4.2 m at 300 km/h — just under the 5 m part grid).
- **A backend knows its own rate; nothing above it may assume one.**
  `format_time` counts milliseconds. `tas.py` kept `PHYSICS_HZ = 60` right
  through the 1000 Hz correction, so every time this project printed for the
  stand-in was 16.67× too long, and `SearchResult.time` was wrong for whichever
  backend it was not built against. Fixed; `SearchResult` now carries the rate
  that produced its frame count, because a bare frame count is not a time.
- **Reward coefficients are per second, not per frame.** With two backends at
  different rates and a tunable control interval, "per frame" means three
  different things. `test_reward_does_not_depend_on_how_finely_it_is_sliced`
  pins it: the same drive cut into twice as many decisions pays the same.
- ~~Run the WASM single-threaded.~~ Resolved, not a risk: the module imports no
  pthread hooks and uses unshared memory. PolyTrack's multithreading is at the
  *JS* level — one worker per car — so a Python host gets a single-threaded
  simulation for free, and parallelism comes from running many instances.

> **Search is an adversarial consumer of the reward signal.** Both progress bugs
> above were found by beam search, not by unit tests. Whenever reward or progress
> logic changes, run a search against it.

---

## 8. Risks and Unknowns

| Risk | Impact | Mitigation |
|---|---|---|
| ~~Collision vertices differ from the game's~~ | Was: silent desync deep into a replay | **Closed.** 185/186 parts and the car hull verified bit-identical against the game's own checksums. `StraightTilted` still open |
| ~~`CarState` declares `velocity` / `angular_velocity`, which the module does not expose~~ | Was: fields with no source | **Closed.** Finite-differenced in the backend; agrees with the module's `speedKmh`. Observation-only, so it cannot affect a replay |
| ~~Save/restore misses hidden state~~ | Was: silent desyncs under search only | **Closed for now.** Whole-heap snapshot; a long tail replays identically after restore. Reopens the moment the ranges are narrowed |
| **Simulation runs at roughly 1× realtime** | Training throughput. This is the big one | ~650–800 µs per 1 ms tick measured (sustained ~2.3 ms under thermal throttling on this laptop). Cost is nearly flat in track size, so it is the module's own fixed work, not our geometry. Python overhead is already down to ~0 via the raw wasmtime call path — the remaining cost is inside the module. Mitigation is parallel collectors, which the Linesight architecture already assumes; measure again on a desktop before drawing conclusions |
| Whole-heap snapshots are ~24 MB | Search throughput, not correctness | Measure before building the refiner; narrow the ranges if it bites |
| `Date.now()` is read once during `__wasm_call_ctors` | If it seeds anything the physics touches, no host can ever match the game | Cheap to disprove: the gate catches it immediately |
| Reward coefficients are untuned placeholders | Training won't converge | **Now actionable.** Real physics is attached and real reference lines exist; nothing has been tuned yet. Run a search against any change |
| Corridor rays are faked from the reference line | Observation says "distance from the lap driven", not "distance to a wall" | `ObservationBuilder._cast_rays` is one method by design. The module has the collision geometry; ask it |
| Physics changed across versions | Times not comparable | **Pinned to 0.6.2** — the worker throws on any other version string |
| Stand-in physics times are meaningless | Misleading benchmarks | Re-measure everything after Stage 0 |

---

## 9. Environment

- **Machine:** Windows, Python 3.14, VS Code + Claude Code extension
- **Install:** `pip install -e .` (no `PYTHONPATH` needed)
- **Tests:** 85 passing — `test_core.py` 24 · `test_search.py` 9 ·
  `test_trackcode.py` 7 · `test_wasm.py` 17 · `test_recording.py` 14 ·
  `test_reference.py` 14. The last three read the game's own files when
  `POLYTRACK_TRACKS` / `geometry.npz` / `polytrack_physics.wasm` are present,
  and skip cleanly when they are not — so a green run without them does not
  prove the backend was exercised
- **Game:** PolyTrack desktop (Electron), extracted to `~\Polytrack\game`
- **Version pinned: `0.6.2`.** The simulation worker throws on any mismatch, and
  replays are version-sensitive. Record it alongside every measured time.
- `app.asar` contents are Kodub's copyrighted code. Extract them somewhere
  outside this repo, keep them local, and do not commit or redistribute them.
- Under git as of the initial import.

---

## 10. The One-Line Summary

> Everything currently runs on a stand-in simulator, so **every time this repo
> produces is meaningless**. The project becomes real the moment a WASM-backed
> physics module replays a known ghost frame-for-frame. That single test is the
> gate between scaffolding and a working agent.
