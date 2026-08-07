# PolyTrack AI Agent — Project Status & Roadmap

**Last updated:** 7 August 2026
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

### Stage 0 — Physics Backend · **API fully mapped, backend not yet written**

This is the only stage that matters right now. Everything else is scaffolding
until it lands.

**Done:**

- Full stack built and tested against a stand-in simulator — **31 tests passing**
  (22 core + 9 search)
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

**Not started:**

- Real track-part geometry, `WasmPhysics`, save/restore, ghost verification

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

**There is no `dt` argument.** The timestep is fixed inside the module, which
confirms the 60 Hz frame-indexing invariant at the source.

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

1. **Get the real collision geometry.** `addTrackPartConfiguration` needs each
   track part's `physicsShapeVertices`, and `initializeCarCollisionShape` needs
   the car hull. Both come from the Draco-compressed `.glb` models. Do **not**
   reimplement GLB+Draco: these tables are static and loaded once, so run the
   game under Electron with remote debugging, intercept the `Init` message the
   main thread posts to `simulation_worker`, and dump `trackParts`,
   `carCollisionShapeVertices` and `carMassOffset` to a `.npz`. One-time cost;
   the Python host then never needs Node.
2. **Port `Track.fromSaveString`** from `main.bundle.js` so a `PolyTrack2…` code
   becomes the 19-byte-per-part buffer `createCarModel` wants. This is now a
   Stage 0 dependency, not a later stage.
3. **Write `WasmPhysics`.** Replace `WasmLayout` (offset-based, wrong model)
   with the call sequence and record parser above.
4. **Implement save/restore.** The module exposes no save-state call, so this is
   a snapshot of linear memory. Start with the whole heap — correct-but-slow
   beats fast-but-lossy, because a lossy save desyncs only under search — then
   measure, because 16.9 MB per search node will dominate beam search.
5. **⛔ THE GATE — replay a known recording and reproduce its finish frame
   exactly.** The game already implements this check: `simulation_worker`'s
   `Verify` path steps a recording to `targetFrames` and asserts
   `hasFinished && frames == targetFrames`. Mirror it. If this fails, everything
   downstream is worthless.
6. **Swap the backend in.** One line — `PolyTrackEnv(backend=WasmPhysics(...))`.
   Nothing above the `PhysicsBackend` protocol changes.

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
| `api_probe.py` | Drives the verified call sequence with synthetic geometry and parses the state record — the reference for how to talk to the module |

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
- **Everything is frame-indexed at 60 Hz, never time-indexed.** Wall-clock never
  enters the simulation. Confirmed at the source: `updateCarModel` takes no `dt`.
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
| **Collision vertices differ from the game's by even one float** | Silent desync hundreds of frames into a replay | Dump the exact arrays the game posts to its worker; never regenerate them |
| `CarState` declares `velocity` / `angular_velocity`, which the module does not expose | `observation.py` reads fields with no source | Decide: finite-difference position & quaternion, or drop them for `speedKmh` + quaternion + a short frame history |
| Save/restore misses hidden state | Silent desyncs under search only | Snapshot generously; narrow later |
| Whole-heap snapshots are 16.9 MB | Search throughput, not correctness | Measure before building the refiner; narrow the ranges if it bites |
| `Date.now()` is read once during `__wasm_call_ctors` | If it seeds anything the physics touches, no host can ever match the game | Cheap to disprove: the gate catches it immediately |
| Reward coefficients are untuned placeholders | Training won't converge | Re-tune once real physics lands |
| Physics changed across versions | Times not comparable | **Pinned to 0.6.2** — the worker throws on any other version string |
| Stand-in physics times are meaningless | Misleading benchmarks | Re-measure everything after Stage 0 |

---

## 9. Environment

- **Machine:** Windows, Python 3.14, VS Code + Claude Code extension
- **Install:** `pip install -e .` (no `PYTHONPATH` needed)
- **Tests:** `python tests\test_core.py` → 22/22 · `python tests\test_search.py` → 9/9
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
