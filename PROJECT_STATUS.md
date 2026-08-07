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

### Stage 0 — Physics Backend · **~60% complete**

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
- **Step function identified:** export `n (f32, i32, i32) -> ()`
- **Clock is not read during stepping** — `n` calls zero imports

**In progress:**

- Running `step_probe.py` to prove determinism and locate car-state offsets

**Not started:**

- Track loading, save/restore, `WasmPhysics` implementation, ghost verification

### What we know about the module

| Export | Signature | Identification |
|---|---|---|
| `j` | Memory (16.9 MB, 258 pages) | Linear memory — car state lives here |
| `k` | `() -> ()` | `__wasm_call_ctors`; calls clock once at init |
| `l` | `(i32) -> i32` | `malloc` — returned pointer 87464 |
| `m` | `(i32) -> ()` | `free` or `stackRestore` |
| **`n`** | **`(f32, i32, i32) -> ()`** | **Physics step — dt + inputs** |
| `o` | 14 args, many f32 | World building (track pieces / car config) |
| `p` | 15 args, many f32 | World building |
| `q` | `(i32) -> ()` | Unknown — trapped on null |
| `r` | `(i32 ×7) -> ()` | Unknown — trapped on null |
| `s` | `() -> i32` | Likely the **determinism self-test** (returns 1, 41,959 clock calls) |
| `t` | `(i32, f64) -> ()` | Unknown — trapped on null |
| `u`, `v`, `w` | — | Emscripten stack helpers |

**Imports:** `a.a`–`a.i`, all stubbable. `a.h` is `() -> f64` — a **clock**.
Called at init only, never during stepping.

---

## 4. Remaining Steps — Stage 0

1. **Prove determinism.** Two identical runs from fresh instances must produce
   byte-identical memory.
2. **Locate car state.** Find position, velocity, orientation, wheel contact,
   and checkpoint offsets in linear memory. Rank offsets by smoothness —
   position integrates velocity, so it evolves continuously while counters jump.
3. **Work out track loading.** Identify what `o`, `p`, and `r` do. They need
   real pointers from `l` (malloc), not nulls.
4. **Implement save/restore.** Snapshot the memory regions that constitute
   simulation state. Start by copying the whole heap — correct-but-slow beats
   fast-but-lossy, because a lossy save desyncs only under search.
5. **Write `WasmPhysics`.** Fill in `WasmLayout`, implement `reset`, `step`,
   `save_state`, `load_state`, `_read_state`.
6. **⛔ THE GATE — replay a known ghost and reproduce its finish frame exactly.**
   Not approximately. Ground truth: `beatrixwashere/polytracktas` ships TAS files
   and a ghost encoder. If this fails, everything downstream is worthless.
7. **Swap the backend in.** One line — `PolyTrackEnv(backend=WasmPhysics(...))`.
   Nothing above the `PhysicsBackend` protocol changes.

---

## 5. Stages After That

### Stage 1 — Track Code Decoder
Parse `PolyTrack1...` base62 codes into a `Track`. Unlocks exact centerline
geometry (no estimation needed) and makes Stage 2 possible.

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
| `step_probe.py` | Determinism proof + state location, fresh instance per experiment |

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
  enters the simulation.
- **Run the WASM single-threaded.** PolyTrack's car sim is multithreaded, and
  threading is the hardest case for deterministic replay.

> **Search is an adversarial consumer of the reward signal.** Both progress bugs
> above were found by beam search, not by unit tests. Whenever reward or progress
> logic changes, run a search against it.

---

## 8. Risks and Unknowns

| Risk | Impact | Mitigation |
|---|---|---|
| Save/restore misses hidden state | Silent desyncs under search only | Snapshot generously; narrow later |
| Track loading (`o`/`p`/`r`) not decodable | Blocks everything | Cross-reference with the JS glue in `app.asar` |
| Reward coefficients are untuned placeholders | Training won't converge | Re-tune once real physics lands |
| Physics changed across versions | Times not comparable | **Pin one game version** and record it |
| Stand-in physics times are meaningless | Misleading benchmarks | Re-measure everything after Stage 0 |

---

## 9. Environment

- **Machine:** Windows, Python 3.14, VS Code + Claude Code extension
- **Install:** `pip install -e .` (no `PYTHONPATH` needed)
- **Tests:** `python tests\test_core.py` → 22/22 · `python tests\test_search.py` → 9/9
- **Game:** PolyTrack desktop (Electron), extracted to `~\Polytrack\game`
- ⚠️ **Not yet under git.** Run `git init && git add -A && git commit` — an
  earlier duplicate-module collision in this project would have been caught
  instantly under version control.

---

## 10. The One-Line Summary

> Everything currently runs on a stand-in simulator, so **every time this repo
> produces is meaningless**. The project becomes real the moment a WASM-backed
> physics module replays a known ghost frame-for-frame. That single test is the
> gate between scaffolding and a working agent.
