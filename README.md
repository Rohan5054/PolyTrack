# PolyTrack Agent

An RL + trajectory-search agent for PolyTrack, built on the design the research
review landed on: **a learned policy for generalisation, test-time search for the
last tenths.**

## Where this stands

**Winter 5, 18.786 s** — 239 ms faster than the human personal best this project
was measured against (19.025), and 636 ms faster than the same policy driving
without search (19.422). It reproduces its finish frame exactly on a fresh
backend, and **the game's own simulation worker agrees**.

| | Winter 5 | verified by |
|---|---|---|
| human PB | 19.025 | the game's own storage (external oracle) |
| Arm 1 — policy alone | 19.422 | gate + client worker |
| **Arm 2 — policy + guided beam** | **18.786** | gate + client worker |
| Arm 3 — continuous optimisation | did not finish, 1089 m of 1747 | n/a — no finish frame |

That is one track. See [What this does not show](#what-this-does-not-show).

## Status

| Component | State |
|---|---|
| Discrete action space | Done |
| `.tas` format read/write | Done |
| Track code **decoder** (`PolyTrack2`) | Done |
| Track code **encoder** | Done — 31/31 shipped tracks re-encode exactly |
| Recording / ghost codec | Done |
| Track geometry, windowed projection | Done |
| Sophy-style observation builder | Done |
| Reward function | Done — every term non-negative, see below |
| Gymnasium environment | Done — runs on real physics |
| Reference stand-in physics | Retained for tests only |
| **Real WASM physics backend** | **Done. Acceptance gate passes, 8/8** |
| **Verification by the client's own worker** | **Done — 3/3 laps PASS** |
| Reference lines from real runs | Done |
| Search layer (beam / CEM / refinement) | Done — beat the human PB |
| Policy-guided search | Done — this is where 18.786 came from |
| IQN training loop | Done — **860,000 gradient steps run** |
| Windows efficiency-core mitigation | Done — `polytrack/priority.py` |
| Three-arm comparison protocol | Done — `EXPERIMENT.md`, `arms/` |
| **Procedural track generator** | **In progress — layout works, drivability bug open** |

**137 tests passing.** Four suites reach for files not in the repo and are
stronger when they find them; without them they skip and say so.

## Quick start

```bash
pip install -e .[learning]      # torch is training-only
python tests/test_core.py       # 24
python tests/test_search.py     # 17
python tests/test_trackcode.py  # 12
python tests/test_wasm.py       # 17  (needs the game files)
python tests/test_recording.py  # 14
python tests/test_reference.py  # 18  (needs the game files)
python tests/test_learning.py   # 20  (needs torch)
python tests/test_spline.py     #  6
python tests/test_generator.py  #  9
python scripts/smoke_test.py
python scripts/verify_baselines.py   # re-verify every banked lap + substrate
```

## Stage 0: the acceptance gate

The gate was: **replay a known recording through the backend and reproduce its
finish frame exactly.** Not approximately — a simulator 0.01% off finishes a
40-second run four milliseconds early.

It passes on eight real runs across four tracks, up to a 61,428-tick run, each
landing on its recorded finish frame to the millisecond. Those runs were played
in the game and read out of its local storage; the game implements the same
check itself, and `polytrack/verify.py` mirrors its loop.

```bash
python tools/extract_records.py -o records.json
POLYTRACK_TRACKS=path/to/tracks python scripts/run_gate.py records.json
```

### The client's own worker verifies our laps too

The gate proves the *simulator* reproduces reality, because a human record's
target frame comes from the game and is an external oracle. For a lap this
project **generated**, the target is our own search's claim — so a gate PASS
means "these inputs reproduce that frame here", and "the game would agree" was
an inference.

`tools/verify_in_client.mjs` measures it instead. It runs the shipped
`simulation_worker.bundle.js` under Node and sends it the `Init` and `Verify`
messages the game sends itself:

```bash
python tools/export_verify_payload.py -o runs/client_check.json
node tools/verify_in_client.mjs --game path/to/extracted/app \
     --payload runs/client_check.json
```

19025, 19422 and 18786 all PASS. The worker decodes the track with its own
`fromSaveString` and deserialises the recording with its own `deserialize`, so
this independently exercises `trackcode.py` and `recording.py` alongside the
`wasm.py` call sequence — everything written here between a lap file and the
module.

**It does not confirm the physics.** Both sides run the same
`polytrack_physics.wasm`, hash-identical. The gate remains the only thing that
can say the module matches the real game. The two are complementary: the gate
says the module reproduces reality, this says we drive the module the way the
game does.

## Results, and the protocol behind them

`EXPERIMENT.md` is the protocol; `arms/` holds the artifacts, version
controlled, unlike `runs/` which is scratch and gitignored. **A number not
backed by a file in `arms/` is not a result.**

Three approaches to the same track:

- **Arm 1 — policy alone.** IQN, forward-only action space, best of 20 greedy
  laps, no test-time search. **19.422.**
- **Arm 2 — policy plus guided beam.** The same checkpoint as an action prior,
  the beam anchored in a known lap and carrying it as a guide so the result is
  bounded below by its seed, ranked on
  `arc + lookahead·speed − line_weight·|lateral| − switch_cost·switches`.
  **18.786.**
- **Arm 3 — continuous trajectory optimisation.** A separable cross-entropy
  method over a per-block steering relaxation, anchored, seeded from the human
  record. No policy, no checkpoint, no Arm 2 artifact. **Did not finish**, 1089 m
  of 1747, for 17,794,080 simulated frames.

Budgets are reported in hardware-independent units, which is the whole reason
the protocol rejects a single-device rule:

| | rollouts | simulated frames |
|---|---|---|
| the run that produced 18786 | 18,309 | 914,944 |
| its 19245 seed | 6,993 | 349,600 |
| every logged search in the project | 186,387 | 9,897,112 |
| Arm 3, all runs | 10,111 | 17,794,080 |

Arm 3 spent nineteen times the winning run's budget and never crossed the line.

### Three caveats that belong in any write-up

**Arm 1 is 19.422, not the 19.105 in the logs.** A 19.105 episode was observed
once during training, but record laps were only written to disk when they beat
the all-time best, so its inputs were discarded and no recording survives. It
cannot be replayed or gate-checked. 19.422 is slower, which makes Arm 2's gain
look larger — so say so.

**Arm 2 was steered toward a human racing line.** The Winter 5 reference line
*is* the 19.025 human lap, and `line_weight` charges candidates for distance
from it. That term found 459 ms in one run where seven runs ranking on arc alone
had found 14 ms in total. The control sequence is the agent's own and drives
that route 239 ms faster than the human did — but it found the route with help.

**Arm 3 is not clean-room.** Same author as Arm 2. The data rule held (no Arm 2
trajectory, coefficient or checkpoint is read) but the insight rule could not:
Arm 2 established by measurement that the entire Winter 5 deficit is *route*,
and that shaped what was built.

## Training

```bash
POLYTRACK_TRACKS=path/to/tracks python scripts/build_library.py   # once, ~3 min
python scripts/train.py --smoke                                   # 90 s
python scripts/train.py --workers 6 --track "Winter 5" --forward-only
```

IQN — distributional Q-learning — with prioritised replay, 5-step returns and
several collector processes feeding one learner. That is Linesight's
architecture, and here the parallelism is not an optimisation: the module runs at
about half realtime per instance, so **one collector yields ~11 transitions a
second** and six give 50–80.

Distributional matters for racing specifically. Committing to a corner at speed
either works or puts the car into a wall, and a mean Q-value of those two futures
describes neither. A quantile critic can represent "usually 90, occasionally 5".

**860,000 gradient steps have been run** on Winter 5 with `--forward-only`. The
policy finishes the lap reliably and its best greedy lap is 19.422. A later run
retrained against a reference line rebuilt from the 18.786 lap rather than the
human one; it reached a 19.286 collector lap in ~72k steps and was stopped
without beating the original.

Watch three things: `ratio` (achieved gradient steps per transition — under ~0.3
and the learner is wasting most of what the collectors produce), `lap`, and the
`failures` histogram. All `off_line` means the car cannot hold the track; all
`stalled` means it cannot leave the line; a sudden flip between them at zero
progress means the reward is being gamed.

### Why no reward term may be negative

The hardest-won rule here, learned from a real run rather than from reasoning.

The first reward charged a per-second time penalty and forfeited part of the
un-driven lap on failure. Twelve minutes of training broke it: with the episode
over, a failure costs a fixed amount whenever it arrives while staying alive
keeps paying the clock, so **the fastest possible failure wins**. The agent
learned to drive sideways off the start line — failures went from 74% `stalled`
to 88% `off_line`, mean return improved, progress fell to zero. It was
optimising correctly; the objective was wrong.

The obvious repair, charging the full time budget on failure, inverts the exploit
rather than fixing it: discounting shrinks any large terminal penalty, so
stalling eight seconds before failing beats failing at once, and the agent learns
to sit still.

With every term non-negative, a terminal state is worth zero and continuing is
worth at least zero, so quitting is never strictly better — for any policy, any
discount, anywhere on the track. Time pressure comes from the finish bonus,
scaled by `par / elapsed`. `RewardConfig.time_penalty` exists and **defaults to
0**; every positive value reopens the exploit in proportion to its size.

The corollary: **encode certainty as a constraint, uncertainty as a reward.**
Since no term may be negative, "stop doing X" cannot be expressed as a penalty at
all — so when you are confident an action is never right, remove it from the
action space. `--forward-only` is the one use so far, and it is a *per-track*
claim with per-track evidence: the 19.025 Winter 5 lap holds throttle for 100.0%
of its frames and never brakes, while every other track in `records.json` brakes
for 8–21% of the lap.

## Search

```bash
python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5" \
    --refine seed.json --beam-window 8.0,21.0 \
    --beam 20 --top-k 3 --reserve 5 --lookahead 1.0 --line-weight 1.0
```

**Score on progress at equal depth, never on reward.** Every candidate at a given
depth has burned exactly the same number of frames, so "furthest along the track"
*is* "fastest to here". No coefficients, and none of the exploits that make
search such an adversarial consumer of a shaped reward.

Three things make a beam across a whole lap affordable:

- **`save_state` is cheap.** 13.9 ms in the original measurement, ~26 ms on a
  loaded machine — against 23–52 s to replay a 19-second lap from frame zero.
  This file previously recorded it as "far too slow for beam search", which had
  never been measured. Measure before designing around a cost.
- **The prior.** Nine actions five blocks ahead is 59,049 futures; the policy's
  top three is 243, and they are 243 the policy thinks are drivable.
- **Bounded memory, two passes per depth.** Scoring children with their 24 MB
  states attached holds `beam × top_k` at once. Pass one keeps only the numbers,
  pass two re-runs the survivors. A third more simulation, peak memory halved.
  Sound only because the backend is exactly deterministic across save/restore.

**Always carry a guide**, and **run the window to the finish; never splice.**
Both were learned by breaking them: without a guide the beam prunes on arc, fills
with the branches carrying most speed, and dies when a corner punishes all of
them at once — 891.8 m on a whole-track run, 1362 m on a windowed one. Splicing
a seed's remaining inputs after a window that moved the car is open-loop and
unsound; the first windowed run put the beam 9 m ahead at 14.0 s and then
reported all sixteen spliced laps as slower.

**Cost is proportional to (finish − anchor), so work backwards.** A late window
is cheap, an early one expensive, and an early anchor subsumes every later one.

### How the 459 ms was actually found

Seven searches ranking on arc alone found 14 ms in total. The breakthrough came
from measuring rather than theorising: a report that the car was "skidding" was
**disproved** — `wheel_skid_info` saturates at 4.000 on both laps (it is a
visual-effects flag, not physics), slip was 0.25° against 0.31°, collision
impulse zero on both, throttle 100% on both, steering effort 0.011 against 0.013.

What was left was the *route*. The agent drove 6.2 m further and wandered 8.3 m
off the reference line, mean offset 2.16 m. Adding `line_weight` to the ranking
found **459 ms in one run**. Four further searches afterwards returned +0 ms, and
a fifth — against a reference line rebuilt from the 18.786 lap itself — returned
a file byte-identical to its seed. Search against this policy has converged.

## Procedural track generator — in progress

The roadmap item that matters most, and the one that would make this
generalisation rather than memorisation. Two thirds exist:

- **`trackcode.encode`** — done and validated three ways. Every shipped track
  that decodes at all (31 of 31) re-encodes to the same parts, environment and
  sun direction; the client's own worker loads a Winter 5 rebuilt from parsed
  parts and all three banked laps still verify frame-exactly on it; five unit
  tests cover the edges.
- **`polytrack/generator.py`** — lays out self-avoiding grid walks with starts,
  straights, sharp turns, checkpoints and a finish, and returns the route by
  construction. Every constant was measured, not guessed: `PART_SIZE` is 5 and a
  `Straight` is 20 world units, so a piece spans exactly four grid units, which
  is why every coordinate in a shipped track is a multiple of four.
- **What does not work yet.** A straight-only generated track drives and
  finishes. A track with turns does not: the car reaches the first corner and
  bounces. The road-surface footprint confirms `TurnSharp` opens on local +z and
  +x exactly as modelled, so something in the placement is still wrong. An
  earlier adjacency check that appeared to pass was measuring vertex extent
  rather than openness and proved nothing.

`tests/test_generator.py` pins the half that works, including a straight-only
track driven end to end on the real module. It deliberately does not assert the
turn failure — that is the open bug, not a property to lock in.

One earlier check is worth recording as a warning: an adjacency test measuring
how far each piece's vertices reached along the connecting direction reported
"0 broken connections" on a track the car could not drive. It was measuring
extent, not openness — a barrier reaches a boundary exactly as well as an
opening does — so it confirmed nothing while looking like proof.

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
  search.py       BeamSearch (+ policy prior), CEMSearch, RefinementSearch
  spline.py       Arm 3: steering relaxation + separable CEM
  generator.py    procedural layout (in progress, uncommitted)
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

`PhysicsBackend` is the seam. Everything above it is written against the protocol
and never touches a simulator directly.

### Design decisions worth knowing

**Frame-indexed, never time-indexed.** Everything counts physics frames, and a
PolyTrack frame is **one millisecond** — the physics runs at 1000 Hz, not the
60 Hz this project assumed until the real backend was built. Wall-clock never
enters the simulation. This is what makes runs reproducible and what made
Polybot's screen-scraping approach fail.

Because 1000 decisions a second is not an action space, an agent holds one action
across a **control interval** — 50 ms by default, 4.2 m at 300 km/h. Reward
coefficients are denominated per second rather than per frame, so the same
configuration means the same thing whichever backend is attached.

**A backend knows its own rate; nothing above it may assume one.** The stand-in's
frames are sixtieths, so anything formatting one without dividing by
`backend.rate` reports a time 16.67× too long. `SearchResult` carries a `rate`
field because a bare frame count is not a time.

**Observations are car-local.** World-frame coordinates would let the network
memorise "at position (412, 88), turn left", which collapses on unseen tracks.

**Speed-scaled lookahead.** GT Sophy samples the track ahead over roughly the
next 6 seconds of travel. Neinders traced TMRL's plateau at "safe but slow"
directly to its lack of this.

**Progress is tracked statefully, with a high-water mark.** Nearest-point
projection is genuinely ambiguous wherever a track passes near itself. On the
oval, a car 1 cm *behind* the start line projects onto the track's end and reads
as a completed lap — beam search found this in seconds and returned a "winning
run" of one frame of reverse. The high-water mark stops an agent oscillating over
one stretch and farming reward forever.

**A reference line is a route, not a centerline.** A track code says where every
piece of road is and nothing about the order they are driven in, so the route
comes from a lap that was actually driven. Sampling that replay must not perturb
it — runs of held input are cut at input boundaries, not on a fixed grid.

**The size of a geometry error is no guide to the size of its consequences.**
Part 159's collision mesh was wrong by 1.8e-4 — invisible, drivable at 308 km/h.
It still threw a 400 km/h wall-ride clean off the track thirteen seconds later,
and Winter 5 was the only one of four tracks it broke.

**This track punishes millisecond input-timing error.** Projecting the 19.025
record onto a fixed block grid and replaying it reaches 26 m at 50 ms, 349 m at
25 ms, 477 m at 10 ms and 468 m at 5 ms — of 1747. Zero timing error replays
exactly. At a two-block spline knot spacing, 97.8% of the actions are identical
and the car still reaches only 72 m against 477.

## Throughput, and a Windows trap worth knowing

The physics module costs **~650–800 µs per 1 ms tick** on a quiet machine,
roughly 1× realtime, nearly independent of track size — so it is the module's own
fixed work, not anything in this repo. Parallel collectors are mandatory rather
than an optimisation.

**Windows confines a detached process to the efficiency cores at their lowest
clock.** Anything launched in the background — `nohup`, a background shell, a
scheduled task — is tagged with background quality of service ("EcoQoS"), and on
a hybrid CPU that is not a small penalty: **52.6 s against 3.6 s** on a fixed
single-threaded benchmark, and about 2.2× on the wasm workload. Every long-running
entry point calls `lift_background_qos()` first.

The symptom is deceptive and cost an afternoon. A tagged training run showed
every python process pegging the four E-cores at 100% while the eight P-core
threads sat at 2–24%, **total CPU 29%**, clocks boosting normally at 110% of
base, no thermal cap, no paging. That reads as "the machine is three-quarters
idle, so the learner must be blocked on a lock". It is not — the learner is
running flat out on a core eight times too slow. The tell is that *everything*
slows by the same factor at once, which a lock does not do. Measured here:
achieved ratio **0.11 → 0.46**, **38 → 75** transitions/second, gradient step
262 ms → roughly 30.

## What this does not show

Stated plainly, because the result above is easy to over-read:

- **One track.** Winter 5 only. Until the generator lands this is a memorisation
  problem, not a generalisation one.
- **The policy alone is slower than the human.** 19.422 against 19.025. Every
  gain came from search, and the search was steered by a human racing line.
- **n = 1 everywhere.** No seeds, no repeats, no error bars. The three-arm
  comparison varies several things at once and two of its arms share an author.
- **No comparison to prior work.** Linesight, TMRL and GT Sophy are the
  references the design derives from; nothing here is measured against them.

## Where this goes next

Ordered by what unblocks what, not by appeal.

### 1. Finish the procedural generator — the blocker for everything else

Until it lands, four tracks is a memorisation problem and no generalisation
claim is available. Two thirds are done: the encoder is validated against the
game's own loader, and layouts are correct, self-avoiding and drivable when
they contain no turns.

**The open bug.** A track containing a `TurnSharp` lays out and encodes
correctly, but the car reaches the first corner and bounces. What is known:

- The start spawns on the first cell facing the second exactly (alignment
  1.000), so `_HEADINGS` is right.
- A straight-only track drives to the finish, so starts, straights,
  checkpoints and finishes are all placed correctly.
- The road-surface footprint confirms `TurnSharp` opens on local +z and +x and
  is closed on local −z, which is the connectivity model the walk assumes.
- The car stops ~1.7 units past the cell boundary it should be entering
  through, losing wheel contact — consistent with striking a step rather than
  falling into a void.

What to try next, cheapest first: compare a generated turn against a turn in a
shipped track at the same rotation, cell-for-cell, since the official tracks are
known-good layouts; check whether the turn's road surface sits at a different
height than the straight's at the shared boundary; and consider `TurnShort` or
`TurnLong` instead, which spread the corner over more than one cell and may be
what official tracks actually use for connected 90° turns.

### 2. Held-out evaluation

Train on generated tracks, evaluate on unseen generated tracks, and keep the
four official tracks as a transfer set that was never trained on. This is the
experiment the whole architecture was designed for and none of it has been run.

### 3. Baselines with variance

Several seeds, error bars, and at least one standard algorithm besides this
IQN. Every number in `arms/` today is n=1, which is fine for engineering and
not fine for a claim.

### 4. The other three tracks

Summer 1, Summer 5 and Desert 3, **without** `--forward-only` — they brake for
8–21% of the lap and would be unfinishable with it.

### 5. Sample efficiency

The largest available speed-up is not throughput. Episodes re-drive the first
half of the track constantly; Linesight starts them from points spread along the
reference line instead. That needs `save_state`, and `save_state` turns out to
be affordable, so this is mostly plumbing.

## Known loose ends

- **`generator.py` turn placement** — the blocker above.
- **Reward coefficients in `RewardConfig` are placeholders.** There is finally a
  real backend to tune them against and nobody has.
- **`ObservationBuilder._cast_rays` approximates the corridor** from the
  reference line and `Track.width`. With the real backend that measures distance
  from *the driven lap*, not from a wall — the module has the collision geometry
  and this should ask it. Isolated in one method for exactly that reason.
- **47 legacy pre-`PolyTrack2` community tracks still do not decode**, which
  shrinks the corpus a generator could learn a style from.
- **`WasmPhysics.save_state` pickles the whole 24 MB heap.** Affordable (13.9 ms)
  but narrowing it would cut search memory, which is the real constraint.
- **`quarantine/duplicate_search_package/`** — a second, function-based search
  implementation written before a module-shadowing collision was noticed. Port
  its anchored windowed refinement or delete it.
- **The machine sleeping has damaged five runs.** `powercfg /change
  standby-timeout-ac 0`. One Arm 3 run lost four hours mid-pass, visible as a
  jump from 77.3 to 323.4 minutes in its log.

## If this is going to be published

The honest framing is **not** "we beat a human on Winter 5". It is *faithful
environment construction from a shipped game binary, validated to
frame-exactness, with a procedural track distribution* — where agent results
demonstrate the environment is usable rather than carrying the claim.

That framing fits a benchmark/environment venue (NeurIPS Datasets & Benchmarks,
IEEE Conference on Games, AIIDE, RLC) and the validation standard is the unusual
asset: most RL environment papers reimplement dynamics and can establish no
fidelity at all, whereas this one can point at the game's own verifier agreeing.

It needs items 1–3 above, and one thing outside this repo entirely:

**Redistribution.** The environment depends on `polytrack_physics.wasm` and
geometry derived from it. *A benchmark nobody can run is not a benchmark*, and
this is the first question a reviewer will ask.

Half of this is now settled: Kodub has confirmed that **using** the physics WASM
directly is fine, under the conditions in
[Permission, attribution, and the leaderboard](#permission-attribution-and-the-leaderboard).
That removes any doubt about the project's legitimacy and makes the
extract-from-your-own-copy path a sanctioned one.

What is still open is **redistribution** — shipping the module inside the repo
so a reviewer can `pip install` and run. The ToS does not address it, and the
permission given was to use. Until that is asked and answered, reproduction
requires a copy of the game, which is a weaker story than a benchmark venue
usually wants.

## Reproducing this

Two files the backend needs are **not in this repo**, because both derive from
Kodub's game: `polytrack_physics.wasm` and `geometry.npz`. Extract them from a
build you own and keep them local. The collision geometry is *captured, not
reimplemented* — `tools/capture_geometry.py` serves your local client over
127.0.0.1 and intercepts the `Init` message it sends its simulation worker, which
carries every part's collision mesh. `tools/extract_geometry.mjs` reimplements
the pipeline in real three.js and got 185 of 186 parts bit-identical; the one it
missed cost more time than everything else in Stage 0 put together.

`tracks.npz` is likewise gitignored — it embeds the game's track content.
`records.json` is committed: it holds input toggles, frame counts and track
content-hashes, nothing of the game's.

Pin game version **0.6.2**. Replays, track codes and times are all
version-sensitive, and the worker throws on a mismatch.

The substrate every result above was produced against, frozen and checked by
`scripts/verify_baselines.py`:

| file | sha256 (first 16) |
|---|---|
| `polytrack_physics.wasm` | `d4ef02676973d41a` |
| `geometry.npz` | `6fb32e3b719811dc` |
| `tracks.npz` | `5cdc6a530cf527c7` |
| `records.json` | `f87af733f13d7b0f` |

**This is the open reproducibility problem.** A benchmark nobody else can run is
not a benchmark, and the extraction path requires a copy of the game.

## Permission, attribution, and the leaderboard

**PolyTrack is by [Kodub](https://www.kodub.com/). Play the official game at
[CrazyGames](https://www.crazygames.com/game/polytrack).** This project is an
unofficial research tool. It is not affiliated with, endorsed by, or sponsored
by Kodub.com, CrazyGames, Poki or itch.io, and it is entirely non-commercial.

Kodub was asked directly before any of this was published, and replied
(Discord, 8 September 2026):

> Using the physics WASM directly should be fine! Please read section 4 of the
> Terms of Service here for details […] Also, please make sure people can't
> cheat using the tool, and that it doesn't upload runs to the leaderboard.

Both conditions are met by construction, not by promise:

- **`tools/watch_run.py` cannot reach the leaderboard.** It serves your own copy
  of the client over `http://127.0.0.1`, and hooks both `window.fetch` and
  `XMLHttpRequest.prototype.open` to reject any request matching
  `vps.kodub.com`, `leaderboard` or `/api/`. The page carries a visible banner
  saying uploads are blocked.
- **It cannot touch your real save.** `http://127.0.0.1` is a different origin
  from the Electron app's `file://`, so the served copy gets its own empty local
  storage. Your personal bests are never read, written or reachable. A fresh
  origin also has no session to upload with — the request block is belt and
  braces on top of that.
- **Nothing in this repo makes an outbound request to Kodub's servers.** The
  only HTTP is a localhost server for your own client copy. `vps.kodub.com`
  returns 403 on every path for this client; that is an answer, not an obstacle
  to route around.

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted
to official leaderboards. The PolyTrack community draws the same line, and
[section 2 of the ToS](https://www.kodub.com/terms/polytrack) prohibits
submitting manipulated records outright.

### What is still an open question

Kodub's answer grants **use**. It does not settle **redistribution**, and ToS
section 4 does not address it either — that section governs mods and reuploads
of the game, and an extracted `.wasm` shipped inside a research repo is neither.

So `polytrack_physics.wasm`, `geometry.npz` and `tracks.npz` stay out of this
repository until that is asked and answered explicitly. Extract them from a copy
you own; see [Reproducing this](#reproducing-this). This is the open
reproducibility problem, and it is a question for Kodub rather than something to
decide unilaterally.
