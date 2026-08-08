# PolyTrack Agent

An RL + trajectory-search agent for PolyTrack, built on the design the research
review landed on: **a learned policy for generalisation, test-time search for the
last tenths.**

## Status

| Component | State |
|---|---|
| Discrete action space | Done |
| `.tas` format read/write | Done |
| Track code decoder (`PolyTrack2`) | Done |
| Recording / ghost codec | Done |
| Track geometry, windowed projection | Done |
| Sophy-style observation builder | Done |
| Reward function | Done — coefficients still placeholders |
| Gymnasium environment | Done — runs on real physics |
| Reference stand-in physics | Done — retained for tests only |
| **Real WASM physics backend** | **Done. Acceptance gate passes, 8/8** |
| Reference lines from real runs | Done |
| Search layer (beam / CEM / refinement) | Done — validated on the stand-in |
| Procedural track generator | Not started |
| IQN training loop | Not started |
| Policy-guided search | Not started |

85 tests passing. Two suites reach for game files and are stronger when they find
them; without those files they skip and say so.

## Quick start

```bash
pip install -e .
python tests/test_core.py       # 24
python tests/test_search.py     #  9
python tests/test_trackcode.py  #  7
python tests/test_wasm.py       # 17  (needs the game files)
python tests/test_recording.py  # 14
python tests/test_reference.py  # 14  (needs the game files)
python scripts/smoke_test.py
```

## Stage 0 is done

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

Two files the backend needs are **not in this repo**, because both derive from
Kodub's game: `polytrack_physics.wasm` and `geometry.npz`. Extract them from a
build you own and keep them local. The collision geometry is *captured, not
reimplemented* — `tools/capture_geometry.py` serves your local client over
127.0.0.1 and intercepts the `Init` message it sends its simulation worker, which
carries every part's collision mesh. See CLAUDE.md for why that replaced a
185-of-186 reimplementation.

Pin game version **0.6.2**. Replays, track codes and times are all
version-sensitive, and the worker throws on a mismatch.

## Reference lines

A track code is a pile of placed parts. It says exactly where every piece of road
is and nothing at all about the route through them. So the route comes from a lap
that was actually driven: replay a recording through the verified backend, sample
where the car went, resample at fixed arc spacing.

```bash
POLYTRACK_TRACKS=path/to/tracks python scripts/reference_demo.py records.json
```

```
  Winter 5
    recorded    0:19.025  (996 samples, built in 46.0s)
    line        1747.0 m over 875 points, 3 checkpoints
    pace        331 km/h average, 482 km/h peak
```

This is Linesight's design, and the same caveat applies: it is a *reference line*,
not a centerline, so "lateral offset" means distance from that lap. It stops
being a constraint at the procedural generator, which knows each track's route by
construction and needs no lap driven for it.

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

`PhysicsBackend` is the seam. Everything above it is written against the protocol
and never touches a simulator directly.

### Design decisions worth knowing

**Frame-indexed, never time-indexed.** Everything counts physics frames, and a
PolyTrack frame is **one millisecond** — the physics runs at 1000 Hz, not the
60 Hz this project assumed until the real backend was built. Wall-clock never
enters the simulation. This is what makes runs reproducible and what made
Polybot's screen-scraping approach fail — it fought display refresh rate and
accumulated 2–3 ms of error per event.

Because 1000 decisions a second is not an action space, an agent holds one action
across a **control interval** — 50 ms by default, which is 4.2 m at 300 km/h,
just under PolyTrack's 5 m part grid. The game works the same way: recordings
store input *changes*, so a held key spans hundreds of ticks. Reward coefficients
are denominated per second rather than per frame, so the same configuration means
the same thing whichever backend is attached and retuning the interval does not
silently rescale the reward.

**`.tas` is the interchange format.** The environment, the search layer and the
verifier all speak it. It round-trips into a ghost the real game can replay,
which is the only trustworthy proof an agent's run is real.

**Observations are car-local.** World-frame coordinates would let the network
memorise "at position (412, 88), turn left" — which collapses on unseen tracks.
Local-frame features force it to learn geometry instead of location.

**Speed-scaled lookahead.** GT Sophy samples the track ahead over roughly the
next 6 seconds of travel, scaling sample distance by speed. Neinders traced
TMRL's plateau at "safe but slow" directly to its lack of this.

**Progress reward with a time penalty.** Dense progress along the reference line
is what every successful racing agent converged on; sparse finish-time reward
never trains because the agent never finishes by chance. The time penalty is what
converts "drive safely" into "drive fast". Progress uses a high-water mark so an
agent cannot oscillate over one stretch of track and farm reward forever.

**Progress is tracked statefully.** Nearest-point projection is genuinely
ambiguous wherever a track passes near itself, and PolyTrack tracks double back
constantly. `ProgressTracker` windows the projection around the last known arc.
Beam search found the alternative's failure in seconds and returned a "winning
run" of one frame of reverse.

**Projection is windowed.** Only segments near the car are considered. Without
this, step cost grows with track length — a 2 km community track would run slower
than a 200 m official one for no reason.

### Throughput

The physics module costs **~650–800 µs per 1 ms tick**, nearly independent of
track size, so it is the module's own fixed work rather than anything in this
repo. That is roughly 1× realtime, which makes parallel collectors mandatory
rather than an optimisation. Building a reference line for a 19-second run takes
about 46 seconds of wall clock for exactly this reason.

The stand-in backend runs orders of magnitude faster and its numbers say nothing
about the real one.

## Roadmap

1. **Procedural track generator.** The editor plus exportable codes gives an
   infinite training distribution. This is the piece that buys generalisation,
   and Trackmania projects never had it cheaply.
2. **IQN training loop.** Distributional Q-learning with PER and n-step, multiple
   collector processes feeding one learner — the Linesight architecture. Start
   from CleanRL or Tianshou; both have readable IQN.
3. **Policy-guided search.** Feed the trained policy into `BeamSearch` as an
   action prior so search only expands plausible actions. This is where record
   times come from: Yosh's RL agent reached human-record level, but going past it
   needed test-time search and exact-state exploitation.

Before any of that, the reward coefficients need tuning against real physics.
They have never been calibrated because until now there was nothing honest to
calibrate them against.

## A note on leaderboards

Keep bot and TAS runs off the human leaderboard. Linesight states the rule
plainly for Trackmania: their runs are tool-assisted and must not be submitted to
official leaderboards. The PolyTrack community draws the same line.

Check Kodub's current Terms of Service before submitting anything — the game
added a ToS in v0.6.0 and enforces server-side replay verification. Label runs
clearly as TAS/AI and you will get help from the Discord rather than a ban.
