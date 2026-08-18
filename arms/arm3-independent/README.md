# Arm 3 -- independent approach

**Winter 5: not yet attempted.**

A separate approach to the same problem, developed without reference to Arm 2's
outputs. Drop the result here as `winter_5.json` in the same shape as the other
arms and add it to `LAPS` in `scripts/verify_baselines.py`.

## What counts as a result

1. `winter_5.json` in this folder with `track`, `frames` and `recording`
2. `python scripts/verify_baselines.py` reports PASS -- the finish frame
   reproduces exactly on a fresh backend, with every checkpoint
3. The budget it consumed is reported: simulated frames for search
   (`SearchResult.n_frames_simulated`), gradient steps for training
4. The substrate hashes in `EXPERIMENT.md` are unchanged

## What isolation means here

Arm 3 must not consume Arm 2 trajectories, tuned coefficients, or checkpoints.
It **may** use the shared substrate: the physics backend, `geometry.npz`,
`tracks.npz`, the environment, and the gate. Those are the problem, not a
solution to it.

`tracks.npz` is frozen for the duration. It carries the Winter 5 reference line,
which is the human 19025 lap, and Arm 2's result comes from ranking against that
line. Rebuilding it from Arm 2's own output would put Arm 2 into the shared
substrate. If Arm 2 wants an improved line it goes in a separate file that only
Arm 2 reads.

Neither `polytrack_physics.wasm` nor `geometry.npz` may be sent to anyone. They
are derived from Kodub's game. Extract them from a copy you own with
`tools/capture_geometry.py` and `tools/build_geometry.py`. Pin game version
**0.6.2** -- v0.6.0 rewrote the physics and times do not cross versions.

## Insight leaks even when data does not

Arm 2 established by measurement that the entire Winter 5 deficit is *route* --
not skidding, braking, wall contact or steering effort. That finding cost seven
failed searches to reach, and anyone who reads it starts here with the answer.

Record what was shared and when. Claiming clean-room independence afterwards is
exactly what peer review is built to catch.

## Approaches that are different in kind

A different RL algorithm is a variation, and a reviewer will say so. Genuinely
distinct decompositions:

* **Geometric racing line plus a tracking controller** -- minimum-curvature path
  through the corridor, then a controller that follows it. The classic F1TENTH
  approach. Note this is also what Arm 2's findings point at, so it is the option
  most exposed to the leak above.
* **Continuous trajectory optimisation** -- CMA-ES over a parameterised spline
  rather than beam search over discrete input blocks. `CEMSearch` exists in
  `polytrack/search.py` as a weak starting point.
* **Behaviour cloning from the human lap, then fine-tuning** -- a supervised
  start instead of reinforcement from scratch.
