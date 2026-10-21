# Plan for the next round of experiments

> Written 20 October 2026, after the Summer 1 transfer experiment. Proposals,
> not decisions: each section says what it would add to PAPER.md, what it
> costs, and what could go wrong. Section 7 lists what we have to decide.

## Summary

1. **Fix three things in the training stack before running anything else.**
   `train.py` overrides the agent's intended discount (0.99 is used by
   default, 0.999 was intended); a resumed run keeps a step-indexed replay schedule the
   docs say does not exist; and every episode starts at the start line, so the
   hard part of a track is rarely practised. All three affect every result from
   here on. (§2)
2. **Rest the results on three claims**, each extending something PAPER.md
   already has:
   - policy + search beats strong human times on several official tracks
     (extends §6.3, which is one track and one sample);
   - widening the generator moves official tracks inside the training support,
     and zero-shot performance follows (§8.1's own falsifiable prediction);
   - transfer to an official track depends on how far it sits outside the
     support (§8.5, measured across tracks instead of on one). (§3)
3. **Ask Kodub about redistribution.** §9.2 calls it the paper's open problem;
   it is the single biggest lever for a benchmark venue, and it costs an email.
   (§4)
4. **Keep the Summer 1 result in the paper.** It is the evidence for the second
   claim, and a diagnosed failure strengthens an environment paper. (§1, §5)

## 1. What the Summer 1 transfer experiment showed

Full results: `arms/transfer/README.md`. Eight runs: a pilot pair at 50k
gradient steps, then three seeds of each arm at 150k. The pretrained arm
started from `arms/generalisation/agent.pt` with weights only (§2.2).

- **No run completed a lap.** The planned primary metric, steps to the first
  lap, is censored in every run, so it cannot compare the arms.
- **A head start that does not last.** The pretrained runs start at 22% of the
  lap, against 1% for scratch. All four of them fall to 13-18% around 25k steps
  and then recover.
- **Then a small, consistent edge.** At 100k steps the pretrained runs reach 44%
  (range 43-45) against 41% (38-43), and mean progress over the run is 35%
  (34-36) against 31% (27-34). All three pretrained runs reach half the lap, at
  105-110k steps; two of three scratch runs do. That is within the noise of
  three seeds.
- **Attempts end on elevation.** Each final policy was driven greedily ten
  times. 55 of the 80 attempts end on a descent or a jump, 33 of them at the
  second jump, 68-76% of the way round. The pretrained policies also lose
  attempts on the 47 m descent where the scratch ones do not: 8 against 4, and
  all 4 scratch ones came from the 50k pilot.

How to report it: **inconclusive at this budget.** Both arms stopped at 150k
steps, inside the range where even the generated-track policy had not learned to
finish (`arms/generalisation/learning_curve.csv`: complete greedy laps only by
~155k, on far easier tracks), so the run cannot say whether transfer helps. The
one finding that does not depend on the budget is the early dip. An earlier
version of this paragraph called it "a weak or null transfer result, consistent
with §6.2"; `updated_paper_plan.md` §0 is right that this over-read it. The
budget question is what `scripts/separator_experiment.py` answers.

## 2. Fix before running anything else

### 2.1 The discount `train.py` uses is not the one the agent was designed for

`IQNConfig` defaults to `gamma = 0.999`, with the reason written beside it
(`polytrack/learning/iqn.py:71-74`): 0.999 at a 50 ms control interval is a
horizon of roughly 35 s, and *"the usual 0.99 is five seconds -- shorter than a
corner sequence, so the agent cannot value the far side of the track at all."*

But `train.py` has its own `--gamma` flag defaulting to **0.99**
(`scripts/train.py:105`) and always passes it in (`scripts/train.py:192`), so
the 0.999 default never takes effect. History: `7ce4bbe` added `train.py` with
0.99 at 17:33 on 8 August; `39c5281`, 35 minutes later, raised `IQNConfig` to
0.999 with that comment and edited `train.py` without changing the flag.
`arms/generalisation/agent.pt` confirms it: its config says 0.99. **Every
policy trained with `train.py`'s defaults -- the generalisation checkpoint
included -- used the horizon the code argues against.** (The Winter 5 and
300k-seed checkpoints are not in the repo to check; unless `--gamma` was passed
explicitly, they did too.)

Fix:

- `--gamma` defaults to `None`, meaning "use `IQNConfig`'s default".
- The collectors' n-step discount must come from `agent.config.gamma`, not from
  `args.gamma` (`scripts/train.py:216`). Today the two agree only because both
  are 0.99. After the fix above, resuming any existing checkpoint without
  `--gamma 0.99` would make the collectors accumulate returns at 0.999 while
  the learner bootstraps at 0.99 -- the exact disagreement the comment below
  that line says must not happen.
- PAPER.md never states the discount. It should, whichever value we keep.

The comparison itself is an ablation worth having (§8.4): 0.99 against 0.999 on
the generated-track setup, three seeds of 300k steps each, about 45 minutes per
seed on the M5.

### 2.2 A resumed run keeps a step-indexed schedule

The prioritised replay anneals its importance-sampling exponent from 0.4 to 1
over the first 500k learner steps (`polytrack/learning/replay.py:158-159, 216`),
keyed on `agent.steps`. `--reset-optimiser` drops Adam's moments but keeps the
counter, so a run resumed from the 1.22M-step checkpoint samples with the
exponent at 1.0 from its first batch while a fresh run starts at 0.4 (and
reaches only 0.58 by 150k). It is the same kind of early step-size difference
the flag exists to remove.

CLAUDE.md ("Two confounds that are already handled") and PAPER.md §8.5 ("Two
things are matched without any work") both say no step-indexed schedule is in
play. This one is.

Fix: make weights-only loading mean weights only -- online weights into both
networks, fresh optimiser, counter at zero -- either inside `--reset-optimiser`
or as a separate `--weights-only`, with a test beside
`test_loading_can_keep_the_weights_and_drop_the_optimiser`.
`scripts/transfer_experiment.py` already does this by writing
`runs/xfer-init/agent.pt`, so the Summer 1 runs above are not affected.

### 2.3 Start episodes along the track, not only at the line

`PolyTrackEnv.reset()` always starts at frame 0 (`polytrack/env.py:217-228`).
On Summer 1, 33 of 80 greedy attempts by the final policies end at the second
jump, 68-76% of the way round. Each of them re-drives two-thirds of the lap to
get a few seconds of practice at the part that fails. CLAUDE.md already calls this "the biggest available
speed-up" and Linesight does it.

Design:

- Per track, replay the reference lap and `save_state` at K evenly spaced arcs
  (13.9 ms each). For an official track the lap is the human recording; for a
  generated track, the scripted driver's.
- At reset, start from a random saved state with probability p, else from the
  line.
- **Memory.** A state is 24 MB. K = 12 on one track is about 290 MB a
  collector, 1.7 GB for six -- fine for single-track training. Across 48
  generated tracks it is not (14 GB a collector), so multi-track training needs
  states built lazily with an LRU, or the narrower `save_state` CLAUDE.md lists
  as a loose end.

Invariants this touches -- each needs a test:

- `ProgressTracker` must be initialised at the start arc. Progress is stateful
  for a reason (CLAUDE.md, "Progress must be tracked statefully").
- The finish bonus is `par / elapsed`. From a mid-lap start, par must be scaled
  to the distance remaining, or the bonus switched off for those episodes.
- A lap exported as a recording must come from a start-line episode.
- Run a search against the changed reward before trusting it (CLAUDE.md, "A note
  on search").

### 2.4 Measure progress with dedicated evaluation episodes

The `lap` and `fin` columns are computed over the last 100 episodes of six
actors whose epsilons run from 0.4 to 0.0007, and "first lap" counts a finish
by any of them. That mixes exploration into the metric and makes curves noisy
(`xfer-scratch-1` went 43% -> 30% -> 60% between 100k and 150k steps).

Add an evaluation pass: every 10k gradient steps, a few greedy attempts from the
start line in a separate worker, logging the fraction reached and whether the
lap finished. It gives clean learning curves for figures, and it is the only
correct way to measure first laps once §2.3 changes where episodes start.

Define the censored metrics before running: steps to reach 25 / 50 / 75 % of
the lap, area under the progress curve, and steps to first lap where one
exists.

### 2.5 Smaller things

- **Collectors crash on Ctrl+C.** Interrupting a run left four macOS crash
  reports at once, each a SIGSEGV in
  `wasmtime::runtime::func::HostFunc::array_call_trampoline`: SIGINT landing
  during a host call. Workers should ignore SIGINT and let the parent shut them
  down.
- **The test suites assume the package is installed.**
  `python tests/test_learning.py` fails with `No module named 'polytrack'`
  unless the repo is on `PYTHONPATH`; the scripts add it themselves.

## 3. The results the paper should rest on

### 3.1 Policy + search beats strong human times on several official tracks

**Why.** It is the most reliable headline we can produce: concrete, verifiable
by the game's own replay checker, and in the frame racing papers are judged in.
§6.3 already has it on one track:

| Winter 5 | |
|---|---|
| Human reference lap | 19.025 s |
| Policy alone, best of 20 | 19.422 s |
| Policy + guided beam search | 18.786 s |

That is one track and one sample (§8.2 says so).

**What.**

- Tracks: Winter 5 (done), Desert 3, Summer 1, Summer 5, and Winter 4 (a local
  lap exists, 19.980 s). Any further track needs one lap played.
- Per track: train with the §2 fixes, all nine actions (only Winter 5 justifies
  forward-only), three seeds, then guided beam search and refinement on the
  best policy.
- Verify every reported lap with the gate and `tools/verify_in_client.mjs`.

**The human baseline needs to be stronger.** The times in `records.json` are our
own laps. A local `records.json` also has an **18.809** on Winter 5; if that is a
human lap, the search's margin is 23 ms, not 239. To read as superhuman, compare
against the top leaderboard times, **read by hand in the game**, and report both.
Never use the leaderboard API -- `vps.kodub.com` returns 403, and CLAUDE.md
is right that "that is an answer, not an obstacle to route around" -- and never
submit a run.

**Cost.** At ~110 gradient steps/s on the M5, 500k steps is about 75 minutes
per seed, so roughly 4 hours of training per track plus search.

**Risk.** Summer 1 and any track with jumps may still not yield a finishing
policy; §2.3 is the main mitigation. Summer 5 is 57 s long.

### 3.2 Widen the generator; show official tracks move inside the support

**Why.** It is the most interesting result if it works, and PAPER.md §8.1
already states it as a falsifiable prediction: the features furthest outside
the training band are vertical velocity (4.4x) and roll rate (10x), so
elevation and banking come first, and adding them should reduce the measured
out-of-support distance on the official tracks before any policy is trained.
Summer 1 now adds evidence: 55 of 80 greedy attempts end on a descent or a
jump, the pretrained policies lose attempts on the descent where the scratch
ones do not, and the generator places no elevation at all.

**What.**

1. Add ramps and jumps to `polytrack/generator.py`; that turns the
   self-avoiding walk into a 3-D problem (§8.1).
2. Teach `polytrack/pursuit.py` to manage speed at a jump, so drivability
   filtering and par times still work.
3. Rebuild the library and run `scripts/measure_support.py` on the official
   tracks. **This is the cheap check:** if the distance does not fall, stop and
   rethink before training anything.
4. Train three seeds of 300k steps (§6.1 found 300k as reliable as 1.22M; about
   45 minutes per seed on the M5), then evaluate zero-shot on the official
   tracks and repeat the transfer test (§3.3).

**Success** is official tracks moving closer to the support *and* zero-shot
progress rising with it -- ideally the first zero-shot laps on an official
track. If the distance falls and performance does not follow, §6.2's mechanism
is wrong, which is still worth reporting.

**Risk.** Mostly engineering: jump geometry, landing, and a scripted driver that
survives both. Then banking.

### 3.3 Transfer as a function of support distance

**Why.** One track gives one censored data point. Several tracks spanning the
support distance turn mixed per-track outcomes into a finding: transfer helps
where the training distribution covers the track.

**What.**

- Tracks spanning §6.2's distance: Summer 1 (least out of distribution, 1.4-1.7
  features per frame), the others (7.4-12.5), and Winter 1, the flattest of the
  17 official tracks (needs a lap played).
- Scratch against pretrained, three seeds each, 500k steps, with §2.1-§2.4 in
  place, measured with §2.4's evaluation episodes.
- Run it twice: once with the current pretrained policy, and once with the
  policy from §3.2. "Transfer improves as support coverage improves" is the
  strongest version of the claim.
- `scripts/transfer_experiment.py` already runs a plan like this (interleaved
  arms, resumable, progress display); it needs a track list and the new
  metrics.

**Cost.** Four tracks x 2 arms x 3 seeds x ~75 minutes is about 30 hours: a few
nights on the M5, or fewer split across machines.

## 4. What else strengthens an environment paper

- **Redistribution (§9.2).** Ask Kodub explicitly whether the physics module and
  derived geometry may be redistributed with the code, under conditions they
  choose. Permission turns the environment into a benchmark others can run --
  the thing a benchmark venue checks first.
- **Baselines (§8.3).** One standard algorithm besides IQN -- PPO or DQN -- on
  the generated tracks and one official track. Reviewers expect at least two.
- **Ablations (§8.4).** The discount (now motivated by §2.1), lookahead horizon,
  action restriction, control interval.
- **Cost figures (§3.6, §8.6) are from the Windows machine** (six collectors,
  50-80 transitions/s). Measured on an M5 MacBook Air with 16 GB: six collectors
  give 190-230 transitions/s and the learner 115-125 gradient steps/s, achieved
  replay ratio about 0.6, about 3 GB of memory in all. Under sustained load the
  fanless machine throttled to 35-100 gradient steps/s; room cooling restored
  most of it. Worth a sentence beside §10's EcoQoS note: wall-clock budgets on
  laptops are not stable.

## 5. Protecting the results

- **Decide before running.** For each experiment, fix the tracks, metrics
  (including the censored ones), budgets and seeds first, and write them into
  PAPER.md §8 before the first run starts.
- **Report everything that ran,** including Summer 1 and any track where
  nothing finishes. A result reviewers can see was not selected is worth more
  than a cleaner one they cannot.
- **Keep diagnoses honest.** "Attempts end on Summer 1's descents and jumps" is
  an observation. "Because the generator has no elevation" is a hypothesis until
  §3.2 tests it, and it cannot be the whole story: the scratch policies never
  saw a generated track and fail at the same jumps. Jumps are hard to learn as
  well as unfamiliar, which is what §2.1 and §2.3 address.

## 6. Suggested order

| | What | Time on the M5 |
|---|---|---|
| 1 | §2.1-§2.5 fixes, with tests | engineering, 1-2 days |
| 2 | Check the fixes on Summer 1: scratch, one seed, 500k steps. Does it get past the jump? | ~75 min |
| 3 | §3.1 on Desert 3 and Summer 1 first, then the rest | ~4 h training per track, plus search |
| 4 | §3.2: elevation in the generator; measure the support distance before training | engineering, about a week |
| 5 | §3.3 across tracks, with the old and the new pretrained policy | ~30 h, overnight runs |
| 6 | Email Kodub (§4) | today |

## 7. Decisions for both of us

1. **Discount.** Keep 0.99 for the existing results and say so, or retrain at
   0.999 (300k steps is about 45 minutes per seed here) and rerun §6.1?
2. **Tracks.** Which official tracks, and who plays laps for the new ones
   (Winter 1 first)?
3. **Human baseline.** Our own laps, leaderboard tops read by hand, or both?
   Is the local 18.809 on Winter 5 a human lap?
4. **Framing.** The title makes this an environment paper. I would keep that --
   the environment is the finished contribution -- and present §3.1-§3.3 as
   evidence it supports real research, not as the contribution itself.
5. **Kodub.** Who writes the email, and what do we ask for?
