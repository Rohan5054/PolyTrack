# Updated plan: what to cut, and what to do instead

> Written 21 October 2026, in response to `PAPER_PLAN.md` on the
> `transfer-summer1` branch. That document is a research programme; this one is
> a scope cut. It keeps every finding in it and drops most of the work.
>
> `PAPER_PLAN.md` is not on `main` yet. Read it there first — the Summer 1
> results in `arms/transfer/README.md` are the evidence this plan assumes.

## The argument in one paragraph

The environment, the three-seed generalisation result, the support measurement
with its control and confound arms, and the gate-verified 18.786 on Winter 5
are already a coherent paper. `PAPER_PLAN.md` §3 would make it a better paper;
nothing in §3 makes it a paper. Executed as written the programme is **5–7
weeks full-time**, or 2½–4 months of evenings. The version below is **about two
days, plus one lap played**, and it spends that time on the one open item that
could *subtract* from what already exists.

## 1. Verified before cutting anything

Three of `PAPER_PLAN.md`'s claims bear on text already pushed to `main`. All
three were checked against the code rather than taken on trust.

**§2.1, the discount — confirmed.** `IQNConfig.gamma` is 0.999 with a comment
arguing that 0.99 is a five-second horizon and cannot value the far side of a
track (`polytrack/learning/iqn.py:68-74`). `scripts/train.py:105` defaults
`--gamma` to 0.99 and `train.py:192` always passes it, so the 0.999 default has
never taken effect. `arms/generalisation/agent.pt` confirms it: its config
records `gamma=0.99`. Every policy trained through `train.py`'s defaults used
the horizon the code argues against.

**§2.2, the step-indexed schedule — confirmed, and CLAUDE.md and PAPER.md §8.5
were wrong.** The prioritised replay anneals its importance-sampling exponent
from 0.4 to 1 over `beta_steps = 500_000`, and `train.py:327` calls
`replay.sample(args.batch_size, agent.steps)`. An arm resumed from the 1.22M-step
checkpoint samples at beta = 1.0 from its first batch while a fresh arm starts at
0.4 and reaches 0.58 by 150k. Both files claimed no step-indexed schedule was in
play; that was concluded from epsilon being Ape-X fixed-per-worker, which does
not follow. Corrected in `bc5e8c5`. The `--reset-optimiser` flag keeps the
counter, so weights-only loading still needs fixing in code.

**§7.3, the 18.809 — not in this repository.** `records.json` holds eight
entries. Winter 5 (`2fe4bd46…`) has **19025 and 21899, and nothing else**; no
entry anywhere in the file is 18809. Whatever holds that lap is on the machine
that ran the Summer 1 experiment. One datum argues against it being human:
`arms/arm2-search/lineage/` contains 19249, 19245, 19405 and 19420, all search
outputs, and an 18.809 would sit naturally in that series.

## 2. Do these, in this order

| | what | cost |
|---|---|---|
| 1 | **Resolve the 18.809.** | ~1 h |
| 2 | **Email Kodub about redistribution** (`PAPER_PLAN.md` §4). | today |
| 3 | §2.1 gamma fix, and state the discount in PAPER.md. | ~1 h |
| 4 | §2.2 weights-only loading, with a test. | ~1 h |
| 5 | `PYTHONPATH` note in the README. | 5 min |
| 6 | M5 cost figures into PAPER.md §3.6/§8.6. | one sentence |
| 7 | **Winter 1**: play a lap, measure support distance, evaluate zero-shot. | ~2 h |
| 8 | *Optional:* §3.1 on Desert 3 only. | ~1 day |

### 2.1 Resolve the 18.809 first

It is the only open item that can reduce a claim the paper already makes. If an
18.809 human lap on Winter 5 exists, the headline margin falls from **239 ms to
23 ms** and §6.3 has to be rewritten. If it is a search output, it belongs in
`arms/arm2-search/lineage/` and the margin stands.

How to tell: a human lap has a `recording` extracted from the client's local
storage by `tools/extract_records.py`, paired with the finish frame the game
itself recorded. A search output has a JSON artifact in `runs/`. Check which
file it came from, then run it through `scripts/run_gate.py` and
`tools/verify_in_client.mjs` either way.

Whatever it turns out to be, report both times in §6.3.

### 2.2 Email Kodub today

`PAPER.md` §9.2 calls redistribution the paper's open problem, and it is the
single biggest lever available: permission turns this from code that describes
an environment into an environment others can run, which is the first thing a
benchmark venue checks. The reply takes days to weeks, so every day of delay is
pure loss and the cost is one message.

Ask explicitly whether `polytrack_physics.wasm` and the derived
`geometry.npz` may be redistributed alongside the code, under whatever
conditions they choose. The existing permission (Discord, 8 September 2026)
covers **use** only.

### 2.3 The gamma fix, and what not to do with it

Fix: `--gamma` defaults to `None`, meaning "use `IQNConfig`'s default", and the
collectors' n-step discount comes from `agent.config.gamma` rather than
`args.gamma` (`train.py:216`). Those two agree today only because both are
0.99; after the first change, resuming an existing checkpoint without
`--gamma 0.99` would have the collectors accumulate returns at 0.999 while the
learner bootstraps at 0.99 — the exact disagreement the comment below that line
forbids.

**Do not retrain §6.1.** Treat 0.99 as the shipped configuration, state it in
the paper, and leave 0.999 as a future ablation. Rebuilding the headline
invalidates comparability with everything banked — including §6.2's support
band, which is a property of one specific policy — for a hypothesis nobody has
tested. If 0.999 later proves clearly better, rerun then and report both.

### 2.4 Winter 1 instead of elevation in the generator

`PAPER_PLAN.md` §3.2 proposes adding ramps and jumps to the generator, which
turns the self-avoiding walk into a 3-D problem, needs the scripted driver to
manage speed at a jump, and is open-ended ("then banking"). One to two weeks.

The same prediction can be tested across tracks instead of by intervention.
Winter 1 is the flattest of the 17 official tracks. Play one lap, build its
reference line, run `scripts/measure_support.py`, and evaluate the existing
policy zero-shot on it.

**Pre-specified, before the lap is played:**

* Support distance is reported as features outside the band per frame, pooled
  over three greedy attempts, against the four official tracks' 7.0–9.2 and the
  held-out generated tracks' 0.0–0.1.
* Zero-shot progress is the furthest fraction of the lap reached in ten greedy
  attempts at a 3× par budget, against Summer 1's 23.5 % and the others' 1.5–3.1 %.
* **Supports the mechanism** if Winter 1 is both measurably closer to the band
  than Summer 1 *and* reaches further.
* **Against it** if Winter 1 is closer and does not reach further, or reaches
  further while sitting no closer. Either outcome is reportable and the second
  is the more interesting.
* **Uninformative** if Winter 1 turns out no flatter in the observation than
  Summer 1, which the measurement itself will say.

This is weaker evidence than §3.2 — a correlation over five tracks, not a
manipulation — and the write-up must say so. It costs about 1 % as much, and a
positive result is the thing that would justify building elevation later.

### 2.5 Desert 3, if there is appetite for one more headline

`PAPER_PLAN.md` §3.1 budgets five tracks, three seeds each and ~40 h, and it is
costed as though each track needs a policy that can complete a lap. **It does
not.** Winter 5's 18.786 came from a beam *guided by the human lap*, with the
policy serving only as the action prior — and Summer 1, Desert 3 and Summer 5
all have human laps in `records.json` already. So the guide exists on each
track without `PAPER_PLAN.md` §2.3 and without any policy ever finishing.

Do Desert 3 and stop. One prior policy at one seed — the lap time is verified
by the gate and does not need error bars — then windowed search, working
backwards from the finish as CLAUDE.md describes. A second gate-verified track
beating a human time is worth more than the fifth track would be.

Every reported lap goes through `scripts/run_gate.py` **and**
`tools/verify_in_client.mjs`. No run is ever submitted to the leaderboard.

## 3. Cut, with the reason

| cut | why |
|---|---|
| **§2.3** start episodes along the track | 2–3 days and the only change here that touches reward and progress semantics: `ProgressTracker` initialised mid-lap, par rescaled for the finish bonus, recordings exported only from start-line episodes, an LRU over 24 MB states. CLAUDE.md then requires a search run against the changed reward, since beam search found both previous progress bugs in seconds. It accelerates training; no claim in the paper depends on it. |
| **§3.3** transfer across tracks | ~60 h — the plan says run it twice and the order table counts it once. Summer 1 already gives the weak/null result and §6.2 explains it. A dose-response curve is a nice-to-have. |
| **§3.2** elevation in the generator | 1–2 weeks, open-ended. Replaced by §2.4 above. |
| **§3.1** beyond one extra track | Five tracks is not five times more convincing for an environment paper, and each new track needs a lap played and raises the human-baseline question again. |
| **§4** PPO or DQN baseline | "Reviewers expect two algorithms" holds for an algorithms paper. This is an environment paper. Expect it as the likeliest reviewer complaint and answer it in §7 rather than running it. |
| **§4** ablations | Lookahead horizon, action restriction, control interval. They defend numbers that already exist; none adds a claim. |
| **§2.4** evaluation episodes | Half a day. Its value was clean curves and correct first-lap measurement, and most of that was needed *because* §2.3 moves where episodes start. With §2.3 cut, `learning_curve.csv` already has the 121 points Appendix A's figure needs. |
| **§2.5** the SIGINT crash | A nuisance during interruption, not a result. The `PYTHONPATH` half of §2.5 is kept — §9.1 claims the tests run from a clone. |

## 4. Answers to `PAPER_PLAN.md` §7

1. **Discount.** Keep 0.99, state it, do not retrain. See §2.3 above.
2. **Tracks.** Winter 1 for the support test, because flat is the whole point of
   choosing it. Desert 3 for the optional second headline, because it already
   has a human lap. Nobody needs to play anything except Winter 1.
3. **Human baseline.** Resolve the 18.809 first; the answer decides this. Report
   our own laps as what they are — laps played by the author — and if leaderboard
   tops are read by hand in the game, report both and label them. Never the API:
   `vps.kodub.com` returns 403 and CLAUDE.md is right that this is an answer and
   not an obstacle.
4. **Framing.** Agreed, keep it an environment paper. The environment is the
   finished contribution; §3.1 and the Winter 1 test are evidence that it
   supports research, not the contribution itself.
5. **Kodub.** Whoever has the existing Discord thread, today. Ask for
   redistribution of the module and derived geometry under conditions of their
   choosing.

## 5. Process, kept in full

`PAPER_PLAN.md` §5 is the part the Summer 1 run earned, and it costs nothing:

* **Decide before running.** Tracks, metrics including the censored ones,
  budgets and seeds fixed and written into PAPER.md §8 before the first run.
  §2.4 above does this for the Winter 1 test.
* **Report everything that ran**, Summer 1 included. A result a reviewer can see
  was not selected is worth more than a cleaner one they cannot.
* **Keep diagnoses honest.** "Attempts end on Summer 1's descents and jumps" is
  an observation; "because the generator has no elevation" is a hypothesis, and
  it cannot be the whole story — the scratch policies never saw a generated
  track and fail at the same jumps. Jumps are hard to learn as well as
  unfamiliar.

## 6. What this plan does not do

* It does not produce zero-shot laps on an official track. That needs §3.2 or
  §2.3, and both are cut.
* It does not add a second algorithm, so the paper claims nothing comparative
  about IQN.
* It leaves the discount at the value the code argues against, documented rather
  than fixed in the results.
* The Winter 1 test is correlational. If a reviewer wants the intervention,
  §3.2 is the answer and it is a week or two away, not an afternoon.

Each of those is a sentence in §7 of PAPER.md rather than a gap in the
argument — which is the test this cut was made against.
