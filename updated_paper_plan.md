# Updated plan: what to cut, and what to do instead

> Written 21 October 2026, in response to `PAPER_PLAN.md` on the
> `transfer-summer1` branch. That document is a research programme; this one is
> a scope cut. It keeps every finding in it and drops most of the work.
>
> `PAPER_PLAN.md` is not on `main` yet. Read it there first — the Summer 1
> results in `arms/transfer/README.md` are the evidence this plan assumes.

> **Revised 21 October 2026**, after comparing the Summer 1 budget against
> `learning_curve.csv`. §0 is new and reorders everything: the Summer 1 runs
> were stopped inside the region where the generated-track policy had not
> learned to finish either, so they cannot yet choose between the options in
> §0b. A 2.5-hour experiment does. §2's order changed, three cuts in §3 became
> conditional, and the discount stopped being a tidy-up.

## The argument in one paragraph

The environment, the three-seed generalisation result, the support measurement
with its control and confound arms, and the gate-verified 18.786 on Winter 5
are already a coherent paper. `PAPER_PLAN.md` §3 would make it a better paper;
nothing in §3 makes it a paper. Executed as written the programme is **5–7
weeks full-time**, or 2½–4 months of evenings. The version below is **about two
days, plus one lap played**, and it spends that time on the one open item that
could *subtract* from what already exists.

## 0. First, two and a half hours that decide which plan to run

**The Summer 1 runs were stopped at a budget the project's own learning curve
says is too short, so they cannot yet distinguish between three multi-week
paths.** `arms/generalisation/learning_curve.csv`, on the generated tracks:

| gradient steps | finish rate | greedy fraction |
|---|---|---|
| 49k | 3 % | 0.30 |
| 73k | 19 % | 0.47 |
| **85k** | 33 % | **0.67** |
| 142k | 55 % | 0.93 |
| **155k** | **70 %** | **1.00** |

Summer 1 at 150k steps reached a greedy fraction of **0.49** (0.37-0.72) with no
finishes — roughly where the generated-track policy sat at **85k**, and far
behind the complete laps it was driving by 155k. The generated figures pool 48
tracks including short ones, so `greedy_fraction` is the fairer column; it is
also the one showing the larger gap. A 23.8-second track with a 47 m descent and
two jumps producing no finishes by 150k is what that curve predicts.

`PAPER_PLAN.md` §1 concedes the budget point — "the first-lap metric needs a
much larger budget" — and then reports the run as "a weak or null transfer
result, consistent with §6.2". **That is over-reading it.** If both arms were
stopped before either could plausibly finish, the honest label is
*inconclusive at this budget*, not weak transfer. The one result that survives
is the early dip: all four pretrained runs start at 22 %, fall to 13-18 % by
25k, and recover. That is a real finding about unlearning and does not depend on
the budget.

Worse, **budget is confounded with the discount**. §1 below confirms every run
ever done used gamma = 0.99, a 5-second horizon at a 50 ms control interval. At
the second jump, where 33 of 80 attempts died, the remaining 7.6 s of lap is
discounted to **0.216** of face value while slowing costs progress reward
immediately at 1.000; at gamma = 0.999 the same remainder is worth 0.859. And
since no episode ever finished, the `par / elapsed` finish bonus — the project's
only reward for finishing rather than progressing — **was never once in the
replay buffer**. A policy optimising progress rate under a 5-second horizon
carries maximum speed into everything, and a jump is precisely the feature that
punishes that.

Nothing in the Summer 1 data separates "horizon too short to value surviving the
jump" from "stopped before it learned the jump".

**The separator: two scratch runs on Summer 1 at 500k steps, gamma 0.99 against
gamma 0.999, nothing else varied.** About 70 minutes each on the M5, 2.5 hours
in total.

| outcome | what it means | what to do |
|---|---|---|
| both finish | it was budget | rerun the transfer experiment at a real budget before claiming anything from it; gamma is a tidy-up |
| only 0.999 finishes | horizon | a finding worth a paragraph, and gamma stops being a tidy-up |
| neither finishes at 500k | the jumps themselves | §3.2 or §2.3 earns its two weeks, and §6.2's mechanism is on firmer ground |

Two and a half hours to learn which of three multi-week paths is the right one.
**This comes before anything in §3 of either document**, and before committing
to any option in the table below.

## 0b. Three options, not two

Which of these is right depends on §0's outcome. The cut below otherwise
optimises for submitting what exists — an assumption about the goal, not a
conclusion from the evidence, and one that should be stated plainly because
changing it changes which cuts are justified.

**One cut item could change what the paper claims. Everything else is rigour.**

§3.2, widening the generator — together with §2.3, which makes the resulting
policy trainable in reasonable time — is the only item that could produce a
first zero-shot lap on an official track. That is a new claim, not better
support for an existing one:

> **Now:** trained on generated tracks, generalises within that distribution,
> fails on all four official tracks, and §6.2 measures why — 147x outside the
> policy's own observation support, with vertical velocity and roll rate the
> features furthest out.
>
> **With §3.2:** and when exactly those features were added to the generator,
> the measured distance fell and the policy drove human-designed tracks it had
> never seen.

The second is a demonstrated mechanism rather than a diagnosis, and it is the
strongest available use of §6.2, which made a falsifiable prediction that
nothing else on the list tests.

It may also fail, and there is already evidence against it: the Summer 1 scratch
policies never saw a generated track and **fail at the same jumps** as the
pretrained ones. If jumps are hard to *learn* rather than merely unfamiliar,
widening the generator buys nothing. Near even odds, which is good for research
and is not a certainty.

Everything else that was cut — §3.3's 60 hours, tracks beyond a second one, the
ablations, a second algorithm, evaluation episodes — makes existing claims
harder to attack without changing any of them.

| option | cost | what it buys |
|---|---|---|
| **this plan** (§2 below) | ~2 days + one lap | submit what exists, honest and defensible |
| **this plan + §2.3 + §3.2** | ~2-3 weeks | one shot at a qualitatively better claim |
| **`PAPER_PLAN.md` in full** | 5-7 weeks | the above, plus rigour that does not change the claim |

The middle option captures nearly all of the upside, because the expensive half
of the programme is also the pure-rigour half. **Do §2 of this document first in
every case** -- the 18.809 and the Kodub email are on the critical path for all
three.

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

**The 150k budget — checked against the project's own curve.** The table in §0
is read from `arms/generalisation/learning_curve.csv`, which is committed, and
from the per-run columns in `arms/transfer/*/history.json`. The comparison is
not exact — one is 48 pooled tracks, the other is one harder track — and it does
not need to be to make the point that 150k is inside the region where the
generated-track policy had not learned to finish either.

**§7.3, the 18.809 — not in this repository.** `records.json` holds eight
entries. Winter 5 (`2fe4bd46…`) has **19025 and 21899, and nothing else**; no
entry anywhere in the file is 18809. Whatever holds that lap is on the machine
that ran the Summer 1 experiment. One datum argues against it being human:
`arms/arm2-search/lineage/` contains 19249, 19245, 19405 and 19420, all search
outputs, and an 18.809 would sit naturally in that series.

## 2. Do these, in this order

| | what | cost |
|---|---|---|
| 1 | **Email Kodub about redistribution** (`PAPER_PLAN.md` §4). | today |
| 2 | **Resolve the 18.809.** | ~1 h |
| 3 | §2.1 gamma fix — now a prerequisite for item 4, not a tidy-up. | ~1 h |
| 4 | **The gamma/budget separator** (§0): two scratch runs, 500k, 0.99 vs 0.999. | ~2.5 h |
| 5 | §2.2 weights-only loading, with a test. | ~1 h |
| 6 | `PYTHONPATH` note in the README. | 5 min |
| 7 | M5 cost figures into PAPER.md §3.6/§8.6. | one sentence |
| 8 | **Winter 1**: play a lap, measure support distance, evaluate zero-shot. | ~2 h |
| 9 | Relabel the Summer 1 result per §0. | ~15 min |
| 10 | *Optional, and read item 4 first:* §3.1 on Desert 3 only. | ~1 day |

Items 1-3 are unconditional. **Item 4 decides whether the rest of this document
is the right plan**, so do not start item 8 or 10 before reading it.

### 2.1 Resolve the 18.809

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

**Do not retrain §6.1 on the strength of the fix alone.** Treat 0.99 as the
shipped configuration for the generalisation results and state it in the paper.
Rebuilding the headline invalidates comparability with everything banked —
including §6.2's support band, which is a property of one specific policy — for
a hypothesis nobody has tested.

§0 is what tests it. An earlier draft of this document filed gamma as a
correctness tidy-up that "defends numbers you already have". That was wrong:
applied to a 24-second track with jumps, a 5-second horizon is a candidate root
cause for the whole 0/4, which makes the fix a prerequisite for item 4 rather
than housekeeping. If 0.999 turns out to matter there, retraining §6.1 becomes
a real question — and §0's first outcome is the one that keeps it closed.

### 2.4 The separator, pre-specified

Fix this before running it, per §5.

* **Runs:** two, both `scratch` on Summer 1 from `tracks.npz`, 500,000 gradient
  steps, six collectors, all nine actions, seed 1 for both. One at
  `--gamma 0.99`, one at `--gamma 0.999`. No pretrained arm — this is about the
  track, not transfer.
* **Primary:** gradient steps to the first completed lap by any collector,
  and the greedy fraction at 150k / 300k / 500k.
* **Censored fallback, named in advance:** steps to reach 25 / 50 / 75 % of the
  lap, and the furthest fraction reached in ten greedy attempts from the final
  checkpoint, using `arms/transfer/where_attempts_end.py` so the section
  breakdown is comparable with the runs already banked.
* **Report both runs whatever happens**, and record that the 0.999 run is the
  first in the project to use the discount the code was designed for.
* One seed is deliberate: this is a triage experiment to choose a path, not a
  result for the paper. If it changes the plan, the version that goes in the
  paper gets three seeds.

### 2.5 Winter 1 instead of elevation in the generator

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

### 2.6 Desert 3, if there is appetite for one more headline

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

Three of these are **conditional on §0** and are marked so. The rest are cut
whatever it returns.

| cut | why |
|---|---|
| **§2.3** start episodes along the track — *conditional* | 2–3 days and the only change here that touches reward and progress semantics: `ProgressTracker` initialised mid-lap, par rescaled for the finish bonus, recordings exported only from start-line episodes, an LRU over 24 MB states. CLAUDE.md then requires a search run against the changed reward, since beam search found both previous progress bugs in seconds. **If §0 returns "neither finishes at 500k", this comes back** — at that point sample efficiency is the binding constraint rather than a convenience. |
| **§3.3** transfer across tracks — *conditional* | ~60 h, and the plan says run it twice while its order table counts it once. **But the Summer 1 run no longer supports a transfer claim at all** (§0), so if a transfer result is wanted, something has to be rerun at a real budget. Rerunning Summer 1 alone at 500k is 2 arms x 3 seeds x ~70 min, about 7 h — a tenth of the four-track sweep and enough for the claim §8.5 asks about. The four-track version stays cut. |
| **§3.2** elevation in the generator — *conditional* | 1–2 weeks, open-ended. Replaced by §2.5 below **unless §0 returns "neither finishes at 500k"**, which is the outcome that makes the jumps themselves the obstacle and earns this its weeks. |
| **§3.1** beyond one extra track | Five tracks is not five times more convincing for an environment paper, and each new track needs a lap played and raises the human-baseline question again. |
| **§4** PPO or DQN baseline | "Reviewers expect two algorithms" holds for an algorithms paper. This is an environment paper. Expect it as the likeliest reviewer complaint and answer it in §7 rather than running it. |
| **§4** ablations | Lookahead horizon, action restriction, control interval. They defend numbers that already exist; none adds a claim. |
| **§2.4** evaluation episodes | Half a day. Its value was clean curves and correct first-lap measurement, and most of that was needed *because* §2.3 moves where episodes start. With §2.3 cut, `learning_curve.csv` already has the 121 points Appendix A's figure needs. |
| **§2.5** the SIGINT crash | A nuisance during interruption, not a result. The `PYTHONPATH` half of §2.5 is kept — §9.1 claims the tests run from a clone. |

## 4. Answers to `PAPER_PLAN.md` §7

1. **Discount.** Keep 0.99 for the banked generalisation results and state it in
   the paper either way. Whether to retrain anything at 0.999 is not a judgement
   call — §0 answers it for 2.5 hours, and it should be answered before the
   question is argued. Fix the flag regardless (§2.3): a default the code's own
   comment argues against is a trap for whoever trains next.
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
  §2.4 and §2.5 above do this, for the separator and the Winter 1 test.
* **Report everything that ran**, Summer 1 included — and report it as what it
  is. §0 is the worked example: the run is sound, the write-up in
  `arms/transfer/README.md` is careful about its censored metric, and
  `PAPER_PLAN.md` §1 still rounds it up to "a weak or null transfer result,
  consistent with §6.2". The slide from "we could not measure it" to "we
  measured something small" happens between a results file and the plan that
  cites it, which is where to watch for it.
* **Keep diagnoses honest.** "Attempts end on Summer 1's descents and jumps" is
  an observation; "because the generator has no elevation" is a hypothesis, and
  it cannot be the whole story — the scratch policies never saw a generated
  track and fail at the same jumps. Jumps are hard to learn as well as
  unfamiliar.

## 6. What this plan does not do

* It does not produce zero-shot laps on an official track, unless §0 returns the
  second outcome and the discount turns out to have been the obstacle.
* It does not add a second algorithm, so the paper claims nothing comparative
  about IQN.
* It leaves the banked results trained at a discount the code's own comment
  argues against. §2.3 fixes the flag and §0 measures what it cost; neither
  retrains §6.1.
* It produces no transfer claim. The Summer 1 runs cannot support one (§0), and
  the ~7 h rerun that could is listed as conditional in §3 rather than
  scheduled.
* The Winter 1 test is correlational. If a reviewer wants the intervention,
  §3.2 is the answer and it is a week or two away, not an afternoon.

Most of those are a sentence in §7 of PAPER.md rather than a gap in the
argument, which is the test this cut was made against.

Two are not. A zero-shot lap on an official track is a result the paper does not
have rather than a limitation it can disclose — §0b is where that is argued, and
if it is wanted, §2.3 and §3.2 come back at two to three weeks. And the
transfer claim §8.5 was written to answer is now unmeasured rather than
measured-and-null, which is a weaker position than the paper currently implies
and should be fixed in §8.5's text whether or not the rerun happens.
