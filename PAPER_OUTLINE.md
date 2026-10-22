# Paper outline

> Written 22 October 2026. This is the skeleton `PAPER.md` should have **after**
> the separator and transfer runs of 21 October, which it does not yet reflect.
> Each section gives its claim, the evidence, and the gap. §8 is the delta list:
> what to change in the current draft and why.
>
> Nothing here is a number that was not produced by code in this repo. Where a
> claim has no evidence yet it says so.

## 1. The claim ladder

Ordered strongest first. The paper should be read as making these five claims
and no others.

| | claim | strength |
|---|---|---|
| **C1** | A faithful RL environment can be built from a shipped game binary by hosting its own physics and capturing rather than reimplementing its geometry, and the result is verifiable to frame-exactness against two independent oracles. | **Strong.** 10 of 10, two oracles, one of them the game's own verifier. |
| **C2** | A procedural generator writes track codes the game loads and yields each track's route by construction, retiring the human-lap bootstrap. | **Strong.** 31/31 re-encode, 24/24 drivable. |
| **C3** | A policy trained on generated tracks generalises within that distribution with no measurable train/held-out gap. | **Strong.** Three seeds, 97.9 ± 0.6 %, 94 % vs 92 %. |
| **C4** | Transfer failure to human-designed tracks is out-of-distribution distance, measured in observation space, and the measurement predicts *where* a policy fails and not merely that it does. | **Medium.** Control arm and confound arm both run; the per-section prediction is a post-hoc reading, not a pre-registered test. |
| **C5** | Policy plus test-time search beats the best human lap on an official track. | **Medium, and narrower than the current draft says.** 23 ms, not 239 ms. One track, n = 1. |

C5 is the claim most likely to be attacked and the least load-bearing. C1 is the
contribution.

## 2. §1-§3 Introduction, related work, the environment

**Unchanged in substance.** C1's evidence is complete and none of the new runs
touch it.

Two additions:

* **The gate becomes 10 of 10 across five tracks.** `records.json` gained two
  laps on 21 October, both played by the author with the game's own finish
  frame: Winter 5 **18809** and track `7d134c93` **19980**. The second is a
  track the gate has never seen, taking the corpus from four tracks to five.
  **Both now pass `scripts/run_gate.py`, 10 of 10** — the cheapest
  strengthening of the paper's central claim available.
* **Throughput figures exist for a second machine.** §3.6 and §8.6 quote the
  Windows laptop (six collectors, 50-80 transitions/s). The M5 Air gives
  190-230 transitions/s and 115-144 gradient steps/s, with a fanless throttle to
  35-100 under sustained load. Report both; wall-clock budgets on laptops are
  not stable, which belongs beside the EcoQoS note.

## 3. §4 The generator

**Unchanged.** Note its narrowness here rather than defending it: five of 186
part types, flat ground. §6.2 measures what that costs and §6.4 is the evidence
it is a generator limit rather than a learning one.

## 4. §5 Agent and search

**Unchanged in method.** One disclosure moves here or to §9:

**Every banked result was trained at a discount the code argues against.**
`IQNConfig.gamma` is 0.999 with a comment that 0.99 is a five-second horizon at
a 50 ms interval; `train.py --gamma` defaulted to 0.99 and always overrode it.
Fixed on 21 October — the flag now defers to the config, and the collectors take
their n-step discount from `agent.config`, with a test.

State the value, and state that **0.999 measured worse**: on Summer 1 from
scratch, first lap 344k against 121k, 38 % against 65 % of the lap at 300k, and
0 of 10 greedy laps against 10 of 10. One seed, so it licenses "0.99 did not
handicap the banked results" and nothing more. Do not retrain §6.1.

## 5. §6 Results

### 6.1 Generalisation to unseen tracks — keep as is

Three seeds, 16 held-out tracks, **97.9 ± 0.6 %**, 100 ± 0 % finishing at least
once, 1.43 ± 0.03 median best/par. The 1.22M run: 94 % held-out against 92 %
training, so no measurable gap. Four times the compute bought pace (1.35x par)
not reliability.

The zero-shot row stays, and its framing changes — see 6.4.

### 6.2 The zero-shot gap, measured — keep, and strengthen the claim

Band = 0.1st-99.9th percentile of each of 62 features over 24,545 training
frames. Control (16 unseen generated tracks, 16/16 finished): **1 %** of frames,
**0.0-0.1** features outside. Test (4 official, 0/4): **57-62 %**, **7.0-9.2**.
**147-154x**, out of support at step 0. Two passes, so about ±15 %.

Confound arm: 12 training roads re-driven with only the reference line's
provenance swapped, 0.03 to 0.77 features per frame, so provenance explains
**at most 8 %**. Unconfoundable corroboration: vertical velocity **4.4x** and
roll rate **10x** beyond the band.

Limitation to keep: **12 of 62 features are bounded and saturated in training**
and can never register a violation, so every distance is a lower bound.

**New, and the strongest thing this section can say.** The measurement predicted
*where* policies would fail. On Summer 1, the pretrained policies' failed
attempts cluster on the valley floor immediately after the 47 m descent — 9 of
16 — while from-scratch policies fail at the jumps, 11 of 18. Vertical velocity
is the feature furthest outside the band that no confound explains. A distance
that also localises the failure is a mechanism; a distance alone is a
correlation. Mark it clearly as a post-hoc reading: the sections were defined
from the reference line's height profile after the runs, not before.

### 6.3 Search on one official track — the headline must change

| | Winter 5 | verified by |
|---|---|---|
| Human laps | 19.025, **18.809** | the game's own storage |
| Policy alone, best of 20 greedy | 19.422 | gate + client verifier |
| **Policy + guided beam search** | **18.786** | gate + client verifier |
| Continuous trajectory optimisation | did not finish, 1089 m of 1747 | — |

**The margin is 23 ms over the best human lap, not 239 ms.** The 18809 was
played by the author and confirmed human on 21 October. Report both human times
and say which the margin is against. The claim survives — 18.786 is still the
fastest lap on the track and still gate-verified — but "239 ms faster than the
human reference" is no longer true of the best human lap available.

Keep the three observations, which matter more than the time:

* **Search contributes everything.** The policy alone is slower than both human
  laps.
* **The search was steered by a human racing line.** The reference line *is* the
  human lap, and the line-distance ranking term found 459 ms in one run where
  seven runs ranking on arc alone found 14 ms total. The control sequence is the
  agent's own; the route was not discovered independently.
* **Continuous optimisation failed informatively.** 17.8M simulated frames
  against the 914,944 that produced the finish, and it never crossed the line.
  Winter 5 is chaotically sensitive to millisecond input timing: the human lap
  projected onto a fixed block grid reaches 26 m at 50 ms, 349 m at 25 ms,
  477 m at 10 ms, 468 m at 5 ms of 1747 m.

Also keep: `--forward-only` is justified per-track, and now by two laps — the
19.025 and the faster 18.809 Winter 5 laps both hold throttle 100.0 % of frames
and never brake; the third Winter 5 lap, 21.899, brakes 0.8 %. The 19.980
Winter 4 lap never brakes either, while Desert 3, Summer 1 and Summer 5 brake
8-21 %.

### 6.4 Official tracks are learnable — NEW SECTION, and it reframes 6.1

**A policy trained on Summer 1 laps it in 24.292 s against the best human
23.805 — 2.0 % off.** Before 21 October no policy in this project had completed
an official lap at all. Seven runs, all lapping within 400k gradient steps,
first laps 121k-344k.

This is what the 0/4 zero-shot row needed beside it. Without 6.4 a reader can
read §6.1 as "official tracks are beyond this agent"; with it the row means what
§6.2 says it means — distributional distance, not a ceiling.

It also retires an earlier null. The 150k Summer 1 runs of 20 October found no
laps in either arm; only one of the seven first laps above falls inside 150k, and
`arms/generalisation/learning_curve.csv` shows the *easier* generated
distribution at a 0.67 greedy fraction and a 33 % finish rate at 85k, reaching
complete laps only by 155k. **The right label for that experiment is
"inconclusive at this budget."** Report it as such, and keep it — the budget
lesson is worth a paragraph in its own right.

### 6.5 Does the representation transfer? — NEW, and genuinely mixed

Three seed pairs on Summer 1 at 400k, gamma 0.99, pretrained arm from the
generalisation checkpoint's weights with a fresh optimiser and the counter at
zero.

| | scratch | pretrained |
|---|---|---|
| first lap, mean | **202k** | 279k |
| share of lap at 300k | 65 / 67 / 65 % | **84 / 93 / 98 %** |
| greedy laps of 10, per seed | 10 / 0 / 2 | 5 / 8 / 1 |
| best lap | 24.652 / 28.010 / 25.092 | 25.988 / **24.292** / 24.833 |

* The **primary metric goes against transfer**: the pretrained arm reached its
  first lap later in all three pairs.
* **Mid-run progress goes the other way** in all three pairs, and the
  pretrained runs' finished-episode counts are far more consistent across seeds
  (157-166 against 16-427).
* **The end is a draw**: 14 of 30 greedy laps against 12 of 30.

Three out of three in the same direction is **not** significance — a two-sided
sign test gives p = 0.25. Report direction and spread, never a p-value dressed
up. Also note "first lap" counts any collector including one exploring at
epsilon 0.4, so it is noisy by construction and the greedy columns are cleaner.

One reading fits all of it and should be labelled as a reading: what transfers
is general driving on the parts of a track the generator resembles; what does
not is the part that decides whether a lap completes. §6.2's per-section failure
split is consistent with it. It is not a test of it.

## 6. §7 What this does not show — keep, with four edits

Keep every existing bullet. Add or amend:

* **The transfer result is mixed, not null, and it is three seeds.** Replace any
  "weak or null transfer" phrasing: the two metrics disagree and neither is
  significant at n = 3.
* **The discount comparison is one seed.**
* **§6.2 is descriptive, not causal.** A policy could in principle be robust to
  inputs outside its support. The experiment that would settle it is a wider
  generator, where the measured distance should fall and performance follow.
* **The per-section failure localisation is post-hoc.** Sections were read off
  the height profile after the runs.
* Amend **"the policy does not beat a human"**: it now laps an official track
  within 2.0 % when trained on it, which is a different statement from the
  Winter 5 one and should not be conflated with it.

## 7. §8-§10 Experiments required, reproducibility, engineering note

* **§8** shrinks. The transfer experiment has run; the budget question is
  settled. What remains: a wider generator (the falsifiable prediction in
  §6.2's own terms), seeds on the discount, a second algorithm, ablations.
  `updated_paper_plan.md` prices all of it and argues most of it is rigour
  rather than claim.
* **§9.1** gains the released policy and laps — `arms/generalisation/agent.pt`,
  2.6 MB, with the 48 training laps — and the fact that both measurement
  scripts default to them, so §6.2 reproduces from a clone plus the two files
  §9.2 explains cannot ship. 168 tests. Note that the suites need the package
  installed or `PYTHONPATH=.`.
* **§9.2** is unchanged and is still the paper's open problem. It is also the
  biggest available lever: permission turns this from a described environment
  into one others can run.
* **§10** keeps the EcoQoS note and gains the M5 throttling figures.

## 8. Delta list: what to change in `PAPER.md`

In rough order of how much each one matters.

| | change | why |
|---|---|---|
| 1 | §6.3: add the 18809 human lap, restate the margin as **23 ms** | A claim in the draft is no longer true of the best available human lap. `records.json` already contains the contradiction. |
| 2 | Add **§6.4**, official tracks are learnable | New result, and it is what makes the 0/4 row readable. |
| 3 | Add **§6.5**, the transfer result | The experiment §8.5 asked for has run. |
| 4 | §6.2: add the per-section failure localisation | Turns a distance into a mechanism, with the post-hoc caveat. |
| 5 | Relabel the 150k Summer 1 runs "inconclusive at this budget" | They do not support a transfer claim, and the plan that cites them rounds them up. |
| 6 | §5 or §9: disclose the discount, and that 0.999 measured worse | A default overrode the config on every banked run. |
| 7 | Run the gate on the two new laps; update §3.3 to 10 of 10 across five tracks | Cheapest strengthening of C1, the actual contribution. |
| 8 | Abstract: margin, §6.4, §6.5 | It currently promises the old §6.3 figure. |
| 9 | §7: the four edits above | |
| 10 | §9.1: policy released, 168 tests, `PYTHONPATH` | |
| 11 | §3.6/§8.6/§10: M5 figures | |

## 9. Figures and tables

| | what | data exists? |
|---|---|---|
| 1 | Environment architecture, marking where each oracle attaches | n/a, diagram |
| 2 | Gate results, all laps, recorded vs reproduced frame, drift zero | yes |
| 3 | Generated track examples with their routes | yes |
| 4 | **Generalisation curve**: held-out finish rate against training transitions | `arms/generalisation/learning_curve.csv`, 121 points — **figure does not exist** |
| 5 | **Out-of-support distance against distance along the lap**, one line per official track, control as a shaded band | per-frame CSVs in `arms/generalisation/support/` — **figure does not exist** |
| 6 | Violations by feature group, control vs official, with the confound arm beside it | yes |
| 7 | **Summer 1 progress against gradient steps**, both arms, three seeds | `arms/separator/*/history.json` — **figure does not exist** |
| 8 | **Where attempts end**, by track section, both arms | `arms/separator/where_attempts_end.jsonl` |
| 9 | Search ranking comparison on Winter 5 | partial, and not a controlled ablation |
| 10 | Input-timing sensitivity: distance reached against grid resolution | yes |

Figures 4, 5 and 7 are the three the paper most needs and all three have their
data committed. None is plotted.

## 10. What blocks submission

Nothing in §5-§7 of this outline. In order:

1. **The 18.809 propagation** (delta 1). A live contradiction between
   `records.json` and the draft.
2. **Kodub on redistribution** (§9.2). Costs a message, gates for weeks.
3. **The gate on two new laps** (delta 7). An hour, strengthens C1.
4. Deltas 2-6, which are writing, not running.

Everything else — a wider generator, a second algorithm, ablations, more seeds
— makes the paper harder to attack without changing what it claims.
