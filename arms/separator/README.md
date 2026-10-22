# Separator and transfer at 400k: Summer 1, from scratch and pretrained

**Every run completed laps of Summer 1 within 400k gradient steps, so the 150k
null in `arms/transfer/` was the budget. The discount the code was designed for,
0.999, learned more slowly than the 0.99 every banked result used. Pretrained
policies reached their first lap later than from-scratch ones in all three seed
pairs (279k against 202k on average), yet were further round the lap at 300k in
all three.**

Run on 21 October 2026 on an M5 MacBook Air (16 GB, room cooled) with
`scripts/separator_experiment.py`, which `updated_paper_plan.md` §0 and §2.4
specify. Each run took 47-54 minutes at a median of 122-144 gradient steps/s.
Seeds 1-2 ran from 17:02; seed 3 ran in a second invocation from 21:13.

## Design

- **Stage 1, the separator.** Two from-scratch runs, seed 1, identical except
  the discount: 0.99 against 0.999. Six collectors, all nine actions, lr 5e-5, a
  log row every 5 s.
- **Budget: 400k gradient steps a run**, not the plan's 500k, so that stage 1 and
  a follow-up fit in five hours. The change was fixed in the runner's docstring
  before anything ran.
- **The decision, specified in advance:** a run *finishes* if any collector
  completes a lap. Both finished, so stage 2 reran the transfer comparison at
  the same budget. `decision.json` records it.
- **Stage 2, transfer.** Seeds 1-3 per arm at gamma 0.99. The 0.99 separator run
  is scratch seed 1: same track, seed, budget and discount. The pretrained arm
  starts from `runs/xfer-init/agent.pt`, which holds `arms/generalisation/agent.pt`'s
  online weights in both networks, a fresh optimiser and the step counter at
  zero, exactly as in `arms/transfer/`. Arms alternate, one pair at a time.
- **Stage 3.** Ten greedy attempts from each final checkpoint, from the start
  line, at a budget of 3x par.

## Results

| run | arm | seed | gamma | first lap | episodes finished | 50% of lap | @150k | @300k | @400k | greedy laps /10 | greedy median | best lap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sep-g099-1` | scratch | 1 | 0.99 | 121k | 427 | 52k | 69% | 65% | 94% | 10 | 100% | 24.652 |
| `sep-g0999-1` | scratch | 1 | 0.999 | 344k | 1 | 128k | 36% | 38% | 72% | 0 | 66% | 45.048 |
| `xlong-pre-1` | pretrained | 1 | 0.99 | 284k | 166 | 172k | 38% | 84% | 85% | 5 | 90% | 25.988 |
| `xlong-scratch-2` | scratch | 2 | 0.99 | 234k | 16 | 242k | 32% | 67% | 84% | 0 | 44% | 28.010 |
| `xlong-pre-2` | pretrained | 2 | 0.99 | 246k | 165 | 132k | 76% | 93% | 65% | 8 | 100% | 24.292 |
| `xlong-scratch-3` | scratch | 3 | 0.99 | 252k | 71 | 198k | 33% | 65% | 82% | 2 | 73% | 25.092 |
| `xlong-pre-3` | pretrained | 3 | 0.99 | 305k | 157 | 138k | 68% | 98% | 94% | 1 | 50% | 24.833 |

Columns:
- **first lap:** gradient steps at the first log row recording a finished episode, by any collector, to within one 5 s log interval (about 700 steps).
- **episodes finished:** training episodes that finished, over the whole run.
- **50% of lap** and **@150k / @300k / @400k:** the near-greedy collector's share of the lap, averaged over ±2.5k steps.
- **greedy laps** and **greedy median:** from stage 3.
- **best lap:** the fastest training episode, in seconds, as kept in each run's `best_summer_1.json`. The human reference is 23.805. `xlong-pre-1`'s 25.988 came at 399,808 gradient steps, after its last 5 s log row at 399,424, so `history.json` and `summary.json`, which take the best lap from that row, still say 26.084.

### 1. The budget

All seven runs completed laps. First laps came at 121k-344k, and only one of them
fell inside the 150k budget of `arms/transfer/`, so "inconclusive at 150k" is
the right label for that experiment. The fastest lap, pretrained seed 2's 24.292,
is 2.0% off the human reference.

### 2. The discount (seed 1 only)

| | 0.99 | 0.999 |
|---|---|---|
| first lap | 121k | 344k |
| @300k | 65% | 38% |
| greedy laps /10 | 10 | 0 |
| best lap | 24.652 | 45.048 |

The longer horizon learned more slowly on every measure. This is one seed. It is
enough to say that 0.99, the value every banked result used, did not handicap
them. It is not enough to make a general claim about the discount.

### 3. Transfer (seeds 1-3, gamma 0.99)

| | scratch | pretrained |
|---|---|---|
| first lap | 121k / 234k / 252k (mean 202k) | 284k / 246k / 305k (mean 279k) |
| @300k | 65% / 67% / 65% | 84% / 93% / 98% |
| greedy laps /10 | 10 / 0 / 2 | 5 / 8 / 1 |
| episodes finished | 427 / 16 / 71 | 166 / 165 / 157 |
| best lap | 24.652 / 28.010 / 25.092 | 25.988 / 24.292 / 24.833 |

- **The primary metric goes against transfer.** The pretrained run reached its
  first lap later in every seed pair.
- **Mid-run progress goes the other way.** At 300k the pretrained runs were
  further round in every pair (84-98% against 65-67%), and their finished-episode
  counts are far more consistent across seeds.
- **The end is a draw.** Greedy laps total 14 of 30 for pretrained against 12 of
  30 for scratch.

One reading fits all of it: what transfers is general driving on the parts of
the track the generator resembles, and what does not is the part that decides
whether a lap is completed. Section 4 is consistent with that, but it is a
reading, not a test.

### 4. Where the final attempts end

Sections are read off the reference line's height profile, as in
`arms/transfer/README.md`.

| section | % of lap | scratch (0.99) | pretrained (0.99) | scratch (0.999) |
|---|---|---|---|---|
| start plateau, 55 m up | 0-16 | 4 | 0 | 0 |
| big descent, 47 m to 1 m | 16-24.5 | 0 | 0 | 0 |
| valley floor | 24.5-41 | 0 | 9 | 0 |
| first jump | 41-53 | 3 | 0 | 0 |
| middle plateau | 53-67.5 | 0 | 1 | 6 |
| second jump | 67.5-76 | 8 | 2 | 4 |
| final plateau | 76-99.5 | 3 | 4 | 0 |
| finished | -- | 12 | 14 | 0 |

- **Scratch failures are at the jumps:** 11 of the 18 failed attempts.
- **Pretrained failures cluster on the valley floor,** right after the 47 m
  descent: 9 of 16. Vertical velocity is among the features furthest outside
  the generated-track band (PAPER.md §6.2, 4.4x), and the same section cost the
  pretrained policies attempts at 150k (`arms/transfer/`).

## What this does and does not show

- **Three seeds a side.** All three pairs point the same way on first lap, and on
  progress at 300k, but three out of three is not significant: a two-sided sign
  test gives p = 0.25. Report the direction and the spread, not a significance
  claim.
- **The discount comparison is one seed.**
- **"First lap" counts any collector,** including the one exploring at epsilon
  0.4, so it is noisy by construction. The greedy columns are the cleaner
  measure of what a policy can do.
- **Throughput was not quite constant.** Five runs had a median of 141-144
  steps/s with an achieved replay ratio of 0.46-0.48. Two ran at about 123 with
  a ratio of 0.56-0.60, one from each arm (`xlong-pre-2` and `xlong-scratch-3`),
  so the difference does not favour either side. Individual 5 s rows range from
  0.25 to 0.63. Every row of each `history.json` records both numbers.
- **Checkpoints are not committed** (2.6 MB each). They are in `runs/` on the
  machine that ran this.

## Files

| file | what it holds |
|---|---|
| `<run>/history.json` | a row every 5 s: steps, transitions, ratio, lap fractions, finishes, failures |
| `<run>/train.log` | `train.py`'s output, which records the gamma and lr each run used |
| `<run>/transfer.json` | arm, seed, gamma, budget, exact command, start time |
| `<run>/transfer.done` | exit code, finish time, steps reached |
| `<run>/best_summer_1.json` | the run's fastest lap: track, finish frame, input recording, gradient steps; `scripts/verify_baselines.py` replays it |
| `decision.json` | stage 1's outcome and the follow-up it chose |
| `summary.json` | every metric in the tables above, per run |
| `where_attempts_end.jsonl` | stage 3's attempts, from `arms/transfer/where_attempts_end.py` |

## Reproduce

```bash
python scripts/separator_experiment.py            # about 6 h on an M5 Air, in sittings of up to 5 h
python scripts/separator_experiment.py --status
```
