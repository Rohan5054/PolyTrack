# Transfer: does the generated-track policy learn Summer 1 faster?

**No run completed a lap of Summer 1 within 150k gradient steps, from scratch
or pretrained. The pretrained policy's head start is gone by 25k steps; from
100k it is slightly ahead and more consistent, within the spread of three
seeds. Most attempts end on Summer 1's elevation: 55 of 80 on a descent or a
jump, 33 of them at the second jump.**

The experiment PAPER.md §8.5 describes, run on 20 October 2026 on an M5 MacBook
Air (16 GB) with `scripts/transfer_experiment.py`.

## Design

- **Track:** Summer 1 from `tracks.npz`, built from the committed `records.json`:
  1300.6 m, par 23.805 s.
- **Arms:** `scratch` starts from a fresh network; `pretrained` starts from
  `arms/generalisation/agent.pt` (1.22M steps on 48 generated tracks).
- **Weights only.** The pretrained arm starts from a copy holding the
  checkpoint's online weights in both networks, a fresh optimiser and the step
  counter at zero (`init_source.json`). This is deliberately not
  `--resume --reset-optimiser`, which keeps the counter. The replay's
  importance-sampling exponent anneals on that counter (0.4 to 1 over 500k
  steps), so a resumed 1.22M-step arm would sample at 1.0 from its first batch
  while the scratch arm starts at 0.4. See `PAPER_PLAN.md` §2.2.
- **Identical otherwise:** lr 5e-5 and gamma 0.99 for both, passed explicitly
  (the checkpoint's values; `train.py` defaulted to 0.99 at the time, and now
  defaults to `IQNConfig`'s 0.999 -- see `PAPER_PLAN.md` §2.1), six collectors,
  all nine actions, a log row every 5 s. With the current code the same start
  is `train.py --resume arms/generalisation/agent.pt --weights-only`.
- **Runs:** a pilot pair at 50k steps (seed 0), then seeds 1-3 at 150k each.
  The arms were interleaved, with the order flipped each seed, so drift over
  the evening did not land on one arm.

## Progress round the lap

Share of the lap reached by the near-greedy collector, averaged over ±2.5k
steps (`<run>/history.json`, column `greedy_fraction`).

| run | 2.5k | 25k | 50k | 100k | 150k | peak | 50% reached |
|---|---|---|---|---|---|---|---|
| pilot scratch, seed 0 | 0% | 2% | 15% | | | 16% | -- |
| pilot pretrained, seed 0 | 23% | 18% | 32% | | | 32% | -- |
| scratch, seed 1 | 1% | 24% | 32% | 43% | 60% | 64% | 85k |
| pretrained, seed 1 | 24% | 15% | 29% | 45% | 40% | 53% | 105k |
| scratch, seed 2 | 0% | 4% | 34% | 38% | 37% | 40% | never |
| pretrained, seed 2 | 22% | 14% | 33% | 44% | 46% | 54% | 107k |
| scratch, seed 3 | 1% | 24% | 34% | 41% | 51% | 57% | 115k |
| pretrained, seed 3 | 20% | 13% | 25% | 43% | 72% | 72% | 110k |

Seeds 1-3, mean (range):

| arm | 2.5k | 25k | 50k | 100k | 150k | mean over the run | reached 50% |
|---|---|---|---|---|---|---|---|
| scratch | 1% | 17% (4-24) | 33% (32-34) | 41% (38-43) | 49% (37-60) | 31% (27-34) | 2 of 3 |
| pretrained | 22% (20-24) | 14% (13-15) | 29% (25-33) | 44% (43-45) | 52% (40-72) | 35% (34-36) | 3 of 3 |

- **A head start that does not last.** The pretrained runs begin at 22%, what
  the policy already does zero-shot. All four of them, pilot included, fall to
  13-18% around 25k steps before recovering. The early steps are spent
  unlearning.
- **Then a small, consistent edge.** From 100k the pretrained runs are slightly
  ahead and tighter, and all three reach half the lap at 105-110k steps against
  two of three scratch runs. Three seeds cannot separate that from noise.

## Where attempts end

The final checkpoint of each run, driven greedily from the start line ten times
with the training budget of 3x par (`where_attempts_end.py`, output in
`where_attempts_end.jsonl`). Sections are read off the reference line's height
profile.

| section | % of lap | all | scratch | pretrained |
|---|---|---|---|---|
| start plateau, 55 m up | 0-16 | 5 | 5 | 0 |
| big descent, 47 m to 1 m | 16-24 | 12 | 4 | 8 |
| valley floor | 24-41 | 6 | 1 | 5 |
| first jump: ramp 41-45, lands ~52 | 41-53 | 10 | 10 | 0 |
| middle plateau | 53-68 | 13 | 5 | 8 |
| second jump | 68-76 | 33 | 15 | 18 |
| final plateau to the finish | 76-100 | 1 | 0 | 1 |

- **The second jump is the wall:** 33 of 80 attempts, from six of the eight
  policies.
- **The first jump stops one policy completely:** all ten of scratch seed 2's
  attempts leave the line at 46-47%.
- **The pretrained policies lose attempts on the descent:** 8 against 4, and all
  4 scratch ones come from the 50k pilot. That fits PAPER.md §6.2, where vertical
  velocity is among the features furthest outside the generated-track band
  (4.4x).
- **Thirteen attempts stop at 63-64%,** on the third checkpoint just before the
  second ramp.
- **Closest to a lap:** one attempt by pretrained seed 1 reached 97%.

Failures: 54 `stalled`, 26 `off_line`.

## What this does and does not show

- **The primary metric is censored in every run.** Steps to the first lap cannot
  compare the arms when neither arm ever finishes. The progress table above is
  the fallback, and it was not specified in advance.
- **Three seeds a side.** The differences after 100k are inside the seed spread.
  The early dip is the one pattern every pretrained run shares.
- **Throughput was not constant.** The fanless laptop throttled to 35-150
  gradient steps/s under sustained load, which moves the achieved replay ratio.
  It stayed near 0.6 apart from the last few minutes of scratch seed 1 (0.23-0.48),
  and every row of `history.json` records it.
- **One attempt was discarded.** The first try at pretrained seed 1 was stopped by
  an accidental Ctrl+C at 42k steps and rerun from zero. It is not included.
- **Checkpoints are not committed** (2.6 MB each). They are in `runs/xfer-*/` on
  the machine that ran this.

## Files

| file | what it holds |
|---|---|
| `<run>/history.json` | a row every 5 s: steps, transitions, ratio, lap fractions, finishes, failures |
| `<run>/train.log` | `train.py`'s output |
| `<run>/transfer.json` | arm, seed, budget, exact command, start time |
| `init_source.json` | the pretrained starting point: source checkpoint, size, timestamp, lr, gamma, step count |
| `where_attempts_end.py`, `.jsonl` | the greedy check above, and its output |

## Reproduce

```bash
python scripts/transfer_experiment.py            # ~2.5 h on an M5 Air
python scripts/transfer_experiment.py --status   # progress, from another terminal
python arms/transfer/where_attempts_end.py . runs/xfer-pre-3/agent.pt xfer-pre-3 10
```

`tracks.npz` is not in the repo. Rebuild it with `scripts/build_library.py`
from the committed `records.json`, which has the Summer 1 lap.
