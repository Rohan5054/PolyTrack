#!/usr/bin/env python3
"""Pool several seeds' evaluations into one figure with a spread.

    python scripts/aggregate_seeds.py runs/seeds_heldout.log

A single run's 94% is a sample. This turns N of them into a mean and a range,
which is the difference between a number and a result, and it is what every
reviewer asks for first.

Reads the output of `scripts/evaluate.py`, one block per seed, separated by a
`######## SEED n ########` header. Deliberately parses the log rather than
recomputing: the log is the artifact that was produced by the measurement, and
a second code path that re-derives the numbers is a second chance to disagree
with them.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
import sys
from pathlib import Path

HEADER = re.compile(r"#+\s*SEED\s+(\S+)\s*#+")
ROW = re.compile(
    r"^\s{4}(\S[^\s].*?)\s{2,}(\d+)%\s+([\d.]+)s\s+([\d.]+)s\s+([\d.]+)", re.M
)
NEVER = re.compile(r"^\s{4}(\S[^\s].*?)\s{2,}(\d+)%\s+\((\d+)%\)", re.M)


def parse(text: str) -> dict[str, dict]:
    """{seed: {track: (finish_rate, best_s, par_s, ratio | None)}}."""
    seeds: dict[str, dict] = {}
    current = None
    for line in text.splitlines(keepends=True):
        found = HEADER.search(line)
        if found:
            current = found.group(1)
            seeds[current] = {}
            continue
        if current is None:
            continue
        finished = ROW.match(line)
        if finished:
            name, rate, best, par, ratio = finished.groups()
            seeds[current][name.strip()] = (
                int(rate) / 100, float(best), float(par), float(ratio)
            )
            continue
        never = NEVER.match(line)
        if never:
            name, rate, _reached = never.groups()
            seeds[current][name.strip()] = (int(rate) / 100, None, None, None)
    return {s: t for s, t in seeds.items() if t}


def spread(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:.2f}" if values else "--"
    return f"{st.mean(values):.2f} ± {(max(values) - min(values)) / 2:.2f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="?",
                        default=Path("runs/seeds_heldout.log"))
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"  no log at {args.log}")
        return 2
    seeds = parse(args.log.read_text(encoding="utf-8", errors="replace"))
    if not seeds:
        print(f"  no seed blocks found in {args.log}")
        return 1

    print(f"\n  {len(seeds)} seeds from {args.log}\n")
    print(f"    {'seed':<8}{'tracks':>8}{'finished >=1':>14}"
          f"{'mean finish':>13}{'median best/par':>18}")

    per_seed_rate: list[float] = []
    per_seed_ratio: list[float] = []
    per_seed_any: list[float] = []
    for name, tracks in seeds.items():
        rates = [r for r, *_ in tracks.values()]
        ratios = [x[3] for x in tracks.values() if x[3] is not None]
        any_rate = sum(1 for r in rates if r > 0) / len(rates)
        per_seed_rate.append(st.mean(rates))
        per_seed_any.append(any_rate)
        if ratios:
            per_seed_ratio.append(st.median(ratios))
        print(f"    {name:<8}{len(tracks):>8}{any_rate:>13.0%}"
              f"{st.mean(rates):>13.0%}"
              f"{(f'{st.median(ratios):.2f}' if ratios else '--'):>18}")

    print(f"\n  across seeds (mean +/- half-range):")
    print(f"    tracks finished at least once : {spread([v * 100 for v in per_seed_any])} %")
    print(f"    mean finish rate per attempt  : {spread([v * 100 for v in per_seed_rate])} %")
    if per_seed_ratio:
        print(f"    median best time / par        : {spread(per_seed_ratio)}")

    # Per-track agreement says something a pooled mean hides: whether the seeds
    # fail on the *same* tracks, which would be a property of those tracks
    # rather than of training.
    common = set.intersection(*(set(t) for t in seeds.values()))
    hard = [
        name for name in sorted(common)
        if all(seeds[s][name][0] < 1.0 for s in seeds)
    ]
    if hard:
        print(f"\n  tracks no seed finished every attempt ({len(hard)} of {len(common)}):")
        for name in hard:
            rates = ", ".join(f"{seeds[s][name][0]:.0%}" for s in seeds)
            print(f"    {name:<16} {rates}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
