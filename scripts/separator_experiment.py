#!/usr/bin/env python3
"""Run the gamma/budget separator, then the follow-up it chooses, unattended.

    python scripts/separator_experiment.py            # everything not yet done
    python scripts/separator_experiment.py --status   # progress; safe against a live run
    python scripts/separator_experiment.py --watch    # the same, every 30 seconds

`updated_paper_plan.md` §0 and §2.4 specify this experiment before it runs.
This script does that, inside a hard wall-clock budget (``--hours``, 5 by
default), and nothing more.

**Stage 1, the separator.** Two from-scratch runs on Summer 1, seed 1, six
collectors, identical except the discount: 0.99, which every banked result
used, against 0.999, which `IQNConfig` was designed for. The Summer 1 transfer
runs stopped at 150k with no laps, and nothing in them separates "too short a
budget" from "too short a horizon". Each run is 400k gradient steps: the plan
asked for 500k, and 400k is what lets the separator and a follow-up fit in five
hours. Read against `arms/generalisation/learning_curve.csv`, Summer 1 at 150k
sat where the generated-track policy was at 85k, which reached complete laps by
155k -- about 270k on Summer 1's pace, so 400k keeps a margin.

**Stage 2, chosen by stage 1.** A run *finishes* if any collector completes a
lap at any point in it. The follow-up runs in the order below for as long as
the time budget allows; a run starts only if it can finish before the
deadline, and whatever does not fit is left for the next invocation.

* Both finish: it was the budget. Rerun the transfer comparison at the same
  budget, at 0.99 like the banked experiment. The 0.99 separator run has the
  scratch arm's seed, track, budget and discount, so it stands in as scratch
  seed 1; the order completes one pair at a time.
* Exactly one finishes: the horizon matters (or, if only 0.99 finishes,
  something stranger). Seed 2, then seed 3, of both discounts.
* Neither finishes: the jumps themselves. Nothing more to run today.

**Stage 3.** Every final checkpoint is driven greedily ten times from the start
line by `arms/transfer/where_attempts_end.py`, and `runs/sep-report/` gets the
metrics §2.4 names: steps to the first lap, steps to 25/50/75% of the lap, the
greedy fraction at 150k/300k/500k, and how far the greedy attempts get.

The pretrained arm starts from `runs/xfer-init/agent.pt`, the weights-only copy
the banked transfer runs used; `scripts/transfer_experiment.py` builds it, and
its runner plumbing is shared. A finished run leaves `transfer.done` and is
skipped next time. A run that stopped part way is moved to
`runs/xfer-aborted/` and started over, and a run that crashes is retried once,
because nobody is there to restart it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import transfer_experiment as tx  # noqa: E402  -- shares the runner's plumbing

ROOT, RUNS, INIT = tx.ROOT, tx.RUNS, tx.INIT
DECISION = RUNS / "sep-decision.json"
DEADLINE = RUNS / "sep-deadline.json"
REPORT = RUNS / "sep-report"
WHERE = ROOT / "arms" / "transfer" / "where_attempts_end.py"

#: Held back from the time budget for stage 3's greedy attempts.
EVAL_RESERVE = 10 * 60
#: Steps/s to plan with until a run of this batch has measured one: what six
#: collectors sustained on an M5 Air with the room cooled.
PLANNING_RATE = 105.0

#: (name, arm, seed, gamma), in the order they run.
SEPARATOR = [
    ("sep-g099-1", "scratch", 1, 0.99),
    ("sep-g0999-1", "scratch", 1, 0.999),
]
FOLLOW_UPS = {
    # One pair at a time, scratch seed 1 being sep-g099-1, so however many runs
    # the time budget allows, the arms stay as balanced as they can be.
    "transfer": [
        ("xlong-pre-1", "pretrained", 1, 0.99),
        ("xlong-scratch-2", "scratch", 2, 0.99),
        ("xlong-pre-2", "pretrained", 2, 0.99),
        ("xlong-scratch-3", "scratch", 3, 0.99),
        ("xlong-pre-3", "pretrained", 3, 0.99),
    ],
    "seeds": [
        ("sep-g099-2", "scratch", 2, 0.99),
        ("sep-g0999-2", "scratch", 2, 0.999),
        ("sep-g0999-3", "scratch", 3, 0.999),
        ("sep-g099-3", "scratch", 3, 0.99),
    ],
    "none": [],
}
WHY = {
    "transfer": "both finished: it was the budget -- rerunning transfer at the same budget",
    "seeds": "one finished: the discount matters -- more seeds of both",
    "none": "neither finished: the jumps themselves -- nothing more to run",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true",
                        help="print progress and exit")
    parser.add_argument("--watch", action="store_true",
                        help="print progress every 30 s until Ctrl+C")
    parser.add_argument("--library", type=Path, default=ROOT / "tracks.npz")
    parser.add_argument("--track", default="Summer 1")
    parser.add_argument("--pretrained", type=Path,
                        default=ROOT / "arms" / "generalisation" / "agent.pt")
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--hours", type=float, default=5.0,
                        help="wall-clock budget for this invocation, stage 3 "
                             "included; no run is started that cannot finish in it")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=10,
                        help="greedy attempts per checkpoint in stage 3")
    parser.add_argument(
        "--follow-up", choices=("auto", *FOLLOW_UPS), default="auto",
        help="override stage 2's choice; only for testing the runner",
    )
    return parser.parse_args(argv)


# -- stage 1 and 2 -----------------------------------------------------------


def final_steps(log: Path) -> int | None:
    """The gradient-step total train.py prints when it exits."""
    if not log.exists():
        return None
    for line in reversed(log.read_text().splitlines()):
        if line.strip().startswith("total:") and "gradient steps" in line:
            return int(line.split()[1].replace(",", ""))
    return None


def measured_rate(specs) -> float:
    """Median steps/s over this batch's runs so far, for scheduling the rest."""
    rates = [r["steps_per_second"] for name, *_ in specs
             for r in tx.read_history(RUNS / name / "history.json")
             if r.get("steps_per_second")]
    return statistics.median(rates) if rates else PLANNING_RATE


def run_one(args, spec, lr: float, deadline: float) -> str:
    """One training run, capped at the deadline; returns done, failed,
    interrupted or skipped."""
    name, arm, seed, gamma = spec
    out = RUNS / name
    if (out / "transfer.done").exists():
        return "done"
    for attempt in (1, 2):
        seconds = deadline - time.time() - tx.STARTUP_SECONDS
        if seconds < 15 * 60:
            print(f"\n  {name}: skipped, {tx.duration(max(seconds, 0))} left in the budget")
            return "skipped"
        if out.exists():
            aside = RUNS / "xfer-aborted" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            out.rename(aside)
            print(f"\n  {name}: an earlier attempt did not finish; moved to {tx.rel(aside)}")
        command = [
            sys.executable, "-u", str(ROOT / "scripts" / "train.py"),
            "--library", str(args.library), "--track", args.track,
            "--workers", str(args.workers), "--max-steps", str(args.steps),
            # A backstop, not the plan: a run is only started when it should
            # reach --max-steps well inside this.
            "--max-seconds", str(int(seconds)),
            "--seed", str(seed), "--name", name,
            "--lr", repr(lr), "--gamma", repr(gamma),
            "--log-every", "5",
        ]
        if arm == "pretrained":
            command += ["--resume", str(INIT)]
        out.mkdir(parents=True)
        (out / "transfer.json").write_text(json.dumps({
            "arm": arm, "seed": seed, "gamma": gamma, "budget": args.steps,
            "command": command, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=1))
        print(f"\n  ===== {name}: {arm}, seed {seed}, gamma {gamma}, "
              f"{args.steps:,} steps (attempt {attempt}) =====")
        code, interrupted = tx.tee(command, out / "train.log")
        if interrupted:
            print(f"\n  stopped. Run the same command again to carry on; "
                  f"{name} will start over.")
            return "interrupted"
        steps = final_steps(out / "train.log")
        marker = "transfer.done" if code == 0 else "transfer.failed"
        (out / marker).write_text(json.dumps({
            "exit": code, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": steps,
            # Stopped by the time budget before its step budget: its metrics
            # are censored at `steps`, not at --steps.
            "cut_short": steps is not None and steps < args.steps,
        }))
        if code == 0:
            if steps is not None and steps < args.steps:
                print(f"\n  {name}: the time budget stopped it at {steps:,} steps")
            return "done"
        print(f"\n  {name} failed (exit {code}); "
              + ("retrying once." if attempt == 1 else "giving up on it."))
    return "failed"


def fits(args, specs, deadline: float) -> bool:
    """Whether one more run can reach --steps before the deadline, with 5% spare."""
    needed = args.steps / measured_rate(specs) * 1.05 + tx.STARTUP_SECONDS
    return time.time() + needed <= deadline


def finished_a_lap(name: str) -> bool | None:
    """Whether any collector completed a lap in the run; None if it is not done."""
    out = RUNS / name
    if not (out / "transfer.done").exists():
        return None
    if any(row.get("finishes") for row in tx.read_history(out / "history.json")):
        return True
    log = out / "train.log"
    return log.exists() and ": finished in " in log.read_text()


def decide(args) -> str:
    """Stage 2's choice, made once and kept, so a restart cannot change it."""
    if args.follow_up != "auto":
        return args.follow_up
    if DECISION.exists():
        return json.loads(DECISION.read_text())["follow_up"]
    low, high = (finished_a_lap(name) for name, *_ in SEPARATOR)
    if low is None or high is None:
        choice = "none"
        why = "a separator run did not complete, so there is nothing to decide on"
    else:
        choice = "transfer" if low and high else "seeds" if low or high else "none"
        why = WHY[choice]
    DECISION.write_text(json.dumps({
        "gamma_0.99_finished": low, "gamma_0.999_finished": high,
        "follow_up": choice, "why": why,
        "decided": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    print(f"\n  ===== decision: {why} =====")
    return choice


def batch_specs(choice: str | None) -> list[tuple]:
    return SEPARATOR + (FOLLOW_UPS[choice] if choice else [])


# -- stage 3 -----------------------------------------------------------------


def evaluate(args, specs) -> None:
    """Greedy attempts from every final checkpoint, all at once: one core each."""
    if args.track != "Summer 1":
        print(f"\n  stage 3 skipped: {tx.rel(WHERE)} drives Summer 1 only")
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, *_ in specs:
        checkpoint = RUNS / name / "agent.pt"
        result = REPORT / f"where_{name}.jsonl"
        if not (RUNS / name / "transfer.done").exists() or not checkpoint.exists():
            continue
        if result.exists() and result.stat().st_size > 0:
            continue
        with open(result, "w") as out, open(REPORT / f"where_{name}.err", "w") as err:
            jobs.append(subprocess.Popen(
                [sys.executable, str(WHERE), str(ROOT), str(checkpoint), name,
                 str(args.attempts)],
                cwd=ROOT, stdout=out, stderr=err,
            ))
    if jobs:
        print(f"\n  ===== stage 3: {len(jobs)} checkpoint(s), "
              f"{args.attempts} greedy attempts each =====")
    for job in jobs:
        job.wait()


def smoothed(history: list[dict], step: int, half: int = 2500) -> float | None:
    values = [r["greedy_fraction"] for r in history
              if r.get("greedy_fraction") is not None and r["gradient_steps"] > 0
              and abs(r["gradient_steps"] - step) <= half]
    return sum(values) / len(values) if values else None


def metrics(args, name: str) -> dict:
    out = RUNS / name
    history = tx.read_history(out / "history.json")
    first_lap = next((r["gradient_steps"] for r in history if r.get("finishes")), None)
    reach = {}
    for level in (0.25, 0.5, 0.75):
        reach[level] = next(
            (s for s in range(2500, args.steps + 1, 2500)
             if (v := smoothed(history, s)) is not None and v >= level), None)
    ends = []
    result = REPORT / f"where_{name}.jsonl"
    if result.exists() and result.stat().st_size:
        ends = json.loads(result.read_text().splitlines()[0])["ends"]
    fractions = [e["fraction"] for e in ends]
    return {
        "run": name,
        "first_lap": first_lap,
        "laps": history[-1].get("finishes") if history else None,
        "best": min((history[-1].get("best") or {}).values(), default=None) if history else None,
        "reach_25": reach[0.25], "reach_50": reach[0.5], "reach_75": reach[0.75],
        "greedy_150k": smoothed(history, 147_500),
        "greedy_300k": smoothed(history, 297_500),
        "greedy_end": smoothed(history, (history[-1]["gradient_steps"] if history else 0) - 2_500),
        "greedy_furthest": max(fractions) if fractions else None,
        "greedy_median": statistics.median(fractions) if fractions else None,
        "greedy_laps": sum(e["failure"] == "finished" for e in ends) if ends else None,
    }


def report(args, specs) -> None:
    rows = [metrics(args, name) for name, *_ in specs
            if (RUNS / name / "transfer.done").exists()]
    if not rows:
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "summary.json").write_text(json.dumps(rows, indent=1))

    def k(v):
        return "--" if v is None else f"{v // 1000}k"

    def pct(v):
        return "--" if v is None else f"{v:.0%}"

    print("\n  run              first lap  laps  25%    50%    75%    "
          "@150k  @300k   @end  greedy: furthest median laps")
    for r in rows:
        print(f"  {r['run']:<16} {k(r['first_lap']):>9} {r['laps'] or 0:>5}  "
              f"{k(r['reach_25']):<6} {k(r['reach_50']):<6} {k(r['reach_75']):<6} "
              f"{pct(r['greedy_150k']):>5}  {pct(r['greedy_300k']):>5}  "
              f"{pct(r['greedy_end']):>5}  {pct(r['greedy_furthest']):>14} "
              f"{pct(r['greedy_median']):>6} {r['greedy_laps'] if r['greedy_laps'] is not None else '--':>4}")
    print(f"\n  written to {tx.rel(REPORT / 'summary.json')}")


# -- the whole day -----------------------------------------------------------


def run(args) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    problems = tx.preflight(args)
    if problems:
        print("\n  not ready to start:")
        for problem in problems:
            print(f"    {problem}")
        return 2
    lr, _ = tx.prepare_init(args)
    started = time.time()
    deadline = started + args.hours * 3600 - EVAL_RESERVE
    DEADLINE.write_text(json.dumps({
        "started": started, "deadline": deadline,
        "ends_by": started + args.hours * 3600, "hours": args.hours,
    }))
    print(f"\n  separator on {args.track}: {args.steps:,} steps a run, "
          f"lr {lr:g}, {args.workers} collectors")
    print(f"  time budget {args.hours:g} h: no run starts that cannot finish by "
          f"{time.strftime('%H:%M', time.localtime(deadline))}, and everything "
          f"ends by {time.strftime('%H:%M', time.localtime(started + args.hours * 3600))}")
    tx.keep_awake()

    failed, left = [], []
    for spec in SEPARATOR:
        outcome = run_one(args, spec, lr, deadline)
        if outcome == "interrupted":
            return 130
        if outcome == "failed":
            failed.append(spec[0])
        if outcome == "skipped":
            left.append(spec[0])

    choice = decide(args) if not left else "none"
    for spec in FOLLOW_UPS[choice]:
        if (RUNS / spec[0] / "transfer.done").exists():
            continue
        if left or not fits(args, batch_specs(choice), deadline):
            left.append(spec[0])
            continue
        outcome = run_one(args, spec, lr, deadline)
        if outcome == "interrupted":
            return 130
        if outcome == "failed":
            failed.append(spec[0])
        if outcome == "skipped":
            left.append(spec[0])

    specs = batch_specs(choice)
    evaluate(args, specs)
    report(args, specs)
    status(args)
    if left:
        print(f"\n  left for later (did not fit in {args.hours:g} h): {', '.join(left)}"
              "\n  run the same command again to do them")
    if failed:
        print(f"\n  failed twice: {', '.join(failed)}")
        return 1
    return 0


def survey(args, specs) -> list[dict]:
    now = time.time()
    rows = []
    for name, arm, seed, gamma in specs:
        out = RUNS / name
        log = out / "train.log"
        if (out / "transfer.done").exists():
            state = "done"
        elif log.exists() and now - log.stat().st_mtime < 90:
            state = "running"
        elif (out / "transfer.failed").exists():
            state = "failed"
        elif out.exists():
            state = "stopped"
        else:
            state = "pending"
        history = tx.read_history(out / "history.json")
        last = history[-1] if history else {}
        rates = [r["steps_per_second"] for r in history[-24:] if r.get("steps_per_second")]
        reached = args.steps
        if state == "done":
            marker = json.loads((out / "transfer.done").read_text() or "{}")
            reached = min(marker.get("steps") or args.steps, args.steps)
        rows.append({
            "name": name, "arm": arm, "seed": seed, "gamma": gamma, "state": state,
            "done": {"done": reached, "running": min(
                last.get("gradient_steps", 0), args.steps)}.get(state, 0),
            "first_lap": next((r["gradient_steps"] for r in history
                               if r.get("finishes")), None),
            "lap": last.get("greedy_fraction"),
            "fin": last.get("finish_rate"),
            "best": min((last.get("best") or {}).values(), default=None),
            "rate": statistics.median(rates) if rates else None,
            "log": log,
        })
    return rows


def status(args) -> None:
    choice = (json.loads(DECISION.read_text())["follow_up"]
              if DECISION.exists() else None)
    runs = survey(args, batch_specs(choice))
    live = next((r for r in runs if r["state"] == "running"), None)
    rate = (live and live["rate"]) or next(
        (r["rate"] for r in reversed(runs) if r["rate"]), None)
    remaining = sum(args.steps - r["done"] for r in runs if r["state"] != "done")
    waiting = sum(r["state"] not in ("done", "running") for r in runs)
    budget = json.loads(DEADLINE.read_text()) if DEADLINE.exists() else None

    print(f"\n  {args.track} separator: "
          f"{sum(r['state'] == 'done' for r in runs)} of {len(runs)} runs done")
    if budget:
        print(f"  time budget: started "
              f"{time.strftime('%H:%M', time.localtime(budget['started']))}, "
              f"everything ends by {time.strftime('%H:%M', time.localtime(budget['ends_by']))}")
    if choice is None:
        print("  stage 2 not decided yet: after the two separator runs it adds "
              "a transfer pair, more seeds, or nothing -- as many runs as fit")
    else:
        print(f"  stage 2: {json.loads(DECISION.read_text())['why']}")
    if rate and remaining and live:
        left = remaining / rate + tx.STARTUP_SECONDS * waiting
        end = time.time() + left
        if budget and end > budget["deadline"]:
            print(f"  at {rate:.0f} steps/s the budget fits the current run and "
                  "whatever else can finish by the deadline; the rest is left for later")
        else:
            print(f"  about {tx.duration(left)} of training left at {rate:.0f} steps/s "
                  f"-- around {time.strftime('%H:%M', time.localtime(end))}"
                  + (", plus stage 2" if choice is None else ""))
    elif not live:
        print("  nothing is running")

    def cell(value, fmt):
        return "--" if value is None else fmt.format(value)

    print(f"\n  {'run':<16} {'arm':<10} {'seed':>4} {'gamma':>6}  {'state':<8} "
          f"{'gradient steps':>17} {'first lap':>9} {'lap':>5} {'fin':>5} {'best':>7}")
    for r in runs:
        line = (f"  {r['name']:<16} {r['arm']:<10} {r['seed']:>4} {r['gamma']:>6}  "
                f"{r['state']:<8} ")
        if r["state"] in ("running", "done"):
            line += (f"{r['done']:>8,} / {args.steps // 1000:>3}k  "
                     f"{cell(r['first_lap'], '{:,}'):>9} "
                     f"{cell(r['lap'], '{:.0%}'):>5} {cell(r['fin'], '{:.0%}'):>5} "
                     f"{cell(r['best'], '{:.3f}'):>7}")
        elif r["state"] in ("stopped", "failed"):
            line += f"{'starts over':>17}"
        print(line.rstrip())
    if live:
        log = str(live["log"]).replace(str(Path.home()), "~", 1)
        print(f"\n  live output: tail -f {log}")
    if (REPORT / "summary.json").exists():
        print(f"  results: {tx.rel(REPORT / 'summary.json')}")


def watch(args) -> int:
    try:
        while True:
            print("\033[2J\033[H", end="")
            status(args)
            print(f"\n  {time.strftime('%H:%M:%S')}, every 30 s. Ctrl+C stops "
                  "watching; the runs carry on.")
            time.sleep(30)
    except KeyboardInterrupt:
        return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    args.library = args.library.resolve()
    args.pretrained = args.pretrained.resolve()
    if args.watch:
        return watch(args)
    if args.status:
        status(args)
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
