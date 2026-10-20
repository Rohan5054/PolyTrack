#!/usr/bin/env python3
"""Run the Summer 1 transfer experiment end to end, and say how far it has got.

    python scripts/transfer_experiment.py            # every run not yet done
    python scripts/transfer_experiment.py --status   # progress; safe against a live run
    python scripts/transfer_experiment.py --watch    # the same, every 30 seconds

Two arms, identical but for the initial weights: ``scratch`` starts from a
fresh network, ``pretrained`` from the generated-track policy. A pilot pair at
50k gradient steps, then three seeds of each at 150k -- eight runs, one at a
time, because six collectors already fill the machine. The primary metric is
gradient steps to the first completed lap.

**The pretrained arm does not resume from the checkpoint as it stands.**
Resuming restores more than the weights, and two of the extras change the
effective step size early, which is exactly where time to first lap is
measured. One is the Adam moments. The other is the step counter: the
prioritised replay anneals its importance-sampling exponent from 0.4 to 1 over
the first 500k steps of it, so a resumed arm would learn fully corrected from
its first batch while the scratch arm starts at 0.4. ``prepare_init`` writes
``runs/xfer-init/agent.pt`` instead -- the checkpoint's online weights in both
networks, a fresh optimiser, the counter at zero -- and both arms get the
checkpoint's lr and gamma as flags, which also keeps the collectors' n-step
discount in step with the learner's. That leaves the weights as the only
difference. train.py's ``--reset-optimiser`` drops the moments but keeps the
counter, so it is not used here.

Runs alternate arm by seed, and the order flips each seed, so whatever drifts
over three hours -- a fanless laptop's clock, and with it the learner's share of
the transitions -- does not land on one arm. A finished run leaves
``transfer.done`` and is skipped next time. A run that stopped part way is moved
to ``runs/xfer-aborted/`` and started over: its replay buffer is gone, and a run
split across sittings is not comparable on time to first lap.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs"
INIT = RUNS / "xfer-init" / "agent.pt"

#: (name, arm, seed, phase), in the order they run. The pilot takes seed 0 so it
#: is a fourth sample of the early window rather than a rerun of seed 1's.
PLAN = [
    ("xfer-pilot-scratch-0", "scratch", 0, "pilot"),
    ("xfer-pilot-pre-0", "pretrained", 0, "pilot"),
    ("xfer-scratch-1", "scratch", 1, "result"),
    ("xfer-pre-1", "pretrained", 1, "result"),
    ("xfer-pre-2", "pretrained", 2, "result"),
    ("xfer-scratch-2", "scratch", 2, "result"),
    ("xfer-scratch-3", "scratch", 3, "result"),
    ("xfer-pre-3", "pretrained", 3, "result"),
]

#: Spawning the collectors and filling the 5k-transition warm-up buffer before
#: the first gradient step: about 30 s on an M5 Air.
STARTUP_SECONDS = 30


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
    parser.add_argument("--pilot-steps", type=int, default=50_000)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def budget(args, phase: str) -> int:
    return args.pilot_steps if phase == "pilot" else args.steps


# -- running -----------------------------------------------------------------


def preflight(args) -> list[str]:
    problems = []
    if not args.library.exists():
        problems.append(f"no {rel(args.library)} -- the track library with {args.track} in it")
    else:
        from polytrack.library import LibraryError, TrackLibrary
        try:
            TrackLibrary.load(args.library).select([args.track])
        except LibraryError as error:
            problems.append(str(error))
    if not args.pretrained.exists():
        problems.append(f"no {rel(args.pretrained)} -- the generated-track "
                        "policy the pretrained arm starts from")
    return problems


def prepare_init(args) -> tuple[float, float]:
    """Write the pretrained arm's starting point; return the lr and gamma."""
    from dataclasses import fields, replace

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    stat = args.pretrained.stat()
    stamp = {"source": str(args.pretrained), "bytes": stat.st_size,
             "mtime_ns": stat.st_mtime_ns}
    sidecar = INIT.parent / "source.json"
    if INIT.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if all(saved.get(k) == v for k, v in stamp.items()):
            return saved["lr"], saved["gamma"]
        if any((RUNS / name / "transfer.done").exists() for name, *_ in PLAN):
            raise SystemExit(
                f"\n  {rel(args.pretrained)} has changed since finished runs "
                "started from it.\n  Move runs/xfer-* aside to start over.\n")

    pretrained = IQNAgent.load(args.pretrained)
    config = replace(pretrained.config, allowed_actions=None)
    # What train.py builds for the scratch arm from --lr and --gamma. Anything
    # else that differs has no flag, so the scratch arm could not match it.
    scratch = IQNConfig(lr=config.lr, gamma=config.gamma)
    differ = [f.name for f in fields(IQNConfig)
              if getattr(config, f.name) != getattr(scratch, f.name)]
    if differ:
        raise SystemExit(
            "\n  the pretrained checkpoint differs from a fresh agent in "
            + ", ".join(f"{n} ({getattr(config, n)} against "
                        f"{getattr(scratch, n)})" for n in differ)
            + "\n  and train.py has no flag to give the scratch arm the same.\n")
    if pretrained.config.allowed_actions is not None:
        print(f"  note: the checkpoint was restricted to actions "
              f"{pretrained.config.allowed_actions}; both arms get all nine")

    init = IQNAgent(config)
    init.online.load_state_dict(pretrained.online.state_dict())
    init.target.load_state_dict(pretrained.online.state_dict())
    INIT.parent.mkdir(parents=True, exist_ok=True)
    init.save(INIT)
    sidecar.write_text(json.dumps({
        **stamp, "lr": config.lr, "gamma": config.gamma,
        "pretrained_steps": pretrained.steps,
    }, indent=1))
    return config.lr, config.gamma


def keep_awake() -> None:
    """Hold off idle sleep while this process lives. Closing the lid still sleeps."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def tee(command: list[str], log_path: Path) -> tuple[int, bool]:
    """Run ``command`` with its output to the terminal and to ``log_path``.

    Returns the exit code and whether the run was interrupted, which train.py
    reports by printing so and exiting 0 -- the same code as a finished run.
    """
    interrupted = False
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
                # --status tells a running run from a dead one by the log's age.
                log.flush()
                interrupted |= line.strip() == "interrupted"
        except KeyboardInterrupt:
            # train.py had the same Ctrl+C and is saving. Let it finish.
            interrupted = True
            try:
                for line in process.stdout:
                    log.write(line)
            except KeyboardInterrupt:
                pass
        code = process.wait()
    return code, interrupted


def run(args) -> int:
    # Through tee or nohup stdout is a file, and block buffering would hold the
    # run's progress lines back for minutes at a time.
    sys.stdout.reconfigure(line_buffering=True)
    problems = preflight(args)
    if problems:
        print("\n  not ready to start:")
        for problem in problems:
            print(f"    {problem}")
        return 2
    lr, gamma = prepare_init(args)
    print(f"\n  both arms: lr {lr:g}, gamma {gamma:g}, "
          f"{args.workers} collectors, {args.track}")
    print(f"  pretrained arm starts from {rel(INIT)}: "
          "weights only, fresh optimiser, step 0")
    keep_awake()

    failed = []
    for index, (name, arm, seed, phase) in enumerate(PLAN, 1):
        out = RUNS / name
        if (out / "transfer.done").exists():
            continue
        if out.exists():
            aside = RUNS / "xfer-aborted" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            out.rename(aside)
            print(f"\n  {name} did not finish last time; moved to {rel(aside)}")

        steps = budget(args, phase)
        command = [
            sys.executable, "-u", str(ROOT / "scripts" / "train.py"),
            "--library", str(args.library), "--track", args.track,
            "--workers", str(args.workers), "--max-steps", str(steps),
            "--seed", str(seed), "--name", name,
            "--lr", repr(lr), "--gamma", repr(gamma),
            # Time to first lap is read off this log, so the interval is the
            # metric's resolution: ~600 gradient steps at 5 s, ~2400 at 20.
            "--log-every", "5",
        ]
        if arm == "pretrained":
            command += ["--resume", str(INIT)]
        out.mkdir(parents=True)
        (out / "transfer.json").write_text(json.dumps({
            "arm": arm, "seed": seed, "phase": phase, "budget": steps,
            "command": command, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=1))

        print(f"\n  ===== [{index}/{len(PLAN)}] {name}: {arm}, seed {seed}, "
              f"{steps:,} steps =====")
        code, interrupted = tee(command, out / "train.log")
        if interrupted:
            print(f"\n  stopped. Run the same command again to carry on; "
                  f"{name} will start over.")
            return 130
        marker = "transfer.done" if code == 0 else "transfer.failed"
        (out / marker).write_text(json.dumps({
            "exit": code, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        }))
        if code:
            failed.append(name)
            print(f"\n  {name} failed (exit {code}); carrying on. "
                  f"Its output is in {rel(out / 'train.log')}")

    status(args)
    if failed:
        print(f"\n  failed: {', '.join(failed)}. Run the same command again "
              "to retry them.")
        return 1
    return 0


# -- reporting ---------------------------------------------------------------


def read_history(path: Path) -> list[dict]:
    # train.py rewrites this in place every few seconds, so a read can land
    # mid-write. The next one will not.
    for _ in range(5):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            time.sleep(0.2)
    return []


def survey(args) -> list[dict]:
    now = time.time()
    runs = []
    for name, arm, seed, phase in PLAN:
        out = RUNS / name
        log = out / "train.log"
        meta = out / "transfer.json"
        steps = (json.loads(meta.read_text())["budget"] if meta.exists()
                 else budget(args, phase))
        if (out / "transfer.done").exists():
            state = "done"
        elif (out / "transfer.failed").exists():
            state = "failed"
        elif log.exists() and now - log.stat().st_mtime < 90:
            state = "running"
        elif out.exists():
            state = "stopped"
        else:
            state = "pending"

        history = read_history(out / "history.json")
        last = history[-1] if history else {}
        rates = [r["steps_per_second"] for r in history[-24:]
                 if r.get("steps_per_second")]
        runs.append({
            "name": name, "arm": arm, "seed": seed, "state": state,
            "budget": steps,
            # A stopped or failed run starts over, so it has done nothing yet.
            "done": {"done": steps, "running": min(
                last.get("gradient_steps", 0), steps)}.get(state, 0),
            "first_lap": next((r["gradient_steps"] for r in history
                               if r.get("finishes")), None),
            "lap": last.get("greedy_fraction"),
            "fin": last.get("finish_rate"),
            "best": min((last.get("best") or {}).values(), default=None),
            "rate": statistics.median(rates) if rates else None,
            "log": log,
        })
    return runs


def duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "under a minute"
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def status(args) -> None:
    runs = survey(args)
    total = sum(r["budget"] for r in runs)
    done = sum(r["done"] for r in runs)
    finished = sum(r["state"] == "done" for r in runs)
    live = next((r for r in runs if r["state"] == "running"), None)

    print(f"\n  {args.track} transfer: {finished} of {len(runs)} runs done, "
          f"{done / total:.0%} of all gradient steps")
    rate = (live and live["rate"]) or next(
        (r["rate"] for r in reversed(runs) if r["rate"]), None)
    if finished == len(runs):
        print("  all done")
    elif rate:
        waiting = sum(r["state"] not in ("done", "running") for r in runs)
        left = (total - done) / rate + STARTUP_SECONDS * waiting
        clock = time.strftime("%H:%M", time.localtime(time.time() + left))
        print(f"  about {duration(left)} to go at {rate:.0f} steps/s "
              f"-- done around {clock}"
              + ("" if live else ", once it is running again"))
    if not live and finished < len(runs):
        print("  nothing is running")

    def cell(value, fmt):
        return "--" if value is None else fmt.format(value)

    print(f"\n  {'run':<21} {'arm':<10} {'seed':>4}  {'state':<8} "
          f"{'gradient steps':>17} {'first lap':>9} {'lap':>5} {'fin':>5} {'best':>7}")
    for r in runs:
        line = f"  {r['name']:<21} {r['arm']:<10} {r['seed']:>4}  {r['state']:<8} "
        if r["state"] in ("running", "done"):
            line += (f"{r['done']:>8,} / {r['budget'] // 1000:>3}k  "
                     f"{cell(r['first_lap'], '{:,}'):>9} "
                     f"{cell(r['lap'], '{:.0%}'):>5} {cell(r['fin'], '{:.0%}'):>5} "
                     f"{cell(r['best'], '{:.3f}'):>7}")
        elif r["state"] in ("stopped", "failed"):
            line += f"{'starts over':>17}"
        print(line.rstrip())

    print("\n  first lap: gradient steps at the first finished lap, to within ~600"
          "\n  lap: how far round the near-greedy collector gets"
          "\n  fin: finished share of the last 100 episodes   best: fastest lap, s")
    if live:
        # In full, so it can be pasted into a terminal that is somewhere else.
        log = str(live["log"]).replace(str(Path.home()), "~", 1)
        print(f"\n  live output: tail -f {log}")


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
