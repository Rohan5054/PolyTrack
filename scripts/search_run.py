#!/usr/bin/env python3
"""Policy-guided beam search against PolyTrack's own physics.

    python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5"

This is the hybrid the literature review recommended and the last item on the
roadmap: the policy generalises and proposes, the search optimises. The policy
alone plateaued on Winter 5 at 19.422 greedy, with a fitted tail saying ~2,400
greedy attempts to reach 19.105 by luck. Search does not sample -- it expands.

Why this is affordable here and is not in Trackmania: the backend saves and
restores exact state. A save is 24 MB but costs **14 ms**, against **14.4 s** to
replay a 19.4-second lap from frame zero. That is a factor of a thousand, and it
is the whole reason a beam can be carried across an entire lap.

Two things make the branching factor survivable:

* **The prior.** Nine actions to the fifth power is 59,049 futures a quarter of
  a second ahead. Expanding only the policy's top ``--top-k`` cuts that to 243
  and, more to the point, cuts them to 243 the policy thinks are *driveable*.
* **The block.** One action is held for ``--block-ms``, matching the control
  interval the policy was trained at, so a 19-second lap is ~390 decisions
  rather than 19,000.

The score is progress, never reward. Every candidate at a given depth has burned
exactly the same number of frames, so "furthest along the track" *is* "fastest to
here" -- no coefficients, and none of the exploits that make search such an
adversarial consumer of a shaped reward.

Output is verified through the acceptance gate before it is written, and lands
in a file ``tools/watch_run.py --from-best`` can install.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="an agent.pt to guide with")
    parser.add_argument("--track", required=True)
    parser.add_argument("--library", type=Path, default=Path("tracks.npz"))
    parser.add_argument(
        "--beam", type=int, default=16,
        help="candidates kept per depth. Peak memory is roughly 2x this many "
             "saved states at 24 MB each, so 16 is about 800 MB",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="actions expanded per candidate, best-rated first. At 1 the search "
             "only replays the policy",
    )
    parser.add_argument(
        "--block-ms", type=int, default=50,
        help="how long one action is held. Defaults to the policy's control "
             "interval, which is the cadence its action values mean anything at",
    )
    parser.add_argument(
        "--budget-ms", type=int, default=30_000,
        help="give up if nothing has finished within this much simulated time",
    )
    parser.add_argument("--max-seconds", type=float, default=float("inf"))
    parser.add_argument("--out", type=Path, default=Path("runs/searched"))
    parser.add_argument(
        "--refine", type=Path, default=None, metavar="JSON",
        help="polish an existing lap instead of searching from the line. Takes a "
             "best_<track>.json or an exported run. Beam search has to re-solve "
             "the whole track and can lose its entire beam to one corner; "
             "refinement starts from a lap that already works and can only "
             "improve on it",
    )
    parser.add_argument(
        "--iterations", type=int, default=300,
        help="--refine only: perturbations to try, spread over the sweep",
    )
    parser.add_argument(
        "--window-blocks", type=int, default=6,
        help="--refine only: how many consecutive blocks a perturbation rewrites",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reserve", type=int, default=4,
        help="beam only: extra candidates kept from behind the leader, one per "
             "band. Insurance against a corner that kills everything arriving at "
             "full speed -- which is how the first Winter 5 beam died, at 51%%",
    )
    return parser.parse_args(argv)


def _beam(args, entry, assets, prior, started, deadline):
    """Search from the start line, expanding only what the policy rates."""
    import time as _time

    from polytrack.physics.wasm import WasmPhysics
    from polytrack.search import BeamSearch

    depths = args.budget_ms // args.block_ms
    print(f"  beam {args.beam} (+{args.reserve} reserve), top-{args.top_k}, "
          f"{args.block_ms} ms blocks, up to {depths} depths\n")

    def on_depth(depth: int, best) -> None:
        if depth % 20 == 0 or depth == depths - 1:
            elapsed = _time.monotonic() - started
            done = best.arc / entry.line.length
            # Extrapolate on distance covered, not depths used: the budget is a
            # cap, the lap length is the actual finish line.
            eta = elapsed / max(done, 1e-6) - elapsed
            print(f"    depth {depth:>4}  {best.frame / 1000:6.2f} s simulated  "
                  f"{best.arc:7.1f} m ({done * 100:5.1f}%)  "
                  f"{elapsed / 60:5.1f} min elapsed, ~{eta / 60:.0f} min left",
                  flush=True)
        deadline()

    return BeamSearch(
        backend=WasmPhysics(entry.track_code(), assets=assets),
        track=entry.line,
        beam_width=args.beam,
        block_frames=args.block_ms,
        prior=prior,
        top_k=args.top_k,
        reserve=args.reserve,
    ).run(max_frames=args.budget_ms, on_depth=on_depth)


def _refine(args, entry, assets, allowed, started, deadline):
    """Polish a lap that already finishes, sweeping a window along it.

    Beam search has to solve the whole track and can lose its entire beam to one
    corner -- on Winter 5 it did, at 51%, and then sat there for twelve seconds
    of simulated time because a forward-only car cannot back out of a wall.
    Refinement cannot fail that way: it starts from a run that works and keeps a
    change only if the run got faster.
    """
    import time as _time

    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.search import RefinementSearch
    from polytrack.tas import TAS, format_time

    saved = json.loads(args.refine.read_text())
    seed_frames = int(saved["frames"])
    masks = list(Recording.deserialize(saved["recording"]).masks(seed_frames))
    seed = TAS.from_frame_masks(masks)
    print(f"  refining {format_time(seed_frames)} from {args.refine.name}")
    print(f"  {args.iterations} perturbations, {args.window_blocks} x "
          f"{args.block_ms} ms windows, actions "
          f"{allowed if allowed else 'all 9'}\n")

    n_blocks = -(-seed_frames // args.block_ms) + 4

    def on_progress(start, total, best_frames, finished):
        elapsed = _time.monotonic() - started
        gain = seed_frames - best_frames if finished else 0
        print(f"    window at {start * args.block_ms / 1000:5.1f} s "
              f"of {n_blocks * args.block_ms / 1000:.1f}  "
              f"best {best_frames / 1000:7.3f} s ({gain:+,} ms)  "
              f"{elapsed / 60:5.1f} min", flush=True)
        deadline()

    return RefinementSearch(
        backend=WasmPhysics(entry.track_code(), assets=assets),
        track=entry.line,
        block_frames=args.block_ms,
        window_blocks=args.window_blocks,
        seed=args.seed,
        actions=allowed,
        on_progress=on_progress,
    ).run(seed, iterations=args.iterations)


def main(argv=None) -> int:
    args = parse_args(argv)

    import numpy as np
    import torch

    torch.set_num_threads(2)

    from polytrack.learning.iqn import IQNAgent
    from polytrack.library import TrackLibrary
    from polytrack.observation import ObservationBuilder
    from polytrack.physics.assets import GeometryAssets
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.search import BeamSearch
    from polytrack.tas import format_time
    from polytrack.verify import verify

    entry = TrackLibrary.load(args.library).by_name(args.track)
    agent = IQNAgent.load(args.checkpoint)
    assets = GeometryAssets.load()

    allowed = agent.config.allowed_actions
    observer = ObservationBuilder(entry.line)

    def prior(state, arc, lateral):
        """What the policy thinks of each action from here.

        Restricted actions come back as ``-inf`` and the search drops them, so a
        ``--forward-only`` checkpoint searches only the three throttle actions.
        That is coherent -- the prior *is* the policy -- but it does mean this
        cannot discover a braking line. Guide with an unrestricted checkpoint if
        you want one considered.
        """
        return agent.action_values(observer.build(state, arc, lateral))

    print(f"\n  {entry.name}: {entry.line.length:.0f} m, par "
          f"{format_time(entry.par_frames)}")
    print(f"  guided by {args.checkpoint} at {agent.steps:,} gradient steps"
          + (f", actions {allowed}" if allowed else ", all 9 actions"))

    started = time.monotonic()
    deadline_hit = False

    def deadline() -> None:
        nonlocal deadline_hit
        if time.monotonic() - started > args.max_seconds:
            deadline_hit = True
            raise KeyboardInterrupt

    try:
        if args.refine:
            result = _refine(args, entry, assets, allowed, started, deadline)
        else:
            result = _beam(args, entry, assets, prior, started, deadline)
    except KeyboardInterrupt:
        print("\n  stopped" + (" at --max-seconds" if deadline_hit else ""))
        return 1

    print(f"\n  {result.summary()}")
    if not result.finished or result.tas is None:
        print("\n  Nothing finished. A wider --beam or a larger --top-k explores "
              "more;\n  a longer --budget-ms just lets a slow line run to the end.")
        return 1

    masks = list(result.tas.expand())
    recording = Recording.from_masks(masks)

    # The gate, on a fresh backend. A searched run that cannot be reproduced from
    # its own reported inputs is worthless, and this is the same check the game
    # runs on a submitted replay.
    check = verify(
        WasmPhysics(entry.track_code(), assets=assets), recording, result.frames
    )
    verdict = "PASS" if check.passed else "FAIL"
    print(f"  replay check: {verdict}  finished on frame {check.frames} "
          f"({check.frames / 1000:.3f} s)")
    if not check.passed:
        print("\n  The search reported a time its own inputs do not reproduce. "
              "That is a\n  bug in the search or the backend, not a slow lap -- "
              "do not install it.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    stem = entry.name.replace(" ", "_").lower()
    path = args.out / f"{stem}.json"
    path.write_text(json.dumps({
        "track": entry.name,
        "name": entry.name,
        "track_id": entry.track_code().track_id,
        "frames": result.frames,
        "finished": True,
        "recording": recording.serialize(),
        "source": f"beam {args.beam} top-{args.top_k} {args.block_ms}ms guided by "
                  f"{args.checkpoint.name} at {agent.steps} steps",
    }, indent=1), encoding="utf-8")
    (args.out / f"{stem}.tas").write_text(result.tas.to_text(), encoding="utf-8")

    par = entry.par_frames
    print(f"\n  {format_time(result.frames)}  ({result.frames - par:+,} ms "
          f"against par {format_time(par)})")
    print(f"  wrote {path}")
    print(f"\n  watch it:\n    python tools/watch_run.py --from-best {path} "
          f"--track \"{entry.name}\" --game path/to/extracted/app\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
