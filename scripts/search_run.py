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
    parser.add_argument(
        "--focus", default=None, metavar="FROM,TO",
        help="--refine only: restrict the sweep to this span of the lap, in "
             "seconds. A uniform sweep spends equal effort everywhere, and a lap "
             "does not lose time evenly -- compare the run against a faster one "
             "and aim at where the gap actually opens",
    )
    parser.add_argument(
        "--beam-window", default=None, metavar="FROM,TO",
        help="with --refine: beam-search this span of the lap in seconds, "
             "leaving the rest untouched. Finds a different line where "
             "refinement can only polish the one it was given",
    )
    parser.add_argument(
        "--lookahead", type=float, default=0.0, metavar="SECONDS",
        help="beam only: credit a candidate this many seconds of its speed when "
             "ranking, as arc + lookahead*speed. Arc alone prefers the candidate "
             "five metres up and twenty km/h down, which is behind two seconds "
             "later -- the shape every Winter 5 beam run has shown",
    )
    parser.add_argument(
        "--line-weight", type=float, default=0.0, metavar="M",
        help="beam only: charge this much virtual arc per metre off the "
             "reference line. Only sound where that line is fast -- on Winter 5 "
             "it is the 19.025 human lap, and the agent's best wanders 8.3 m "
             "off it while matching on throttle, slip, contact and steering",
    )
    parser.add_argument(
        "--switch-cost", type=float, default=0.0, metavar="M",
        help="beam only: charge this much virtual arc per input change. The "
             "agent saws where the human holds an arc -- 157 changes against 85 "
             "-- and steering scrubs speed",
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
        lookahead=args.lookahead,
        line_weight=args.line_weight,
        switch_cost=args.switch_cost,
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
    focus = None
    if args.focus:
        lo, hi = (float(x) for x in args.focus.split(","))
        focus = (int(lo * 1000 / args.block_ms), int(hi * 1000 / args.block_ms))

    print(f"  refining {format_time(seed_frames)} from {args.refine.name}"
          + (f", focused on {args.focus} s" if focus else ""))
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
        focus_blocks=focus,
    ).run(seed, iterations=args.iterations)


def _beam_window(args, entry, assets, prior, allowed, started, deadline):
    """Beam-search one stretch of a known lap, then time each line to the finish.

    The middle ground between the two methods that have already failed. Beam
    search over a whole track lost its entire beam to one corner; refinement
    polished the policy's line and could not leave it -- four sweeps, a thousand
    perturbations, three milliseconds. A window does what neither can: it finds a
    genuinely *different* line through the stretch that is losing time, while the
    rest of the lap stays exactly as it was.

    **Set the window to reach the finish.** The window's own score is arc at
    equal depth, which is only a proxy for lap time, so the honest way to judge a
    line is to drive the rest of the lap from it. Splicing the seed's remaining
    inputs back on instead is *open loop* and unsound: those inputs were tuned
    for the seed's trajectory, and a car nine metres further along at a different
    heading and speed diverges from them immediately. The first run here did
    exactly that -- it put the beam 9 m ahead at 14.0 s and then reported every
    one of the sixteen spliced laps as slower.

    Splicing is kept only for windows that end early, where it is at least a
    cheap screen, and it says so when it uses one.
    """
    import time as _time

    from polytrack.actions import ACTION_TO_MASK, mask_to_action
    from polytrack.physics.wasm import WasmPhysics
    from polytrack.recording import Recording
    from polytrack.search import BeamSearch, Candidate, SearchResult
    from polytrack.tas import TAS, format_time
    from polytrack.track import ProgressTracker

    lo, hi = (float(x) for x in args.beam_window.split(","))
    first = int(lo * 1000 / args.block_ms)
    last = int(hi * 1000 / args.block_ms)

    saved = json.loads(args.refine.read_text())
    seed_frames = int(saved["frames"])
    masks = list(Recording.deserialize(saved["recording"]).masks(seed_frames))
    blocks = [
        mask_to_action(masks[b * args.block_ms])
        for b in range(seed_frames // args.block_ms)
    ]
    print(f"  beam window {lo:.1f}-{hi:.1f} s of {format_time(seed_frames)} "
          f"from {args.refine.name}")
    print(f"  beam {args.beam} (+{args.reserve} reserve), top-{args.top_k}, "
          f"{last - first} depths\n")

    backend = WasmPhysics(entry.track_code(), assets=assets)
    tracker = ProgressTracker(entry.line)
    state = backend.reset()
    tracker.reset()
    arc, lateral = tracker.update(state.position)
    for action in blocks[:first]:
        result = backend.step_many(ACTION_TO_MASK[action], args.block_ms)
        state = result.state
        arc, lateral = tracker.update(state.position)
    anchor = backend.save_state()
    print(f"    anchor at {state.frame / 1000:.2f} s, {arc:.1f} m")

    root = Candidate(
        blob=anchor, frame=state.frame, checkpoint=state.checkpoint, arc=arc,
        actions=(), position=state.position.copy(),
        velocity=state.velocity.copy(), state=state, lateral=lateral,
    )
    search = BeamSearch(
        backend=backend, track=entry.line, beam_width=args.beam,
        block_frames=args.block_ms, prior=prior, top_k=args.top_k,
        reserve=args.reserve, lookahead=args.lookahead,
        line_weight=args.line_weight, switch_cost=args.switch_cost,
        # The seed's own actions from the anchor on. Kept in the beam at every
        # depth, so the search is bounded below by the lap it started from and
        # cannot lose every candidate to a corner the seed drives through.
        guide=tuple(int(a) for a in blocks[first:]),
    )

    def on_depth(depth, best):
        if depth % 10 == 0:
            print(f"    depth {depth:>3}/{last - first}  {best.arc:7.1f} m  "
                  f"{(_time.monotonic() - started) / 60:5.1f} min", flush=True)
        deadline()

    search.winner = None
    search.descend(root, last - first, on_depth)
    tail = blocks[last:]

    if search.winner is not None:
        # The beam ran all the way to the line. No splice, and no open-loop tail
        # -- which is the point. Replaying the seed's remaining inputs from a
        # state the beam has changed is unsound: they were tuned for the seed's
        # trajectory, and a car nine metres further on at a different heading
        # diverges from them at once. That is how the first windowed run threw
        # away a line it had already put 9 m ahead at 14.0 s.
        best_frames = search.winner.frame
        best_actions = list(search.winner.actions)
        print(f"\n    beam reached the line: {best_frames / 1000:.3f} s "
              f"({best_frames - seed_frames:+,} ms)")
    else:
        print(f"\n    {len(search.beam)} lines through the window; splicing each "
              f"back and timing to the finish")
        print("    (open-loop tail -- only trustworthy where the window barely "
              "moved the car)")
        best_frames, best_actions = seed_frames, None
    for i, candidate in enumerate(search.beam):
        backend.load_state(candidate.blob)
        result = None
        for action in tail:
            result = backend.step_many(ACTION_TO_MASK[int(action)], args.block_ms)
            if result.finished:
                break
        if result is not None and result.finished and result.state.frame < best_frames:
            best_frames = result.state.frame
            best_actions = list(candidate.actions)
            print(f"      line {i}: {best_frames / 1000:.3f} s "
                  f"({best_frames - seed_frames:+,} ms)", flush=True)
        deadline()

    if best_actions is None:
        print(f"\n  No line through the window beat {format_time(seed_frames)}.")
        return SearchResult(
            tas=None, frames=None, finished=False, n_rollouts=search._rollouts,
            n_frames_simulated=search._frames, best_arc=0.0,
            rate=backend.rate,
        )

    whole = blocks[:first] + best_actions + tail
    frame_masks = []
    for action in whole:
        frame_masks.extend([ACTION_TO_MASK[int(action)]] * args.block_ms)
    return SearchResult(
        tas=TAS.from_frame_masks(frame_masks[:best_frames]),
        frames=best_frames, finished=True, n_rollouts=search._rollouts,
        n_frames_simulated=search._frames, best_arc=entry.line.length,
        rate=backend.rate,
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    import numpy as np
    import torch

    from polytrack.priority import lift_background_qos

    # A search is normally launched detached and left for half an hour,
    # which is exactly what Windows tags as background work.
    lift_background_qos()
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
        if args.refine and args.beam_window:
            result = _beam_window(
                args, entry, assets, prior, allowed, started, deadline
            )
        elif args.refine:
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
