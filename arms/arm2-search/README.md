# Arm 2 -- policy plus guided beam search

**Winter 5: 18786** (`winter_5.json`, gate PASS, both checkpoints)

239 ms under the 19025 human lap and 636 ms under Arm 1.

```bash
python scripts/search_run.py runs/w5-fwd/agent.pt --track "Winter 5" \
    --refine <seed>.json --beam-window 8.0,21.0 \
    --beam 20 --top-k 3 --reserve 5 --lookahead 1.0 --line-weight 1.0
```

## How it works

The policy is an action **prior**: only its top-k rated actions are expanded from
each candidate, so nine actions do not mean 59,049 futures a quarter-second
ahead. The beam is anchored part way through a lap that already finishes and
carries that lap as a **guide** at every depth, which makes the result bounded
below by its seed.

Candidates are ranked `arc + lookahead*speed - line_weight*|lateral|`. All three
terms were added because a measurement said so, and the third is the one that
mattered.

## Lineage

`lineage/` holds the seeds, because 18786 is the end of a chain and each link is
independently verifiable.

```
19422  Arm 1, policy alone
  |
  +-- 19420 -> 19405   refinement, then guided beam on the last 2 s
  |                    superseded: this branch stalled
  |
  +-- 19249            fastest collector episode of a training sitting
        |
        19245          guided beam from that seed
        |
        18786          same beam, ranked on distance from the reference line
```

The branch that mattered starts from a **training** lap, not a searched one. The
collectors' epsilon noise produced 19249 while seven beam runs over ~2.5M
simulated frames were managing 19405 from a worse seed. A collector perturbs an
action and the policy then drives on from the perturbed state -- closed loop, so
the perturbation gets corrected. A beam optimises an open-loop action sequence,
where it compounds.

## What `line_weight` is, and its caveat

The Winter 5 reference line **is** the 19025 human lap, because
`build_library.py` derives it from that record. So `|lateral|` is distance from
the fastest known route, and charging arc for it pulls candidates onto that
route.

It was added after measuring what actually separated the laps. Not skidding:
0.3 degrees of slip on both, and `wheel_skid_info` saturates at 4.000 for both,
so that field is a visual-effects flag rather than physics. Not wall contact:
zero collision impulse on both. Not throttle: 100% on both. Not steering effort:
0.011 against 0.013. What was left was the route -- the agent wandered **8.3 m**
off the line, mean 2.16 m, and drove **6.2 m further** over the lap.

Adding the term found 459 ms in one run. Seven previous runs ranking on arc alone
had found 14 ms in total.

**The caveat, and it belongs in the write-up.** The search was steered toward a
human racing line. The control sequence is the agent's own and it drives that
route 239 ms faster than the human did, so it is not copying inputs -- but it
found the route with help rather than discovering it. On a track whose reference
line came from a mediocre lap, this term pulls toward mediocrity, which is why it
defaults to zero.

## Known remaining loss

Measured against the human lap with `scripts/compare_laps.py`, roughly 150 ms is
still available and all of it is early, because the search window started at
8.0 s and never saw the first 460 m:

| stretch | lost |
|---|---|
| 0-100 m (the launch) | +39 ms |
| 400-500 m | +50 ms |
| 200-300 m | +14 ms |
| 1200-1300 m | +50 ms |
