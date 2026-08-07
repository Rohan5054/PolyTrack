"""Cross-entropy method over action sequences.

Where beam search explores a tree breadth-first, CEM optimises a *distribution*
over whole input sequences: sample a population, keep the best few, refit toward
them, repeat.

The two are complementary, and which one wins depends on the track:

* **Beam search** is better when the track has hard commit points -- a jump you
  either clear or don't. Its lockstep expansion compares candidates fairly and
  it never throws away a promising line early.
* **CEM** is better over long horizons and when the payoff surface is smooth. It
  has no branching factor, so cost is flat in horizon length rather than
  exponential, and it parallelises trivially: every sample is independent.

CEM also takes a warm start, which is how it slots into the hybrid design --
seed it from a trained policy's rollout and it refines around that line instead
of searching from noise.
"""

from __future__ import annotations

import numpy as np

from ..actions import ACTION_TO_MASK, N_ACTIONS
from ..physics.base import PhysicsBackend
from ..track import Track
from .base import SearchResult, rollout


def cem_search(
    backend: PhysicsBackend,
    track: Track,
    *,
    horizon: int = 600,
    block_frames: int = 6,
    n_samples: int = 64,
    n_elites: int = 8,
    n_iterations: int = 20,
    smoothing: float = 0.7,
    warm_start: list[int] | None = None,
    seed: int = 0,
) -> SearchResult:
    """Optimise an input sequence by iterated sampling and refitting.

    Args:
        horizon: Frames to plan over. Must comfortably exceed the expected
            finish time, or the search can never see a completion.
        block_frames: Frames per action decision, as in beam search.
        n_samples: Population per iteration. More is better and parallelisable.
        n_elites: How many top performers shape the next distribution.
        smoothing: Interpolation weight toward the elite distribution. Below 1.0
            keeps the distribution from collapsing onto the first decent line it
            finds and then being unable to escape it.
        warm_start: Per-frame masks to bias the initial distribution toward --
            pass a policy rollout or a previous search result here.
        seed: Sampling seed. Search is fully reproducible given the same seed.

    Returns:
        The best trajectory found across all iterations.
    """
    rng = np.random.default_rng(seed)
    n_blocks = max(horizon // block_frames, 1)

    # probs[b, a] = probability of taking action a in block b.
    probs = np.full((n_blocks, N_ACTIONS), 1.0 / N_ACTIONS, dtype=np.float64)
    if warm_start:
        probs = _bias_toward(probs, warm_start, block_frames, weight=0.6)

    best: dict | None = None
    evaluations = 0

    for _ in range(n_iterations):
        scored: list[tuple[float, np.ndarray, dict]] = []

        for _ in range(n_samples):
            actions = _sample(rng, probs)
            masks = _to_masks(actions, block_frames)
            candidate = rollout(backend, track, masks)
            evaluations += candidate.frame

            outcome = {
                "frames": candidate.frame,
                "finished": candidate.finished,
                "arc": candidate.arc,
                "masks": candidate.masks,
            }
            scored.append((_score(outcome, horizon), actions, outcome))

            if best is None or _score(outcome, horizon) > _score(best, horizon):
                best = outcome

        # Refit toward the elites.
        scored.sort(key=lambda item: item[0], reverse=True)
        elites = np.stack([actions for _, actions, _ in scored[:n_elites]])
        counts = np.zeros_like(probs)
        for block in range(n_blocks):
            counts[block] = np.bincount(elites[:, block], minlength=N_ACTIONS)
        # Laplace smoothing keeps every action's probability non-zero, so an
        # action ruled out early can still be rediscovered later.
        counts += 1.0
        counts /= counts.sum(axis=1, keepdims=True)
        probs = smoothing * counts + (1.0 - smoothing) * probs
        probs /= probs.sum(axis=1, keepdims=True)

    if best is None:
        return SearchResult([], 0, False, 0.0, evaluations)

    return SearchResult(
        masks=best["masks"],
        frames=best["frames"],
        finished=best["finished"],
        arc=best["arc"],
        evaluations=evaluations,
    )


def _score(outcome: dict, horizon: int) -> float:
    """Rank trajectories: finishing dominates, then speed, then progress.

    Finishers are offset by a large constant so that *any* completion outranks
    *any* non-completion. Without that, a non-finishing run that coasted a long
    way can outscore a slow but valid finish, and the population drifts away
    from the only outcome that actually counts.
    """
    if outcome["finished"]:
        return 1e6 + (horizon - outcome["frames"])
    return outcome["arc"]


def _sample(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Draw one action per block from the current per-block distributions."""
    cumulative = probs.cumsum(axis=1)
    draws = rng.random((len(probs), 1))
    return (draws < cumulative).argmax(axis=1)


def _to_masks(actions: np.ndarray, block_frames: int) -> list[int]:
    masks: list[int] = []
    for action in actions:
        masks.extend([ACTION_TO_MASK[int(action)]] * block_frames)
    return masks


def _bias_toward(
    probs: np.ndarray, masks: list[int], block_frames: int, weight: float
) -> np.ndarray:
    """Tilt the initial distribution toward an existing input sequence."""
    from ..actions import mask_to_action

    biased = probs.copy()
    for block in range(len(probs)):
        frame = block * block_frames
        if frame >= len(masks):
            break
        action = mask_to_action(masks[frame])
        biased[block] *= 1.0 - weight
        biased[block, action] += weight
    return biased / biased.sum(axis=1, keepdims=True)
