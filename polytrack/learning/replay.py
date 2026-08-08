"""n-step returns and prioritised experience replay.

Both are standard, and both are here rather than pulled from a library because
the interesting parameters are ones a racing task needs to set unusually:

**n-step is doing more work than usual.** A decision here covers 50 ms and a lap
is hundreds of decisions, so credit for "braked early enough for that corner"
has to travel a long way back. One-step bootstrapping propagates it one decision
per gradient step, which is far too slow. n=3 is the common default; racing
agents run longer, and this defaults to 5.

**Priorities are sampled, not just weighted.** A track is mostly straights the
agent gets right immediately, and a few corners it does not. Uniform sampling
spends most of the batch on the straights. Prioritising by TD error puts it on
the corners, which is where the remaining time is.

The sum tree is the usual array trick: a complete binary tree in a flat array,
internal node ``i`` holding the sum of ``2i`` and ``2i+1``, so both update and
prefix-sum search are O(log n) rather than O(n). At 200k transitions that is the
difference between 18 comparisons and 200,000 per sample.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SumTree:
    """Fixed-capacity array of non-negative values, searchable by prefix sum.

    ``capacity`` is rounded up to a power of two so the tree is complete and the
    index arithmetic needs no bounds cases.
    """

    __slots__ = ("capacity", "_nodes")

    def __init__(self, capacity: int) -> None:
        self.capacity = 1 << (capacity - 1).bit_length()
        self._nodes = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self._nodes[1])

    def __setitem__(self, index: int, value: float) -> None:
        node = index + self.capacity
        self._nodes[node] = value
        node //= 2
        while node >= 1:
            self._nodes[node] = self._nodes[2 * node] + self._nodes[2 * node + 1]
            node //= 2

    def __getitem__(self, index: int) -> float:
        return float(self._nodes[index + self.capacity])

    def update_many(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Set many leaves, then rebuild the affected internal nodes once.

        Rebuilding a level at a time is much faster than walking each leaf to the
        root separately when a whole batch changes, which is every learner step.
        """
        nodes = np.asarray(indices, dtype=np.int64) + self.capacity
        self._nodes[nodes] = values
        parents = np.unique(nodes // 2)
        while parents.size and parents[0] >= 1:
            self._nodes[parents] = (
                self._nodes[2 * parents] + self._nodes[2 * parents + 1]
            )
            if parents[0] == 1:
                break
            parents = np.unique(parents // 2)

    def find(self, prefix: np.ndarray) -> np.ndarray:
        """Leaf indices whose cumulative sum first exceeds each prefix value.

        Vectorised descent: every query walks the same number of levels, so the
        whole batch moves down together.
        """
        values = np.asarray(prefix, dtype=np.float64).copy()
        nodes = np.ones(values.shape, dtype=np.int64)
        while nodes[0] < self.capacity:
            left = 2 * nodes
            go_right = values > self._nodes[left]
            values -= np.where(go_right, self._nodes[left], 0.0)
            nodes = left + go_right
        return nodes - self.capacity


class NStepAccumulator:
    """Turns a stream of one-step transitions into n-step ones.

    Holds the last ``n`` transitions and emits, for the oldest of them, the
    discounted sum of the rewards since and the state ``n`` steps later. On
    episode end the remainder is flushed with whatever horizon is left, so the
    terminal reward is not thrown away.
    """

    __slots__ = ("n", "gamma", "_window", "_discounts")

    def __init__(self, n: int = 5, gamma: float = 0.99) -> None:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self.gamma = gamma
        self._window: deque = deque(maxlen=n)
        self._discounts = gamma ** np.arange(n, dtype=np.float64)

    def push(self, obs, action: int, reward: float, next_obs, done: bool):
        """Add a one-step transition; yield any n-step transitions it completes."""
        self._window.append((obs, action, reward, next_obs, done))
        if done:
            # Every entry still in the window ends at this terminal state.
            while self._window:
                yield self._collapse()
                self._window.popleft()
        elif len(self._window) == self.n:
            yield self._collapse()
            self._window.popleft()

    def _collapse(self):
        rewards = np.fromiter(
            (entry[2] for entry in self._window), dtype=np.float64,
            count=len(self._window),
        )
        total = float(rewards @ self._discounts[: len(rewards)])
        obs, action = self._window[0][0], self._window[0][1]
        _, _, _, next_obs, done = self._window[-1]
        # The discount the learner must apply to its bootstrap: one per step
        # actually accumulated, which is fewer than n on a flushed tail.
        return obs, action, total, next_obs, done, self.gamma ** len(self._window)

    def clear(self) -> None:
        self._window.clear()


class PrioritisedReplay:
    """Fixed-capacity replay buffer sampled in proportion to priority.

    Args:
        capacity: Transitions held. At ~60 transitions/second aggregate, 200k is
            about an hour of collection.
        obs_dim: Length of the observation vector.
        alpha: How sharply priority is followed. 0 is uniform, 1 is fully
            proportional. 0.5 is the value the Ape-X paper settles on.
        beta0: Starting importance-sampling exponent, annealed to 1. Prioritising
            biases the gradient; the weights correct it, and correcting fully
            from the start is unstable while the value estimates are noise.
        beta_steps: Learner steps over which beta reaches 1.
    """

    def __init__(
        self,
        capacity: int = 200_000,
        obs_dim: int = 62,
        alpha: float = 0.5,
        beta0: float = 0.4,
        beta_steps: int = 500_000,
        epsilon: float = 1e-3,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta0 = beta0
        self.beta_steps = beta_steps
        self.epsilon = epsilon

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)

        self._tree = SumTree(capacity)
        self._cursor = 0
        self._size = 0
        self._max_priority = 1.0

    def __len__(self) -> int:
        return self._size

    def add(self, obs, action, reward, next_obs, done, discount) -> None:
        i = self._cursor
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.discounts[i] = discount
        # New transitions enter at the highest priority seen, so everything is
        # replayed at least once before its priority is ever believed.
        self._tree[i] = self._max_priority ** self.alpha
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, step: int = 0):
        """Draw a batch. Returns ``(indices, weights, tensors...)``.

        Stratified: the total priority is cut into ``batch_size`` equal segments
        and one sample drawn from each. Cheaper variance than independent draws,
        and it guarantees the batch spans the whole buffer rather than piling
        onto whichever few transitions currently have the largest error.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        total = self._tree.total
        edges = np.linspace(0.0, total, batch_size + 1)
        prefix = edges[:-1] + np.random.random(batch_size) * np.diff(edges)
        indices = self._tree.find(prefix)
        # find() can land on an unfilled leaf when priorities are degenerate.
        indices = np.clip(indices, 0, max(self._size - 1, 0))

        priorities = self._tree._nodes[indices + self._tree.capacity]
        probabilities = np.maximum(priorities / max(total, 1e-12), 1e-12)
        beta = min(1.0, self.beta0 + (1.0 - self.beta0) * step / max(self.beta_steps, 1))
        weights = (self._size * probabilities) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return (
            indices,
            weights,
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
            self.discounts[indices],
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.epsilon
        self._max_priority = max(self._max_priority, float(priorities.max()))
        self._tree.update_many(indices, priorities**self.alpha)
