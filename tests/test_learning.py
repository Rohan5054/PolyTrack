"""The training machinery: replay, n-step returns, and the quantile critic.

None of this touches the simulator, so it all runs anywhere torch is installed.
The interesting test is the last one -- it fits the critic to a distribution
whose quantiles are known in closed form, which is the only way to tell a working
quantile loss from a plausible-looking one. A subtly wrong loss (the sum and the
mean over the two quantile axes swapped, say) still trains, still produces a
falling curve, and still learns nothing but the mean.
"""

from __future__ import annotations

import numpy as np

from polytrack.learning.replay import (
    NStepAccumulator,
    PrioritisedReplay,
    SumTree,
)

_SKIPPED: list[str] = []


# ------------------------------------------------------------------ sumtree --


def test_sumtree_totals_and_lookup():
    tree = SumTree(8)
    values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    for i, v in enumerate(values):
        tree[i] = v
    assert abs(tree.total - sum(values)) < 1e-12

    # A prefix inside leaf i's slice must return i. Boundaries are the only
    # place this can be off by one, so probe just inside each.
    edges = np.concatenate([[0.0], np.cumsum(values)])
    probes = (edges[:-1] + edges[1:]) / 2.0
    found = tree.find(probes)
    assert list(found) == list(range(len(values)))


def test_sumtree_batch_update_matches_one_at_a_time():
    single = SumTree(16)
    batched = SumTree(16)
    rng = np.random.default_rng(0)
    values = rng.random(16) * 10

    for i, v in enumerate(values):
        single[i] = v
    batched.update_many(np.arange(16), values)

    assert abs(single.total - batched.total) < 1e-12
    for i in range(16):
        assert abs(single[i] - batched[i]) < 1e-12


def test_sumtree_sampling_follows_priority():
    """A leaf with 90% of the mass must win about 90% of the draws."""
    tree = SumTree(4)
    tree[0] = 90.0
    for i in (1, 2, 3):
        tree[i] = 10.0 / 3.0
    rng = np.random.default_rng(1)
    draws = tree.find(rng.random(4000) * tree.total)
    share = float((draws == 0).mean())
    assert 0.86 < share < 0.94, f"leaf 0 drawn {share:.1%} of the time, wanted ~90%"


# ------------------------------------------------------------------- n-step --


def test_nstep_discounts_the_reward_sequence():
    accumulator = NStepAccumulator(n=3, gamma=0.5)
    emitted = []
    for step in range(3):
        emitted += list(
            accumulator.push(np.array([step]), step, float(step + 1),
                             np.array([step + 1]), False)
        )
    assert len(emitted) == 1
    obs, action, reward, next_obs, done, discount = emitted[0]
    assert obs[0] == 0 and action == 0
    # 1 + 0.5*2 + 0.25*3 = 2.75, bootstrapping from three steps later.
    assert abs(reward - 2.75) < 1e-12
    assert next_obs[0] == 3
    assert abs(discount - 0.125) < 1e-12


def test_nstep_flushes_the_tail_at_an_episode_end():
    """The terminal reward must reach every transition still in the window.

    Dropping the tail is a quiet bug: it throws away exactly the transitions
    adjacent to the finish line, which are the ones carrying the outcome.
    """
    accumulator = NStepAccumulator(n=4, gamma=1.0)
    emitted = []
    for step in range(3):
        emitted += list(
            accumulator.push(np.array([step]), 0, 1.0, np.array([step + 1]),
                             step == 2)
        )
    assert len(emitted) == 3, "the window was not flushed on termination"
    assert all(t[4] for t in emitted), "flushed transitions must be terminal"
    # Horizons shorten toward the end: 3, 2, then 1 reward accumulated.
    assert [t[2] for t in emitted] == [3.0, 2.0, 1.0]


# ------------------------------------------------------------------- replay --


def test_replay_round_trips_a_transition():
    replay = PrioritisedReplay(capacity=64, obs_dim=4)
    obs = np.arange(4, dtype=np.float32)
    replay.add(obs, 3, 1.5, obs + 1, False, 0.9)
    assert len(replay) == 1

    _, weights, sampled_obs, actions, rewards, next_obs, dones, discounts = (
        replay.sample(1)
    )
    assert np.allclose(sampled_obs[0], obs)
    assert actions[0] == 3
    assert abs(float(rewards[0]) - 1.5) < 1e-6
    assert np.allclose(next_obs[0], obs + 1)
    assert dones[0] == 0.0
    assert abs(float(discounts[0]) - 0.9) < 1e-6
    assert 0.0 < weights[0] <= 1.0


def test_replay_prefers_high_error_transitions():
    """The whole point of prioritising: the batch must land on the errors."""
    replay = PrioritisedReplay(capacity=256, obs_dim=2, alpha=1.0)
    for i in range(64):
        replay.add(np.array([i, 0.0]), 0, 0.0, np.array([i, 1.0]), False, 0.99)

    # One transition is wrong by a lot; the rest are already learned.
    errors = np.full(64, 0.001)
    errors[7] = 50.0
    replay.update_priorities(np.arange(64), errors)

    indices, *_ = replay.sample(512)
    share = float((indices == 7).mean())
    assert share > 0.5, f"the high-error transition took only {share:.1%} of the batch"


def test_replay_wraps_without_losing_capacity():
    replay = PrioritisedReplay(capacity=32, obs_dim=2)
    for i in range(100):
        replay.add(np.array([i, 0.0]), 0, float(i), np.array([i, 1.0]), False, 0.99)
    assert len(replay) == 32
    indices, *_ = replay.sample(64)
    assert indices.max() < 32


# ---------------------------------------------------------------------- iqn --


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        _SKIPPED.append("torch is not installed: pip install torch")
        return None
    import torch

    torch.set_num_threads(2)
    return torch


def test_network_shapes():
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNConfig, IQNNetwork

    config = IQNConfig(obs_dim=62, n_actions=9, hidden=64, n_cos=16)
    net = IQNNetwork(config)
    obs = torch.zeros(5, 62)
    taus = torch.rand(5, 7)
    assert net(obs, taus).shape == (5, 7, 9)
    assert net.q_values(obs, 11).shape == (5, 9)


def test_the_quantile_embedding_actually_changes_the_output():
    """tau must reach the output.

    The paper merges state and tau features multiplicatively for this reason: an
    additive merge lets the network learn to ignore tau, and it does, collapsing
    to a mean estimator that still trains and still looks fine.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNConfig, IQNNetwork

    net = IQNNetwork(IQNConfig(obs_dim=4, n_actions=2, hidden=32, n_cos=16))
    obs = torch.randn(1, 4)
    low = net(obs, torch.full((1, 1), 0.05))
    high = net(obs, torch.full((1, 1), 0.95))
    assert not torch.allclose(low, high, atol=1e-6)


def test_the_critic_learns_a_known_return_distribution():
    """Fit a two-point distribution and check the learned quantiles.

    One state, one action, return 0 or 10 with equal probability. A correct
    quantile loss puts the 10th percentile near 0, the 90th near 10, and the mean
    near 5. A mean-only estimator -- which is what a mis-wired quantile loss
    degenerates to -- puts all three at 5, so this separates them.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNConfig, IQNNetwork, quantile_huber_loss

    torch.manual_seed(0)
    config = IQNConfig(obs_dim=1, n_actions=1, hidden=64, n_cos=32, kappa=1.0)
    net = IQNNetwork(config)
    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)

    batch = 256
    obs = torch.zeros(batch, 1)
    for _ in range(1500):
        returns = (torch.rand(batch, 1) < 0.5).float() * 10.0  # (B, 1) targets
        taus = torch.rand(batch, 8)
        quantiles = net(obs, taus).squeeze(-1)  # (B, 8)
        td = returns.unsqueeze(1) - quantiles.unsqueeze(1)  # (B, 1, 8)
        loss = quantile_huber_loss(td, taus, config.kappa).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        probe = torch.tensor([[0.1, 0.5, 0.9]])
        learned = net(torch.zeros(1, 1), probe)[0, :, 0].numpy()

    low, middle, high = (float(v) for v in learned)
    assert low < 2.5, f"the 10th percentile came out at {low:.2f}, wanted near 0"
    assert high > 7.5, f"the 90th percentile came out at {high:.2f}, wanted near 10"
    assert 3.0 < middle < 7.0, f"the median came out at {middle:.2f}"
    assert high - low > 5.0, (
        f"learned a spread of only {high - low:.2f}: this is a mean estimator, "
        "not a distributional one"
    )


def test_a_gradient_step_runs_and_reduces_loss_on_a_fixed_batch():
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    torch.manual_seed(0)
    np.random.seed(0)
    agent = IQNAgent(IQNConfig(obs_dim=8, n_actions=4, hidden=64, n_cos=32, lr=1e-3))

    rng = np.random.default_rng(0)
    batch = (
        rng.standard_normal((32, 8)).astype(np.float32),
        rng.integers(0, 4, 32),
        rng.standard_normal(32).astype(np.float32) * 5.0,
        rng.standard_normal((32, 8)).astype(np.float32),
        np.zeros(32, dtype=np.float32),
        np.full(32, 0.99, dtype=np.float32),
    )
    weights = np.ones(32, dtype=np.float32)

    first = None
    for _ in range(120):
        td_errors, loss = agent.learn(batch, weights)
        first = loss if first is None else first
    assert np.isfinite(loss) and loss < first, (
        f"loss went {first:.4f} -> {loss:.4f} on a fixed batch"
    )
    assert td_errors.shape == (32,)
    assert np.isfinite(td_errors).all()


def test_weights_survive_a_round_trip():
    """Collectors receive weights as plain numpy across a process boundary."""
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    config = IQNConfig(obs_dim=6, n_actions=3, hidden=32, n_cos=16)
    source, destination = IQNAgent(config), IQNAgent(config)
    observation = np.random.default_rng(0).standard_normal(6).astype(np.float32)

    destination.load_weights(source.weights())
    torch.manual_seed(7)
    a = source.action_values(observation)
    torch.manual_seed(7)
    b = destination.action_values(observation)
    assert np.allclose(a, b, atol=1e-6)


def test_epsilon_spread_covers_greedy_and_exploratory_actors():
    from polytrack.learning.collector import CollectorConfig, worker_epsilon

    config = CollectorConfig()
    values = [worker_epsilon(i, 6, config) for i in range(6)]
    assert values[0] > 0.35, "the noisiest actor is barely exploring"
    assert values[-1] < 0.01, "the greediest actor is still mostly random"
    assert all(a > b for a, b in zip(values, values[1:])), "not monotonic"


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if _SKIPPED:
        print(f"\ntorch-backed tests skipped: {_SKIPPED[0]}")
    sys.exit(1 if failures else 0)
