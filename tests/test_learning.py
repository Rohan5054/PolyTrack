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


def test_a_restricted_policy_never_takes_a_forbidden_action():
    """Both branches of epsilon-greedy, not just the argmax.

    A random BRAKE that the greedy policy can never choose is a transition the
    critic is asked to value for a policy that does not exist -- and at a 50 ms
    control interval one of them is half a car length.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    allowed = {3, 4, 5}
    agent = IQNAgent(IQNConfig(obs_dim=8, hidden=32, allowed_actions=tuple(allowed)))
    observation = np.random.randn(8).astype(np.float32)

    greedy = {agent.act(observation) for _ in range(100)}
    explored = {agent.act(observation, epsilon=1.0) for _ in range(400)}
    assert greedy <= allowed, f"greedy chose {greedy - allowed}"
    assert explored == allowed, f"exploration chose {explored ^ allowed}"

    values = agent.action_values(observation)
    assert np.isneginf(values[[0, 1, 2, 6, 7, 8]]).all(), "blocked action has a value"
    assert np.isfinite(values[[3, 4, 5]]).all(), "allowed action lost its value"


def test_the_bootstrap_does_not_reach_through_a_forbidden_action():
    """The subtle half of a restriction, and the half that fails silently.

    Masking only the acting argmax still leaves every TD target bootstrapped
    through ``max_a Q(s', a)`` over *all nine* actions. The critic then converges
    to the value of a policy that brakes while the car never brakes, so every
    state is optimistic by whatever braking was worth -- which is precisely the
    overestimation Double Q is in the loop to prevent.

    Rigged so the failure is unmissable: BRAKE is worth 1000 and nothing else is
    worth anything, so a target that leaks through it is three orders of
    magnitude out rather than subtly wrong.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    def rigged(allowed):
        torch.manual_seed(0)
        agent = IQNAgent(IQNConfig(obs_dim=8, hidden=32, allowed_actions=allowed))
        for net in (agent.online, agent.target):
            with torch.no_grad():
                net.head[-1].weight.zero_()
                net.head[-1].bias.zero_()
                net.head[-1].bias[6] = 1000.0  # BRAKE, and only BRAKE
        return agent

    batch = (
        np.zeros((16, 8), dtype=np.float32),
        np.full(16, 3, dtype=np.int64),
        np.zeros(16, dtype=np.float32),
        np.zeros((16, 8), dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        np.full(16, 0.999, dtype=np.float32),
    )
    weights = np.ones(16, dtype=np.float32)

    leaked, _ = rigged(None).learn(batch, weights)
    masked, _ = rigged((3, 4, 5)).learn(batch, weights)
    assert leaked.mean() > 900, "the rig did not work; unmasked should bootstrap 1000"
    assert masked.mean() < 1.0, (
        f"the target bootstrapped through BRAKE anyway (td {masked.mean():.1f})"
    )


def test_a_restriction_can_be_changed_without_losing_the_run():
    """``restrict`` mid-run, on a checkpoint worth hundreds of thousands of steps.

    Rebuilding the agent would apply the restriction just as well and discard the
    Adam state, which by then is worth as much as the weights.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    agent = IQNAgent(IQNConfig(obs_dim=8, hidden=32))
    batch = (
        np.random.randn(8, 8).astype(np.float32),
        np.random.randint(0, 9, 8).astype(np.int64),
        np.random.rand(8).astype(np.float32),
        np.random.randn(8, 8).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        np.full(8, 0.999, dtype=np.float32),
    )
    agent.learn(batch, np.ones(8, dtype=np.float32))
    exp_avg = agent.optimiser.state_dict()["state"][0]["exp_avg"].clone()

    agent.restrict((3, 4, 5))
    assert agent.steps == 1, "restricting reset the step count"
    assert torch.equal(agent.optimiser.state_dict()["state"][0]["exp_avg"], exp_avg)
    assert {agent.act(batch[0][0], 1.0) for _ in range(200)} == {3, 4, 5}

    agent.restrict(None)
    assert len({agent.act(batch[0][0], 1.0) for _ in range(400)}) == 9


def test_a_checkpoint_written_before_the_field_existed_still_loads():
    """``IQNConfig`` is ``slots=True``, so a new field is not merely defaulted.

    Pickle stores exactly the slots that existed at write time and the generated
    ``__setstate__`` zips them back, so a field added later ends up *unset* and
    raises ``AttributeError`` from wherever it is first read. These runs are
    hundreds of thousands of gradient steps and cannot be regenerated, so a
    checkpoint has to keep loading across any field addition.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNConfig, _upgrade_config

    stale = IQNConfig.__new__(IQNConfig)
    stale.__setstate__(IQNConfig(lr=1e-3).__getstate__()[:-1])  # drop the newest
    try:
        stale.allowed_actions
        raise AssertionError("the premise is gone: the slot is set after all")
    except AttributeError:
        pass

    upgraded = _upgrade_config(stale)
    assert upgraded.allowed_actions is None, "did not default the missing field"
    assert upgraded.lr == 1e-3, "lost a field that was in the checkpoint"


def test_an_unusable_restriction_is_rejected_at_construction():
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNAgent, IQNConfig

    for bad in ((), (9,), (-1,)):
        try:
            IQNAgent(IQNConfig(obs_dim=8, hidden=32, allowed_actions=bad))
        except ValueError:
            continue
        raise AssertionError(f"allowed_actions={bad} was accepted")


def test_epsilon_spread_covers_greedy_and_exploratory_actors():
    from polytrack.learning.collector import CollectorConfig, worker_epsilon

    config = CollectorConfig()
    values = [worker_epsilon(i, 6, config) for i in range(6)]
    assert values[0] > 0.35, "the noisiest actor is barely exploring"
    assert values[-1] < 0.01, "the greediest actor is still mostly random"
    assert all(a > b for a, b in zip(values, values[1:])), "not monotonic"

def test_lifting_background_qos_never_raises_and_reports_honestly():
    """It is an optimisation, so it must not be able to stop a run starting.

    A detached process on Windows is confined to the efficiency cores at their
    lowest clock -- 52.6 s against 3.6 s on a fixed single-threaded benchmark.
    Worth asking for; never worth failing over, which is why this checks the
    contract rather than the effect.
    """
    import sys

    from polytrack.priority import lift_background_qos

    applied = lift_background_qos()
    assert isinstance(applied, bool)
    # Idempotent: collectors inherit it and then ask again themselves.
    assert lift_background_qos() is applied
    if sys.platform == "win32":
        assert applied, "the API has been present since Windows 10 1709"


def test_closing_the_pool_lets_the_process_exit():
    """A finished run must actually exit, not deadlock having done its work.

    Every put onto a multiprocessing queue starts a feeder thread in the
    writing process, and Python joins those at interpreter exit. The learner
    writes weights to each worker, so a worker that dies holding unread data
    leaves the parent blocked on a full pipe forever. One seeded run printed
    its summary, saved its checkpoint and then sat at zero CPU until killed,
    which stopped the two runs queued behind it.

    Checked without starting any worker: closing a pool that never ran must
    still cancel the feeder threads, and must be safe to call twice, because
    train.py closes in a finally block that can run after an earlier close.
    """
    from polytrack.learning.collector import CollectorConfig, CollectorPool

    pool = CollectorPool(2, CollectorConfig())
    pool.broadcast({"nothing": 1})
    pool.close(timeout=0.1)
    assert pool.stop.is_set()
    pool.close(timeout=0.1)


def test_the_environment_cache_does_not_grow_with_the_library():
    """Memory per worker must not scale with how many tracks there are.

    Every environment owns a WasmPhysics with its own ~24 MB heap, and the
    cache that held one per track was unbounded. On 48 generated tracks that
    measured 1.1 GB a worker and 6.7 GB across six, which took a 15.7 GB
    machine to 0.2 GB free and 2000 pages a second. It never showed on the
    single-track runs, where the same code caches exactly one thing -- so the
    guard has to be against library size, not against absolute bytes.
    """
    from types import SimpleNamespace

    from polytrack.learning.collector import EnvCache

    cache = EnvCache(lambda entry: object(), limit=4)
    tracks = [SimpleNamespace(name=f"Generated {i}") for i in range(48)]
    for _ in range(3):           # round-robin, as a collector does
        for entry in tracks:
            cache.get(entry)
    assert len(cache) == 4, len(cache)

    # A repeated track is served from the cache rather than rebuilt.
    before = cache.builds
    for _ in range(10):
        cache.get(tracks[-1])
    assert cache.builds == before, "a cached track should not be rebuilt"

    # And the bound is never zero, however it is configured.
    tiny = EnvCache(lambda entry: object(), limit=0)
    tiny.get(tracks[0])
    assert len(tiny) == 1


def test_the_cache_evicts_least_recently_used():
    from types import SimpleNamespace

    from polytrack.learning.collector import EnvCache

    cache = EnvCache(lambda entry: entry.name, limit=2)
    a, b, c = (SimpleNamespace(name=n) for n in "abc")
    cache.get(a)
    cache.get(b)
    cache.get(a)        # a is now the most recent, b the least
    cache.get(c)        # so b is the one evicted
    builds = cache.builds
    cache.get(a)
    assert cache.builds == builds, "a was most recently used and should survive"
    cache.get(b)
    assert cache.builds == builds + 1, "b was least recently used and should go"


def test_loading_can_keep_the_weights_and_drop_the_optimiser():
    """A resume keeps Adam's moments; ``reset_optimiser`` drops them alone.

    The moments set the effective step size over a run's early steps, so they
    are worth dropping on purpose and never by accident. Resuming an
    interrupted run wants them kept, so the default has to stay False -- a flag
    that silently discarded them would quietly damage every resume in the
    project. This is not the transfer experiment's load: ``reset_optimiser``
    keeps the step count and the stale target, which
    ``test_loading_weights_only_starts_a_new_run_from_the_online_weights``
    covers.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    agent = IQNAgent(IQNConfig(obs_dim=8, hidden=32))
    batch = (
        np.random.randn(8, 8).astype(np.float32),
        np.random.randint(0, 9, 8).astype(np.int64),
        np.random.rand(8).astype(np.float32),
        np.random.randn(8, 8).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        np.full(8, 0.999, dtype=np.float32),
    )
    agent.learn(batch, np.ones(8, dtype=np.float32))
    weights = {k: v.clone() for k, v in agent.online.state_dict().items()}
    moments = agent.optimiser.state_dict()["state"][0]["exp_avg"].clone()
    assert moments.abs().sum() > 0, "the fixture never built any Adam state"

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        agent.save(path)

        kept = IQNAgent.load(path)
        assert kept.optimiser.state_dict()["state"], "a plain resume lost Adam"
        assert torch.equal(
            kept.optimiser.state_dict()["state"][0]["exp_avg"], moments
        ), "a plain resume changed the moment estimates"

        fresh = IQNAgent.load(path, reset_optimiser=True)
        assert not fresh.optimiser.state_dict()["state"], "the moments survived"
        # The weights are the thing being transferred, so they must be intact,
        # and the step count stays so the checkpoint's provenance is visible.
        for key, value in fresh.online.state_dict().items():
            assert torch.equal(value, weights[key]), f"{key} did not survive"
        assert fresh.steps == agent.steps


def _learn_a_little(agent, steps: int = 1) -> None:
    """A few gradient steps on random data, so Adam has state and steps > 0."""
    obs_dim, n_actions = agent.config.obs_dim, agent.config.n_actions
    batch = (
        np.random.randn(8, obs_dim).astype(np.float32),
        np.random.randint(0, n_actions, 8).astype(np.int64),
        np.random.rand(8).astype(np.float32),
        np.random.randn(8, obs_dim).astype(np.float32),
        np.zeros(8, dtype=np.float32),
        np.full(8, 0.99, dtype=np.float32),
    )
    for _ in range(steps):
        agent.learn(batch, np.ones(8, dtype=np.float32))


def test_loading_weights_only_starts_a_new_run_from_the_online_weights():
    """What a transfer comparison loads: the weights, and nothing of the run.

    ``reset_optimiser`` drops Adam but keeps two things a from-scratch control
    does not have. The step counter keys the replay's importance-sampling
    schedule, so a load from 1.22M steps samples at beta 1.0 from its first
    batch while the control starts at 0.4. And the saved target is a stale
    copy, where a fresh agent's target is its online network -- so the target
    has to come from the saved *online* weights.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    config = IQNConfig(
        obs_dim=8, hidden=32, lr=1e-3, gamma=0.99, allowed_actions=(3, 4, 5)
    )
    agent = IQNAgent(config)
    _learn_a_little(agent, steps=3)
    # Three steps leave the target at its initial copy, so it already differs
    # from the online network. Move it further anyway, so that a load which
    # took the saved target fails below whatever target_update_every is.
    with torch.no_grad():
        for parameter in agent.target.parameters():
            parameter.add_(1.0)
    online = {k: v.clone() for k, v in agent.online.state_dict().items()}
    assert agent.optimiser.state_dict()["state"], "the fixture built no Adam state"
    assert agent.steps == 3

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        agent.save(path)
        loaded = IQNAgent.load(path, weights_only=True)

    for key, value in loaded.online.state_dict().items():
        assert torch.equal(value, online[key]), f"online {key} did not survive"
    for key, value in loaded.target.state_dict().items():
        assert torch.equal(value, online[key]), (
            f"target {key} is not the saved online network"
        )
    assert not loaded.optimiser.state_dict()["state"], "Adam's moments survived"
    assert loaded.steps == 0, "the step count came forward"
    assert loaded.source_steps == 3, "lost the count the weights were trained for"
    assert loaded.config == config, "the config did not come from the checkpoint"


def test_the_discount_can_be_changed_without_losing_the_run():
    """``--gamma`` on a resume replaces the checkpoint's, as ``restrict`` does.

    The new value has to reach every holder of the config and the next
    checkpoint, so a later resume continues at it, and nothing else may move:
    by then Adam's state is worth as much as the weights.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    agent = IQNAgent(IQNConfig(obs_dim=8, hidden=32, gamma=0.99))
    _learn_a_little(agent)
    exp_avg = agent.optimiser.state_dict()["state"][0]["exp_avg"].clone()
    weights = {k: v.clone() for k, v in agent.online.state_dict().items()}

    agent.set_gamma(0.999)
    assert agent.config.gamma == 0.999
    assert agent.online.config is agent.config
    assert agent.target.config is agent.config
    assert agent.steps == 1, "changing the discount reset the step count"
    assert torch.equal(agent.optimiser.state_dict()["state"][0]["exp_avg"], exp_avg)
    for key, value in agent.online.state_dict().items():
        assert torch.equal(value, weights[key]), f"{key} changed"

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        agent.save(path)
        assert IQNAgent.load(path).config.gamma == 0.999, (
            "the checkpoint recorded the old discount"
        )


def _train_script():
    """``scripts/train.py`` as a module, loaded from its path without running.

    It is a script rather than part of the package, and ``main`` only runs
    under ``__main__``, so its argument parsing and agent start-up can be
    called directly -- nothing here starts a collector.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(train, *argv):
    """``train.parse_args`` on these flags, with a rejection made a failure.

    argparse rejects by raising SystemExit, which the runner below does not
    catch, so a regression in the CLI would end the whole suite at its usage
    message -- reading as though this file had been given a bad argument --
    instead of failing one test and running the rest.
    """
    import contextlib
    import io

    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            return train.parse_args(list(argv))
    except SystemExit as stopped:
        raise AssertionError(
            f"argparse rejected {list(argv)} (exit {stopped.code}):\n"
            + errors.getvalue()
        ) from None


def _start(train, *argv):
    """``train.start_agent`` on these flags: its four results, then its output."""
    import contextlib
    import io

    args = _parse(train, *argv)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        started = train.start_agent(args)
    return (*started, printed.getvalue())


def test_train_takes_the_discount_from_the_config_unless_told():
    """An unset ``--gamma`` has to mean IQNConfig's, not a default of its own.

    The flag defaulted to 0.99 and was always passed in, so the 0.999 that
    ``IQNConfig`` was designed around never once took effect. Every banked
    result is a 0.99 run, and reproduces with ``--gamma 0.99``.
    """
    torch = _torch()
    if torch is None:
        return
    from polytrack.learning.iqn import IQNConfig

    train = _train_script()
    assert _parse(train).gamma is None
    fresh, *_ = _start(train)
    assert fresh.config.gamma == IQNConfig().gamma
    told, *_ = _start(train, "--gamma", "0.99")
    assert told.config.gamma == 0.99


def test_train_refuses_a_load_flag_without_a_checkpoint():
    """``--weights-only`` and ``--reset-optimiser`` say how to load ``--resume``'s
    checkpoint.

    Ignored when there is none, either would let the run quietly start from
    scratch: a pretrained arm that is really a second control.
    ``--reset-optimiser`` was the transfer flag first, so an old
    pretrained-arm command that lost its ``--resume`` is the likely way in.
    """
    torch = _torch()
    if torch is None:
        return
    import contextlib
    import io

    train = _train_script()
    for flag in ("--weights-only", "--reset-optimiser"):
        errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(errors):
                train.parse_args([flag])
        except SystemExit as stopped:
            assert stopped.code == 2, f"{flag}: exited with {stopped.code}"
        else:
            raise AssertionError(f"{flag} was accepted without --resume")
        # The message itself: the usage block argparse prints ahead of every
        # error names --resume too, so finding that would prove nothing.
        assert f"{flag} needs --resume" in errors.getvalue(), errors.getvalue()
    assert _parse(train, "--resume", "agent.pt", "--weights-only").weights_only
    assert _parse(train, "--resume", "agent.pt", "--reset-optimiser").reset_optimiser


def test_a_weights_only_start_carries_nothing_of_the_earlier_run():
    """A later sitting continues a run's log; a new run from its weights must not.

    A run split across sittings brings its log, clock and best laps forward so
    that it plots as one curve. A run started from another run's weights is a
    new run, and the same carry-forward would graft that run's curve onto it.
    """
    torch = _torch()
    if torch is None:
        return
    import json
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    train = _train_script()
    source = IQNAgent(IQNConfig(obs_dim=8, hidden=32))
    _learn_a_little(source, steps=2)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        source.save(path)
        (Path(folder) / "history.json").write_text(json.dumps(
            [{"seconds": 600.0, "best": {"Summer 1": 30.5}}]
        ))

        resumed, history, elapsed, best, printed = _start(
            train, "--resume", str(path)
        )
        assert resumed.steps == 2 and len(history) == 1
        assert elapsed == 600.0 and best == {"Summer 1": 30500}
        assert "resumed from" in printed, printed

        fresh, history, elapsed, best, printed = _start(
            train, "--resume", str(path), "--weights-only"
        )
        assert fresh.steps == 0, "the step count came forward"
        assert (history, elapsed, best) == ([], 0.0, {}), (
            "the earlier run's log came forward"
        )
        assert "weights from" in printed, printed
        assert "(2 gradient steps), starting at step 0" in printed, printed


def test_gamma_on_a_resume_overrides_the_checkpoint_and_says_so():
    """The checkpoint's discount stands unless ``--gamma`` names another.

    Overriding has to change the agent, not only the collectors, or the run
    trains at one discount while every checkpoint it writes records another.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    train = _train_script()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        IQNAgent(IQNConfig(obs_dim=8, hidden=32, gamma=0.99)).save(path)

        kept, *_, printed = _start(train, "--resume", str(path))
        assert kept.config.gamma == 0.99, "a resume without --gamma changed it"
        assert "overrides" not in printed, printed

        same, *_, printed = _start(train, "--resume", str(path), "--gamma", "0.99")
        assert same.config.gamma == 0.99 and "overrides" not in printed, printed

        changed, *_, printed = _start(
            train, "--resume", str(path), "--gamma", "0.999"
        )
        assert changed.config.gamma == 0.999
        assert changed.online.config.gamma == 0.999
        assert "--gamma 0.999 overrides the checkpoint's 0.99" in printed, printed


def test_lr_on_a_resume_overrides_the_checkpoint_and_says_so():
    """``--lr`` works as ``--gamma`` does: the checkpoint's unless it is given.

    It used to be read by a fresh start alone, so any resume -- a
    ``--weights-only`` one included -- dropped it without a word, and the same
    flags could give a transfer experiment's two arms different learning
    rates. Given, it has to reach Adam's parameter groups, which are what the
    optimiser steps at, as well as the config, and nothing else may move.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    def rates(agent):
        return {group["lr"] for group in agent.optimiser.param_groups}

    train = _train_script()
    assert _parse(train).lr is None
    fresh, *_ = _start(train)
    assert fresh.config.lr == IQNConfig().lr
    told, *_ = _start(train, "--lr", "1e-4")
    assert told.config.lr == 1e-4 and rates(told) == {1e-4}

    source = IQNAgent(IQNConfig(obs_dim=8, hidden=32, lr=5e-5))
    _learn_a_little(source, steps=2)
    exp_avg = source.optimiser.state_dict()["state"][0]["exp_avg"].clone()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "agent.pt"
        source.save(path)

        kept, *_, printed = _start(train, "--resume", str(path))
        assert kept.config.lr == 5e-5 and rates(kept) == {5e-5}
        assert "overrides" not in printed, printed
        same, *_, printed = _start(train, "--resume", str(path), "--lr", "5e-5")
        assert same.config.lr == 5e-5 and "overrides" not in printed, printed

        changed, *_, printed = _start(
            train, "--resume", str(path), "--lr", "1e-4"
        )
        assert changed.config.lr == 1e-4
        assert rates(changed) == {1e-4}, "Adam kept the checkpoint's rate"
        assert "--lr 0.0001 overrides the checkpoint's 5e-05" in printed, printed
        assert changed.steps == 2, "changing the rate reset the step count"
        assert torch.equal(
            changed.optimiser.state_dict()["state"][0]["exp_avg"], exp_avg
        ), "changing the rate lost Adam's moments"

        # A --weights-only start is a transfer experiment's pretrained arm, so
        # it has to take the --lr its from-scratch control is given.
        moved, *_ = _start(
            train, "--resume", str(path), "--weights-only", "--lr", "1e-4"
        )
        assert moved.config.lr == 1e-4 and rates(moved) == {1e-4}

        changed.save(path)
        again = IQNAgent.load(path)
        assert again.config.lr == 1e-4 and rates(again) == {1e-4}, (
            "the checkpoint recorded the old rate"
        )


def test_the_collectors_discount_is_the_agents():
    """The collectors' n-step discount is the one the critic trains at.

    ``learn`` reads no gamma of its own -- every transition arrives discounted
    by its collector -- and ``CollectorConfig`` keeps a gamma of its own that
    defaults to 0.99. Left at that default, a fresh run would train at 0.99
    while every checkpoint recorded 0.999; built from the flag instead, an
    unset ``--gamma`` would hand the workers None, which fails only once they
    are running.
    """
    torch = _torch()
    if torch is None:
        return
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    train = _train_script()
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "agent.pt")
        IQNAgent(IQNConfig(obs_dim=8, hidden=32, gamma=0.99)).save(path)
        for argv, expected in (
            ((), IQNConfig().gamma),
            (("--gamma", "0.99"), 0.99),
            (("--resume", path), 0.99),
            (("--resume", path, "--gamma", "0.999"), 0.999),
            (("--resume", path, "--weights-only"), 0.99),
        ):
            args = _parse(train, *argv)
            with contextlib.redirect_stdout(io.StringIO()):
                agent, *_ = train.start_agent(args)
            config = train.collector_config_for(args, agent)
            assert config.gamma == agent.config.gamma == expected, (
                f"{argv}: collectors at {config.gamma}, agent at "
                f"{agent.config.gamma}, wanted {expected}"
            )
            assert config.iqn_config is agent.config, argv


def test_a_weights_only_start_will_not_write_over_a_run():
    """A new run's log starts empty, so it must not start in a run's directory.

    Pointed at one -- the run its weights come from, or a banked run that
    shares the name -- it would write over that run's history.json at the
    first log line and its agent.pt on any exit. A plain resume into the same
    directory carries the log forward, so that stays allowed.
    """
    torch = _torch()
    if torch is None:
        return
    import tempfile
    from pathlib import Path

    from polytrack.learning.iqn import IQNAgent, IQNConfig

    train = _train_script()
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "source"
        logged = Path(folder) / "logged"     # a banked run kept without its agent.pt
        source.mkdir()
        logged.mkdir()
        path = str(source / "agent.pt")
        IQNAgent(IQNConfig(obs_dim=8, hidden=32)).save(path)
        (logged / "history.json").write_text("[]")

        # --name is joined onto runs/, and an absolute one replaces it.
        for taken in (source, logged):
            args = _parse(
                train, "--resume", path, "--weights-only", "--name", str(taken)
            )
            try:
                train.run_directory(args)
            except SystemExit as stopped:
                assert "already holds a run" in str(stopped.code), stopped.code
            else:
                raise AssertionError(f"a --weights-only start was let into {taken}")

        args = _parse(train, "--resume", path, "--name", str(source))
        assert train.run_directory(args) == source
        fresh = Path(folder) / "fresh"
        args = _parse(train, "--resume", path, "--weights-only", "--name", str(fresh))
        assert train.run_directory(args) == fresh


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
