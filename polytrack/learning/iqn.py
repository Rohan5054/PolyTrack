"""Implicit Quantile Networks: the critic, its loss, and the agent around it.

IQN (Dabney et al., ICML 2018) learns the quantile function of the return rather
than its mean. Instead of one number per action it learns a map from a quantile
fraction tau to the return at that quantile, sampling fresh taus every forward
pass -- so the distribution is represented implicitly, at arbitrary resolution,
by a network that takes tau as an input.

Why distributional, for this task specifically: racing outcomes are bimodal in a
way that averages badly. Committing to a corner at speed either works or puts the
car into a wall, and a mean Q-value of those two futures describes neither of
them. A quantile critic can say "usually 90, occasionally 5", which is what is
actually true, and the richer per-sample gradient signal is the usual explanation
for why distributional critics learn faster even when only the mean is used to
act on.

Three implementation choices worth naming:

* **Double Q-learning.** The bootstrap action comes from the online network and
  its value from the target network. Standard, and the overestimation it fixes is
  worse than usual here: with nine actions and a dense shaped reward, a single
  optimistic corner-cutting Q-value propagates a long way before anything
  contradicts it.
* **Quantile Huber loss**, not plain Huber. The asymmetric ``|tau - 1(u<0)|``
  weight is the whole mechanism -- it is what makes each output head converge to
  its own quantile rather than all of them to the mean.
* **One thread per process.** Torch defaults to using every core, and with six
  collector processes each doing that the machine spends its time in thread
  contention. The collectors set this; a learner sharing the box should be given
  a deliberate budget too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class IQNConfig:
    """Network and optimisation hyperparameters.

    The defaults are the paper's where the paper has an opinion, and CleanRL's
    or Linesight's where it does not. ``n_cos`` at 64 and the quantile counts at
    8 are Dabney's; ``lr`` at 5e-5 is lower than Atari IQN's 5e-5 * 4 because a
    62-float observation needs far less network than a frame stack and tolerates
    less noise.
    """

    obs_dim: int = 62
    n_actions: int = 9
    hidden: int = 256
    #: Cosine basis functions embedding tau. 64 is the paper's; fewer visibly
    #: degrades the quantile function's resolution near the tails.
    n_cos: int = 64
    #: Quantile samples for the online net, the target, and action selection.
    n_tau: int = 8
    n_tau_prime: int = 8
    n_tau_action: int = 32
    lr: float = 5e-5
    adam_eps: float = 1e-4  # 1e-8 underflows in float32 on sparse gradients
    gamma: float = 0.99
    kappa: float = 1.0  # Huber transition point
    grad_clip: float = 10.0
    target_update_every: int = 1000


class IQNNetwork(nn.Module):
    """State features, a cosine embedding of tau, and their product.

    The multiplicative merge is the paper's and matters: adding the two
    embeddings lets the network ignore tau, and it will, collapsing straight back
    to a mean estimator with extra steps.
    """

    def __init__(self, config: IQNConfig) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
        )
        self.tau_embedding = nn.Linear(config.n_cos, config.hidden)
        self.head = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(),
            nn.Linear(config.hidden, config.n_actions),
        )
        # cos(pi * i * tau) for i = 0..n_cos-1, precomputed as a constant row.
        self.register_buffer(
            "_pi_range",
            torch.arange(config.n_cos, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    def forward(
        self, obs: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        """Quantiles of the return.

        Args:
            obs: (B, obs_dim)
            taus: (B, T) quantile fractions in [0, 1].

        Returns:
            (B, T, n_actions)
        """
        batch, n_taus = taus.shape
        state = self.features(obs)  # (B, H)

        cosines = torch.cos(taus.unsqueeze(-1) * self._pi_range)  # (B, T, n_cos)
        tau_features = F.relu(self.tau_embedding(cosines))  # (B, T, H)

        merged = state.unsqueeze(1) * tau_features  # (B, T, H)
        return self.head(merged.view(batch * n_taus, -1)).view(
            batch, n_taus, self.config.n_actions
        )

    def q_values(self, obs: torch.Tensor, n_taus: int) -> torch.Tensor:
        """Mean over sampled quantiles: the ordinary Q-value, for acting."""
        taus = torch.rand(obs.shape[0], n_taus, device=obs.device)
        return self.forward(obs, taus).mean(dim=1)


def quantile_huber_loss(
    td_errors: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0
) -> torch.Tensor:
    """Per-sample quantile regression loss.

    Args:
        td_errors: (B, T', T) target quantile minus online quantile.
        taus: (B, T) the online net's quantile fractions.
        kappa: Huber transition point.

    Returns:
        (B,) loss per transition -- not reduced, because prioritised replay needs
        to weight each one by its importance-sampling correction.

    The sum is over the *online* quantile dimension and the mean over the target
    one. Getting that the wrong way round still trains, just to a badly scaled
    objective, and nothing about the resulting curves looks obviously wrong --
    which is why it is worth spelling out.
    """
    absolute = td_errors.abs()
    huber = torch.where(
        absolute <= kappa, 0.5 * td_errors.pow(2), kappa * (absolute - 0.5 * kappa)
    )
    # |tau - 1(u < 0)|: the asymmetry that pins each output to its own quantile.
    weight = (taus.unsqueeze(1) - (td_errors.detach() < 0).float()).abs()
    return (weight * huber / kappa).sum(dim=2).mean(dim=1)


class IQNAgent:
    """Online and target networks, the optimiser, and the two operations on them.

    Deliberately free of any environment, replay buffer or process concern, so
    it is the same object in the learner and in every collector -- the collectors
    simply never call :meth:`learn`.
    """

    def __init__(
        self, config: IQNConfig | None = None, device: str = "cpu"
    ) -> None:
        self.config = config or IQNConfig()
        self.device = torch.device(device)
        self.online = IQNNetwork(self.config).to(self.device)
        self.target = IQNNetwork(self.config).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(
            self.online.parameters(), lr=self.config.lr, eps=self.config.adam_eps
        )
        self.steps = 0

    # -- acting --------------------------------------------------------------

    @torch.no_grad()
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if epsilon > 0.0 and np.random.random() < epsilon:
            return int(np.random.randint(self.config.n_actions))
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return int(self.online.q_values(obs, self.config.n_tau_action).argmax())

    @torch.no_grad()
    def action_values(self, observation: np.ndarray) -> np.ndarray:
        """Q per action. For diagnostics and for seeding a search prior."""
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return self.online.q_values(obs, self.config.n_tau_action)[0].cpu().numpy()

    # -- learning ------------------------------------------------------------

    def learn(self, batch, weights: np.ndarray) -> tuple[np.ndarray, float]:
        """One gradient step. Returns per-sample TD errors and the loss.

        ``batch`` is ``(obs, actions, rewards, next_obs, dones, discounts)``.
        Discounts come per-transition rather than as a constant because n-step
        tails flushed at an episode boundary are shorter than n steps.
        """
        cfg = self.config
        obs, actions, rewards, next_obs, dones, discounts = (
            torch.as_tensor(x, device=self.device) for x in batch
        )
        obs = obs.float()
        next_obs = next_obs.float()
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        batch_size = obs.shape[0]

        with torch.no_grad():
            # Double Q: the online net picks, the target net values.
            best = self.online.q_values(next_obs, cfg.n_tau_action).argmax(dim=1)
            taus_prime = torch.rand(batch_size, cfg.n_tau_prime, device=self.device)
            next_quantiles = self.target(next_obs, taus_prime)
            next_quantiles = next_quantiles.gather(
                2, best.view(-1, 1, 1).expand(-1, cfg.n_tau_prime, 1)
            ).squeeze(2)  # (B, T')
            target = (
                rewards.float().unsqueeze(1)
                + discounts.float().unsqueeze(1)
                * (1.0 - dones.float().unsqueeze(1))
                * next_quantiles
            )

        taus = torch.rand(batch_size, cfg.n_tau, device=self.device)
        quantiles = self.quantiles_for(obs, taus, actions)  # (B, T)

        td = target.unsqueeze(2) - quantiles.unsqueeze(1)  # (B, T', T)
        per_sample = quantile_huber_loss(td, taus, cfg.kappa)
        loss = (per_sample * weights_t).mean()

        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimiser.step()

        self.steps += 1
        if self.steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        # Priority is the mean absolute TD error over the quantile grid.
        return td.detach().abs().mean(dim=(1, 2)).cpu().numpy(), loss.detach().item()

    def quantiles_for(
        self, obs: torch.Tensor, taus: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Online quantiles for the actions actually taken. (B, T)"""
        all_quantiles = self.online(obs, taus)  # (B, T, A)
        index = actions.view(-1, 1, 1).expand(-1, taus.shape[1], 1)
        return all_quantiles.gather(2, index).squeeze(2)

    # -- weight transfer -----------------------------------------------------

    def weights(self) -> dict:
        """Online parameters as numpy, for shipping to collector processes."""
        return {k: v.cpu().numpy() for k, v in self.online.state_dict().items()}

    def load_weights(self, weights: dict) -> None:
        self.online.load_state_dict(
            {k: torch.as_tensor(v) for k, v in weights.items()}
        )

    def save(self, path) -> None:
        torch.save(
            {
                "config": self.config,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "steps": self.steps,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "cpu") -> "IQNAgent":
        blob = torch.load(path, map_location=device, weights_only=False)
        agent = cls(blob["config"], device=device)
        agent.online.load_state_dict(blob["online"])
        agent.target.load_state_dict(blob["target"])
        agent.optimiser.load_state_dict(blob["optimiser"])
        agent.steps = blob["steps"]
        return agent
