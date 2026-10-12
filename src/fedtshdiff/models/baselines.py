"""Non-diffusion probabilistic forecasters used to justify the diffusion choice (Reviewer 1, C2/C5).

* :class:`GaussianLSTM` -- a DeepAR-style autoregressive LSTM with a Gaussian likelihood.
* :class:`CondGenerator` / :class:`CondCritic` -- a conditional WGAN-GP forecaster
  (Gulrajani et al., 2017): the generator maps (past, z) to a future; the critic scores
  the concatenated [past | future] sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianLSTM(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def _params(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(h)
        return out[..., 0], F.softplus(out[..., 1]) + 1e-3

    def nll(self, x_full: torch.Tensor, past_len: int) -> torch.Tensor:
        """Teacher-forced Gaussian NLL of the future positions; x_full: [b, L]."""
        inp = x_full[:, :-1].unsqueeze(-1)
        h, _ = self.lstm(inp)
        mu, sigma = self._params(h)
        target = x_full[:, 1:]
        sl = slice(past_len - 1, None)
        # no argument validation: NaNs must reach the trainer's divergence check (recorded as
        # "diverged") instead of raising here and failing the worker
        dist = torch.distributions.Normal(mu[:, sl], sigma[:, sl], validate_args=False)
        return -dist.log_prob(target[:, sl]).mean()

    @torch.no_grad()
    def sample(self, past: torch.Tensor, future_len: int, gen: torch.Generator) -> torch.Tensor:
        """past: [n, P] -> [n, F]."""
        h, state = self.lstm(past.unsqueeze(-1))
        mu, sigma = self._params(h[:, -1])
        outs = []
        for _ in range(future_len):
            y = mu + sigma * torch.randn(mu.shape, device=mu.device, generator=gen)
            y = y.clamp(-1, 1)
            outs.append(y)
            h, state = self.lstm(y.view(-1, 1, 1), state)
            mu, sigma = self._params(h[:, -1])
        return torch.stack(outs, dim=1)


class CondGenerator(nn.Module):
    def __init__(self, past_len: int, future_len: int, noise_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.noise_dim = noise_dim
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
        )
        enc_len = (past_len + 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(64 * enc_len + noise_dim, 4 * hidden), nn.LeakyReLU(0.2),
            nn.Linear(4 * hidden, 4 * hidden), nn.LeakyReLU(0.2),
            nn.Linear(4 * hidden, future_len),
        )

    def forward(self, past: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.enc(past.unsqueeze(1)).flatten(1)
        return torch.tanh(self.mlp(torch.cat([h, z], dim=1)))


class CondCritic(nn.Module):
    def __init__(self, window_len: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
        )
        self.mlp = nn.Sequential(nn.Linear(64 * ((window_len + 1) // 2), 4 * hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(4 * hidden, 1))

    def forward(self, full: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.conv(full.unsqueeze(1)).flatten(1)).squeeze(1)


class GANPair(nn.Module):
    """Generator then critic, so each is one contiguous parameter segment."""

    def __init__(self, past_len: int, future_len: int, noise_dim: int, hidden: int):
        super().__init__()
        self.G = CondGenerator(past_len, future_len, noise_dim, hidden)
        self.D = CondCritic(past_len + future_len, hidden)
