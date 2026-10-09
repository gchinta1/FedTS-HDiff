"""Gaussian diffusion on the future segment, conditioned on the past (DDPM training, DDIM/DDPM sampling)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Schedule:
    def __init__(self, T: int, beta_start: float, beta_end: float, kind: str, device: torch.device):
        if kind == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        elif kind == "quad":  # CSDI
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T, dtype=torch.float64) ** 2
        else:
            raise ValueError(kind)
        self.T = T
        self.betas = betas.float().to(device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, 0)
        self.alphas_cumprod = ac.float().to(device)
        self.sqrt_ac = ac.sqrt().float().to(device)
        self.sqrt_om = (1.0 - ac).sqrt().float().to(device)


def ddim_timesteps(T: int, steps: int, spacing: str) -> list[int]:
    """Descending timestep list for DDIM.

    ``trailing`` always starts at T-1 (the step whose marginal is ~N(0, I)).  ``legacy``
    reproduces the original code, ``range(0, T, T // steps)[:steps]``, which for T=500,
    steps=200 starts at t=398 instead of 499.
    """
    if spacing == "legacy":
        ratio = max(1, T // steps)
        return list(reversed(list(range(0, T, ratio))[:steps]))
    ts = np.round(np.arange(T, 0, -T / steps)).astype(int) - 1
    out: list[int] = []
    for t in ts.tolist():
        if not out or t < out[-1]:
            out.append(max(0, t))
    return out


def diffusion_loss(eps_net: nn.Module, sched: Schedule, past: torch.Tensor, future: torch.Tensor,
                   gen: torch.Generator) -> torch.Tensor:
    b = future.size(0)
    t = torch.randint(0, sched.T, (b,), device=future.device, generator=gen)
    noise = torch.randn(future.shape, device=future.device, generator=gen)
    x_t = sched.sqrt_ac[t].view(-1, 1, 1) * future + sched.sqrt_om[t].view(-1, 1, 1) * noise
    return F.mse_loss(eps_net(x_t, past, t), noise)


@torch.no_grad()
def ddim_sample(eps_net: nn.Module, sched: Schedule, past: torch.Tensor, future_len: int,
                gen: torch.Generator, steps: int, eta: float = 0.0, spacing: str = "trailing") -> torch.Tensor:
    """past: [n, 1, P] -> futures [n, 1, F] clamped to [-1, 1] (as the original)."""
    n = past.size(0)
    ts = ddim_timesteps(sched.T, steps, spacing)
    x = torch.randn((n, 1, future_len), device=past.device, generator=gen)
    ac = sched.alphas_cumprod
    for i, t_int in enumerate(ts):
        t = torch.full((n,), t_int, device=past.device, dtype=torch.long)
        eps = eps_net(x, past, t)
        a_t = ac[t_int]
        a_prev = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=past.device)
        x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
        if eta > 0:
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            noise = torch.randn(x.shape, device=x.device, generator=gen)
        else:
            sigma = torch.tensor(0.0, device=x.device)
            noise = 0.0
        x = torch.sqrt(a_prev) * x0 + torch.sqrt(torch.clamp(1 - a_prev - sigma ** 2, min=0.0)) * eps + sigma * noise
    return x.clamp(-1, 1)


@torch.no_grad()
def ddpm_sample(eps_net: nn.Module, sched: Schedule, past: torch.Tensor, future_len: int,
                gen: torch.Generator) -> torch.Tensor:
    """Ancestral sampling over all T steps (CSDI's sampler)."""
    n = past.size(0)
    x = torch.randn((n, 1, future_len), device=past.device, generator=gen)
    ac, betas = sched.alphas_cumprod, sched.betas
    for t_int in range(sched.T - 1, -1, -1):
        t = torch.full((n,), t_int, device=past.device, dtype=torch.long)
        eps = eps_net(x, past, t)
        a_t, b_t = ac[t_int], betas[t_int]
        x = (x - b_t / torch.sqrt(1 - a_t) * eps) / torch.sqrt(1 - b_t)
        if t_int > 0:
            sigma = torch.sqrt((1 - ac[t_int - 1]) / (1 - a_t) * b_t)
            x = x + sigma * torch.randn(x.shape, device=x.device, generator=gen)
    return x.clamp(-1, 1)


def sinusoidal(positions: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / max(half - 1, 1))
    ang = positions.float().unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
