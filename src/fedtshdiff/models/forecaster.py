"""Uniform interface over the forecasting models used by the federated algorithms.

A :class:`Forecaster` owns one ``nn.Module`` (all trainable state is in parameters,
never in buffers, so a model *is* its flat parameter vector), a list of named
parameter *segments* with their optimiser settings, and two operations:

``train_batch(x_full, ctx)``   one minibatch of local training; every optimiser step
                               goes through ``ctx.step(loss, segment)`` so that the
                               federated algorithm can inject its gradient correction.
``sample(past, K, gen, steps)`` K forecast trajectories per past window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn as nn

from ..config import Config
from .baselines import GANPair, GaussianLSTM
from .csdi import CSDIEps
from .diffusion import Schedule, ddim_sample, ddpm_sample, diffusion_loss
from .unet import UNetEps


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    prefixes: tuple[str, ...]  # parameter-name prefixes ("" = everything)
    beta1: float
    beta2: float
    weight_decay: float


class StepContext(Protocol):
    gen: torch.Generator
    batches: int  # minibatches already processed in this client's local training call

    def step(self, loss: torch.Tensor, segment: str) -> None: ...

    def extra_loss(self) -> torch.Tensor | None: ...


class Forecaster:
    kind = "base"
    module: nn.Module
    segments: list[SegmentSpec]

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.past_len, self.future_len = cfg.past_len, cfg.future_len

    def train_batch(self, x_full: torch.Tensor, ctx: StepContext) -> torch.Tensor:
        """Returns the detached minibatch loss on the device (no host synchronisation)."""
        raise NotImplementedError

    def loss(self, x_full: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, past: torch.Tensor, K: int, gen: torch.Generator, steps: int | None = None,
               chunk: int = 16384, x_init: torch.Tensor | None = None) -> torch.Tensor:
        """past: [B, P] -> [B, K, F]; sequences are processed in chunks of ``chunk``.

        ``x_init`` ([B, K, F]) is the starting noise of deterministic samplers (DDIM, eta = 0);
        models without such a sampler ignore it."""
        B = past.size(0)
        rep = past.repeat_interleave(K, dim=0)
        init = x_init.reshape(B * K, self.future_len) if x_init is not None else None
        outs = []
        for s in range(0, rep.size(0), chunk):
            outs.append(self._sample_flat(rep[s:s + chunk], gen, steps,
                                          None if init is None else init[s:s + chunk]))
        return torch.cat(outs, 0).view(B, K, self.future_len)

    @property
    def deterministic_given_init(self) -> bool:
        return False

    def _sample_flat(self, past: torch.Tensor, gen: torch.Generator, steps: int | None,
                     x_init: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def gradient_segment(self) -> str:
        """Segment whose gradient defines the 'client gradient' in dispersion diagnostics."""
        return self.segments[0].name


def _split(x_full: torch.Tensor, P: int) -> tuple[torch.Tensor, torch.Tensor]:
    return x_full[:, None, :P], x_full[:, None, P:]


class DiffusionForecaster(Forecaster):
    kind = "diffusion"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        if cfg.model == "unet":
            self.module = UNetEps(cfg.base_ch, cfg.time_dim).to(device)
        else:
            self.module = CSDIEps(cfg.past_len, cfg.future_len, cfg.csdi_channels, cfg.csdi_layers,
                                  cfg.csdi_heads).to(device)
        self.sched = Schedule(cfg.diffusion_steps, cfg.beta_start, cfg.beta_end, cfg.beta_schedule, device)
        self.segments = [SegmentSpec("all", ("",), cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay)]
        if cfg.model == "unet":
            head = UNetEps.head_prefix
            self.segments += [
                SegmentSpec("body", ("net.time_emb", "net.in_conv", "net.down", "net.mid", "net.up"),
                            cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay),
                SegmentSpec("head", (head,), cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay),
            ]

    def loss(self, x_full: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        past, fut = _split(x_full, self.past_len)
        return diffusion_loss(self.module, self.sched, past, fut, gen)

    def train_batch(self, x_full: torch.Tensor, ctx: StepContext, segment: str = "all") -> torch.Tensor:
        loss = self.loss(x_full, ctx.gen)
        extra = ctx.extra_loss()
        total = loss if extra is None else loss + extra
        ctx.step(total, segment)
        return loss.detach()

    @property
    def deterministic_given_init(self) -> bool:
        return self.cfg.sampler == "ddim" and self.cfg.ddim_eta == 0.0

    def _sample_flat(self, past: torch.Tensor, gen: torch.Generator, steps: int | None,
                     x_init: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        p = past[:, None, :]
        if cfg.sampler == "ddpm":
            out = ddpm_sample(self.module, self.sched, p, self.future_len, gen)
        else:
            out = ddim_sample(self.module, self.sched, p, self.future_len, gen, steps or cfg.ddim_steps,
                              cfg.ddim_eta, cfg.ddim_spacing,
                              None if x_init is None else x_init[:, None, :])
        return out[:, 0, :]


class LSTMForecaster(Forecaster):
    kind = "lstm"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.module = GaussianLSTM(cfg.lstm_hidden, cfg.lstm_layers).to(device)
        self.segments = [SegmentSpec("all", ("",), cfg.adam_beta1, cfg.adam_beta2, cfg.weight_decay)]

    def loss(self, x_full: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        return self.module.nll(x_full, self.past_len)

    def train_batch(self, x_full: torch.Tensor, ctx: StepContext, segment: str = "all") -> torch.Tensor:
        loss = self.loss(x_full, ctx.gen)
        ctx.step(loss, segment)
        return loss.detach()

    def _sample_flat(self, past: torch.Tensor, gen: torch.Generator, steps: int | None,
                     x_init: torch.Tensor | None = None) -> torch.Tensor:
        return self.module.sample(past, self.future_len, gen)


class GANForecaster(Forecaster):
    """Conditional WGAN-GP.  One critic update per minibatch and one generator update every
    ``gan_n_critic`` minibatches; Adam(0.5, 0.9) without weight decay, as in WGAN-GP."""

    kind = "gan"

    def __init__(self, cfg: Config, device: torch.device):
        super().__init__(cfg, device)
        self.module = GANPair(cfg.past_len, cfg.future_len, cfg.gan_noise_dim, cfg.gan_hidden).to(device)
        self.segments = [SegmentSpec("G", ("G.",), 0.5, 0.9, 0.0), SegmentSpec("D", ("D.",), 0.5, 0.9, 0.0)]

    def _z(self, n: int, gen: torch.Generator) -> torch.Tensor:
        return torch.randn((n, self.cfg.gan_noise_dim), device=self.device, generator=gen)

    def loss(self, x_full: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Generator loss -D(G(z)) (used for gradient-dispersion diagnostics)."""
        past = x_full[:, : self.past_len]
        fake = self.module.G(past, self._z(past.size(0), gen))
        return -self.module.D(torch.cat([past, fake], dim=1)).mean()

    def train_batch(self, x_full: torch.Tensor, ctx: StepContext, segment: str = "all") -> torch.Tensor:
        G, D = self.module.G, self.module.D
        past = x_full[:, : self.past_len]
        n = past.size(0)
        with torch.no_grad():
            fake = G(past, self._z(n, ctx.gen))
        real_full = x_full
        fake_full = torch.cat([past, fake], dim=1)
        eps = torch.rand((n, 1), device=self.device, generator=ctx.gen)
        interp = (eps * real_full + (1 - eps) * fake_full).requires_grad_(True)
        grad = torch.autograd.grad(D(interp).sum(), interp, create_graph=True)[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        d_loss = D(fake_full).mean() - D(real_full).mean() + self.cfg.gan_gp * gp
        ctx.step(d_loss, "D")
        ctx.batches += 1  # per local-training call: every client gets G steps at the same minibatches
        if ctx.batches % self.cfg.gan_n_critic == 0:
            g_loss = self.loss(x_full, ctx.gen)
            ctx.step(g_loss, "G")
        return d_loss.detach()

    def _sample_flat(self, past: torch.Tensor, gen: torch.Generator, steps: int | None,
                     x_init: torch.Tensor | None = None) -> torch.Tensor:
        return self.module.G(past, self._z(past.size(0), gen))

    @property
    def gradient_segment(self) -> str:
        return "G"


def build_forecaster(cfg: Config, device: torch.device) -> Forecaster:
    if cfg.model in ("unet", "csdi"):
        return DiffusionForecaster(cfg, device)
    if cfg.model == "lstm":
        return LSTMForecaster(cfg, device)
    if cfg.model == "gan":
        return GANForecaster(cfg, device)
    raise ValueError(cfg.model)


Factory = Callable[[Config, torch.device], Forecaster]
