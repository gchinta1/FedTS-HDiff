"""Simulated concept drift (Section 5.8): at each drift round a seeded subset of clients
changes amplitude, variance and periodic behaviour.  The transform is applied to the
client's training *and* test windows from that round on (cumulatively across events);
noise realisations are fixed per (client, event), so every method sees identical drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import Config
from .federated import FederatedData


@dataclass(frozen=True)
class DriftEvent:
    round: int
    clients: tuple[int, ...]


def drift_plan(cfg: Config, n_clients: int) -> list[DriftEvent]:
    events = []
    for e, r in enumerate(sorted(cfg.drift_rounds)):
        rng = np.random.default_rng([cfg.drift_seed, e])
        m = max(1, int(round(cfg.drift_fraction * n_clients)))
        events.append(DriftEvent(int(r), tuple(sorted(rng.choice(n_clients, m, replace=False).tolist()))))
    return events


def _params(L: int, rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    amp = rng.uniform(0.5, 1.5)
    sigma = rng.uniform(0.05, 0.15)
    period = rng.uniform(L / 8, L / 2)
    phase = rng.uniform(0, 2 * np.pi)
    periodic = 0.25 * np.sin(2 * np.pi * np.arange(L) / period + phase)
    return amp, sigma, periodic


def _transform(X: np.ndarray, params: tuple[float, float, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    amp, sigma, periodic = params
    out = amp * X + periodic[None, :] + sigma * rng.standard_normal(X.shape)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def apply_event(data: FederatedData, event: DriftEvent, cfg: Config, index: int, device: torch.device) -> None:
    for i in event.clients:
        c = data.clients[i]
        rng = np.random.default_rng([cfg.drift_seed, index, i])
        params = _params(c.train_X.shape[1], rng)  # one shift per (client, event), shared by train and test
        c.train_X = _transform(c.train_X, params, rng)
        c.test_X = _transform(c.test_X, params, rng)
        c.train_t = torch.from_numpy(np.ascontiguousarray(c.train_X)).to(device)
