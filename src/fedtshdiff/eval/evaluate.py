"""Evaluation driver: sample K forecasts per evaluation window with the right model(s)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..data.federated import EvalSet
from ..fl.trainer import FederatedTrainer
from ..utils import derive_seed, make_generator
from .metrics import summarise, window_metrics


@torch.no_grad()
def evaluate(trainer: FederatedTrainer, es: EvalSet, K: int, steps: int | None, seed: int,
             metric_chunk: int = 256) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Returns (summary, per-window arrays).  The sampling noise is a fixed function of
    ``seed`` so every method/round sees the same evaluation randomness."""
    fore, fm, cfg = trainer.fore, trainer.fm, trainer.cfg
    keep = fm.snapshot()
    gen = make_generator(trainer.device, seed)
    fore.module.eval()
    M = es.X.shape[0]
    samples = torch.empty((M, K, cfg.future_len), device=trainer.device)
    t0 = time.perf_counter()
    for vec, pos in trainer.model_groups(es.client_index):
        fm.load(vec)
        pos_t = torch.as_tensor(pos, device=trainer.device)
        past = es.X[pos_t, : cfg.past_len]
        samples[pos_t] = fore.sample(past, K, gen, steps, chunk=cfg.eval_batch)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    sample_seconds = time.perf_counter() - t0
    fm.load(keep)
    fore.module.train()
    truth = es.X[:, cfg.past_len:]
    per: dict[str, list[np.ndarray]] = {}
    mgen = make_generator(trainer.device, derive_seed(seed, "pit"))
    for s in range(0, M, metric_chunk):
        wm = window_metrics(truth[s:s + metric_chunk], samples[s:s + metric_chunk], mgen)
        for k, v in wm.items():
            per.setdefault(k, []).append(v.cpu().numpy())
    arrays = {k: np.concatenate(v, axis=-1 if k == "coverage" else 0) for k, v in per.items()}
    summary = summarise(arrays)
    summary["sample_seconds"] = sample_seconds
    summary["sample_seconds_per_window"] = sample_seconds / max(M, 1)
    summary["windows"] = M
    summary["samples_per_window"] = K
    summary["ddim_steps"] = steps or cfg.ddim_steps
    return summary, arrays


@torch.no_grad()
def diffusion_test_loss(trainer: FederatedTrainer, es: EvalSet, seed: int) -> float:
    """Average training objective on evaluation windows (fixed t/noise), using the global model
    (or each client's own model for personalised methods)."""
    fore, fm = trainer.fore, trainer.fm
    keep = fm.snapshot()
    gen = make_generator(trainer.device, seed)
    total, n = 0.0, 0
    for vec, pos in trainer.model_groups(es.client_index):
        fm.load(vec)
        X = es.X[torch.as_tensor(pos, device=trainer.device)]
        for s in range(0, X.shape[0], 512):
            xb = X[s:s + 512]
            total += float(fore.loss(xb, gen)) * xb.shape[0]
            n += xb.shape[0]
    fm.load(keep)
    return total / max(n, 1)
