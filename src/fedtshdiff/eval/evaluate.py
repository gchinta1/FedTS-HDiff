"""Evaluation driver: sample K forecasts per evaluation window with the right model(s)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from typing import TYPE_CHECKING

from ..data.federated import EvalSet

if TYPE_CHECKING:  # fl.runner imports this module; a runtime import would be circular
    from ..fl.trainer import FederatedTrainer
from ..utils import derive_seed, make_generator
from .metrics import summarise, window_metrics


@torch.no_grad()
def evaluate(trainer: FederatedTrainer, es: EvalSet, K: int, steps: int | None, seed: int,
             metric_chunk: int = 256) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Returns (summary, per-window arrays).

    The sampling noise is a fixed function of ``seed``: for deterministic samplers (DDIM, eta = 0)
    the starting noise of every (window, sample) is drawn once for the whole set, so each window
    gets the same noise under every method, whatever the model grouping (FedRep heads, CFL
    clusters) or ``eval_batch`` chunking -- common random numbers for the paired tests."""
    fore, fm, cfg = trainer.fore, trainer.fm, trainer.cfg
    keep = fm.snapshot()
    gen = make_generator(trainer.device, seed)
    fore.module.eval()
    M = es.X.shape[0]
    samples = torch.empty((M, K, cfg.future_len), device=trainer.device)
    x_init = (torch.randn((M, K, cfg.future_len), device=trainer.device, generator=gen)
              if fore.deterministic_given_init else None)
    t0 = time.perf_counter()
    groups = trainer.model_groups(es.client_index)
    many_small = len(groups) > 1 and 2 * max(len(pos) for _, pos in groups) * K <= cfg.eval_batch
    if x_init is not None and trainer._batched_net is not None and many_small:
        _sample_groups_batched(trainer, es, groups, K, steps, x_init, samples)
    else:
        for vec, pos in groups:
            fm.load(vec)
            pos_t = torch.as_tensor(pos, device=trainer.device)
            past = es.X[pos_t, : cfg.past_len]
            samples[pos_t] = fore.sample(past, K, gen, steps, chunk=cfg.eval_batch,
                                         x_init=None if x_init is None else x_init[pos_t])
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
def _sample_groups_batched(trainer: FederatedTrainer, es: EvalSet, groups, K: int, steps: int | None,
                           x_init: torch.Tensor, samples: torch.Tensor) -> None:
    """Deterministic DDIM for many parameter groups at once (e.g. FedRep's per-client heads).
    Each group's windows are padded to a common length; padded slots never interact with real
    ones (GroupNorm is per sample).  Identical to sampling group by group with the same x_init."""
    from ..fl.batched import ddim_batched

    cfg, dev = trainer.cfg, trainer.device
    P, Fl = cfg.past_len, cfg.future_len
    n_max = max(len(pos) for _, pos in groups) * K
    per_pass = max(1, cfg.eval_batch // n_max)
    for g0 in range(0, len(groups), per_pass):
        chunk = groups[g0:g0 + per_pass]
        W = torch.stack([vec for vec, _ in chunk])
        past = torch.zeros((len(chunk), n_max, 1, P), device=dev)
        x0 = torch.zeros((len(chunk), n_max, 1, Fl), device=dev)
        for g, (_, pos) in enumerate(chunk):
            pos_t = torch.as_tensor(pos, device=dev)
            n = len(pos) * K
            past[g, :n, 0] = es.X[pos_t, :P].repeat_interleave(K, dim=0)
            x0[g, :n, 0] = x_init[pos_t].reshape(n, Fl)
        out = ddim_batched(trainer._batched_net, W, trainer.fore.sched, past, x0, steps or cfg.ddim_steps,
                           cfg.ddim_spacing)
        for g, (_, pos) in enumerate(chunk):
            pos_t = torch.as_tensor(pos, device=dev)
            samples[pos_t] = out[g, : len(pos) * K, 0].view(len(pos), K, Fl)


@torch.no_grad()
def diffusion_test_loss(trainer: FederatedTrainer, es: EvalSet, seed: int) -> float:
    """Average training objective on evaluation windows (fixed t/noise), using the global model
    (or each client's own model for personalised methods)."""
    fore, fm = trainer.fore, trainer.fm
    keep = fm.snapshot()
    gen = make_generator(trainer.device, seed)
    fore.module.eval()  # no dropout (CSDI's transformer) in the evaluation loss
    total, n = 0.0, 0
    for vec, pos in trainer.model_groups(es.client_index):
        fm.load(vec)
        X = es.X[torch.as_tensor(pos, device=trainer.device)]
        for s in range(0, X.shape[0], 512):
            xb = X[s:s + 512]
            total += float(fore.loss(xb, gen)) * xb.shape[0]
            n += xb.shape[0]
    fm.load(keep)
    fore.module.train()
    return total / max(n, 1)
