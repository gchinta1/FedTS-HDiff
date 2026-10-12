"""Gradient-dispersion diagnostics (Reviewer 2, comments 3-4).

At selected rounds, every client's full gradient of its local objective is computed at
the current global model on a fixed subset of its training windows, with *common random
numbers* (identical diffusion timesteps and noise across clients), so that differences
between client gradients reflect data, not sampling noise.

From the Gram matrix G G^T of the client gradients we evaluate, for many partitions of
the same clients and with data-proportional weights p:

  delta_global = sum_i p_i ||g_i - g||^2
  delta_in     = sum_k q_k sum_{i in C_k} p_{i|k} ||g_i - g_k||^2      (within-group)
  delta_out    = sum_k q_k ||g_k - g||^2                               (between-group)

(Lemma 1: delta_global = delta_in + delta_out for every partition), the aggregation bias
||sum_i a_i g_i - sum_i p_i g_i||^2 of the adaptive weights, and the rank correlation
between pairwise descriptor distances and pairwise gradient distances (with a Mantel
permutation test).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import stats

from ..models.diffusion import diffusion_loss
from ..utils import derive_seed, make_generator
from .grouping import aggregation, assign_edges, compute_descriptors, reference
from .trainer import FederatedTrainer

MAX_DENSE = 1_500_000_000  # floats; above this, gradients are randomly projected
PROJ_DIM = 65_536


def client_gradients(tr: FederatedTrainer, r: int, clients: list[int]) -> torch.Tensor:
    cfg, fm, fore = tr.cfg, tr.fm, tr.fore
    keep = fm.snapshot()
    sl = tr.opt.slices[fore.gradient_segment]
    P = sl.stop - sl.start
    project = len(clients) * P > MAX_DENSE
    if project:
        pg = make_generator(tr.device, derive_seed(cfg.seed, r, "proj"))
    rows = []
    for i in clients:
        fm.load(tr.global_vec)
        X = tr.train_clients[i].train_t
        sel = torch.randperm(X.shape[0], device=tr.device,
                             generator=make_generator(tr.device, derive_seed(cfg.seed, r, i, "diagsel")))
        xb = X[sel[: cfg.diag_windows]]
        gen = make_generator(tr.device, derive_seed(cfg.seed, r, "crn"))  # same draws for every client
        fm.grad.zero_()
        if fore.kind == "diffusion":
            # CRN: draw t/noise for diag_windows samples and use the first b, so a client with fewer
            # windows still gets exactly the same draws for its samples as every other client
            t_all = torch.randint(0, fore.sched.T, (cfg.diag_windows,), device=tr.device, generator=gen)
            n_all = torch.randn((cfg.diag_windows, 1, cfg.future_len), device=tr.device, generator=gen)
            b = xb.shape[0]
            diffusion_loss(fore.module, fore.sched, xb[:, None, : cfg.past_len], xb[:, None, cfg.past_len:], gen,
                           t=t_all[:b], noise=n_all[:b]).backward()
        else:
            fore.loss(xb, gen).backward()
        g = fm.grad[sl].detach().clone()
        rows.append(g)
    fm.load(keep)
    G = torch.stack(rows)
    if project:
        R = torch.randn((P, PROJ_DIM), device=tr.device, generator=pg) / np.sqrt(PROJ_DIM)
        G = G @ R
    return G


def _dispersion(Kmat: np.ndarray, p: np.ndarray, groups: list[np.ndarray]) -> dict[str, float]:
    sq = np.diag(Kmat)
    glob = float(p @ sq - p @ Kmat @ p)
    within = 0.0
    between = 0.0
    gbar_sq = float(p @ Kmat @ p)
    for g in groups:
        if g.size == 0:
            continue
        q = p[g].sum()
        pk = p[g] / q
        Kg = Kmat[np.ix_(g, g)]
        within += q * float(pk @ sq[g] - pk @ Kg @ pk)
        # ||g_k - g||^2 = pk K pk - 2 pk K_{g,:} p + p K p
        between += q * float(pk @ Kg @ pk - 2 * pk @ Kmat[g] @ p + gbar_sq)
    return {"delta_global": glob, "delta_in": within, "delta_out": between}


def _mantel(D1: np.ndarray, D2: np.ndarray, rng: np.random.Generator, perms: int = 999) -> dict[str, float]:
    iu = np.triu_indices_from(D1, 1)
    a, b = D1[iu], D2[iu]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"spearman": float("nan"), "pearson": float("nan"), "mantel_p": float("nan")}
    rho = stats.spearmanr(a, b).statistic
    r = stats.pearsonr(a, b).statistic
    n = D1.shape[0]
    hits = 0
    rb = stats.rankdata(b)
    for _ in range(perms):
        perm = rng.permutation(n)
        ap = D1[np.ix_(perm, perm)][iu]
        hits += stats.spearmanr(ap, rb).statistic >= rho
    return {"spearman": float(rho), "pearson": float(r), "mantel_p": float((hits + 1) / (perms + 1))}


def run_diagnostics(tr: FederatedTrainer, r: int) -> dict[str, Any]:
    cfg = tr.cfg
    clients = list(range(len(tr.train_clients)))
    G = client_gradients(tr, r, clients)
    Kmat = (G.double() @ G.double().T).cpu().numpy()
    sizes = tr.sizes
    p = sizes / sizes.sum()
    out: dict[str, Any] = {"round": r, "clients": len(clients), "grad_norms": np.sqrt(np.diag(Kmat)).tolist()}
    sq = np.diag(Kmat)
    Dg = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Kmat, 0.0))
    rng = np.random.default_rng(derive_seed(cfg.seed, r, "diag"))
    kinds = ["signal", "psd"] + (["label"] if tr.data.n_classes > 0 else [])
    tx = [c.train_X for c in tr.train_clients]
    ty = [c.train_y for c in tr.train_clients]
    partitions: dict[str, Any] = {}
    single = _dispersion(Kmat, p, [np.arange(len(clients))])
    out["single_group"] = single
    rand = [_dispersion(Kmat, p, [np.array(e, dtype=int) for e in
                                  assign_edges(cfg, clients, np.zeros((len(clients), 1)), sizes, np.zeros(1), 1.0,
                                               rng, grouping="random")])
            for _ in range(cfg.diag_random_partitions)]
    partitions["random"] = {k: float(np.mean([d[k] for d in rand])) for k in rand[0]}
    partitions["random_sd"] = {k: float(np.std([d[k] for d in rand], ddof=1)) for k in rand[0]}
    partitions["round_robin"] = _dispersion(Kmat, p, [np.array(e, dtype=int) for e in assign_edges(
        cfg, clients, np.zeros((len(clients), 1)), sizes, np.zeros(1), 1.0, rng, grouping="round_robin")])
    correlations: dict[str, Any] = {}
    n_bar = float(sizes.mean())
    for kind in kinds:
        phi = compute_descriptors(kind, tx, ty, tr.data.n_classes)
        u = reference(kind, phi)
        Dphi = np.sqrt(((phi[:, None, :] - phi[None, :, :]) ** 2).sum(-1))
        correlations[kind] = _mantel(Dphi, Dg, rng, perms=999 if len(clients) <= 500 else 199)
        iu = np.triu_indices_from(Dphi, 1)
        ok = Dphi[iu] > 1e-9
        if ok.any():
            ratio = Dg[iu][ok] / Dphi[iu][ok]
            correlations[kind]["lipschitz_q95"] = float(np.quantile(ratio, 0.95))
        variants = [("balance", u), ("similarity", u), ("kmeans", u)]
        if kind == "label":  # balancing towards the population label mix instead of the uniform one
            variants.append(("balance_popmean", reference(kind, phi, "mean", sizes)))
        for grouping, uu in variants:
            edges = assign_edges(cfg, clients, phi, sizes, uu, n_bar, rng, grouping=grouping.split("_")[0])
            groups = [np.array(e, dtype=int) for e in edges]
            d = _dispersion(Kmat, p, groups)
            asg = aggregation(cfg, edges, phi, sizes, uu, adaptive=True)
            eff = asg.effective_weights()
            a = np.array([eff[i] for i in clients])
            d["agg_bias"] = float((a - p) @ Kmat @ (a - p))
            partitions[f"{grouping}/{kind}"] = d
    covered = tr.last_assignment is not None and sum(len(e) for e in tr.last_assignment.edges) == len(clients)
    if covered:
        groups = [np.array(e, dtype=int) for e in tr.last_assignment.edges]
        d = _dispersion(Kmat, p, groups)
        eff = tr.last_assignment.effective_weights()
        a = np.zeros(len(clients))
        for i, w in eff.items():
            a[i] = w
        d["agg_bias"] = float((a - p) @ Kmat @ (a - p))
        partitions["actual"] = d
    out["partitions"] = partitions
    out["descriptor_gradient_correlation"] = correlations
    return out
