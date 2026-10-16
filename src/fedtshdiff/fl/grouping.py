"""Client descriptors, edge assignment and homogeneity-aware aggregation weights.

``balance`` is the mechanism of the original code (and of FedPhD): clients are placed
sequentially, and each edge's running descriptor is pushed towards a *reference*
``u`` (uniform label distribution, or the mean descriptor).  It therefore makes every
edge resemble the population -- it mixes dissimilar clients -- which reduces the
*between-group* term of Lemma 1.  ``similarity`` and ``kmeans`` instead put similar
clients together and target the *within-group* term.  Both are kept so that the
revision can measure which term each mechanism actually reduces.

Selection forms (``select_form``):
  ``softmax`` (paper eq. / PTB-XL code)  gamma = a * mu' - (n_k + n_i) / n_bar + b,  P ~ softmax(gamma / tau)
  ``linear``  (MIT-BIH code)             s = max(0, a * mu' - (n_k + n_i) + b),     P ~ s / sum(s)
Aggregation forms (``agg_form``):
  ``exp``    (paper eq. / PTB-XL code)   w_i = n_i^alpha * exp(beta * (mu_i - 1))
  ``linear`` (MIT-BIH code)              w_i = max(0, n_i + alpha * mu_i + beta)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Config


# ======================================================================================
# Descriptors
# ======================================================================================


def label_descriptor(y: np.ndarray, n_classes: int) -> np.ndarray:
    if n_classes == 0:
        raise ValueError("label descriptors need a labeled dataset")
    if y.size == 0:
        return np.zeros(n_classes)
    return np.bincount(y, minlength=n_classes)[:n_classes].astype(np.float64) / y.size


def signal_stats(X: np.ndarray) -> np.ndarray:
    """[mean, std, min, max, lag-1 autocorrelation, high-frequency energy share] of a client's
    training windows (paper eq. for unlabeled data; the spectral term is the share of power
    in the upper half of the frequency bins)."""
    if X.size == 0:
        return np.zeros(6)
    flat = X.reshape(-1).astype(np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    num = (Xc[:, 1:] * Xc[:, :-1]).sum(axis=1)
    den = (Xc ** 2).sum(axis=1) + 1e-12
    rho = float(np.mean(num / den))
    P = np.abs(np.fft.rfft(Xc, axis=1)) ** 2
    half = P.shape[1] // 2
    e = float(np.mean(P[:, half:].sum(axis=1) / (P.sum(axis=1) + 1e-12)))
    return np.array([flat.mean(), flat.std(), flat.min(), flat.max(), rho, e])


def psd_descriptor(X: np.ndarray, bands: int = 8) -> np.ndarray:
    """Average normalised power spectrum in ``bands`` equal bands (a distribution)."""
    if X.size == 0:
        return np.full(bands, 1.0 / bands)
    P = np.abs(np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)) ** 2
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    edges = np.linspace(0, P.shape[1], bands + 1).astype(int)
    out = np.array([P[:, a:b].sum(axis=1).mean() for a, b in zip(edges[:-1], edges[1:])])
    return out / max(out.sum(), 1e-12)


def compute_descriptors(kind: str, train_X: list[np.ndarray], train_y: list[np.ndarray],
                        n_classes: int) -> np.ndarray:
    """[N, d] descriptor matrix.  ``signal`` features are min-max scaled across clients and
    divided by sqrt(d), so distances lie in [0, 1] like those of distributions."""
    if kind == "label":
        return np.stack([label_descriptor(y, n_classes) for y in train_y])
    if kind == "psd":
        return np.stack([psd_descriptor(X) for X in train_X])
    raw = np.stack([signal_stats(X) for X in train_X])
    lo, hi = raw.min(axis=0), raw.max(axis=0)
    scaled = (raw - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return scaled / np.sqrt(raw.shape[1])


def reference(kind: str, phi: np.ndarray, mode: str = "auto", sizes: np.ndarray | None = None) -> np.ndarray:
    """Homogeneity reference u.  ``auto`` (paper): uniform for label distributions, the mean
    descriptor otherwise.  ``mean``: the size-weighted population mean, i.e. the descriptor of the
    global objective, which is what bounds between-group gradient dissimilarity."""
    if mode == "uniform" or (mode == "auto" and kind == "label"):
        return np.full(phi.shape[1], 1.0 / phi.shape[1])
    if mode == "mean" and sizes is not None:
        return (sizes[:, None] * phi).sum(0) / sizes.sum()
    return phi.mean(axis=0)


def homogeneity(q: np.ndarray, u: np.ndarray) -> float:
    return 2.0 - float(np.linalg.norm(q - u))


# ======================================================================================
# Assignment
# ======================================================================================


@dataclass
class Assignment:
    edges: list[list[int]]  # client indices per edge (global client indices)
    client_weight: dict[int, float]  # normalised within-edge weight
    edge_weight: np.ndarray  # normalised cloud weight per edge
    mu_client: dict[int, float]
    mu_edge: np.ndarray
    edge_descriptor: np.ndarray
    extra: dict = field(default_factory=dict)

    def edge_of(self) -> dict[int, int]:
        return {i: k for k, mem in enumerate(self.edges) for i in mem}

    def effective_weights(self) -> dict[int, float]:
        return {i: float(self.edge_weight[k] * self.client_weight[i]) for k, mem in enumerate(self.edges) for i in mem}


def _sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Inverse-CDF draw exactly like the original loop (first edge whose cumulative >= u)."""
    u = rng.random()
    cum = np.cumsum(probs)
    hit = np.flatnonzero(u <= cum)
    return int(hit[0]) if hit.size else 0


def _softmax(scores: np.ndarray, tau: float) -> np.ndarray:
    s = (scores - scores.max()) / max(tau, 1e-8)
    p = np.exp(s)
    return p / (p.sum() + 1e-12)


def _probs(scores: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.select_form == "softmax":
        return _softmax(scores, cfg.tau_sel)
    s = np.maximum(scores, 0.0)
    tot = s.sum()
    return np.full(len(s), 1.0 / len(s)) if tot <= 0 else s / tot


def kmeans(phi: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    """Weighted Lloyd's algorithm with k-means++ seeding; returns [k, d] centroids."""
    n = phi.shape[0]
    k = min(k, n)
    cents = [phi[rng.choice(n, p=w / w.sum())]]
    for _ in range(1, k):
        d2 = np.min(((phi[:, None, :] - np.array(cents)[None]) ** 2).sum(-1), axis=1) * w
        cents.append(phi[rng.choice(n, p=d2 / d2.sum())] if d2.sum() > 0 else phi[rng.integers(n)])
    C = np.array(cents)
    for _ in range(iters):
        lab = np.argmin(((phi[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([np.average(phi[lab == j], axis=0, weights=w[lab == j]) if np.any(lab == j) else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def assign_edges(cfg: Config, clients: list[int], phi: np.ndarray, sizes: np.ndarray, u: np.ndarray,
                 n_bar: float, rng: np.random.Generator, grouping: str | None = None,
                 num_edges: int | None = None) -> list[list[int]]:
    """Partition ``clients`` (global indices; ``phi``/``sizes`` are indexed globally)."""
    grouping = grouping or cfg.grouping
    K = 1 if grouping == "none" else (num_edges or cfg.num_edges)
    order = list(clients)
    if cfg.assign_order == "shuffled":
        order = [order[j] for j in rng.permutation(len(order))]
    edges: list[list[int]] = [[] for _ in range(K)]
    if grouping == "none" or K == 1:
        edges[0] = list(order)
        return edges
    if grouping == "random":
        for i in order:
            edges[int(rng.integers(K))].append(i)
        return edges
    if grouping == "round_robin":
        for j, i in enumerate(order):
            edges[j % K].append(i)
        return edges
    d = phi.shape[1]
    if grouping in ("kmeans", "similarity"):
        C = kmeans(phi[order], sizes[order].astype(np.float64), K, rng)
        if grouping == "kmeans":
            for i in order:
                edges[int(np.argmin(((C - phi[i]) ** 2).sum(-1)))].append(i)
            return edges
    qe = np.zeros((K, d))
    ne = np.zeros(K)
    for i in order:
        ni = float(sizes[i])
        if grouping == "balance":
            q_new = (ne[:, None] * qe + ni * phi[i][None]) / np.maximum(ne + ni, 1e-12)[:, None]
            mu_new = 2.0 - np.linalg.norm(q_new - u[None], axis=1)
            sim = mu_new
        else:  # similarity: closeness to the edge's cluster centre
            sim = -np.linalg.norm(C[: K] - phi[i][None], axis=1)
        n_term = (ne + ni) / max(n_bar, 1.0) if cfg.select_form == "softmax" else (ne + ni)
        scores = cfg.a_sel * sim - cfg.size_penalty * n_term + cfg.b_sel
        k = _sample(_probs(scores, cfg), rng)
        edges[k].append(i)
        qe[k] = (ne[k] * qe[k] + ni * phi[i]) / (ne[k] + ni) if ne[k] + ni > 0 else qe[k]
        ne[k] += ni
    return edges


def aggregation(cfg: Config, edges: list[list[int]], phi: np.ndarray, sizes: np.ndarray, u: np.ndarray,
                adaptive: bool = True) -> Assignment:
    """Within-edge and cloud weights for a partition.  ``adaptive=False`` gives data-proportional
    weights at both levels (then the effective weights equal FedAvg's)."""
    K = len(edges)
    mu_client = {i: homogeneity(phi[i], u) for mem in edges for i in mem}
    edge_desc = np.zeros((K, phi.shape[1]))
    mu_edge = np.zeros(K)
    client_w: dict[int, float] = {}
    raw_edge = np.zeros(K)
    for k, mem in enumerate(edges):
        if not mem:
            continue
        n = sizes[mem].astype(np.float64)
        edge_desc[k] = (n[:, None] * phi[mem]).sum(0) / n.sum()
        mu_edge[k] = homogeneity(edge_desc[k], u)
        if adaptive:
            raw = np.array([_agg_weight(cfg, float(sizes[i]), mu_client[i]) for i in mem])
            raw_edge[k] = _agg_weight(cfg, float(n.sum()), mu_edge[k]) if cfg.edge_weighting == "homogeneity" else n.sum()
        else:
            raw = n
            raw_edge[k] = n.sum()
        tot = raw.sum()
        raw = raw / tot if tot > 0 else np.full(len(mem), 1.0 / len(mem))
        client_w.update({i: float(w) for i, w in zip(mem, raw)})
    tot = raw_edge.sum()
    edge_w = raw_edge / tot if tot > 0 else np.array([1.0 / sum(1 for m in edges if m) if m else 0.0 for m in edges])
    return Assignment(edges, client_w, edge_w, mu_client, mu_edge, edge_desc)


def _agg_weight(cfg: Config, n: float, mu: float) -> float:
    if cfg.agg_form == "exp":
        return float(max(n, 1.0) ** cfg.alpha_agg * np.exp(cfg.beta_agg * (mu - 1.0)))
    return float(max(0.0, n + cfg.alpha_agg * mu + cfg.beta_agg))


class Grouper:
    """Owns the round-wise (dynamic) or persistent (static) assignment for a run."""

    def __init__(self, cfg: Config, phi: np.ndarray, sizes: np.ndarray, kind: str):
        self.cfg, self.kind = cfg, kind
        self.phi, self.sizes = phi, sizes
        self.n_bar = float(np.mean(sizes[sizes > 0])) if np.any(sizes > 0) else 1.0
        self._static: dict[int, int] | None = None
        self._static_K = 0

    def update_descriptors(self, phi: np.ndarray, sizes: np.ndarray) -> None:
        self.phi, self.sizes = phi, sizes

    def assign(self, r: int, participants: list[int], rng: np.random.Generator) -> Assignment:
        cfg = self.cfg
        u = reference(self.kind, self.phi[participants], cfg.homogeneity_ref, self.sizes[participants])
        dynamic = cfg.regroup_every == 1
        if dynamic:
            edges = assign_edges(cfg, participants, self.phi, self.sizes, u, self.n_bar, rng)
        else:
            regroup = self._static is None or (cfg.regroup_every > 1 and (r - 1) % cfg.regroup_every == 0)
            if regroup:
                everyone = list(range(len(self.sizes)))
                full = assign_edges(cfg, everyone, self.phi, self.sizes,
                                    reference(self.kind, self.phi, cfg.homogeneity_ref, self.sizes), self.n_bar, rng)
                self._static = {i: k for k, mem in enumerate(full) for i in mem}
                self._static_K = len(full)  # fixed edge identities (also when an edge ends up empty)
            K = self._static_K
            edges = [[] for _ in range(K)]
            for i in participants:
                edges[self._static[i]].append(i)
        adaptive = cfg.beta_agg != 0.0 or cfg.alpha_agg != 1.0 or cfg.agg_form != "exp"
        return aggregation(cfg, edges, self.phi, self.sizes, u, adaptive=adaptive)
