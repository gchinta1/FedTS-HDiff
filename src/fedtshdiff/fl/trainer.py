"""Federated training of one run: every method shares the same local-training loop,
optimiser protocol, data, model and random streams; methods differ only in the
gradient correction they inject, in how clients are grouped, and in aggregation.

Methods
-------
fedavg      data-proportional averaging (McMahan et al., 2017)
fedprox     + mu/2 ||theta - theta_start||^2 (Li et al., 2020)
fednova     normalised averaging with tau_eff (Wang et al., 2020)
feddyn      dynamic regulariser, server state h (Acar et al., 2021)
scaffold    control variates, option II (Karimireddy et al., 2020)
fedrep      shared body, personal head; head epochs then body epochs (Collins et al., 2021)
fedtdd      adapted FedTDD: model averaging plus an exchanged pool of fully synthetic windows
            (two-step generation f1 = G(past), f2 = G(f1), window = [f1 | f2]) mixed into
            local training with weight lambda (Soi et al., 2024; see docs/METHODS.md)
fedts_hdiff hierarchical: grouping -> kappa edge rounds -> cloud aggregation (ours)
cfl         clustered FL: static k-means of descriptors, one model per cluster
central     one pseudo-client holding the union of all training data
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from ..config import Config
from ..data.federated import Client, FederatedData
from ..models.forecaster import Forecaster
from ..utils import Stopwatch, derive_seed, make_generator
from .batched import BatchedUNet, ddim_batched, supports_batched, train_batched
from .flat import FlatAdamW, FlatModel, OptState
from .grouping import Assignment, Grouper, aggregation, compute_descriptors, kmeans, reference


class Diverged(RuntimeError):
    pass


@dataclass
class ClientState:
    opt: OptState | None = None
    h: torch.Tensor | None = None  # feddyn
    c: torch.Tensor | None = None  # scaffold
    head: torch.Tensor | None = None  # fedrep personal head


@dataclass
class LocalResult:
    vec: torch.Tensor
    steps: int
    loss: float
    seconds: float


@dataclass
class RoundLog:
    round: int
    train_loss: float
    participants: int
    stats: dict = field(default_factory=dict)


class _Ctx:
    """StepContext handed to Forecaster.train_batch."""

    def __init__(self, trainer: "FederatedTrainer", state: OptState, gen: torch.Generator,
                 correction: Callable[[slice, torch.Tensor], torch.Tensor] | None,
                 extra: Callable[[], torch.Tensor | None] | None):
        self.t, self.state, self.gen, self.correction, self._extra = trainer, state, gen, correction, extra
        self.batches = 0

    def extra_loss(self) -> torch.Tensor | None:
        return self._extra() if self._extra else None

    def step(self, loss: torch.Tensor, segment: str) -> None:
        fm, opt = self.t.fm, self.t.opt
        fm.grad.zero_()
        loss.backward()
        sl = opt.slices[segment]
        g = fm.grad[sl]
        if self.correction is not None:
            g = self.correction(sl, g)
        clip = self.t.cfg.grad_clip
        if clip > 0:
            norm = g.norm()
            if norm > clip:
                g = g * (clip / (norm + 1e-12))
        opt.update(self.state, segment, g)


class FederatedTrainer:
    def __init__(self, cfg: Config, data: FederatedData, fore: Forecaster, device: torch.device):
        self.cfg, self.data, self.fore, self.device = cfg, data, fore, device
        self.fm = FlatModel(fore.module, device)
        self.opt = FlatAdamW(self.fm, fore.segments, cfg.lr, cfg.adam_eps)
        self.gen = make_generator(device, derive_seed(cfg.seed, "train"))
        self.rng = np.random.default_rng(derive_seed(cfg.seed, "server"))
        self.global_vec = self.fm.snapshot()
        self.watch = Stopwatch(device)
        self.method = cfg.method
        self.current_round = 0
        self.train_clients: list[Client] = self._training_clients()
        self.states: dict[int, ClientState] = {}
        self.sizes = np.array([c.n_train for c in self.train_clients], dtype=np.float64)
        self.param_bytes = self.fm.numel * 4
        self.descriptor_kind = cfg.descriptor
        self.phi: np.ndarray | None = None
        self.grouper: Grouper | None = None
        self.last_assignment: Assignment | None = None
        if self.method in ("fedts_hdiff", "cfl"):
            self.refresh_descriptors()
        # method state
        self.server_h = torch.zeros_like(self.global_vec) if self.method == "feddyn" else None
        self.server_c = torch.zeros_like(self.global_vec) if self.method == "scaffold" else None
        self.syn_pool: dict[int, torch.Tensor] = {}
        if self.method == "fedrep":
            self.head_slice = self.opt.slices["head"]
            self.body_slice = self.opt.slices["body"]
        if self.method == "cfl":
            C = kmeans(self.phi, self.sizes, cfg.num_edges, np.random.default_rng(derive_seed(cfg.seed, "cfl")))
            self.cluster_of = np.argmin(((self.phi[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            self.cluster_vecs = [self.global_vec.clone() for _ in range(C.shape[0])]
        self._batched_net = BatchedUNet.from_trainer(self) if supports_batched(self) else None

    def train_many(self, ids: list[int], starts: list[torch.Tensor], sub: int = 0) -> list[LocalResult]:
        """Local training of several clients (client-batched when supported, else one by one)."""
        if self._batched_net is not None and len(ids) > 1:
            return train_batched(self, ids, starts, sub)
        return [self.local_train(i, st, sub=sub) for i, st in zip(ids, starts)]

    # ----------------------------------------------------------------------------------
    def _training_clients(self) -> list[Client]:
        if self.method != "central":
            return self.data.clients
        X = np.concatenate([c.train_X for c in self.data.clients])
        y = np.concatenate([c.train_y for c in self.data.clients])
        merged = Client(0, "central", X, y, X[:0], X[:0], y[:0])
        merged.train_t = torch.from_numpy(np.ascontiguousarray(X)).to(self.device)
        return [merged]

    def refresh_descriptors(self) -> None:
        """(Re)compute descriptors from the clients' current training data (after drift)."""
        self.phi = compute_descriptors(self.descriptor_kind, [c.train_X for c in self.train_clients],
                                       [c.train_y for c in self.train_clients], self.data.n_classes)
        if self.grouper is None:
            self.grouper = Grouper(self.cfg, self.phi, self.sizes, self.descriptor_kind)
        else:
            self.grouper.update_descriptors(self.phi, self.sizes)

    def state(self, i: int) -> ClientState:
        st = self.states.get(i)
        if st is None:
            st = self.states[i] = ClientState()
        return st

    # ----------------------------------------------------------------------------------
    # Local training
    # ----------------------------------------------------------------------------------
    def local_train(self, i: int, start: torch.Tensor, epochs: int | None = None, sub: int = 0) -> LocalResult:
        """Train client ``i`` from ``start``.  Its randomness (minibatch order, diffusion t/noise)
        comes from a stream keyed by (seed, round, client, sub-round), so results do not depend on
        the order in which clients are trained, and all methods share the same client streams."""
        cfg, fm = self.cfg, self.fm
        gen = make_generator(self.device, derive_seed(cfg.seed, "local", self.current_round, i, sub))
        client = self.train_clients[i]
        st = self.state(i)
        if cfg.client_opt_state == "persistent":
            if st.opt is None:
                st.opt = OptState.zeros(fm.numel, self.device)
            opt_state = st.opt
        else:
            opt_state = OptState.zeros(fm.numel, self.device)
        fm.load(start)
        correction = self._correction(i, start)
        extra = self._fedtdd_extra(i, gen) if self.method == "fedtdd" and self.syn_pool else None
        ctx = _Ctx(self, opt_state, gen, correction, extra)
        X = client.train_t
        n = X.shape[0]
        E = epochs or cfg.local_epochs
        steps = 0
        loss_sum = torch.zeros((), device=self.device)
        t0 = time.perf_counter()
        self.fore.module.train()
        if self.method == "fedrep":
            fm.flat[self.head_slice].copy_(st.head if st.head is not None else start[self.head_slice])
        for e in range(E):
            segment = "all"
            if self.method == "fedrep":
                segment = "head" if e < cfg.fedrep_head_epochs else "body"
            perm = torch.randperm(n, device=self.device, generator=gen)
            for b in range(0, n, cfg.batch_size):
                loss_sum += self.fore.train_batch(X[perm[b:b + cfg.batch_size]], ctx, segment=segment)
                steps += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        mean_loss = float(loss_sum) / max(steps, 1)
        if not math.isfinite(mean_loss) or not torch.isfinite(fm.flat).all():
            raise Diverged(f"client {client.cid}: non-finite loss/parameters")
        vec = fm.snapshot()
        if self.method == "fedrep":
            st.head = vec[self.head_slice].clone()
        self._post_local(i, start, vec, steps)
        return LocalResult(vec, steps, mean_loss, seconds)

    def _correction(self, i: int, start: torch.Tensor):
        cfg, m = self.cfg, self.method
        if m == "fedprox":
            mu = cfg.prox_mu
            return lambda sl, g: g + mu * (self.fm.flat[sl] - start[sl])
        if m == "feddyn":
            st = self.state(i)
            if st.h is None:
                st.h = torch.zeros_like(start)
            a = cfg.dyn_alpha
            return lambda sl, g: g - st.h[sl] + a * (self.fm.flat[sl] - start[sl])
        if m == "scaffold":
            st = self.state(i)
            if st.c is None:
                st.c = torch.zeros_like(start)
            return lambda sl, g: g - st.c[sl] + self.server_c[sl]
        return None

    def _post_local(self, i: int, start: torch.Tensor, vec: torch.Tensor, steps: int) -> None:
        if self.method == "feddyn":
            self.state(i).h.sub_(self.cfg.dyn_alpha * (vec - start))
        elif self.method == "scaffold":
            st = self.state(i)
            # option II: c_i+ = c_i - c + (x - y_i) / (K lr)  =>  delta_c_i = -c + (x - y_i) / (K lr)
            st._c_delta = (start - vec) / (max(steps, 1) * self.cfg.lr) - self.server_c  # type: ignore[attr-defined]
            st.c.add_(st._c_delta)  # type: ignore[attr-defined]

    def _fedtdd_extra(self, i: int, gen: torch.Generator):
        others = [v for j, v in self.syn_pool.items() if j != i]
        if not others:
            return None
        pool = torch.cat(others, 0)
        lam, B = self.cfg.fedtdd_lambda, self.cfg.batch_size

        def extra() -> torch.Tensor:
            idx = torch.randint(0, pool.shape[0], (min(B, pool.shape[0]),), device=self.device, generator=gen)
            return lam * self.fore.loss(pool[idx], gen)

        return extra

    @torch.no_grad()
    def _fedtdd_generate_many(self, ids: list[int], vecs: list[torch.Tensor]) -> dict[int, torch.Tensor]:
        """Synthetic windows of several clients.  Client-batched when possible: the random draws are
        taken from ``self.gen`` in exactly the sequential order (per client: indices, f1 noise, f2
        noise), and DDIM with eta = 0 is deterministic given them, so both paths agree."""
        cfg, dev = self.cfg, self.device
        if self._batched_net is None or len(ids) < 2 or not self.fore.deterministic_given_init:
            out = {}
            for i, v in zip(ids, vecs):
                self.fm.load(v)
                out[i] = self._fedtdd_generate(i)
            return out
        S, P, Fl = cfg.fedtdd_syn_per_client, cfg.past_len, cfg.future_len
        draws = []
        for i in ids:
            X = self.train_clients[i].train_t
            idx = torch.randint(0, X.shape[0], (S,), device=dev, generator=self.gen)
            n1 = torch.randn((S, 1, Fl), device=dev, generator=self.gen)
            n2 = torch.randn((S, 1, Fl), device=dev, generator=self.gen)
            draws.append((X[idx, :P], n1, n2))
        out = {}
        for c0 in range(0, len(ids), cfg.client_batch):
            sl = slice(c0, c0 + cfg.client_batch)
            W = torch.stack(vecs[sl])
            past = torch.stack([d[0] for d in draws[sl]])[:, :, None, :]
            f1 = ddim_batched(self._batched_net, W, self.fore.sched, past,
                              torch.stack([d[1] for d in draws[sl]]), cfg.fedtdd_syn_ddim_steps, cfg.ddim_spacing)
            f2 = ddim_batched(self._batched_net, W, self.fore.sched, f1,
                              torch.stack([d[2] for d in draws[sl]]), cfg.fedtdd_syn_ddim_steps, cfg.ddim_spacing)
            for k, i in enumerate(ids[sl]):
                out[i] = torch.cat([f1[k, :, 0], f2[k, :, 0]], dim=1)
        return out

    @torch.no_grad()
    def _fedtdd_generate(self, i: int) -> torch.Tensor:
        """Fully synthetic windows from client i's freshly trained model (loaded in fm)."""
        cfg = self.cfg
        X = self.train_clients[i].train_t
        idx = torch.randint(0, X.shape[0], (cfg.fedtdd_syn_per_client,), device=self.device, generator=self.gen)
        self.fore.module.eval()
        f1 = self.fore.sample(X[idx, : cfg.past_len], 1, self.gen, cfg.fedtdd_syn_ddim_steps)[:, 0]
        f2 = self.fore.sample(f1, 1, self.gen, cfg.fedtdd_syn_ddim_steps)[:, 0]
        self.fore.module.train()
        return torch.cat([f1, f2], dim=1)

    # ----------------------------------------------------------------------------------
    # One communication round
    # ----------------------------------------------------------------------------------
    def participants(self) -> list[int]:
        N = len(self.train_clients)
        if self.cfg.participation >= 1.0 or self.method == "central":
            return list(range(N))
        m = max(1, int(round(self.cfg.participation * N)))
        # own stream per (seed, round): every method samples the same clients, independent of how
        # much randomness grouping consumed (self.rng is the grouping/server stream)
        rng = np.random.default_rng(derive_seed(self.cfg.seed, "participants", self.current_round))
        return sorted(rng.choice(N, size=m, replace=False).tolist())

    def run_round(self, r: int) -> RoundLog:
        self.current_round = r
        parts = self.participants()
        if self.method == "fedts_hdiff":
            return self._round_hier(r, parts)
        if self.method == "cfl":
            return self._round_cfl(r, parts)
        return self._round_flat(r, parts)

    def _round_flat(self, r: int, parts: list[int]) -> RoundLog:
        cfg, m = self.cfg, self.method
        sizes = self.sizes[parts]
        p = sizes / sizes.sum()
        start = self.global_vec
        acc = torch.zeros_like(start)
        sq, losses, secs, steps_list = 0.0, [], [], []
        new_pool: dict[int, torch.Tensor] = {}
        nova_acc = torch.zeros_like(start) if m == "fednova" else None
        dyn_sum = torch.zeros_like(start) if m == "feddyn" else None
        c_delta = torch.zeros_like(start) if m == "scaffold" else None
        with self.watch.time("local"):
            results = self.train_many(parts, [start] * len(parts))
            for w, i, res in zip(p, parts, results):
                delta = res.vec - start
                if m == "fednova":
                    nova_acc.add_(delta, alpha=float(w) / max(res.steps, 1))
                if m == "feddyn":
                    dyn_sum.add_(res.vec)
                if m == "scaffold":
                    c_delta.add_(self.state(i)._c_delta)  # type: ignore[attr-defined]
                acc.add_(res.vec, alpha=float(w))
                sq += float(w) * float(delta.pow(2).sum())
                losses.append(res.loss)
                secs.append(res.seconds)
                steps_list.append(res.steps)
        if m == "fedtdd":
            with self.watch.time("synthetic"):
                new_pool = self._fedtdd_generate_many(parts, [res.vec for res in results])
        with self.watch.time("aggregate"):
            mean_delta_sq = float((acc - start).pow(2).sum())
            if m == "fednova":
                tau_eff = float(np.dot(p, steps_list))
                new = start + tau_eff * nova_acc
            elif m == "feddyn":
                N = len(self.train_clients)
                self.server_h.sub_(cfg.dyn_alpha / N * (dyn_sum - len(parts) * start))
                new = dyn_sum / len(parts) - self.server_h / cfg.dyn_alpha
            else:
                new = acc
            if m == "scaffold":
                self.server_c.add_(c_delta / len(self.train_clients))
            if m == "fedrep":  # only the body is shared; keep the global head unchanged
                new[self.head_slice] = start[self.head_slice]
            self.global_vec = new
            if m == "fedtdd":
                self.syn_pool = new_pool
        P = self.param_bytes
        up = len(parts) * (P if m != "fedrep" else (self.body_slice.stop - self.body_slice.start) * 4)
        extra_up = 0
        if m == "scaffold":
            extra_up = len(parts) * P
        if m == "fedtdd":
            syn_bytes = cfg.fedtdd_syn_per_client * cfg.window_len * 4
            extra_up = len(parts) * syn_bytes
        down = up + (len(parts) * P if m == "scaffold" else 0)
        if m == "fedtdd":
            down += len(parts) * max(len(new_pool) - 1, 0) * cfg.fedtdd_syn_per_client * cfg.window_len * 4
        stats = {
            "update_disp_total": sq - mean_delta_sq,
            "bytes_client_up": up + extra_up, "bytes_client_down": down,
            "bytes_cloud": up + extra_up + down,
            "local_seconds_sum": float(np.sum(secs)), "local_seconds_max": float(np.max(secs)),
            "local_steps_mean": float(np.mean(steps_list)),
        }
        return RoundLog(r, float(np.average(losses, weights=p)), len(parts), stats)

    def _round_hier(self, r: int, parts: list[int]) -> RoundLog:
        cfg = self.cfg
        with self.watch.time("group"):
            asg = self.grouper.assign(r, parts, self.rng)
        self.last_assignment = asg
        K = len(asg.edges)
        start = self.global_vec
        edge_vecs = [start.clone() for _ in range(K)]
        losses, weights, secs_parallel, secs_sum = [], [], 0.0, 0.0
        within, between = 0.0, 0.0
        edge_of = asg.edge_of()
        order = [i for i in parts if i in edge_of]
        for s in range(cfg.edge_rounds):
            round_max = 0.0
            accs = [torch.zeros_like(start) if mem else None for mem in asg.edges]
            sqs = [0.0] * K
            with self.watch.time("local"):
                # index order, independent of the partition (same streams as flat methods)
                results = self.train_many(order, [edge_vecs[edge_of[i]] for i in order], sub=s)
                for i, res in zip(order, results):
                    k = edge_of[i]
                    w = asg.client_weight[i]
                    accs[k].add_(res.vec, alpha=w)
                    sqs[k] += w * float((res.vec - edge_vecs[k]).pow(2).sum())
                    losses.append(res.loss)
                    weights.append(asg.edge_weight[k] * w)
                    secs_sum += res.seconds
                    round_max = max(round_max, res.seconds)
            with self.watch.time("aggregate"):
                edge_means: list[torch.Tensor | None] = [None] * K
                for k in range(K):
                    if accs[k] is None:
                        continue
                    if s == cfg.edge_rounds - 1:
                        within += asg.edge_weight[k] * (sqs[k] - float((accs[k] - edge_vecs[k]).pow(2).sum()))
                        edge_means[k] = accs[k] - start
                    edge_vecs[k] = accs[k]
            secs_parallel += round_max
        with self.watch.time("aggregate"):
            new = torch.zeros_like(start)
            for k in range(K):
                if asg.edges[k]:
                    new.add_(edge_vecs[k], alpha=float(asg.edge_weight[k]))
            glob_delta = new - start
            for k in range(K):
                if edge_means[k] is not None:
                    between += asg.edge_weight[k] * float((edge_means[k] - glob_delta).pow(2).sum())
            self.global_vec = new
        P, m, d = self.param_bytes, len(parts), self.phi.shape[1]
        eff = asg.effective_weights()
        p = self.sizes[parts] / self.sizes[parts].sum()
        a = np.array([eff[i] for i in parts])
        stats = {
            "edge_sizes": [len(e) for e in asg.edges],
            "edge_mu": asg.mu_edge.tolist(),
            "edge_weights": asg.edge_weight.tolist(),
            "update_disp_within": within, "update_disp_between": between,
            "eff_weight_l1_from_p": float(np.abs(a - p).sum()),
            "bytes_client_up": cfg.edge_rounds * m * P + m * d * 4, "bytes_client_down": cfg.edge_rounds * m * P,
            "bytes_cloud": 2 * sum(1 for e in asg.edges if e) * P,
            "local_seconds_sum": secs_sum, "local_seconds_max": secs_parallel,
        }
        return RoundLog(r, float(np.average(losses, weights=weights)), m, stats)

    def _round_cfl(self, r: int, parts: list[int]) -> RoundLog:
        losses, weights, secs = [], [], []
        within = 0.0
        with self.watch.time("local"):  # all clusters' clients trained together (each from its cluster model)
            trained = dict(zip(parts, self.train_many(parts, [self.cluster_vecs[self.cluster_of[i]] for i in parts])))
        for k in range(len(self.cluster_vecs)):
            mem = [i for i in parts if self.cluster_of[i] == k]
            if not mem:
                continue
            start = self.cluster_vecs[k]
            w = self.sizes[mem] / self.sizes[mem].sum()
            acc = torch.zeros_like(start)
            sq = 0.0
            with self.watch.time("aggregate"):
                for wi, i in zip(w, mem):
                    res = trained[i]
                    acc.add_(res.vec, alpha=float(wi))
                    sq += float(wi) * float((res.vec - start).pow(2).sum())
                    losses.append(res.loss)
                    weights.append(self.sizes[i])
                    secs.append(res.seconds)
            share = self.sizes[mem].sum() / self.sizes[parts].sum()
            within += share * (sq - float((acc - start).pow(2).sum()))
            self.cluster_vecs[k] = acc
        P = self.param_bytes
        stats = {"update_disp_within": within, "bytes_client_up": len(parts) * P, "bytes_client_down": len(parts) * P,
                 "bytes_cloud": 2 * len(parts) * P, "local_seconds_sum": float(np.sum(secs)),
                 "local_seconds_max": float(np.max(secs)),
                 "cluster_sizes": np.bincount(self.cluster_of, minlength=len(self.cluster_vecs)).tolist()}
        return RoundLog(r, float(np.average(losses, weights=weights)), len(parts), stats)

    # ----------------------------------------------------------------------------------
    # Models used for evaluation
    # ----------------------------------------------------------------------------------
    def model_groups(self, client_indices: np.ndarray) -> list[tuple[torch.Tensor, np.ndarray]]:
        """(parameter vector, positions in ``client_indices`` it must serve) pairs."""
        if self.method == "cfl":
            out = []
            for k, vec in enumerate(self.cluster_vecs):
                pos = np.flatnonzero(self.cluster_of[client_indices] == k)
                if pos.size:
                    out.append((vec, pos))
            return out
        if self.method == "fedrep":
            out = []
            for ci in np.unique(client_indices):
                vec = self.global_vec.clone()
                head = self.state(int(ci)).head
                if head is not None:
                    vec[self.head_slice] = head
                out.append((vec, np.flatnonzero(client_indices == ci)))
            return out
        return [(self.global_vec, np.arange(len(client_indices)))]

    def checkpoint(self) -> dict:
        out = {"global": self.global_vec.detach().cpu()}
        if self.method == "cfl":
            out["clusters"] = [v.cpu() for v in self.cluster_vecs]
            out["cluster_of"] = self.cluster_of
        if self.method == "fedrep":
            out["heads"] = {i: st.head.cpu() for i, st in self.states.items() if st.head is not None}
        return out
