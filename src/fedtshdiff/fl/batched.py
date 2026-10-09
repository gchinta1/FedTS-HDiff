"""Client-batched local training for the U-Net diffusion model.

The model is tiny (0.5 M parameters, batch 64 x length 64), so training one client at a
time is limited by kernel-launch latency, not by the GPU.  Here C clients are trained
simultaneously: their parameters are the rows of a [C, P] matrix, and a functional U-Net
evaluates all C models in one pass (convolutions as batched matrix products).

Exactness: every client draws its minibatch permutation, diffusion timesteps and noise
from its own generator in exactly the order of the sequential path
(``FederatedTrainer.local_train``), padded minibatch slots are masked out of the loss
(GroupNorm normalises per sample, so padding never affects real samples), and the
optimiser is the same AdamW with per-client step counters.  Results therefore equal the
sequential path up to floating-point summation order (``tests/test_batched.py``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..utils import derive_seed, make_generator
from .flat import OptState

if TYPE_CHECKING:  # pragma: no cover
    from .trainer import FederatedTrainer, LocalResult


# ======================================================================================
# Functional, client-batched U-Net (same parameters and maths as models.unet.UNet1D)
# ======================================================================================


def _conv(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """x [C,B,Ci,L], w [C,Co,Ci,k], b [C,Co] -> [C,B,Co,L] ('same' padding, stride 1)."""
    C, B, Ci, L = x.shape
    Co, k = w.shape[1], w.shape[3]
    if k == 1:
        y = torch.einsum("cbil,coi->cbol", x, w[..., 0])
    else:
        cols = F.pad(x, (k // 2, k // 2)).unfold(3, k, 1)  # [C,B,Ci,L,k]
        cols = cols.permute(0, 1, 3, 2, 4).reshape(C, B * L, Ci * k)
        y = torch.bmm(cols, w.reshape(C, Co, Ci * k).transpose(1, 2)).view(C, B, L, Co).permute(0, 1, 3, 2)
    return y + b[:, None, :, None]


def _gn(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor, groups: int = 8, eps: float = 1e-5) -> torch.Tensor:
    C, B, Ch, L = x.shape
    xg = x.reshape(C, B, groups, -1)
    mean = xg.mean(-1, keepdim=True)
    var = xg.var(-1, unbiased=False, keepdim=True)
    return ((xg - mean) * torch.rsqrt(var + eps)).view(C, B, Ch, L) * g[:, None, :, None] + b[:, None, :, None]


def _lin(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.baddbmm(b[:, None, :], x, w.transpose(1, 2))


class BatchedUNet:
    """Evaluates the U-Net noise predictor for C parameter vectors at once."""

    def __init__(self, names: list[str], slices: dict[str, slice], shapes: dict[str, torch.Size], time_dim: int):
        self.names, self.slices, self.shapes, self.time_dim = names, slices, shapes, time_dim

    @classmethod
    def from_trainer(cls, tr: "FederatedTrainer") -> "BatchedUNet":
        shapes = {n: p.shape for n, p in tr.fore.module.named_parameters()}
        return cls(tr.fm.names, tr.fm.slices, shapes, tr.cfg.time_dim)

    def unpack(self, W: torch.Tensor) -> dict[str, torch.Tensor]:
        # One split (backward = a single concatenation) instead of per-parameter slices, whose
        # backward would materialise a full [C, P] zero gradient per parameter.
        C = W.shape[0]
        parts = torch.split(W, [self.slices[n].stop - self.slices[n].start for n in self.names], dim=1)
        return {n[len("net."):]: t.view(C, *self.shapes[n]) for n, t in zip(self.names, parts)}

    def _res(self, p: dict, pre: str, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = _conv(F.silu(_gn(x, p[pre + "block1.0.weight"], p[pre + "block1.0.bias"])),
                  p[pre + "block1.2.weight"], p[pre + "block1.2.bias"])
        h = h + _lin(temb, p[pre + "time_mlp.weight"], p[pre + "time_mlp.bias"])[..., None]
        h = _conv(F.silu(_gn(h, p[pre + "block2.0.weight"], p[pre + "block2.0.bias"])),
                  p[pre + "block2.2.weight"], p[pre + "block2.2.bias"])
        r = _conv(x, p[pre + "res_conv.weight"], p[pre + "res_conv.bias"]) if pre + "res_conv.weight" in p else x
        return h + r

    def eps(self, W: torch.Tensor, x_t: torch.Tensor, past: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """W [C,P]; x_t, past [C,B,1,L]; t [C,B] -> predicted noise [C,B,1,L]."""
        p = self.unpack(W)
        half = self.time_dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1)))
        ang = t.float()[..., None] * freqs
        e = torch.cat([torch.sin(ang), torch.cos(ang)], -1)
        temb = _lin(F.silu(_lin(e, p["time_emb.lin1.weight"], p["time_emb.lin1.bias"])),
                    p["time_emb.lin2.weight"], p["time_emb.lin2.bias"])
        x0 = _conv(torch.cat([x_t, past], dim=2), p["in_conv.weight"], p["in_conv.bias"])
        d1 = self._res(p, "down1.", x0, temb)
        d2 = self._res(p, "down2.", _pool(d1), temb)
        m = self._res(p, "mid2.", self._res(p, "mid1.", _pool(d2), temb), temb)
        u1 = self._res(p, "up1.", torch.cat([_up(m), d2], 2), temb)
        u2 = self._res(p, "up2.", torch.cat([_up(u1), d1], 2), temb)
        return _conv(F.silu(_gn(u2, p["out_conv.0.weight"], p["out_conv.0.bias"])),
                     p["out_conv.2.weight"], p["out_conv.2.bias"])


def _pool(z: torch.Tensor) -> torch.Tensor:
    return z.view(*z.shape[:3], -1, 2).amax(-1)


def _up(z: torch.Tensor) -> torch.Tensor:
    return z.repeat_interleave(2, dim=-1)


# ======================================================================================
# Pre-drawn randomness (identical draw order to the sequential path)
# ======================================================================================


@dataclass
class _Plan:
    """Per-client sequence of optimiser steps with all of their random draws."""

    idx: list[torch.Tensor] = field(default_factory=list)  # minibatch row indices
    t: list[torch.Tensor] = field(default_factory=list)
    noise: list[torch.Tensor] = field(default_factory=list)
    syn: list[torch.Tensor | None] = field(default_factory=list)  # FedTDD synthetic windows
    syn_t: list[torch.Tensor | None] = field(default_factory=list)
    syn_noise: list[torch.Tensor | None] = field(default_factory=list)
    segment: list[str] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return len(self.idx)


def _draw_plan(tr: "FederatedTrainer", i: int, sub: int, pool: torch.Tensor | None) -> _Plan:
    cfg, dev = tr.cfg, tr.device
    gen = make_generator(dev, derive_seed(cfg.seed, "local", tr.current_round, i, sub))
    n = tr.train_clients[i].train_t.shape[0]
    T, Fl, B = tr.fore.sched.T, cfg.future_len, cfg.batch_size
    plan = _Plan()
    for e in range(cfg.local_epochs):
        seg = "all"
        if tr.method == "fedrep":
            seg = "head" if e < cfg.fedrep_head_epochs else "body"
        perm = torch.randperm(n, device=dev, generator=gen)
        for b0 in range(0, n, B):
            idx = perm[b0:b0 + B]
            b = idx.numel()
            plan.idx.append(idx)
            plan.t.append(torch.randint(0, T, (b,), device=dev, generator=gen))
            plan.noise.append(torch.randn((b, 1, Fl), device=dev, generator=gen))
            plan.segment.append(seg)
            if pool is not None:
                k = min(B, pool.shape[0])
                sidx = torch.randint(0, pool.shape[0], (k,), device=dev, generator=gen)
                plan.syn.append(pool[sidx])
                plan.syn_t.append(torch.randint(0, T, (k,), device=dev, generator=gen))
                plan.syn_noise.append(torch.randn((k, 1, Fl), device=dev, generator=gen))
            else:
                plan.syn.append(None)
                plan.syn_t.append(None)
                plan.syn_noise.append(None)
    return plan


def _pad(rows: list[torch.Tensor], B: int, fill_shape: tuple[int, ...], dtype, dev) -> tuple[torch.Tensor, torch.Tensor]:
    if all(r.shape[0] == B for r in rows):  # common case: every minibatch is full
        return torch.stack(rows), torch.ones((len(rows), B), device=dev)
    out = torch.zeros((len(rows), B, *fill_shape), dtype=dtype, device=dev)
    mask = torch.zeros((len(rows), B), device=dev)
    for c, r in enumerate(rows):
        out[c, : r.shape[0]] = r
        mask[c, : r.shape[0]] = 1.0
    return out, mask


# ======================================================================================
# Batched local training
# ======================================================================================


def supports_batched(tr: "FederatedTrainer") -> bool:
    cfg = tr.cfg
    return (cfg.client_batch > 1 and cfg.model == "unet" and tr.fore.kind == "diffusion"
            and tr.method != "central")


def train_batched(tr: "FederatedTrainer", ids: list[int], starts: list[torch.Tensor], sub: int) -> list["LocalResult"]:
    from .trainer import Diverged, LocalResult

    cfg, dev = tr.cfg, tr.device
    net = tr._batched_net
    out: list[LocalResult] = []
    for c0 in range(0, len(ids), cfg.client_batch):
        chunk = ids[c0:c0 + cfg.client_batch]
        cstarts = starts[c0:c0 + cfg.client_batch]
        out += _train_chunk(tr, net, chunk, cstarts, sub, Diverged, LocalResult)
    return out


def _train_chunk(tr, net: BatchedUNet, ids, starts, sub, Diverged, LocalResult):
    cfg, dev, m = tr.cfg, tr.device, tr.method
    C, Bsz, P_len, F_len = len(ids), cfg.batch_size, cfg.past_len, cfg.future_len
    sched = tr.fore.sched
    t0 = time.perf_counter()
    S = torch.stack(starts)  # start models [C, P]
    W = S.clone()
    states = []
    for i in ids:
        st = tr.state(i)
        if cfg.client_opt_state == "persistent":
            if st.opt is None:
                st.opt = OptState.zeros(tr.fm.numel, dev)
            states.append(st.opt)
        else:
            states.append(OptState.zeros(tr.fm.numel, dev))
    M = torch.stack([s.m for s in states])
    V = torch.stack([s.v for s in states])
    segs = list(tr.opt.slices)
    stepc = {seg: torch.tensor([float(s.steps.get(seg, 0)) for s in states], device=dev) for seg in segs}
    if m == "fedrep":
        hs = tr.head_slice
        for c, i in enumerate(ids):
            st = tr.state(i)
            W[c, hs] = st.head if st.head is not None else S[c, hs]
    H = Cc = None
    if m == "feddyn":
        for i, s in zip(ids, starts):
            if tr.state(i).h is None:
                tr.state(i).h = torch.zeros_like(s)
        H = torch.stack([tr.state(i).h for i in ids])
    if m == "scaffold":
        for i, s in zip(ids, starts):
            if tr.state(i).c is None:
                tr.state(i).c = torch.zeros_like(s)
        Cc = torch.stack([tr.state(i).c for i in ids])
    plans = []
    for i in ids:
        pool = None
        if m == "fedtdd" and tr.syn_pool:
            others = [v for j, v in tr.syn_pool.items() if j != i]
            pool = torch.cat(others, 0) if others else None
        plans.append(_draw_plan(tr, i, sub, pool))
    Xs = [tr.train_clients[i].train_t for i in ids]
    T_max = max(p.steps for p in plans)
    loss_sum = torch.zeros(C, device=dev)
    lam = cfg.fedtdd_lambda
    for s in range(T_max):
        act = [c for c in range(C) if s < plans[c].steps]
        full = len(act) == C
        rows = torch.arange(C, device=dev) if full else torch.tensor(act, device=dev)
        Wa = (W if full else W[rows]).detach().requires_grad_(True)
        xb, mask = _pad([Xs[c][plans[c].idx[s]] for c in act], Bsz, (P_len + F_len,), torch.float32, dev)
        tt, _ = _pad([plans[c].t[s] for c in act], Bsz, (), torch.long, dev)
        nz, _ = _pad([plans[c].noise[s] for c in act], Bsz, (1, F_len), torch.float32, dev)
        past, fut = xb[:, :, None, :P_len], xb[:, :, None, P_len:]
        x_t = sched.sqrt_ac[tt][..., None, None] * fut + sched.sqrt_om[tt][..., None, None] * nz
        err = (net.eps(Wa, x_t, past, tt) - nz).pow(2).mean(dim=(2, 3))  # [Ca, B]
        per_client = (err * mask).sum(1) / mask.sum(1)  # == F.mse_loss over each client's real batch
        total = per_client
        if plans[act[0]].syn[s] is not None:
            sx, smask = _pad([plans[c].syn[s] for c in act], Bsz, (P_len + F_len,), torch.float32, dev)
            st_, _ = _pad([plans[c].syn_t[s] for c in act], Bsz, (), torch.long, dev)
            sn, _ = _pad([plans[c].syn_noise[s] for c in act], Bsz, (1, F_len), torch.float32, dev)
            sxt = sched.sqrt_ac[st_][..., None, None] * sx[:, :, None, P_len:] + sched.sqrt_om[st_][..., None, None] * sn
            serr = (net.eps(Wa, sxt, sx[:, :, None, :P_len], st_) - sn).pow(2).mean(dim=(2, 3))
            total = per_client + lam * (serr * smask).sum(1) / smask.sum(1)
        (G,) = torch.autograd.grad(total.sum(), Wa)
        loss_sum.index_add_(0, rows, per_client.detach())
        with torch.no_grad():
            by_seg: dict[str, list[int]] = {}
            for k, c in enumerate(act):
                by_seg.setdefault(plans[c].segment[s], []).append(k)
            for seg, ks in by_seg.items():
                sl = tr.opt.slices[seg]
                kk = torch.tensor(ks, device=dev)
                cr = rows[kk]
                g = G[kk][:, sl]
                if m == "fedprox":
                    g = g + cfg.prox_mu * (W[cr][:, sl] - S[cr][:, sl])
                elif m == "feddyn":
                    g = g - H[cr][:, sl] + cfg.dyn_alpha * (W[cr][:, sl] - S[cr][:, sl])
                elif m == "scaffold":
                    g = g - Cc[cr][:, sl] + tr.server_c[sl]
                if cfg.grad_clip > 0:
                    nrm = g.norm(dim=1, keepdim=True)
                    g = g * torch.clamp(cfg.grad_clip / (nrm + 1e-12), max=1.0)
                _adamw_rows(tr, W, M, V, stepc[seg], cr, sl, g, tr.opt.specs[seg])
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    seconds = (time.perf_counter() - t0) / C
    means = (loss_sum / torch.tensor([max(p.steps, 1) for p in plans], device=dev)).tolist()
    finite = torch.isfinite(W).all(dim=1).tolist()
    results = []
    for c, i in enumerate(ids):
        if not math.isfinite(means[c]) or not finite[c]:
            raise Diverged(f"client {tr.train_clients[i].cid}: non-finite loss/parameters")
        if cfg.client_opt_state == "persistent":
            states[c].m.copy_(M[c])
            states[c].v.copy_(V[c])
            for seg in segs:
                k = int(stepc[seg][c].item())
                if k:
                    states[c].steps[seg] = k
        vec = W[c].clone()
        if m == "fedrep":
            tr.state(i).head = vec[tr.head_slice].clone()
        tr._post_local(i, starts[c], vec, plans[c].steps)
        results.append(LocalResult(vec, plans[c].steps, means[c], seconds))
    return results


def _adamw_rows(tr, W, M, V, steps, rows, sl, g, spec) -> None:
    """torch.optim.AdamW on rows ``rows`` and columns ``sl`` of the stacked state."""
    lr, eps = tr.opt.lr, tr.opt.eps
    b1, b2, wd = spec.beta1, spec.beta2, spec.weight_decay
    steps.index_add_(0, rows, torch.ones(rows.numel(), device=rows.device))
    k = steps[rows][:, None]
    all_rows = rows.numel() == W.shape[0]
    if all_rows:  # views: update in place (rows are 0..C-1 in order)
        p, m, v = W[:, sl], M[:, sl], V[:, sl]
    else:
        p, m, v = W[rows][:, sl], M[rows][:, sl], V[rows][:, sl]
    if wd:
        p.mul_(1.0 - lr * wd)
    m.mul_(b1).add_(g, alpha=1.0 - b1)
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    bc1 = 1.0 - b1 ** k
    denom = (v.sqrt() / torch.sqrt(1.0 - b2 ** k)).add_(eps)
    p.addcdiv_(m, denom * (bc1 / lr), value=-1.0)  # p -= (lr / bc1) * m / denom
    if not all_rows:
        W[rows, sl.start:sl.stop] = p
        M[rows, sl.start:sl.stop] = m
        V[rows, sl.start:sl.stop] = v
