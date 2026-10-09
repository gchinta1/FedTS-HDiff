"""Optimiser, diffusion schedule, grouping and configuration invariants."""

import copy

import numpy as np
import pytest
import torch

from fedtshdiff.config import Config, build_config, parse_overrides
from fedtshdiff.fl.flat import FlatAdamW, FlatModel, OptState
from fedtshdiff.fl.grouping import aggregation, assign_edges, homogeneity, reference
from fedtshdiff.models import build_forecaster
from fedtshdiff.models.diffusion import ddim_timesteps


@pytest.mark.parametrize("model", ["unet", "lstm", "gan"])
def test_flat_adamw_equals_torch_adamw(model):
    cfg = Config(model=model, past_len=16, future_len=16, method="fedavg").validate()
    torch.manual_seed(0)
    fore = build_forecaster(cfg, torch.device("cpu"))
    ref = copy.deepcopy(fore.module)
    fm = FlatModel(fore.module, torch.device("cpu"))
    seg = fore.segments[0]
    opt = FlatAdamW(fm, fore.segments, 1e-3, 1e-8)
    state = OptState.zeros(fm.numel, torch.device("cpu"))
    sl = fm.segment_slice(seg)
    params = [p for n, p in ref.named_parameters() if seg.prefixes == ("",) or n.startswith(seg.prefixes)]
    topt = torch.optim.AdamW(params, lr=1e-3, betas=(seg.beta1, seg.beta2), weight_decay=seg.weight_decay)
    x = torch.rand(8, 32) * 2 - 1
    other = copy.copy(fore)
    other.module = ref
    for it in range(4):
        fm.grad.zero_()
        fore.loss(x, torch.Generator().manual_seed(it)).backward()
        topt.zero_grad()
        other.loss(x, torch.Generator().manual_seed(it)).backward()
        assert fm.grads_are_views()
        opt.update(state, seg.name, fm.grad[sl])
        topt.step()
    flat_ref = torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
    assert torch.allclose(flat_ref, fm.flat, atol=1e-6)


def test_ddim_spacing():
    assert ddim_timesteps(500, 200, "trailing")[0] == 499
    assert ddim_timesteps(500, 200, "legacy")[0] == 398  # the original code's truncated schedule
    ts = ddim_timesteps(200, 200, "trailing")
    assert ts == list(range(199, -1, -1))
    assert ddim_timesteps(500, 25, "trailing")[0] == 499


def _original_balance(client_dists, client_sizes, num_edges, a_sel, b_sel, tau, target, rand):
    """The assignment loop of FedTS-HDiff_PTB-XL.py (softmax form), with an injected uniform source."""
    keys = list(range(len(target)))
    qe = [{k: 0.0 for k in keys} for _ in range(num_edges)]
    ne = [0] * num_edges
    avg_n = float(np.mean(client_sizes))
    out = []

    def upd(q, n, qn, nn):
        new_n = n + nn
        return {k: (q[k] * n + qn[k] * nn) / new_n for k in keys}, new_n

    def sh(q):
        return 2.0 - np.sqrt(sum((q[k] - target[k]) ** 2 for k in keys))

    for cid in range(len(client_sizes)):
        scores = []
        for e in range(num_edges):
            q2, n2 = upd(qe[e], ne[e], client_dists[cid], client_sizes[cid])
            scores.append(a_sel * sh(q2) - n2 / max(avg_n, 1.0) + b_sel)
        s = np.array(scores)
        p = np.exp((s - s.max()) / tau)
        p = p / (p.sum() + 1e-12)
        u, cum, chosen = rand(), 0.0, 0
        for e, pe in enumerate(p):
            cum += pe
            if u <= cum:
                chosen = e
                break
        out.append(chosen)
        qe[chosen], ne[chosen] = upd(qe[chosen], ne[chosen], client_dists[cid], client_sizes[cid])
    return out


def test_balance_grouping_reproduces_original_loop():
    rng = np.random.default_rng(3)
    N, C, K = 40, 5, 4
    phi = rng.dirichlet(np.ones(C) * 0.3, size=N)
    sizes = rng.integers(50, 200, size=N).astype(float)
    cfg = Config(num_edges=K, method="fedts_hdiff").validate()
    u = reference("label", phi)
    mine = assign_edges(cfg, list(range(N)), phi, sizes, u, float(sizes.mean()), np.random.default_rng(9))
    draws = np.random.default_rng(9)
    orig = _original_balance([dict(enumerate(r)) for r in phi], sizes.tolist(), K, cfg.a_sel, cfg.b_sel,
                             cfg.tau_sel, dict(enumerate(u)), draws.random)
    assert [next(k for k, m in enumerate(mine) if i in m) for i in range(N)] == orig


def test_data_proportional_weights_equal_fedavg():
    rng = np.random.default_rng(0)
    phi = rng.dirichlet(np.ones(4), size=12)
    sizes = rng.integers(10, 100, size=12).astype(float)
    cfg = Config(num_edges=3, method="fedts_hdiff", beta_agg=0.0).validate()
    edges = assign_edges(cfg, list(range(12)), phi, sizes, reference("label", phi), sizes.mean(),
                         np.random.default_rng(1))
    asg = aggregation(cfg, edges, phi, sizes, reference("label", phi), adaptive=False)
    eff = asg.effective_weights()
    assert np.allclose([eff[i] for i in range(12)], sizes / sizes.sum())


def test_balance_moves_edges_towards_reference():
    # balancing must yield edge descriptors closer to u than similarity clustering does
    rng = np.random.default_rng(2)
    phi = np.repeat(np.eye(4), 10, axis=0) * 0.9 + 0.025
    sizes = np.full(40, 100.0)
    u = reference("label", phi)
    out = {}
    for grouping in ("balance", "kmeans"):
        cfg = Config(num_edges=4, method="fedts_hdiff", grouping=grouping).validate()
        edges = assign_edges(cfg, list(range(40)), phi, sizes, u, 100.0, np.random.default_rng(0))
        asg = aggregation(cfg, edges, phi, sizes, u)
        out[grouping] = float(np.mean([homogeneity(asg.edge_descriptor[k], u) for k in range(4) if edges[k]]))
    assert out["balance"] > out["kmeans"]


def test_config_rejects_unknown_and_parses_overrides():
    with pytest.raises(KeyError):
        build_config({"dataset": "synthetic", "learning_rate": 1.0})
    ov = parse_overrides(["drift_rounds=[3, 5]", "lr=0.001", "method=fedavg"])
    cfg = build_config({"dataset": "synthetic"}, overrides=ov)
    assert cfg.drift_rounds == (3, 5) and cfg.lr == 0.001
    assert cfg.fingerprint() == cfg.with_updates(device="cuda", data_root="x").fingerprint()
    assert cfg.fingerprint() != cfg.with_updates(lr=0.01).fingerprint()
    with pytest.raises(ValueError):
        build_config({"dataset": "synthetic", "method": "nope"})
