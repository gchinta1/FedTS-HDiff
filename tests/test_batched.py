"""Client-batched training equals one-client-at-a-time training (up to floating point)."""

import pytest
import torch

from fedtshdiff.data.federated import build_federated
from fedtshdiff.fl.batched import BatchedUNet
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.utils import seed_everything


def _trainer(cfg):
    data = build_federated(cfg)
    data.to_device(torch.device("cpu"))
    seed_everything(cfg.seed)
    return FederatedTrainer(cfg, data, build_forecaster(cfg, torch.device("cpu")), torch.device("cpu"))


def _run(cfg, rounds=2):
    tr = _trainer(cfg)
    for r in range(1, rounds + 1):
        tr.run_round(r)
    return tr


def test_batched_unet_matches_module(syn_cfg):
    tr = _trainer(syn_cfg(method="fedavg"))
    net = BatchedUNet.from_trainer(tr)
    g = torch.Generator().manual_seed(0)
    W = torch.stack([tr.global_vec + 0.01 * torch.randn(tr.fm.numel, generator=g) for _ in range(3)])
    x_t = torch.randn(3, 5, 1, 16, generator=g)
    past = torch.randn(3, 5, 1, 16, generator=g)
    t = torch.randint(0, 50, (3, 5), generator=g)
    out = net.eps(W, x_t, past, t)
    for c in range(3):
        tr.fm.load(W[c])
        ref = tr.fore.module(x_t[c], past[c], t[c])
        assert torch.allclose(out[c], ref, atol=1e-5)


@pytest.mark.parametrize("ov", [
    {"method": "fedavg"},
    {"method": "fedprox", "prox_mu": 0.1},
    {"method": "feddyn", "dyn_alpha": 0.1},
    {"method": "scaffold"},
    {"method": "fedrep"},
    {"method": "fednova"},
    {"method": "fedtdd"},
    {"method": "cfl"},
    {"method": "fedts_hdiff", "edge_rounds": 2},
    {"method": "fedts_hdiff", "participation": 0.5, "client_opt_state": "reset"},
], ids=lambda o: "-".join(f"{k}={v}" for k, v in o.items()))
def test_batched_equals_sequential(syn_cfg, ov):
    seq = _run(syn_cfg(client_batch=1, **ov))
    bat = _run(syn_cfg(client_batch=5, **ov))  # chunks of 5 -> several chunks, unequal client sizes
    assert bat._batched_net is not None and seq._batched_net is None
    def rel(a, b):
        return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()

    # single-step gradients agree to ~1e-6; after two rounds the models agree to ~1e-6 relative
    assert rel(seq.global_vec, bat.global_vec) < 1e-5
    if ov["method"] == "cfl":
        for a, b in zip(seq.cluster_vecs, bat.cluster_vecs):
            assert rel(a, b) < 1e-5
    # persistent optimiser state was written back (Adam moments accumulate float noise in
    # near-zero coordinates, hence the looser relative tolerance)
    for i, st in seq.states.items():
        if st.opt is not None:
            assert st.opt.steps == bat.states[i].opt.steps
            assert rel(st.opt.m, bat.states[i].opt.m) < 1e-2
            assert rel(st.opt.v, bat.states[i].opt.v) < 1e-2
        if st.head is not None:
            assert rel(st.head, bat.states[i].head) < 1e-5
