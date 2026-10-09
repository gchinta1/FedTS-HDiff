"""End-to-end training on synthetic data (CPU)."""

import numpy as np
import pytest
import torch

from fedtshdiff.data.federated import build_federated
from fedtshdiff.fl.diagnostics import _dispersion, run_diagnostics
from fedtshdiff.fl.runner import is_sealed, run_experiment, run_fingerprint
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.utils import read_json, seed_everything


def _train(cfg, rounds=2):
    data = build_federated(cfg)
    data.to_device(torch.device("cpu"))
    seed_everything(cfg.seed)
    tr = FederatedTrainer(cfg, data, build_forecaster(cfg, torch.device("cpu")), torch.device("cpu"))
    for r in range(1, rounds + 1):
        tr.run_round(r)
    return tr


def test_hierarchy_with_data_proportional_weights_is_fedavg(syn_cfg):
    """With kappa = 1 and beta_agg = 0, any grouping gives exactly FedAvg (paper Remark 3)."""
    a = _train(syn_cfg(method="fedavg"))
    for grouping in ("balance", "random", "kmeans"):
        b = _train(syn_cfg(method="fedts_hdiff", beta_agg=0.0, grouping=grouping, num_edges=3))
        assert torch.allclose(a.global_vec, b.global_vec, atol=1e-5), grouping


def test_adaptive_weights_change_the_model(syn_cfg):
    a = _train(syn_cfg(method="fedavg"))
    b = _train(syn_cfg(method="fedts_hdiff", beta_agg=2.0))
    assert not torch.allclose(a.global_vec, b.global_vec, atol=1e-5)


def test_edge_rounds_matter(syn_cfg):
    a = _train(syn_cfg(method="fedts_hdiff", beta_agg=0.0, edge_rounds=1, grouping="kmeans"))
    b = _train(syn_cfg(method="fedts_hdiff", beta_agg=0.0, edge_rounds=2, grouping="kmeans"))
    assert not torch.allclose(a.global_vec, b.global_vec, atol=1e-5)


def test_lemma1_decomposition(syn_cfg):
    tr = _train(syn_cfg(method="fedts_hdiff"), rounds=1)
    d = run_diagnostics(tr, 2)
    for name, part in d["partitions"].items():
        if name.endswith("_sd") or name == "random":
            continue
        assert part["delta_in"] + part["delta_out"] == pytest.approx(part["delta_global"], rel=1e-6, abs=1e-9)
    rng = np.random.default_rng(0)
    G = rng.standard_normal((6, 5))
    p = np.full(6, 1 / 6)
    out = _dispersion(G @ G.T, p, [np.arange(3), np.arange(3, 6)])
    gbar = p @ G
    assert out["delta_global"] == pytest.approx(float(p @ ((G - gbar) ** 2).sum(1)))


@pytest.mark.parametrize("method", ["fedavg", "fedprox", "fednova", "feddyn", "scaffold", "fedrep", "fedtdd",
                                    "fedts_hdiff", "cfl", "central"])
def test_every_method_runs_and_seals(syn_cfg, tmp_path, method):
    cfg = syn_cfg(method=method, rounds=2)
    res = run_experiment(cfg, tmp_path / method)
    assert res["status"] == "complete"
    final = read_json(tmp_path / method / "final_metrics.json")["default"]
    assert np.isfinite(final["crps"]) and 0 <= final["picp_90"] <= 1
    fp = run_fingerprint(cfg, read_json(tmp_path / method / "meta.json")["data_hash"])
    assert is_sealed(tmp_path / method, fp)
    assert not is_sealed(tmp_path / method, run_fingerprint(cfg.with_updates(lr=0.5), "x"))


@pytest.mark.parametrize("model", ["csdi", "lstm", "gan"])
def test_other_models_run(syn_cfg, tmp_path, model):
    extra = {"sampler": "ddpm", "beta_schedule": "quad", "beta_end": 0.5} if model == "csdi" else {}
    res = run_experiment(syn_cfg(method="fedavg", model=model, rounds=1, **extra), tmp_path / model)
    assert res["status"] == "complete"


def test_drift_static_vs_dynamic(syn_cfg, tmp_path):
    for regroup in (1, 0):
        res = run_experiment(syn_cfg(method="fedts_hdiff", rounds=3, drift_rounds=[2], regroup_every=regroup),
                             tmp_path / f"r{regroup}")
        assert res["status"] == "complete"


def test_same_seed_is_reproducible(syn_cfg):
    a = _train(syn_cfg(method="fedts_hdiff"))
    b = _train(syn_cfg(method="fedts_hdiff"))
    assert torch.equal(a.global_vec, b.global_vec)
