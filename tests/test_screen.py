"""Follow-up screen: group personalisation (edge_mix), screen campaign shape and screen analysis."""

import numpy as np
import pandas as pd
import pytest
import torch

from fedtshdiff.config import build_config
from fedtshdiff.data.federated import build_federated, eval_set
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.orchestrate.suites import GROUPS, SCREEN_KAPPA, campaign, screen_specs
from fedtshdiff.utils import seed_everything

PERS = {"method": "fedts_hdiff", "grouping": "similarity", "regroup_every": 0}


def _run(cfg, rounds=2):
    data = build_federated(cfg)
    data.to_device(torch.device("cpu"))
    seed_everything(cfg.seed)
    tr = FederatedTrainer(cfg, data, build_forecaster(cfg, torch.device("cpu")), torch.device("cpu"))
    for r in range(1, rounds + 1):
        tr.run_round(r)
    return tr


def test_edge_mix_default_keeps_fingerprints(syn_cfg):
    """edge_mix=1 is the standard method and is left out of the fingerprint (round-1 runs stay valid)."""
    a, b = syn_cfg(**PERS), syn_cfg(**PERS, edge_mix=1.0)
    assert "edge_mix" not in a.scientific_dict() and a.fingerprint() == b.fingerprint()
    c = syn_cfg(**PERS, edge_mix=0.5)
    assert c.scientific_dict()["edge_mix"] == 0.5 and c.fingerprint() != a.fingerprint()


def test_edge_mix_one_is_standard_training(syn_cfg):
    std, one = _run(syn_cfg(**PERS)), _run(syn_cfg(**PERS, edge_mix=1.0))
    assert torch.equal(std.global_vec, one.global_vec)
    assert one.edge_models is None and len(one.model_groups(np.arange(5))) == 1


@pytest.mark.parametrize("ov", [{}, {"method": "fedavg"}, {"regroup_every": 1}, {"diag_rounds": (1,)}],
                         ids=["valid", "fedavg", "dynamic", "diagnostics"])
def test_edge_mix_validation(syn_cfg, ov):
    if not ov:
        syn_cfg(**PERS, edge_mix=0.0)
        return
    with pytest.raises(ValueError, match="edge_mix"):
        syn_cfg(**{**PERS, **ov}, edge_mix=0.5)


@pytest.mark.parametrize("mix", [0.0, 0.5])
def test_group_personalisation(syn_cfg, mix):
    tr = _run(syn_cfg(**PERS, edge_mix=mix, edge_rounds=2))
    K = tr.grouper._static_K
    assert K > 1 and len(tr.edge_models) == K
    # the global model is the edge-weighted mean of the trained edge models (full participation)
    q = tr.last_assignment.edge_weight
    mean = sum(float(q[k]) * tr.edge_models[k] for k in range(K) if tr.last_assignment.edges[k])
    assert torch.allclose(tr.global_vec, mean, atol=1e-6)
    # evaluation serves every client with its edge's (mixed) model
    es = eval_set(tr.data, 24, 7, torch.device("cpu"))
    groups = tr.model_groups(es.client_index)
    assert len(groups) == len({tr.grouper._static[int(c)] for c in es.client_index}) > 1
    assert sorted(np.concatenate([p for _, p in groups]).tolist()) == list(range(len(es.client_index)))
    for vec, pos in groups:
        k = tr.grouper._static[int(es.client_index[pos[0]])]
        assert torch.allclose(vec, mix * tr.global_vec + (1 - mix) * tr.edge_models[k])
    ck = tr.checkpoint()
    assert len(ck["edge_models"]) == K and ck["edge_of"] == tr.grouper._static
    # personalised training actually departs from the standard method
    std = _run(syn_cfg(**PERS, edge_rounds=2))
    assert not torch.allclose(std.global_vec, tr.global_vec)


def test_group_personalisation_batched_equals_sequential(syn_cfg):
    ov = {**PERS, "edge_mix": 0.5, "edge_rounds": 2}
    seq, bat = _run(syn_cfg(client_batch=1, **ov)), _run(syn_cfg(client_batch=5, **ov))
    assert bat._batched_net is not None and seq._batched_net is None

    def rel(a, b):
        return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()

    assert rel(seq.global_vec, bat.global_vec) < 1e-5
    for a, b in zip(seq.edge_models, bat.edge_models):
        assert rel(a, b) < 1e-5


def test_screen_campaign_shape():
    specs = campaign(groups=("screen",), seeds=(1, 2, 3, 4, 5), datasets=("mitbih", "ptbxl", "metrla"))
    assert len(specs) == 171 and len({s.task_id for s in specs}) == 171
    per = {}
    for s in specs:
        per.setdefault((s.dataset, s.seed), []).append(s)
    assert set(k[1] for k in per) == {1, 2, 3} and all(len(v) == 19 for v in per.values())
    for s in specs:
        cfg = build_config({"dataset": s.dataset, **s.overrides})
        label = s.label
        R = build_config({"dataset": s.dataset}).rounds
        k = cfg.edge_rounds
        # equal computation: kappa edge rounds x R/kappa cloud rounds; FedAvg references at R/kappa rounds
        if label.startswith("screen_fedavg_c"):
            assert cfg.method == "fedavg" and cfg.rounds == R // int(label[len("screen_fedavg_c"):])
        elif label == "screen_fedavg":
            assert cfg.method == "fedavg" and cfg.rounds == R
        else:
            assert cfg.method == "fedts_hdiff" and cfg.rounds == R // k and k in SCREEN_KAPPA
            assert cfg.eval_every == max(1, cfg.rounds // 10)
        if label.startswith("screen_pers"):
            assert cfg.edge_mix < 1 and cfg.regroup_every == 0
        if s.dataset == "ptbxl" and cfg.method == "fedts_hdiff":
            assert cfg.descriptor == "signal"
    # the screen is opt-in: the default campaign is unchanged
    assert "screen" not in GROUPS and not any("screen" in s.groups for s in campaign())


def test_screen_specs_fall_back_to_given_seeds():
    specs = screen_specs(["synthetic"], (7, 8))
    assert {s.seed for s in specs} == {7} and len(specs) == 19


def test_screen_analysis(tmp_path):
    from fedtshdiff.analysis import figures, tables

    rng = np.random.default_rng(0)
    rows = []
    for s in screen_specs(["synthetic", "mitbih"], (1, 2, 3)):
        label = s.label
        _, k, kind = tables.screen_parse(label)
        rows.append({"task_id": s.task_id, "dataset": s.dataset, "seed": s.seed, "group": "screen", "label": label,
                     "status": "complete", "crps": 0.07 + 0.002 * (k or 1) + 0.001 * rng.standard_normal(),
                     "dtw_mean": 1.0 + 0.01 * rng.standard_normal(),
                     "cost_total_bytes_cloud": 2**20 * (100.0 if kind.startswith("fedavg") else 20.0) / (k or 1)})
    rows.append({**rows[0], "label": "fedavg"})  # a task shared with the main group carries its labels too
    df = pd.DataFrame(rows)
    summ = tables.screen_table(df, tmp_path / "t")
    assert len(summ) == 38
    tex = (tmp_path / "t" / "tab_screen.tex").read_text(encoding="utf-8")
    assert "Similarity grouping" in tex and "Group-personalised (mix 0.5)" in tex and "MIT-BIH" in tex
    comm = pd.read_csv(tmp_path / "t" / "tab_screen_tests_equal_cloud.csv")
    assert set(comm["reference"]) == {"screen_fedavg_c2", "screen_fedavg_c5", "screen_fedavg_c10"}
    assert not comm["label"].str.endswith("_k1").any()
    comp = pd.read_csv(tmp_path / "t" / "tab_screen_tests_equal_compute.csv")
    assert set(comp["reference"]) == {"screen_fedavg"} and len(comp) == 2 * 2 * 18
    figures.screen(df, tmp_path / "f")
    assert (tmp_path / "f" / "fig_screen.pdf").exists()
