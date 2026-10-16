"""Personalised hierarchy (personal_head): reduction to FedRep, evaluation, cost accounting, batched
training, the personal / phsel campaign groups and their analysis."""

import numpy as np
import pandas as pd
import pytest
import torch

from fedtshdiff.data.federated import build_federated, eval_set
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.orchestrate.suites import GROUPS, TUNE_SEED, campaign, personal_specs, phsel_specs
from fedtshdiff.utils import seed_everything

PH = {"method": "fedts_hdiff", "grouping": "similarity", "regroup_every": 0, "personal_head": True}


def _run(cfg, rounds=2):
    data = build_federated(cfg)
    data.to_device(torch.device("cpu"))
    seed_everything(cfg.seed)
    tr = FederatedTrainer(cfg, data, build_forecaster(cfg, torch.device("cpu")), torch.device("cpu"))
    for r in range(1, rounds + 1):
        tr.run_round(r)
    return tr


def _rel(a, b):
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


def _assert_same_as_fedrep(rep, ph):
    body = rep.body_slice
    assert _rel(rep.global_vec[body], ph.global_vec[body]) < 1e-6
    for i, st in rep.states.items():
        assert torch.allclose(st.head, ph.states[i].head, atol=1e-6)
    es = eval_set(rep.data, 24, 7, torch.device("cpu"))
    for (va, pa), (vb, pb) in zip(rep.model_groups(es.client_index), ph.model_groups(es.client_index)):
        assert np.array_equal(pa, pb) and _rel(va, vb) < 1e-6


def test_defaults_keep_fingerprints(syn_cfg):
    a = syn_cfg(method="fedts_hdiff")
    b = syn_cfg(method="fedts_hdiff", personal_head=False)
    assert a.fingerprint() == b.fingerprint() and "personal_head" not in a.scientific_dict()
    c = syn_cfg(**PH)
    assert c.scientific_dict()["personal_head"] is True and c.fingerprint() != a.fingerprint()


@pytest.mark.parametrize("ov", [{"method": "fedavg", "personal_head": True},
                                {"method": "fedts_hdiff", "personal_head": True, "edge_mix": 0.5, "regroup_every": 0},
                                {"method": "fedts_hdiff", "personal_head": True, "diag_rounds": (1,)},
                                {"method": "fedts_hdiff", "personal_head": True, "fedrep_head_epochs": 9}],
                         ids=["fedavg", "edge_mix", "diagnostics", "head-epochs"])
def test_validation(syn_cfg, ov):
    with pytest.raises(ValueError):
        syn_cfg(**ov)


def test_head_mix_is_gone(syn_cfg):
    with pytest.raises(Exception):
        syn_cfg(**PH, head_mix=0.5)


@pytest.mark.parametrize("ov", [{"num_edges": 1, "beta_agg": 0.0, "regroup_every": 1, "grouping": "balance"},
                                {"beta_agg": 0.0}],  # two static similarity groups, data weights (pha_beta0)
                         ids=["one-edge", "two-groups-data-weights"])
def test_data_weights_with_one_edge_round_is_fedrep(syn_cfg, ov):
    """kappa = 1 and data-proportional weights: the hierarchy is FedAvg on the body, so personal heads
    must reproduce FedRep (bodies, heads and evaluation), whatever the grouping."""
    rep = _run(syn_cfg(method="fedrep"))
    ph = _run(syn_cfg(**{**PH, **ov}))
    _assert_same_as_fedrep(rep, ph)


def test_personal_start_and_evaluation(syn_cfg):
    tr = _run(syn_cfg(**PH, edge_rounds=2))
    hs = tr.head_slice
    edge = tr.global_vec.clone()
    edge[hs] = 3.0
    v = tr._personal_start(0, edge)
    assert torch.equal(v[hs], tr.states[0].head) and torch.equal(v[tr.body_slice], edge[tr.body_slice])
    es = eval_set(tr.data, 24, 7, torch.device("cpu"))
    for vec, pos in tr.model_groups(es.client_index):
        i = int(es.client_index[pos[0]])
        assert torch.equal(vec[hs], tr.states[i].head) and torch.equal(vec[tr.body_slice], tr.global_vec[tr.body_slice])
    assert len(tr.checkpoint()["heads"]) == len(tr.train_clients)
    tr.states.pop(0)  # a client that has not trained starts from the edge head
    assert torch.equal(tr._personal_start(0, edge), edge)


def test_cost_counts_only_the_body(syn_cfg):
    std = _run(syn_cfg(**{k: v for k, v in PH.items() if k != "personal_head"}), rounds=1)
    ph = _run(syn_cfg(**PH), rounds=1)
    log_std, log_ph = std.run_round(2), ph.run_round(2)
    K = sum(1 for e in ph.last_assignment.edges if e)
    m, d = len(ph.train_clients), ph.phi.shape[1]
    assert log_ph.stats["bytes_cloud"] == 2 * K * ph.body_bytes
    assert log_std.stats["bytes_cloud"] == 2 * K * std.param_bytes
    assert log_ph.stats["bytes_client_up"] == m * ph.body_bytes + m * d * 4
    assert ph.body_bytes < ph.param_bytes


def test_batched_equals_sequential(syn_cfg):
    ov = {**PH, "edge_rounds": 2}
    seq, bat = _run(syn_cfg(client_batch=1, **ov)), _run(syn_cfg(client_batch=5, **ov))
    assert bat._batched_net is not None and seq._batched_net is None
    assert _rel(seq.global_vec, bat.global_vec) < 1e-5
    for i, st in seq.states.items():
        assert _rel(st.head, bat.states[i].head) < 1e-5


def test_campaign_shapes():
    from fedtshdiff.config import build_config

    per = campaign(groups=("personal",))
    assert len(per) == 4 * 5 * 5 and len({s.task_id for s in per}) == len(per)
    sel = campaign(groups=("phsel",))
    assert len(sel) == 4 * 2 + 4 * 5 * 2 + 5 * 3 and len({s.task_id for s in sel}) == len(sel)
    for s in per + sel:
        cfg = build_config({"dataset": s.dataset, **s.overrides})
        R = build_config({"dataset": s.dataset}).rounds
        if s.label.startswith(("ph_client", "ph_off", "phval")):
            assert cfg.personal_head == s.label.startswith(("ph_client", "phval_on"))
            assert cfg.regroup_every == 0 and cfg.grouping == "similarity" and cfg.rounds == R // cfg.edge_rounds
        if s.label.startswith("phval"):
            assert s.seed == TUNE_SEED and cfg.eval_split == "val" and cfg.edge_rounds == 5
        if s.label.startswith("pha"):
            assert s.dataset == "mitbih" and cfg.personal_head and cfg.edge_rounds == 1 and cfg.rounds == R
        if s.label in ("ph_fedavg", "ph_fedrep", "ph_cfl"):
            assert cfg.method == s.label[3:] and cfg.rounds == R and not cfg.personal_head
    assert not ({"personal", "phsel"} & set(GROUPS)) and not any({"personal", "phsel"} & s.groups for s in campaign())


def _fake_rows(specs, rng, group_crps):
    from fedtshdiff.analysis import tables

    rows = []
    for s in specs:
        k = tables.personal_parse(s.label)[1] if s.label.startswith("ph_") else 1
        rows.append({"task_id": s.task_id, "dataset": s.dataset, "seed": s.seed, "group": s.group, "label": s.label,
                     "status": "complete", "crps": group_crps(s) + 0.001 * rng.standard_normal(),
                     "dtw_mean": 1.0 + 0.01 * rng.standard_normal(), "wis_90": 0.8, "picp_90": 0.9,
                     "cost_total_bytes_cloud": 2**20 * 100.0 / (k or 1)})
    return pd.DataFrame(rows)


def test_personal_analysis_with_reference_runs(tmp_path):
    """phsel rows + the personal campaign's exported rows: selection, selected rows, tables, figure."""
    from fedtshdiff.analysis import figures, tables
    from fedtshdiff.analysis.report import _add_reference_runs

    rng = np.random.default_rng(0)
    # personal heads help on mitbih (validation and test), hurt on synthetic
    crps = lambda s: (0.05 if s.dataset == "mitbih" else 0.09) if "client" in s.label or s.label == "phval_on" else 0.07
    ref = _fake_rows(personal_specs(["synthetic", "mitbih"], (1, 2, 3)), rng, crps)
    ref.to_csv(tmp_path / "ref_runs.csv", index=False)
    new = _fake_rows(phsel_specs(["synthetic", "mitbih"], (1, 2, 3)), rng, crps)
    df = _add_reference_runs(new, [tmp_path / "ref_runs.csv"])
    assert len(df) == len(new) + len(ref)
    sel = tables.personal_selection(df).set_index("dataset")["personal_heads"]
    assert sel["mitbih"] and not sel["synthetic"]
    runs = tables.personal_runs(df)
    for ds, src in (("mitbih", "ph_client"), ("synthetic", "ph_off")):
        a = runs[(runs.dataset == ds) & (runs.label == "ph_sel_k5")].sort_values("seed")["crps"].to_numpy()
        b = runs[(runs.dataset == ds) & (runs.label == f"{src}_k5")].sort_values("seed")["crps"].to_numpy()
        assert np.array_equal(a, b)
    summ = tables.personal_table(df, tmp_path / "t")
    assert len(summ) == 2 * 9
    tex = (tmp_path / "t" / "tab_personal.tex").read_text(encoding="utf-8")
    assert "heads chosen on validation (personal)" in tex and "heads chosen on validation (shared)" in tex
    abl = tables.personal_ablation_table(df, tmp_path / "t")
    assert set(abl["label"]) == {"ph_fedrep", "ph_client_k1", "pha_beta0", "pha_nogroup", "pha_balance"}
    assert (tmp_path / "t" / "tab_personal_ablation.tex").exists()
    figures.personal(df, tmp_path / "f")
    assert (tmp_path / "f" / "fig_personal.pdf").exists()
