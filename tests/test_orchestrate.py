"""Slot parsing, campaign construction, planning and resume semantics."""

import pytest

from fedtshdiff.orchestrate.planner import build_plan, try_select
from fedtshdiff.orchestrate.scheduler import parse_slots
from fedtshdiff.orchestrate.suites import GROUPS, campaign
from fedtshdiff.utils import atomic_write_json


def test_parse_slots_respects_inherited_mask():
    slots = parse_slots("auto", 2, env={"CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7"}, device_count=lambda: 7)
    assert len(slots) == 14
    assert {s.device for s in slots} == {str(i) for i in range(1, 8)}
    slots = parse_slots("0,2", 1, env={"CUDA_VISIBLE_DEVICES": "3,5,7"})
    assert [s.device for s in slots] == ["3", "7"]
    with pytest.raises(ValueError):
        parse_slots("4", 1, env={"CUDA_VISIBLE_DEVICES": "3,5"})
    assert [s.device for s in parse_slots("cpu", 3)] == ["", "", ""]


def test_full_campaign_shape():
    specs = campaign()
    ids = [s.task_id for s in specs]
    assert len(ids) == len(set(ids))
    groups = {g for s in specs for g in s.groups}
    assert groups == set(GROUPS)
    # the full model is shared by main and ablation (one run, two groups)
    shared = [s for s in specs if {"main", "ablation"} <= s.groups]
    assert len(shared) == 4 * 5
    main = [s for s in specs if "main" in s.groups]
    assert len(main) == 4 * 10 * 5
    tuned = [s for s in main if s.select]
    assert {next(iter(s.select)) for s in tuned} == {"prox_mu", "dyn_alpha", "fedtdd_lambda", "beta_agg"}


def test_plan_waits_for_selection_then_resolves(tmp_path, data_root):
    specs = campaign(groups=("tune", "main"), seeds=(1,), datasets=("synthetic",))
    plan = build_plan(tmp_path, specs, str(data_root))
    waiting = plan.by_state("waiting")
    assert waiting and all(w.startswith("selection:") for t in waiting for w in t.waiting_on)
    # fake-complete the FedProx tuning runs with validation CRPS 3, 1, 2 -> mu = 0.01 wins
    tune = [t for t in plan.tasks if "tune" in t.spec.groups and t.spec.overrides["method"] == "fedprox"]
    for t, crps in zip(sorted(tune, key=lambda t: t.spec.overrides["prox_mu"]), (3.0, 1.0, 2.0)):
        t.out_dir.mkdir(parents=True)
        atomic_write_json(t.out_dir / "final_metrics.json", {"default": {"crps": crps}})
        from fedtshdiff.utils import sha256_file

        atomic_write_json(t.out_dir / "done.json", {"status": "complete", "fingerprint": t.fingerprint,
                                                     "final_metrics_sha256": sha256_file(t.out_dir / "final_metrics.json")})
    plan = build_plan(tmp_path, specs, str(data_root))
    prox = [t for t in plan.tasks if "main" in t.spec.groups and t.spec.overrides.get("method") == "fedprox"]
    assert prox and prox[0].state == "ready" and prox[0].config.prox_mu == 0.01
    assert all(t.state == "done" for t in plan.tasks if t in tune or t.task_id in {x.task_id for x in tune})
    sel = try_select(tmp_path, specs, "synthetic", "fedprox", {t.task_id: t.fingerprint for t in plan.tasks})
    assert sel["value"] == 0.01 and sel["scores"]["0.001"] == 3.0
