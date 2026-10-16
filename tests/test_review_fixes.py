"""Regression tests for the issues found in the pre-deployment code review."""

import argparse
import threading

import numpy as np
import pytest
import torch

from fedtshdiff.config import Config, build_config
from fedtshdiff.data.federated import _split_indices, build_federated, eval_set, split_gap
from fedtshdiff.eval.evaluate import diffusion_test_loss, evaluate
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.orchestrate.planner import build_plan, specs_from_manifest, write_manifest
from fedtshdiff.orchestrate.suites import campaign
from fedtshdiff.utils import seed_everything


def _trainer(cfg):
    data = build_federated(cfg)
    data.to_device(torch.device("cpu"))
    seed_everything(cfg.seed)
    return FederatedTrainer(cfg, data, build_forecaster(cfg, torch.device("cpu")), torch.device("cpu"))


def test_participants_identical_across_methods(syn_cfg):
    """Partial participation: grouping must not consume the participant-sampling stream."""
    a = _trainer(syn_cfg(method="fedavg", participation=0.5))
    b = _trainer(syn_cfg(method="fedts_hdiff", participation=0.5, grouping="balance"))
    for r in (1, 2, 3):
        a.current_round = b.current_round = r
        pa, pb = a.participants(), b.participants()
        assert pa == pb
        b.grouper.assign(r, pb, b.rng)  # consumes the server stream


def test_gan_generator_steps_do_not_depend_on_client_order(syn_cfg):
    cfg = syn_cfg(method="fedavg", model="gan")
    a = _trainer(cfg)
    a.current_round = 1
    alone = a.local_train(1, a.global_vec).vec
    b = _trainer(cfg)
    b.current_round = 1
    b.local_train(0, b.global_vec)
    after = b.local_train(1, b.global_vec).vec
    assert torch.allclose(alone, after)


def test_eval_noise_is_common_across_model_groupings(syn_cfg):
    """FedRep evaluates per-client groups; with identical parameters it must give exactly the
    samples of a single-group method (same starting noise for every window)."""
    flat = _trainer(syn_cfg(method="fedavg"))
    rep = _trainer(syn_cfg(method="fedrep"))
    assert torch.equal(flat.global_vec, rep.global_vec)
    es = eval_set(flat.data, 24, 7, torch.device("cpu"))
    assert len(rep.model_groups(es.client_index)) > 1
    s1, a1 = evaluate(flat, es, 6, None, seed=11)
    s2, a2 = evaluate(rep, es, 6, None, seed=11)
    assert np.allclose(a1["crps"], a2["crps"]) and np.allclose(a1["dtw_mean"], a2["dtw_mean"])


def test_csdi_test_loss_is_deterministic(syn_cfg):
    tr = _trainer(syn_cfg(method="central", model="csdi", sampler="ddpm", beta_schedule="quad", beta_end=0.5))
    es = eval_set(tr.data, 16, 3, torch.device("cpu"))
    assert diffusion_test_loss(tr, es, 5) == diffusion_test_loss(tr, es, 5)


def test_time_split_embargo_removes_target_overlap():
    cfg = Config(split="time", val_ratio=0.1, test_ratio=0.2, future_len=64, past_len=64).validate()
    gap = split_gap(cfg, {"preprocessing": {"stride": 8}})
    assert gap == 7
    tr, va, te = _split_indices(110, cfg, np.random.default_rng(0), gap)
    stride, L, P = 8, 128, 64
    last_train_end = tr.max() * stride + L
    first_heldout_target = min(va.min(), te.min()) * stride + P
    assert last_train_end <= first_heldout_target
    assert split_gap(cfg.with_updates(split="random"), {"preprocessing": {"stride": 8}}) == 0
    assert split_gap(cfg.with_updates(split_gap=0), {"preprocessing": {"stride": 8}}) == 0


def test_tuned_groups_bring_their_tuning_runs():
    specs = campaign(groups=("main",), seeds=(1,), datasets=("synthetic",))
    assert any("tune" in s.groups for s in specs)
    assert not any("tune" in s.groups for s in campaign(groups=("main",), tuned=False, seeds=(1,),
                                                        datasets=("synthetic",)))


def test_manifest_merges_staged_launches(tmp_path, data_root):
    a = campaign(groups=("main",), seeds=(1,), datasets=("synthetic",))
    b = campaign(groups=("ablation",), seeds=(1,), datasets=("synthetic",))
    write_manifest(tmp_path, a, {"data_root": str(data_root)})
    write_manifest(tmp_path, b, {"data_root": str(data_root)})
    specs = specs_from_manifest(tmp_path)
    ids = {s.task_id for s in specs}
    assert {s.task_id for s in a} <= ids and {s.task_id for s in b} <= ids
    shared = [s for s in specs if {"main", "ablation"} <= s.groups]
    assert shared, "the full model is in both groups"
    build_plan(tmp_path, specs, str(data_root))  # ids reproduce from the manifest


def test_forbidden_gpu_is_refused(monkeypatch):
    from fedtshdiff.cli import _checked_slots

    ns = argparse.Namespace(gpus="0,1", slots_per_gpu=1, forbid_gpus="0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        _checked_slots(ns)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
    ok = _checked_slots(argparse.Namespace(gpus="0,1", slots_per_gpu=1, forbid_gpus="0"))
    assert [s.device for s in ok] == ["1", "2"]
    with pytest.raises(SystemExit):  # auto without a mask would expose everything
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
        _checked_slots(argparse.Namespace(gpus="auto", slots_per_gpu=1, forbid_gpus="0"))


def test_scheduler_exits_after_failure_with_continue_on_error(tmp_path, data_root):
    """A failed task stays 'ready' in the plan; the scheduler must still terminate."""
    from fedtshdiff.orchestrate.scheduler import Running, Scheduler, Slot

    specs = campaign(groups=("generative",), seeds=(1,), datasets=("synthetic",))[:1]

    class FailedProc:
        pid = 999999

        def poll(self):
            return 1

    sched = Scheduler(tmp_path, specs, str(data_root), [Slot("cpu.0", "")], continue_on_error=True, poll=0.01)

    def fake_launch(task, slots):
        task.out_dir.mkdir(parents=True, exist_ok=True)
        sched.running[task.task_id] = Running(task, slots, FailedProc(), 0.0, task.out_dir / "worker.log",
                                              tmp_path / "t.json")

    sched.launch = fake_launch
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("state", sched._loop()), daemon=True)
    th.start()
    th.join(timeout=30)
    assert not th.is_alive(), "scheduler did not terminate"
    assert out["state"] == "failed"


def test_later_invocation_supersedes_same_label(tmp_path, data_root):
    tuned = campaign(groups=("main",), seeds=(1,), datasets=("synthetic",))
    untuned = campaign(groups=("main",), seeds=(1,), datasets=("synthetic",), tuned=False)
    write_manifest(tmp_path, tuned, {"data_root": str(data_root)})
    write_manifest(tmp_path, untuned, {"data_root": str(data_root)})
    specs = specs_from_manifest(tmp_path)
    keys = [(s.dataset, s.seed, s.label) for s in specs]
    assert len(keys) == len(set(keys)), "superseded tasks must be dropped"
    by_label = {s.label: s for s in specs}
    assert by_label["fedprox"].overrides.get("prox_mu") == 0.01 and not by_label["fedprox"].select
    assert any(s.group == "tune" for s in specs)  # tune runs of the first invocation are kept


def test_analysis_without_prepared_data(tmp_path, data_root):
    """Exported results analysed on a machine without the data: fall back to the runs' data hash."""
    from fedtshdiff.analysis.collect import collect
    from fedtshdiff.fl.runner import run_experiment

    specs = [s for s in campaign(groups=("generative",), seeds=(1,), datasets=("synthetic",), tuned=False)
             if s.label == "fedavg_lstm"]
    write_manifest(tmp_path, specs, {"data_root": str(data_root)})
    task = build_plan(tmp_path, specs, str(data_root)).tasks[0]
    run_experiment(task.config.with_updates(rounds=1, device="cpu"), task.out_dir)
    # rounds changed -> fingerprint differs from the plan: nothing matches and that is an error
    with pytest.raises(ValueError):
        collect(tmp_path, tmp_path / "no-data-here")
    run_experiment(task.config.with_updates(device="cpu"), task.out_dir)
    res = collect(tmp_path, tmp_path / "no-data-here")
    assert len(res.runs) == 1 and res.runs["status"].iloc[0] == "complete"
