"""Run one experiment end to end and write its outputs.

``<out>/``
    resolved_config.yaml   the validated configuration
    meta.json              fingerprint, data hash, code digest, environment, data summary
    rounds.csv             per-round training log (loss, dispersion, bytes, timings)
    curve.csv              periodic evaluation on the fixed curve subset
    diagnostics.jsonl      gradient-dispersion diagnostics (when ``diag_rounds`` is set)
    final_metrics.json     final evaluation (default DDIM steps, plus ``eval_extra_ddim``)
    per_window.npz         per-window final metrics (for window-level paired tests)
    cost.json              communication and computation cost summary
    model.pt               final parameters (global / per-cluster / personal heads)
    progress.json          live progress for the scheduler
    done.json              written last: status complete|diverged + fingerprint (the seal)

Partial outputs of an interrupted run are removed when the run is restarted.
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Config, save_config
from ..data.drift import apply_event, drift_plan
from ..data.federated import build_federated, eval_set
from ..eval.evaluate import diffusion_test_loss, evaluate
from ..models.forecaster import build_forecaster
from ..utils import (atomic_write_json, append_jsonl, code_digest, derive_seed, read_json, resolve_device,
                     runtime_record, seed_everything, set_numerics, sha256_file)
from .diagnostics import run_diagnostics
from .trainer import Diverged, FederatedTrainer

# Bump when a change alters scientific results of an unchanged configuration.
SCIENCE_VERSION = 1
OUTPUT_FILES = ("rounds.csv", "curve.csv", "diagnostics.jsonl", "final_metrics.json", "per_window.npz",
                "cost.json", "model.pt", "progress.json", "done.json", "meta.json", "resolved_config.yaml")
PER_WINDOW_KEYS = ("dtw_mean", "dtw_point", "psd_l2", "div_dtw", "crps", "wis_90", "wis_multi", "picp_90",
                   "mpiw_90", "mse", "mae")


def run_fingerprint(cfg: Config, data_hash: str) -> str:
    return cfg.fingerprint({"data_hash": data_hash, "science_version": SCIENCE_VERSION})


def is_sealed(out_dir: Path, fingerprint: str) -> bool:
    p = Path(out_dir) / "done.json"
    if not p.exists():
        return False
    try:
        done = read_json(p)
    except (OSError, ValueError):
        return False
    if done.get("fingerprint") != fingerprint or done.get("status") not in ("complete", "diverged"):
        return False
    if done.get("status") == "complete":
        fm = Path(out_dir) / "final_metrics.json"
        return fm.exists() and sha256_file(fm) == done.get("final_metrics_sha256")
    return True


class _CSV:
    def __init__(self, path: Path):
        self.path, self.fields = path, None

    def write(self, row: dict[str, Any]) -> None:
        flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        new = self.fields is None
        if new:
            self.fields = list(flat)
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(flat)


def _clean(out: Path) -> None:
    for name in OUTPUT_FILES:
        p = out / name
        if p.exists():
            p.unlink()


class FingerprintMismatch(RuntimeError):
    pass


def run_experiment(cfg: Config, out_dir: str | Path, *, prepared=None, data_hash: str | None = None,
                   expected_fingerprint: str | None = None) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clean(out)
    set_numerics()
    device = resolve_device(cfg.device)
    t_start = time.time()

    data = build_federated(cfg, prepared, data_hash)
    fingerprint = run_fingerprint(cfg, data.data_hash)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise FingerprintMismatch(
            f"worker fingerprint {fingerprint[:12]} != planner fingerprint {expected_fingerprint[:12]}: the code or "
            "data changed under a running scheduler; restart the campaign with the frozen release")
    data.to_device(device)
    save_config(cfg, out / "resolved_config.yaml")
    meta = {"fingerprint": fingerprint, "data_hash": data.data_hash, "science_version": SCIENCE_VERSION,
            "code_digest": code_digest(), "started": t_start, "runtime": runtime_record(),
            "data": data.summary()}
    atomic_write_json(out / "meta.json", meta)

    seed_everything(cfg.seed)  # identical model initialisation for every method with this seed
    fore = build_forecaster(cfg, device)
    trainer = FederatedTrainer(cfg, data, fore, device)
    curve_set = eval_set(data, cfg.eval_curve_windows, cfg.eval_seed, device, cfg.eval_split)
    events = drift_plan(cfg, data.n_clients)
    rounds_csv, curve_csv = _CSV(out / "rounds.csv"), _CSV(out / "curve.csv")
    status, fail_round, fail_msg = "complete", None, None
    peak = 0

    def progress(r: int, phase: str) -> None:
        try:  # best effort: monitoring must never fail a run
            atomic_write_json(out / "progress.json", {"round": r, "rounds": cfg.rounds, "phase": phase,
                                                      "elapsed": time.time() - t_start, "time": time.time()})
        except OSError:
            pass

    def curve_eval(r: int) -> None:
        nonlocal curve_set
        t0 = time.perf_counter()
        summ, _ = evaluate(trainer, curve_set, cfg.eval_curve_samples, None, derive_seed(cfg.eval_seed, "curve"))
        loss = diffusion_test_loss(trainer, curve_set, derive_seed(cfg.eval_seed, "loss"))
        row = {"round": r, "test_loss": loss, "eval_seconds": time.perf_counter() - t0}
        row.update({k: v for k, v in summ.items() if not isinstance(v, list)})
        row["coverage_levels"] = summ["coverage_levels"]
        curve_csv.write(row)

    try:
        progress(0, "eval")
        curve_eval(0)
        for r in range(1, cfg.rounds + 1):
            for idx, ev in enumerate(events):
                if ev.round == r:
                    apply_event(data, ev, cfg, idx, device)
                    if trainer.grouper is not None:
                        trainer.refresh_descriptors()
                    curve_set = eval_set(data, cfg.eval_curve_windows, cfg.eval_seed, device, cfg.eval_split)
            if r in cfg.diag_rounds:
                progress(r, "diagnostics")
                t_diag = time.perf_counter()
                diag = run_diagnostics(trainer, r)
                diag["seconds"] = time.perf_counter() - t_diag
                append_jsonl(out / "diagnostics.jsonl", diag)
            progress(r, "train")
            t0 = time.perf_counter()
            log = trainer.run_round(r)
            timings = trainer.watch.pop()
            row = {"round": r, "train_loss": log.train_loss, "participants": log.participants,
                   "round_seconds": time.perf_counter() - t0}
            row.update({f"t_{k}": v for k, v in timings.items()})
            row.update(log.stats)
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))
            rounds_csv.write(row)
            if r % cfg.eval_every == 0 or r == cfg.rounds:
                progress(r, "eval")
                curve_eval(r)
        if cfg.rounds + 1 in cfg.diag_rounds:  # diagnostics at the final model
            append_jsonl(out / "diagnostics.jsonl", run_diagnostics(trainer, cfg.rounds + 1))
    except Diverged as exc:
        status, fail_round, fail_msg = "diverged", r, str(exc)

    result: dict[str, Any] = {"status": status}
    if status == "complete":
        progress(cfg.rounds, "final")
        final_set = eval_set(data, cfg.eval_final_windows, cfg.eval_seed, device, cfg.eval_split)
        final: dict[str, Any] = {}
        summ, arrays = evaluate(trainer, final_set, cfg.eval_final_samples, None, derive_seed(cfg.eval_seed, "final"))
        final["default"] = summ
        np.savez_compressed(out / "per_window.npz", client_index=final_set.client_index,
                            window_index=final_set.window_index,
                            **{k: arrays[k] for k in PER_WINDOW_KEYS})
        for steps in cfg.eval_extra_ddim:
            if fore.kind == "diffusion" and cfg.sampler == "ddim" and steps != cfg.ddim_steps:
                s2, _ = evaluate(trainer, final_set, cfg.eval_final_samples, steps,
                                 derive_seed(cfg.eval_seed, "final"))
                final[f"ddim_{steps}"] = s2
        final["test_loss"] = diffusion_test_loss(trainer, final_set, derive_seed(cfg.eval_seed, "loss"))
        atomic_write_json(out / "final_metrics.json", final)
        if cfg.save_checkpoint:
            torch.save(trainer.checkpoint(), out / "model.pt")
        result["final"] = final["default"]
    _write_cost(out, cfg, trainer, peak)
    done = {"status": status, "fingerprint": fingerprint, "elapsed": time.time() - t_start,
            "finished": time.time(), "diverged_round": fail_round, "message": fail_msg}
    if status == "complete":
        done["final_metrics_sha256"] = sha256_file(out / "final_metrics.json")
    atomic_write_json(out / "done.json", done)
    return result


def _write_cost(out: Path, cfg: Config, trainer: FederatedTrainer, peak: int) -> None:
    import pandas as pd

    path = out / "rounds.csv"
    cost: dict[str, Any] = {"params": trainer.fm.numel, "param_bytes": trainer.param_bytes,
                            "peak_gpu_bytes": int(peak), "method": cfg.method}
    if path.exists():
        df = pd.read_csv(path)
        for col in df.columns:
            if col.startswith(("t_", "bytes_", "local_seconds", "round_seconds")):
                cost[f"mean_{col}"] = float(df[col].mean())
                cost[f"total_{col}"] = float(df[col].sum())
        cost["rounds_completed"] = int(len(df))
    curve = out / "curve.csv"
    if curve.exists():
        dfc = pd.read_csv(curve)
        if "sample_seconds_per_window" in dfc:
            cost["inference_seconds_per_window_curve"] = float(dfc["sample_seconds_per_window"].iloc[-1])
    fm = out / "final_metrics.json"
    if fm.exists():
        final = read_json(fm)
        cost["inference"] = {k: {"steps": v.get("ddim_steps"), "seconds_per_window": v.get("sample_seconds_per_window"),
                                 "samples_per_window": v.get("samples_per_window")}
                             for k, v in final.items() if isinstance(v, dict)}
    atomic_write_json(out / "cost.json", cost)


def run_safely(cfg: Config, out_dir: str | Path, expected_fingerprint: str | None = None) -> int:
    """Worker entry: returns a process exit code; failures leave a traceback in failure.txt."""
    out = Path(out_dir)
    try:
        run_experiment(cfg, out, expected_fingerprint=expected_fingerprint)
        return 0
    except Exception:  # noqa: BLE001 - recorded for the scheduler
        out.mkdir(parents=True, exist_ok=True)
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1
