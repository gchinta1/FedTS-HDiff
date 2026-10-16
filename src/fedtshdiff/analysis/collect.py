"""Collect sealed run outputs of a campaign root into tidy DataFrames."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import read_json

CONFIG_COLS = ("method", "model", "grouping", "descriptor", "num_edges", "edge_rounds", "regroup_every",
               "max_clients", "participation", "beta_agg", "alpha_agg", "prox_mu", "dyn_alpha", "fedtdd_lambda",
               "rounds", "split", "ddim_spacing", "eval_split", "drift_rounds")


@dataclass
class Results:
    runs: pd.DataFrame  # one row per (task, group, label), final default metrics
    ddim: pd.DataFrame  # one row per (task, ddim variant)
    curves: pd.DataFrame
    rounds: pd.DataFrame  # per-run means of the round log
    diagnostics: list[dict[str, Any]]
    root: Path

    def group(self, name: str) -> pd.DataFrame:
        return self.runs[self.runs["group"] == name]


def _task_rows(root: Path, data_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Tasks of the campaign whose sealed outputs are *current*: the seal's fingerprint must equal
    the fingerprint the planner computes now (configuration, selected hyperparameters, data hash,
    science version), so superseded results of rerun tasks never enter the tables."""
    from ..orchestrate.planner import build_plan, specs_from_manifest

    manifest = read_json(root / "campaign.json")
    data_root = str(data_root or manifest["data_root"])
    # validate against the data hash recorded by the sealed runs (local data may be prepared differently)
    fallback: dict[str, set[str]] = {}
    for t in manifest["tasks"]:
        meta = root / "runs" / t["task_id"] / "meta.json"
        if (root / "runs" / t["task_id"] / "done.json").exists() and meta.exists():
            fallback.setdefault(t["dataset"], set()).add(read_json(meta)["data_hash"])
    ambiguous = {ds: h for ds, h in fallback.items() if len(h) > 1}
    if ambiguous:
        raise ValueError(f"runs of one dataset were produced from different prepared data: {ambiguous}")
    plan = build_plan(root, specs_from_manifest(root), data_root, {ds: h.pop() for ds, h in fallback.items()})
    done = {t.task_id for t in plan.by_state("done")}
    rows = [t for t in manifest["tasks"] if t["task_id"] in done]
    sealed = [t for t in manifest["tasks"] if (root / "runs" / t["task_id"] / "done.json").exists()]
    if sealed and not rows:
        raise ValueError(f"{len(sealed)} sealed runs but none matches the current plan (code or data changed?)")
    keys = [(g, t["label"], t["dataset"], t["seed"]) for t in rows for g in t["groups"]]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"duplicate results for {sorted(dup)[:5]}: campaign.json lists superseded tasks")
    return rows


def collect(root: str | Path, data_root: str | Path | None = None) -> Results:
    root = Path(root)
    rows, ddim_rows, curves, rounds, diags = [], [], [], [], []
    for t in _task_rows(root, data_root):
        run = root / "runs" / t["task_id"]
        done_p = run / "done.json"
        done = read_json(done_p)
        cfg = {}
        if (run / "resolved_config.yaml").exists():
            import yaml

            cfg = yaml.safe_load((run / "resolved_config.yaml").read_text(encoding="utf-8"))
        base = {"task_id": t["task_id"], "dataset": t["dataset"], "seed": t["seed"], "status": done["status"],
                "elapsed": done.get("elapsed")}
        base.update({k: (json.dumps(cfg.get(k)) if isinstance(cfg.get(k), list) else cfg.get(k)) for k in CONFIG_COLS})
        metrics: dict[str, Any] = {}
        if done["status"] == "complete" and (run / "final_metrics.json").exists():
            final = read_json(run / "final_metrics.json")
            for k, v in final["default"].items():
                if isinstance(v, list):
                    metrics[k] = json.dumps(v)
                else:
                    metrics[k] = v
            metrics["test_loss"] = final.get("test_loss")
            for key, v in final.items():
                if isinstance(v, dict):
                    ddim_rows.append({**base, "labels": ",".join(t["labels"]), "variant": key, "ddim_steps": v.get("ddim_steps"),
                                      **{m: v[m] for m in ("dtw_mean", "psd_l2", "div_dtw", "crps", "wis_90",
                                                           "picp_90", "mpiw_90", "calib_error",
                                                           "sample_seconds_per_window") if m in v}})
        cost = read_json(run / "cost.json") if (run / "cost.json").exists() else {}
        metrics.update({f"cost_{k}": v for k, v in cost.items() if isinstance(v, (int, float))})
        for g in t["groups"]:
            for lab in t["labels"]:
                rows.append({**base, "group": g, "label": lab, **metrics})
        if (run / "curve.csv").exists():
            c = pd.read_csv(run / "curve.csv")
            c["task_id"], c["dataset"], c["seed"] = t["task_id"], t["dataset"], t["seed"]
            c["labels"] = ",".join(t["labels"])
            c["groups"] = ",".join(t["groups"])
            curves.append(c)
        if (run / "rounds.csv").exists():
            r = pd.read_csv(run / "rounds.csv")
            num = r.select_dtypes(include=[np.number]).mean().to_dict()
            rounds.append({"task_id": t["task_id"], "dataset": t["dataset"], "seed": t["seed"],
                           "labels": ",".join(t["labels"]), "groups": ",".join(t["groups"]),
                           **{f"mean_{k}": v for k, v in num.items()}})
        if (run / "diagnostics.jsonl").exists():
            for line in (run / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    d.update(task_id=t["task_id"], dataset=t["dataset"], seed=t["seed"], labels=t["labels"],
                             method=cfg.get("method"))
                    diags.append(d)
    return Results(pd.DataFrame(rows), pd.DataFrame(ddim_rows),
                   pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
                   pd.DataFrame(rounds), diags, root)


def per_window(root: Path, task_id: str) -> dict[str, np.ndarray] | None:
    p = Path(root) / "runs" / task_id / "per_window.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}
