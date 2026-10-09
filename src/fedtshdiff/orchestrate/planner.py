"""Turn campaign task specs into a plan against the files on disk.

A task is *done* when its run directory holds a seal (``done.json``) whose fingerprint
matches the fully resolved configuration (including selected hyperparameters and the
prepared-data hash).  A task that needs a validation-selected hyperparameter is
*waiting* until every tuning run of that (dataset, method) is sealed; the selection is
then computed deterministically and written to ``selection/<id>.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, build_config
from ..data.store import data_hash, dataset_dir
from ..fl.runner import is_sealed, run_fingerprint
from ..utils import atomic_write_json, read_json
from .suites import GROUPS, PAPER_DEFAULTS, SELECT_METRIC, TUNE_SEED, TUNING_GRID, TaskSpec

GROUP_PRIORITY = {"cost": 0, "tune": 1, "main": 2, "ablation": 3, "descriptors": 4, "drift": 5, "generative": 6,
                  "scalability": 7, "reproduce": 8, "smoke": 0}


@dataclass
class PlannedTask:
    spec: TaskSpec
    config: Config | None
    out_dir: Path
    fingerprint: str | None
    state: str  # done | ready | waiting
    waiting_on: list[str] = field(default_factory=list)

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    @property
    def priority(self) -> tuple[int, float]:
        g = min(GROUP_PRIORITY.get(x, 9) for x in (self.spec.groups or {self.spec.group}))
        return g, -self.cost_hint

    @property
    def cost_hint(self) -> float:
        """Relative run cost: rounds x local epochs x edge rounds x client count x model factor."""
        c = self.config
        if c is None:
            return 1.0
        clients = c.max_clients or 50
        factor = {"unet": 1.0, "csdi": 2.0, "lstm": 0.5, "gan": 0.6}[c.model]
        if c.method == "fedtdd":
            factor *= 1.5
        return c.rounds * c.local_epochs * c.edge_rounds * clients * c.participation * factor


@dataclass
class Plan:
    tasks: list[PlannedTask]

    def by_state(self, state: str) -> list[PlannedTask]:
        return [t for t in self.tasks if t.state == state]

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for t in self.tasks:
            for g in sorted(t.spec.groups or {t.spec.group}):
                c = out.setdefault(g, {"done": 0, "ready": 0, "waiting": 0})
                c[t.state] += 1
        return out


def selection_path(root: Path, sel_id: str) -> Path:
    return root / "selection" / f"{sel_id}.json"


def _tune_run_dirs(root: Path, specs: list[TaskSpec], dataset: str, method: str) -> list[tuple[float, TaskSpec]]:
    key = TUNING_GRID[method][0]
    out = []
    for sp in specs:
        if "tune" in (sp.groups or {sp.group}) and sp.dataset == dataset and sp.overrides.get("method") == method:
            out.append((float(sp.overrides[key]), sp))
    return out


def try_select(root: Path, specs: list[TaskSpec], dataset: str, method: str, fingerprints: dict[str, str]) -> dict | None:
    """Selection = argmin validation CRPS over the method's grid (ties -> paper default)."""
    sel_id = f"{dataset}__{method}"
    path = selection_path(root, sel_id)
    cands = _tune_run_dirs(root, specs, dataset, method)
    if not cands:
        return None
    scores: dict[float, float] = {}
    for value, sp in cands:
        run = root / "runs" / sp.task_id
        fp = fingerprints.get(sp.task_id)
        if fp is None or not is_sealed(run, fp):
            return None
        done = read_json(run / "done.json")
        if done["status"] == "diverged":
            scores[value] = math.inf
        else:
            scores[value] = float(read_json(run / "final_metrics.json")["default"][SELECT_METRIC])
    key = TUNING_GRID[method][0]
    best = min(scores.values())
    winners = [v for v, s in scores.items() if s == best]
    value = PAPER_DEFAULTS[key] if PAPER_DEFAULTS[key] in winners else min(winners)
    rec = {"selection": sel_id, "key": key, "value": value, "metric": f"validation {SELECT_METRIC}",
           "scores": {str(k): v for k, v in sorted(scores.items())}, "seed": TUNE_SEED,
           "runs": {str(v): sp.task_id for v, sp in cands}}
    if not path.exists() or read_json(path) != rec:
        atomic_write_json(path, rec)
    return rec


def resolve_config(spec: TaskSpec, data_root: str, selections: dict[str, Any]) -> Config:
    ov = dict(spec.overrides)
    ov["data_root"] = data_root
    ov["seed"] = spec.seed
    ov["name"] = spec.task_id
    ov["tags"] = sorted(spec.groups or {spec.group})
    for key, sel_id in spec.select.items():
        ov[key] = selections[sel_id]["value"]
    return build_config(dataset=spec.dataset, overrides=ov)


def build_plan(root: Path, specs: list[TaskSpec], data_root: str) -> Plan:
    root = Path(root)
    hashes: dict[str, str] = {}
    for ds in sorted({s.dataset for s in specs}):
        if (dataset_dir(data_root, ds) / "SHA256SUMS").exists():
            hashes[ds] = data_hash(data_root, ds)
    planned: list[PlannedTask] = []
    fingerprints: dict[str, str] = {}
    # first pass: tasks without selections (tuning runs included)
    for sp in specs:
        if sp.select:
            continue
        cfg = resolve_config(sp, data_root, {})
        fp = run_fingerprint(cfg, hashes[sp.dataset]) if sp.dataset in hashes else None
        fingerprints[sp.task_id] = fp
        out = root / "runs" / sp.task_id
        state = "done" if fp and is_sealed(out, fp) else ("ready" if fp else "waiting")
        planned.append(PlannedTask(sp, cfg, out, fp, state, [] if fp else [f"data:{sp.dataset}"]))
    selections: dict[str, Any] = {}
    for sp in specs:
        for sel_id in sp.select.values():
            if sel_id not in selections:
                ds, method = sel_id.split("__", 1)
                rec = try_select(root, specs, ds, method, fingerprints)
                if rec is not None:
                    selections[sel_id] = rec
    for sp in specs:
        if not sp.select:
            continue
        missing = [s for s in sp.select.values() if s not in selections]
        out = root / "runs" / sp.task_id
        if missing or sp.dataset not in hashes:
            planned.append(PlannedTask(sp, None, out, None, "waiting",
                                       [f"selection:{m}" for m in missing] or [f"data:{sp.dataset}"]))
            continue
        cfg = resolve_config(sp, data_root, selections)
        fp = run_fingerprint(cfg, hashes[sp.dataset])
        planned.append(PlannedTask(sp, cfg, out, fp, "done" if is_sealed(out, fp) else "ready"))
    planned.sort(key=lambda t: t.priority)
    return Plan(planned)


def write_manifest(root: Path, specs: list[TaskSpec], meta: dict[str, Any]) -> None:
    atomic_write_json(Path(root) / "campaign.json", {
        **meta,
        "groups": sorted({g for s in specs for g in (s.groups or {s.group})}, key=lambda g: GROUPS.index(g) if g in GROUPS else 99),
        "tasks": [{"task_id": s.task_id, "dataset": s.dataset, "seed": s.seed, "groups": sorted(s.groups),
                   "labels": sorted(s.labels), "overrides": s.overrides, "select": s.select,
                   "exclusive": s.exclusive} for s in specs],
    })
