"""The revision campaign: every experiment requested by the reviewers, as a task list.

Each task is one (dataset x method/variant x seed) run, described by overrides on top of
the dataset's paper configuration.  Identical configurations requested by several
groups (e.g. the full model in the main table and in the ablation) collapse into one
task, because the task id is derived from the configuration itself.

Groups (``--only`` selects a subset; order = scheduling priority)
------------------------------------------------------------------
tune         validation-set selection of each method's extra hyperparameter with the same
             grid size for every method (FedProx mu, FedDyn alpha, FedTDD lambda,
             FedTS-HDiff beta_agg); one selection seed, metric = validation CRPS  [R2-C5, fairness]
main         all methods x 4 datasets x 5 seeds, mean +- sd and significance tests,
             gradient diagnostics on FedAvg and FedTS-HDiff, DDIM-steps evaluation      [R1-C3, R2-C5]
ablation     the ablation variants on all four datasets                                   [R2-C8]
descriptors  label-free (signal / spectral) descriptors on MIT-BIH and PTB-XL             [R2-C2]
scalability  PTB-XL with N clients x K edges, edge rounds and partial participation      [R2-C10]
drift        concept drift on METR-LA and Electricity                                     [Sec. 5.8]
generative   non-diffusion probabilistic forecasters: Gaussian LSTM and WGAN-GP           [R1-C2, R1-C5]
cost         short exclusive-GPU runs of every method for clean timing measurements       [R1-C1, R2-C6]
reproduce    the submitted configuration (legacy DDIM spacing, MIT-BIH linear forms,
             METR-LA random split) for provenance
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable

DATASETS = ("mitbih", "ptbxl", "electricity", "metrla")
SEEDS = (1, 2, 3, 4, 5)
TUNE_SEED = 101
SELECT_METRIC = "crps"
GROUPS = ("tune", "main", "ablation", "descriptors", "scalability", "drift", "generative", "cost", "reproduce")
# Follow-up groups: never part of the default campaign; launch them explicitly with --only.
EXTRA_GROUPS = ("screen", "personal", "phsel")
ALL_GROUPS = GROUPS + EXTRA_GROUPS

# Extra hyperparameter of each method, tuned on validation with the same 3-point grid size.
TUNING_GRID: dict[str, tuple[str, tuple[float, ...]]] = {
    "fedprox": ("prox_mu", (0.001, 0.01, 0.1)),
    "feddyn": ("dyn_alpha", (0.001, 0.01, 0.1)),
    "fedtdd": ("fedtdd_lambda", (0.01, 0.1, 1.0)),
    "fedts_hdiff": ("beta_agg", (0.5, 1.0, 2.0)),
}
PAPER_DEFAULTS = {"prox_mu": 0.01, "dyn_alpha": 0.01, "fedtdd_lambda": 0.1, "beta_agg": 1.0}

CSDI = {"method": "central", "model": "csdi", "diffusion_steps": 50, "beta_schedule": "quad",
        "beta_start": 1e-4, "beta_end": 0.5, "sampler": "ddpm", "ddim_steps": 50}
MAIN_METHODS: dict[str, dict[str, Any]] = {
    "fedavg": {"method": "fedavg"},
    "fedprox": {"method": "fedprox"},
    "fednova": {"method": "fednova"},
    "feddyn": {"method": "feddyn"},
    "fedrep": {"method": "fedrep"},
    "fedtdd": {"method": "fedtdd"},
    "cfl": {"method": "cfl"},
    "fedts_hdiff": {"method": "fedts_hdiff"},
    "central_unet": {"method": "central"},
    "csdi": CSDI,
}
DIAG = {"diag_rounds": "@diag"}  # filled per dataset: [1, 25, R/2, R+1]
# Groups whose convergence curves no table/figure reads evaluate only at round 0 and R.
NO_CURVES = {"eval_every": "@rounds"}
DDIM_EXTRA = [25, 50, 100]

ABLATIONS: dict[str, dict[str, Any]] = {
    # FedAvg-equivalent under kappa = 1 (data-proportional weights at both levels)
    "group_std_agg": {"method": "fedts_hdiff", "beta_agg": 0.0, "alpha_agg": 1.0},
    "adaptive_only": {"method": "fedts_hdiff", "grouping": "none"},  # = w/o hierarchy, adaptive weights kept
    "random_group": {"method": "fedts_hdiff", "grouping": "random"},
    "round_robin": {"method": "fedts_hdiff", "grouping": "round_robin"},  # "w/o similarity"
    "static_group": {"method": "fedts_hdiff", "regroup_every": 0},
    "similarity_group": {"method": "fedts_hdiff", "grouping": "similarity"},
    "edge_rounds2": {"method": "fedts_hdiff", "edge_rounds": 2, "rounds": "@half_rounds"},
}

SCALE_N = (100, 200, 400, 800)
SCALE_K = (2, 4, 8, 16)
SCALE_Q = (0.1, 0.25, 0.5)
SCALE_SEEDS = (1, 2, 3)


@dataclass
class TaskSpec:
    group: str
    label: str  # human-readable variant name
    dataset: str
    seed: int
    overrides: dict[str, Any]
    exclusive: bool = False
    select: dict[str, str] = field(default_factory=dict)  # config key -> selection id
    groups: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)

    @property
    def task_id(self) -> str:
        blob = json.dumps({"d": self.dataset, "s": self.seed, "o": self.overrides, "sel": self.select,
                           "x": self.exclusive}, sort_keys=True, default=str)
        h = hashlib.sha1(blob.encode()).hexdigest()[:10]
        return f"{self.dataset}__{self.label}__s{self.seed}__{h}"


# The synthetic dataset stands in for every real dataset so the whole campaign can be tested on a laptop.
DESCRIPTOR_DATASETS = ("mitbih", "ptbxl", "synthetic")
DRIFT_DATASETS = ("metrla", "electricity", "synthetic")
SCALE_DATASETS = {"ptbxl": (SCALE_N, SCALE_K), "synthetic": ((6, 12), (2, 3))}


def _rounds(dataset: str) -> int:
    return {"mitbih": 200, "ptbxl": 200, "synthetic": 3}.get(dataset, 250)


def _resolve_macros(ds: str, ov: dict[str, Any]) -> dict[str, Any]:
    out = dict(ov)
    if out.get("rounds") == "@half_rounds":
        out["rounds"] = max(1, _rounds(ds) // 2)
    elif isinstance(out.get("rounds"), str) and out["rounds"].startswith("@div"):  # equal computation
        out["rounds"] = max(1, _rounds(ds) // int(out["rounds"][4:]))
    R = int(out.get("rounds", _rounds(ds)))  # the run's own round count, after halving/dividing
    for k, v in out.items():
        if v == "@diag":
            out[k] = sorted({1, min(25, R), max(1, R // 2), R + 1})
        elif v == "@rounds":
            out[k] = R
        elif v == "@tenth":
            out[k] = max(1, R // 10)
    return out


def _with_selection(spec: TaskSpec, tuned: bool) -> TaskSpec:
    """Attach the validation-selected hyperparameter of the spec's method (if tuned)."""
    method = spec.overrides.get("method")
    if method in TUNING_GRID and TUNING_GRID[method][0] not in spec.overrides:
        key = TUNING_GRID[method][0]
        if tuned:
            spec.select[key] = selection_id(spec.dataset, method)
        else:
            spec.overrides[key] = PAPER_DEFAULTS[key]
    return spec


def selection_id(dataset: str, method: str) -> str:
    return f"{dataset}__{method}"


def tune_specs(datasets: Iterable[str] = DATASETS) -> list[TaskSpec]:
    out = []
    for ds in datasets:
        for method, (key, grid) in TUNING_GRID.items():
            for v in grid:
                ov = {"method": method, key: v, "eval_split": "val", "eval_extra_ddim": [], **NO_CURVES}
                out.append(TaskSpec("tune", f"tune_{method}_{key}{v}", ds, TUNE_SEED, ov))
    return out


def campaign(groups: Iterable[str] = GROUPS, tuned: bool = True, seeds: Iterable[int] = SEEDS,
             datasets: Iterable[str] = DATASETS) -> list[TaskSpec]:
    groups, seeds, datasets = list(groups), tuple(seeds), tuple(datasets)
    specs: list[TaskSpec] = []
    if "tune" in groups and tuned:
        specs += tune_specs(datasets)
    for ds in datasets:
        if "main" in groups:
            for label, ov in MAIN_METHODS.items():
                extra = dict(ov)
                if label in ("fedavg", "fedts_hdiff"):
                    extra.update(DIAG)
                if label == "fedts_hdiff":
                    extra["eval_extra_ddim"] = DDIM_EXTRA
                for s in seeds:
                    specs.append(_with_selection(TaskSpec("main", label, ds, s, extra), tuned))
        if "ablation" in groups:
            for label, ov in ABLATIONS.items():
                for s in seeds:
                    specs.append(_with_selection(TaskSpec("ablation", label, ds, s, {**ov, **NO_CURVES}), tuned))
            if ds in DESCRIPTOR_DATASETS:  # labeled data: balance towards the population label mix
                for s in seeds:
                    specs.append(_with_selection(TaskSpec("ablation", "popmean_ref", ds, s,
                                                          {"method": "fedts_hdiff", "homogeneity_ref": "mean",
                                                           **NO_CURVES}), tuned))
            for s in seeds:  # the full model (deduplicated with main)
                specs.append(_with_selection(TaskSpec("ablation", "fedts_hdiff", ds, s,
                                                      {"method": "fedts_hdiff", **DIAG,
                                                       "eval_extra_ddim": DDIM_EXTRA}), tuned))
        if "descriptors" in groups and ds in DESCRIPTOR_DATASETS:
            for desc in ("signal", "psd"):
                for s in seeds:
                    specs.append(_with_selection(TaskSpec("descriptors", f"fedts_hdiff_{desc}", ds, s,
                                                          {"method": "fedts_hdiff", "descriptor": desc,
                                                           **NO_CURVES}), tuned))
        if "drift" in groups and ds in DRIFT_DATASETS:
            R = _rounds(ds)
            drift = {"drift_rounds": [int(R * 0.4), int(R * 0.7)], "drift_fraction": 0.3}
            for label, ov in {"drift_dynamic": {"method": "fedts_hdiff"},
                              "drift_static": {"method": "fedts_hdiff", "regroup_every": 0},
                              "drift_random": {"method": "fedts_hdiff", "grouping": "random"},
                              "drift_fedavg": {"method": "fedavg"}}.items():
                for s in seeds:
                    specs.append(_with_selection(TaskSpec("drift", label, ds, s, {**ov, **drift}), tuned))
        if "generative" in groups:
            for label, ov in {"fedavg_lstm": {"method": "fedavg", "model": "lstm"},
                              "fedavg_gan": {"method": "fedavg", "model": "gan"}}.items():
                for s in seeds:
                    specs.append(TaskSpec("generative", label, ds, s, dict(ov)))
        if "cost" in groups:
            for label, ov in MAIN_METHODS.items():
                # one client at a time, so per-client compute times are those of a single device
                cost = {**ov, "rounds": 10, "eval_every": 10,
                        "eval_extra_ddim": DDIM_EXTRA if label == "fedts_hdiff" else [],  # read by tab_ddim
                        "save_checkpoint": False, "client_batch": 1}
                specs.append(_with_selection(TaskSpec("cost", f"cost_{label}", ds, 1, cost, exclusive=True), False))
            specs.append(TaskSpec("cost", "cost_fedts_hdiff_k2", ds, 1,
                                  {"method": "fedts_hdiff", "edge_rounds": 2, "rounds": 10, "eval_every": 10,
                                   "save_checkpoint": False, "beta_agg": 1.0, "client_batch": 1}, exclusive=True))
        if "reproduce" in groups:
            legacy = {"ddim_spacing": "legacy", "val_ratio": 0.0, "split_gap": 0}
            if ds == "mitbih":
                legacy.update(select_form="linear", agg_form="linear", alpha_agg=20.0, beta_agg=0.0)
            if ds == "metrla":
                legacy.update(split="random")
            for label, ov in {"repro_fedavg": {"method": "fedavg"},
                              "repro_fedts_hdiff": {"method": "fedts_hdiff"}}.items():
                for s in seeds[:3]:
                    spec = TaskSpec("reproduce", label, ds, s, {**ov, **legacy, **NO_CURVES})
                    if "beta_agg" not in spec.overrides and spec.overrides["method"] == "fedts_hdiff":
                        spec.overrides["beta_agg"] = 1.0
                    specs.append(spec)
    for sds in [d for d in datasets if d in SCALE_DATASETS and "scalability" in groups]:
        Ns, Ks = SCALE_DATASETS[sds]
        n_part = Ns[len(Ns) // 2]
        scale_seeds = tuple(s for s in SCALE_SEEDS if s in seeds) or seeds[:1]
        for N in Ns:
            for s in scale_seeds:
                specs.append(_with_selection(TaskSpec("scalability", f"scale_fedavg_N{N}", sds, s,
                                                      {"method": "fedavg", "max_clients": N, **NO_CURVES}), tuned))
                for K in Ks:
                    specs.append(_with_selection(TaskSpec("scalability", f"scale_hdiff_N{N}_K{K}", sds, s,
                                                          {"method": "fedts_hdiff", "max_clients": N,
                                                           "num_edges": K, **NO_CURVES}), tuned))
                specs.append(_with_selection(TaskSpec("scalability", f"scale_hdiff_N{N}_K{Ks[1]}_kappa2", sds, s,
                                                      {"method": "fedts_hdiff", "max_clients": N, "num_edges": Ks[1],
                                                       "edge_rounds": 2, "rounds": "@half_rounds", **NO_CURVES}), tuned))
        for q in SCALE_Q:
            for s in scale_seeds:
                for label, ov in {"fedavg": {"method": "fedavg"}, "hdiff": {"method": "fedts_hdiff"}}.items():
                    specs.append(_with_selection(TaskSpec("scalability", f"scale_{label}_N{n_part}_q{q}", sds, s,
                                                          {**ov, "max_clients": n_part, "participation": q, **NO_CURVES}), tuned))
    if "screen" in groups:
        specs += screen_specs([d for d in datasets if d in SCREEN_DATASETS], seeds)
    if "personal" in groups:
        specs += personal_specs([d for d in datasets if d in PERSONAL_DATASETS], seeds)
    if "phsel" in groups:
        specs += phsel_specs([d for d in datasets if d in PERSONAL_DATASETS], seeds)
    if tuned:  # tasks that need a validation-selected value always bring their tuning runs along
        needed = {sel.split("__", 1)[0] for sp in specs for sel in sp.select.values()}
        have = {sp.dataset for sp in specs if sp.group == "tune"}
        specs += tune_specs(sorted(needed - have))
    for sp in specs:
        sp.overrides = _resolve_macros(sp.dataset, sp.overrides)
    return _dedupe(specs)


# ---------------------------------------------------------------------------------------------
# Screen (follow-up after round 1): grouping x edge rounds at equal computation, group
# personalisation, and FedAvg references at equal computation and at equal cloud traffic.
# ---------------------------------------------------------------------------------------------
# descriptor per dataset: PTB-XL clients carry a single label, so they are grouped by signal statistics
SCREEN_DATASETS: dict[str, dict[str, Any]] = {"mitbih": {}, "metrla": {}, "ptbxl": {"descriptor": "signal"},
                                              "synthetic": {}}
# beta_agg selected on validation CRPS in round 1 (results/revision-8385228cff1b/selection/*__fedts_hdiff.json)
SCREEN_BETA = {"mitbih": 2.0, "ptbxl": 0.5, "metrla": 0.5, "electricity": 0.5, "synthetic": 1.0}
SCREEN_GROUPINGS = ("balance", "similarity", "kmeans")
SCREEN_KAPPA = (1, 2, 5, 10)
# (edge_mix, kappa). With edge_mix = 0 the global model never feeds back into training, so kappa only
# changes the random draws: independent group models are run once, at kappa = 1.
SCREEN_PERS = ((0.0, 1), (0.5, 1), (0.5, 5))
SCREEN_SEEDS = (1, 2, 3)
CURVES_TENTH = {"eval_every": "@tenth"}  # 10 curve points per run (accuracy vs cloud traffic)


def screen_specs(datasets: Iterable[str], seeds: Iterable[int]) -> list[TaskSpec]:
    """Equal computation: kappa edge rounds per cloud round and R/kappa cloud rounds, so every
    configuration performs R x E local epochs per client (FedAvg: R rounds).  FedAvg with R/kappa
    rounds is the equal-cloud-traffic reference of hierarchical runs with kappa edge rounds and
    K edges (cloud traffic per round: K vs N models)."""
    seeds = tuple(s for s in SCREEN_SEEDS if s in tuple(seeds)) or tuple(seeds)[:1]
    out: list[TaskSpec] = []
    for ds in datasets:
        base = {"beta_agg": SCREEN_BETA[ds], **SCREEN_DATASETS[ds], **CURVES_TENTH}
        for s in seeds:
            out.append(TaskSpec("screen", "screen_fedavg", ds, s, {"method": "fedavg", **CURVES_TENTH}))
            for k in SCREEN_KAPPA[1:]:
                out.append(TaskSpec("screen", f"screen_fedavg_c{k}", ds, s,
                                    {"method": "fedavg", "rounds": f"@div{k}", **CURVES_TENTH}))
            for g in SCREEN_GROUPINGS:
                for k in SCREEN_KAPPA:
                    out.append(TaskSpec("screen", f"screen_{g}_k{k}", ds, s,
                                        {"method": "fedts_hdiff", "grouping": g, "edge_rounds": k,
                                         "rounds": f"@div{k}", **base}))
            for mix, k in SCREEN_PERS:
                out.append(TaskSpec("screen", f"screen_pers_m{mix}_k{k}", ds, s,
                                    {"method": "fedts_hdiff", "grouping": "similarity", "regroup_every": 0,
                                     "edge_mix": mix, "edge_rounds": k, "rounds": f"@div{k}", **base}))
    return out


# Personalised hierarchy: client output heads on top of a hierarchically aggregated body, against flat
# FedRep, clustered FL and FedAvg trained under the same protocol.  (Release d9ec7fd3a876 also ran group
# and mixed heads, ph_group_* / ph_mix_*; they were dropped, see docs/METHODS.md.)
PERSONAL_DATASETS: dict[str, dict[str, Any]] = {"mitbih": {}, "ptbxl": {"descriptor": "signal"}, "metrla": {},
                                                "electricity": {}, "synthetic": {}}
PERSONAL_KAPPA = (1, 5)


def _personal_base(ds: str, personal: bool) -> dict[str, Any]:
    """Static similarity groups, beta at its round-1 validation choice."""
    return {"method": "fedts_hdiff", "grouping": "similarity", "regroup_every": 0, "personal_head": personal,
            "beta_agg": SCREEN_BETA[ds], **PERSONAL_DATASETS[ds]}


def personal_specs(datasets: Iterable[str], seeds: Iterable[int]) -> list[TaskSpec]:
    """kappa edge rounds per cloud round for R/kappa cloud rounds (equal computation)."""
    out: list[TaskSpec] = []
    for ds in datasets:
        for s in seeds:
            for m in ("fedavg", "fedrep", "cfl"):
                out.append(TaskSpec("personal", f"ph_{m}", ds, s, {"method": m, **CURVES_TENTH}))
            for k in PERSONAL_KAPPA:
                out.append(TaskSpec("personal", f"ph_client_k{k}", ds, s,
                                    {**_personal_base(ds, True), "edge_rounds": k, "rounds": f"@div{k}",
                                     **CURVES_TENTH}))
    return out


# Validation switch for personal heads, the same hierarchy with a shared head, and the MIT-BIH ablation of
# client heads at kappa = 1.  The references and ph_client_* runs come from the `personal` campaign.
PHSEL_VAL_KAPPA = 5


def phsel_specs(datasets: Iterable[str], seeds: Iterable[int]) -> list[TaskSpec]:
    out: list[TaskSpec] = []
    for ds in datasets:
        for on in (True, False):  # validation runs: personal heads on / off (tuning seed, validation split)
            out.append(TaskSpec("phsel", f"phval_{'on' if on else 'off'}", ds, TUNE_SEED,
                                {**_personal_base(ds, on), "edge_rounds": PHSEL_VAL_KAPPA,
                                 "rounds": f"@div{PHSEL_VAL_KAPPA}", "eval_split": "val", "eval_extra_ddim": [],
                                 **NO_CURVES}))
        for s in seeds:
            for k in PERSONAL_KAPPA:
                out.append(TaskSpec("phsel", f"ph_off_k{k}", ds, s,
                                    {**_personal_base(ds, False), "edge_rounds": k, "rounds": f"@div{k}",
                                     **CURVES_TENTH}))
            if ds in ("mitbih", "synthetic"):  # what makes client heads beat FedRep on MIT-BIH (kappa = 1)?
                base = _personal_base(ds, True)
                for label, ov in {"pha_beta0": {"beta_agg": 0.0},  # data weights: algebraically FedRep
                                  "pha_nogroup": {"num_edges": 1},  # adaptive weights without grouping
                                  "pha_balance": {"grouping": "balance", "regroup_every": 1}}.items():
                    out.append(TaskSpec("phsel", label, ds, s, {**base, **ov, **CURVES_TENTH}))
    return out


def _dedupe(specs: list[TaskSpec]) -> list[TaskSpec]:
    by_id: dict[str, TaskSpec] = {}
    for sp in specs:
        tid = sp.task_id
        if tid in by_id:
            by_id[tid].groups.add(sp.group)
            by_id[tid].labels.add(sp.label)
        else:
            sp.groups.add(sp.group)
            sp.labels.add(sp.label)
            by_id[tid] = sp
    return list(by_id.values())


SMOKE_GROUPS = ("main", "ablation", "descriptors", "drift", "generative", "screen", "personal", "phsel")


def smoke_campaign(datasets: Iterable[str], groups: Iterable[str] = SMOKE_GROUPS) -> list[TaskSpec]:
    """Two-round versions of every variant of ``groups`` on real data (preflight)."""
    specs = campaign(groups=tuple(groups), tuned=False, seeds=(1,), datasets=datasets)
    out = []
    for sp in specs:
        ov = dict(sp.overrides)
        ov.update(rounds=2, eval_every=2, eval_curve_windows=32, eval_curve_samples=4, eval_final_windows=64,
                  eval_final_samples=8, save_checkpoint=False)
        if "diag_rounds" in ov:
            ov["diag_rounds"] = [1, 3]
        if "drift_rounds" in ov:
            ov["drift_rounds"] = [2]
        if ov.get("edge_rounds", 1) > 1:
            ov["rounds"] = 1
        out.append(TaskSpec("smoke", sp.label, sp.dataset, sp.seed, ov))
    return _dedupe(out)
