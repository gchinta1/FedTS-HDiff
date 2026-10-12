"""``fedts analyse``: rebuild every table, figure and the report from stored results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import atomic_write_json, read_json
from . import figures, tables
from .collect import collect, per_window
from .stats import METRICS, friedman, seed_summary, window_level_test

ABLATION_LABELS = ["fedts_hdiff", "group_std_agg", "adaptive_only", "random_group", "round_robin", "static_group",
                   "similarity_group", "popmean_ref", "edge_rounds2", "fedavg"]
REVIEWER_MAP = [
    ("R1-C1 / R2-C6 cost and overhead", ["tab_cost.tex", "tab_ddim.tex"]),
    ("R1-C3 / R2-C5 multi-seed results and significance", ["tab_main_ecg.tex", "tab_main_other.tex",
                                                           "tab_main_ecg_tests.csv", "friedman.json",
                                                           "window_level_tests.csv"]),
    ("R1-C5 diffusion vs GAN/LSTM", ["tab_generative.tex"]),
    ("R2-C2 label-free descriptors", ["tab_descriptors.tex"]),
    ("R2-C3/C4 similarity vs gradient dispersion", ["tab_dispersion.tex", "tab_descriptor_gradient_corr.csv",
                                                    "fig_dispersion.pdf"]),
    ("R2-C7 partition details", ["tab_data.tex"]),
    ("R2-C8 ablations on all datasets", ["tab_ablation.tex"]),
    ("R2-C9 uncertainty calibration", ["tab_calibration.tex", "fig_reliability.pdf", "fig_pit.pdf"]),
    ("R2-C10 scalability", ["tab_scalability.tex", "fig_scalability.pdf"]),
    ("R2-C11 configurations", ["selections.csv", "configs/"]),
    ("Sec. 5.8 concept drift", ["tab_drift.tex", "fig_drift.pdf"]),
]


def _with_reference_rows(res, group: str, ref_labels: list[str]) -> pd.DataFrame:
    """Rows of ``group`` plus rows of the main group for ``ref_labels`` (same seeds)."""
    g = res.runs[res.runs["group"] == group]
    m = res.runs[(res.runs["group"] == "main") & res.runs["label"].isin(ref_labels)]
    m = m[m["dataset"].isin(set(g["dataset"]))]
    return pd.concat([g, m], ignore_index=True).drop_duplicates(subset=["task_id", "label"])


def window_tests(res, out: Path, reference: str = "fedts_hdiff", metrics=("dtw_mean", "crps", "wis_90")) -> pd.DataFrame:
    rows = []
    main = res.runs[(res.runs["group"] == "main") & (res.runs["status"] == "complete")]
    for ds, g in main.groupby("dataset"):
        ref = g[g["label"] == reference].set_index("seed")["task_id"]
        for lab, h in g.groupby("label"):
            if lab == reference:
                continue
            other = h.set_index("seed")["task_id"]
            seeds = sorted(set(ref.index) & set(other.index))
            if not seeds:
                continue
            A = [per_window(res.root, other[s]) for s in seeds]
            B = [per_window(res.root, ref[s]) for s in seeds]
            if any(x is None for x in A + B):
                continue
            for m in metrics:
                r = window_level_test(np.stack([a[m] for a in A]), np.stack([b[m] for b in B]), m)
                rows.append({"dataset": ds, "label": lab, "metric": m, "seeds": len(seeds), **r})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(out / "window_level_tests.csv", index=False)
    return t


def selections(root: Path, out: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((root / "selection").glob("*.json")):
        s = read_json(p)
        rows.append({"selection": s["selection"], "parameter": s["key"], "value": s["value"],
                     "scores": json.dumps(s["scores"])})
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(out / "selections.csv", index=False)
    return t


def analyse(root: str | Path, data_root: str | Path | None = None) -> Path:
    root = Path(root)
    out = root / "analysis"
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    res = collect(root, data_root)
    if res.runs.empty:
        (out / "report.md").write_text("# FedTS-HDiff revision analysis\n\nNo sealed runs yet.\n", encoding="utf-8")
        return out
    res.runs.to_csv(out / "runs.csv", index=False)
    if not res.curves.empty:
        res.curves.to_csv(out / "curves.csv.gz", index=False)
    main = res.runs[res.runs["group"] == "main"]
    notes: list[str] = []

    if len(main):
        tables.method_table(main, tables.MAIN_ORDER, ["mitbih", "ptbxl", "synthetic"], tables.MAIN_METRICS,
                            "Forecasting performance on MIT-BIH and PTB-XL (64/64), mean $\\pm$ SD over 5 seeds.",
                            "tab_main_ecg", tdir)
        tables.method_table(main, tables.MAIN_ORDER, ["electricity", "metrla"], tables.MAIN_METRICS,
                            "Forecasting performance on Electricity (96/96) and METR-LA (64/64), mean $\\pm$ SD over 5 seeds.",
                            "tab_main_other", tdir)
        tables.method_table(main, tables.MAIN_ORDER, list(tables.DATASET_NAMES),
                            ["mse", "mae", "dtw_point", "wis_multi"],
                            "Point-forecast errors of the ensemble mean and multi-level WIS.", "tab_main_point", tdir)
        summ = seed_summary(main[main["label"].isin(tables.MAIN_ORDER)], ["dataset", "label"], METRICS)
        fed = summ[~summ["label"].isin(tables.CENTRAL)]
        atomic_write_json(out / "friedman.json", [friedman(fed, m) for m in ("dtw_mean", "crps", "wis_90", "psd_l2")])
        window_tests(res, out)
        tables.calibration_table(res.runs, tdir, [l for l in tables.MAIN_ORDER])
        figures.convergence(res.curves, fdir)
        tables.convergence_table(res.curves, tdir)
        figures.reliability(res.runs, fdir, ["fedavg", "fedprox", "feddyn", "fedts_hdiff", "csdi"])
        figures.pit_histograms(res.runs, fdir, ["fedavg", "fedts_hdiff", "csdi"])
    if len(res.ddim):
        tables.ddim_table(res.ddim, tdir)
    if (res.runs["group"] == "ablation").any():
        tables.method_table(_with_reference_rows(res, "ablation", ["fedavg"]), ABLATION_LABELS,
                            list(tables.DATASET_NAMES), ["dtw_mean", "psd_l2", "crps", "wis_90", "picp_90", "mpiw_90"],
                            "Ablation of FedTS-HDiff on all datasets (mean $\\pm$ SD over 5 seeds). DDIM-step "
                            "variants are reported in Table~\\ref{tab:ddim}.", "tab_ablation", tdir)
    if (res.runs["group"] == "descriptors").any():
        tables.method_table(_with_reference_rows(res, "descriptors", ["fedavg", "fedts_hdiff"]),
                            ["fedavg", "fedts_hdiff", "fedts_hdiff_signal", "fedts_hdiff_psd"], ["mitbih", "ptbxl", "synthetic"],
                            ["dtw_mean", "crps", "wis_90", "picp_90"],
                            "Label-based vs.\\ label-free client descriptors on the clinical datasets.",
                            "tab_descriptors", tdir)
    if (res.runs["group"] == "drift").any():
        d = res.runs[res.runs["group"] == "drift"]
        tables.method_table(d, ["drift_fedavg", "drift_random", "drift_static", "drift_dynamic"],
                            ["metrla", "electricity", "synthetic"], ["dtw_mean", "psd_l2", "crps", "wis_90", "picp_90", "mpiw_90"],
                            "Concept drift: forecasting after two drift events affecting 30\\% of the clients.",
                            "tab_drift", tdir, reference="drift_dynamic")
        figures.drift_curves(res.curves, res.runs, fdir)
    if (res.runs["group"] == "generative").any():
        tables.method_table(_with_reference_rows(res, "generative", ["fedavg", "fedts_hdiff"]),
                            ["fedavg_lstm", "fedavg_gan", "fedavg", "fedts_hdiff"], list(tables.DATASET_NAMES),
                            ["dtw_mean", "crps", "wis_90", "picp_90", "mpiw_90", "div_dtw"],
                            "Diffusion vs.\\ non-diffusion probabilistic forecasters trained with FedAvg.",
                            "tab_generative", tdir)
    if (res.runs["group"] == "reproduce").any():
        tables.method_table(res.runs[res.runs["group"] == "reproduce"], ["repro_fedavg", "repro_fedts_hdiff"],
                            list(tables.DATASET_NAMES), tables.MAIN_METRICS,
                            "Submitted protocol (legacy DDIM spacing, original MIT-BIH forms, random METR-LA split).",
                            "tab_reproduce", tdir, reference="repro_fedts_hdiff")
    tables.cost_table(res.runs, tdir)
    tables.scalability_table(res.runs, tdir)
    figures.scalability(res.runs, fdir)
    long = tables.dispersion_table(res.diagnostics, tdir)
    figures.dispersion_bars(long, fdir)
    selections(root, tdir)
    if data_root is not None:
        from .datasets import dataset_table

        dataset_table(data_root, out=tdir)

    status = res.runs.drop_duplicates("task_id")["status"].value_counts().to_dict()
    lines = ["# FedTS-HDiff revision analysis", "",
             f"Sealed runs: {sum(status.values())} ({', '.join(f'{k}: {v}' for k, v in status.items())}).", "",
             "## Reviewer comment -> evidence", ""]
    for title, files in REVIEWER_MAP:
        present = [f for f in files if (tdir / f).exists() or (fdir / f).exists() or (out / f).exists()]
        lines.append(f"- **{title}**: " + (", ".join(f"`{f}`" for f in present) if present else "_not yet available_"))
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    lines += ["", "Tables are in `tables/` (LaTeX + CSV), figures in `figures/` (PDF + PNG).",
              "Significance: seed-paired t-tests, Holm-adjusted per dataset and metric (`*_tests.csv`), plus "
              "window-level Wilcoxon tests with bootstrap intervals (`window_level_tests.csv`).", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out
