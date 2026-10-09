"""LaTeX (booktabs) and CSV tables for the revised manuscript.

Conventions: mean +- sample SD over seeds; the best *federated* value per dataset and metric
is bold (centralised upper bounds are excluded); a dagger marks a baseline that is
significantly worse than FedTS-HDiff and a double dagger one that is significantly better
(seed-paired t-test, Holm-adjusted over the baselines of that dataset and metric, p < 0.05).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .stats import METRICS, badness, paired_tests, seed_summary

DATASET_NAMES = {"mitbih": "MIT-BIH", "ptbxl": "PTB-XL", "electricity": "Electricity", "metrla": "METR-LA",
                 "synthetic": "Synthetic"}
METHOD_NAMES = {
    "fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova", "feddyn": "FedDyn", "fedrep": "FedRep",
    "fedtdd": "FedTDD$^\\ast$", "cfl": "Clustered FL", "central_unet": "Centralised (same U-Net)",
    "csdi": "CSDI (centralised)", "fedts_hdiff": "\\textbf{FedTS-HDiff (ours)}",
    "group_std_agg": "Grouping + standard agg. ($\\beta_{agg}{=}0$)", "adaptive_only": "Adaptive weights, no grouping",
    "random_group": "Random grouping", "round_robin": "w/o similarity (round-robin)",
    "static_group": "Static grouping", "similarity_group": "Similarity (clustered) grouping",
    "edge_rounds2": "$\\kappa{=}2$ edge rounds, $R/2$",
    "popmean_ref": "Balance towards population label mix",
    "fedts_hdiff_signal": "Signal descriptors", "fedts_hdiff_psd": "Spectral descriptors",
    "fedavg_lstm": "FedAvg + Gaussian LSTM", "fedavg_gan": "FedAvg + WGAN-GP",
    "drift_dynamic": "Dynamic grouping", "drift_static": "Static grouping", "drift_random": "Random grouping",
    "drift_fedavg": "FedAvg",
}
MAIN_ORDER = ["fedavg", "fedprox", "fednova", "feddyn", "fedrep", "fedtdd", "cfl", "csdi", "central_unet",
              "fedts_hdiff"]
CENTRAL = {"csdi", "central_unet"}
MAIN_METRICS = ["dtw_mean", "psd_l2", "div_dtw", "crps", "wis_90", "picp_90", "mpiw_90"]
REFERENCE = "fedts_hdiff"
NO_BEST = {"mpiw_90"}  # interval width is only meaningful together with coverage: never bolded


def _fmt(mean: float, sd: float, digits: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    s = f"{mean:.{digits}f}"
    if sd is not None and not (isinstance(sd, float) and math.isnan(sd)):
        s += f"{{\\scriptsize$\\pm${sd:.{digits}f}}}"
    return s


def _arrow(metric: str) -> str:
    d = METRICS[metric][1]
    return "$\\downarrow$" if d < 0 else ("$\\uparrow$" if d > 0 else "$\\to0.9$")


def _write(out: Path, name: str, latex: str, csv: pd.DataFrame | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.tex").write_text(latex, encoding="utf-8")
    if csv is not None:
        csv.to_csv(out / f"{name}.csv", index=False)


def method_table(df: pd.DataFrame, labels: list[str], datasets: Iterable[str], metrics: list[str], caption: str,
                 tag: str, out: Path, reference: str = REFERENCE, exclude_from_best: set[str] = CENTRAL) -> pd.DataFrame:
    df = df[df["label"].isin(labels)]
    summ = seed_summary(df, ["dataset", "label"], metrics)
    tests = paired_tests(df, reference, metrics) if reference in set(df["label"]) else pd.DataFrame()
    cols = "ll" + "c" * len(metrics)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{tab:{tag}}}",
             "\\resizebox{\\columnwidth}{!}{", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             "Dataset & Method & " + " & ".join(f"{METRICS[m][0]} {_arrow(m)}" for m in metrics) + " \\\\",
             "\\midrule"]
    for ds in datasets:
        s = summ[summ["dataset"] == ds].set_index("label")
        if s.empty:
            continue
        best = {}
        for m in metrics:
            if m in NO_BEST:
                continue
            cand = s[~s.index.isin(exclude_from_best)][f"{m}_mean"].dropna()
            if len(cand):
                best[m] = cand.index[int(np.argmin(badness(m, cand.values)))]
        first = True
        for lab in labels:
            if lab not in s.index:
                continue
            row = s.loc[lab]
            cells = []
            for m in metrics:
                cell = _fmt(row.get(f"{m}_mean"), row.get(f"{m}_sd"))
                if best.get(m) == lab:
                    cell = f"\\textbf{{{cell}}}"
                if not tests.empty and lab != reference:
                    t = tests[(tests.dataset == ds) & (tests.metric == m) & (tests.label == lab)]
                    if len(t) and t.iloc[0]["p_t_holm"] < 0.05:
                        cell += "$^\\dagger$" if t.iloc[0]["mean_diff"] > 0 else "$^\\ddagger$"
                cells.append(cell)
            n_note = "" if row["n"] >= 5 else f" (n={int(row['n'])})"
            if row.get("diverged", 0):
                n_note += f" [{int(row['diverged'])} div.]"
            name = METHOD_NAMES.get(lab, lab) + n_note
            lines.append(f"{DATASET_NAMES.get(ds, ds) if first else ''} & {name} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, tag, "\n".join(lines), summ)
    if not tests.empty:
        tests.to_csv(out / f"{tag}_tests.csv", index=False)
    return summ


def ddim_table(ddim: pd.DataFrame, out: Path) -> None:
    d = ddim[ddim["task_id"].str.contains("__fedts_hdiff__")]
    if d.empty:
        return
    metrics = ["dtw_mean", "crps", "wis_90", "picp_90", "mpiw_90"]
    rows = []
    for (ds, steps), g in d.groupby(["dataset", "ddim_steps"]):
        row = {"dataset": ds, "steps": int(steps), "n": len(g)}
        for m in metrics + ["sample_seconds_per_window"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1)
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(["dataset", "steps"])
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Inference cost--quality trade-off of DDIM sampling for FedTS-HDiff (same trained models, "
             "100 samples per window; seconds per window on one A100).}", "\\label{tab:ddim}",
             "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{llcccccc}", "\\toprule",
             "Dataset & Steps & " + " & ".join(f"{METRICS[m][0]}" for m in metrics) + " & s/window \\\\", "\\midrule"]
    for ds, g in t.groupby("dataset", sort=False):
        for k, (_, r) in enumerate(g.iterrows()):
            lines.append(f"{DATASET_NAMES.get(ds, ds) if k == 0 else ''} & {r['steps']} & " +
                         " & ".join(_fmt(r[f'{m}_mean'], r[f'{m}_sd']) for m in metrics) +
                         f" & {r['sample_seconds_per_window_mean']:.3f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_ddim", "\n".join(lines), t)


def calibration_table(df: pd.DataFrame, out: Path, labels: list[str]) -> None:
    d = df[(df["group"] == "main") & df["label"].isin(labels) & (df["status"] == "complete")].copy()
    if d.empty:
        return
    cov = d["coverage_levels"].apply(lambda s: json.loads(s) if isinstance(s, str) else [math.nan] * 9)
    for lv, idx in (("50", 4), ("80", 7), ("90", 8)):
        d[f"cov{lv}"] = cov.apply(lambda c: c[idx])
    metrics = ["crps", "wis_multi", "calib_error", "pit_tv", "cov50", "cov80", "cov90"]
    METRICS.setdefault("cov50", ("Cov@50", 0))
    METRICS.setdefault("cov80", ("Cov@80", 0))
    METRICS.setdefault("cov90", ("Cov@90", 0))
    summ = seed_summary(d, ["dataset", "label"], metrics)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Probabilistic calibration: CRPS, multi-level WIS, mean absolute calibration error over "
             "nominal central intervals 10--90\\%, total-variation distance of the PIT histogram from uniform, "
             "and empirical coverage of the 50/80/90\\% intervals.}", "\\label{tab:calibration}",
             "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{llccccccc}", "\\toprule",
             "Dataset & Method & CRPS & WIS & Cal. err. & PIT TV & Cov@50 & Cov@80 & Cov@90 \\\\", "\\midrule"]
    for ds in DATASET_NAMES:
        s = summ[summ.dataset == ds].set_index("label")
        for k, lab in enumerate([l for l in labels if l in s.index]):
            r = s.loc[lab]
            lines.append(f"{DATASET_NAMES[ds] if k == 0 else ''} & {METHOD_NAMES.get(lab, lab)} & " +
                         " & ".join(_fmt(r[f'{m}_mean'], r[f'{m}_sd'], 3) for m in metrics) + " \\\\")
        if len(s):
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_calibration", "\n".join(lines), summ)


def cost_table(df: pd.DataFrame, out: Path) -> None:
    d = df[(df["group"] == "cost") & (df["status"] == "complete")].copy()
    if d.empty:
        return
    d["method_label"] = d["label"].str.replace("cost_", "", regex=False)
    cols = {
        "cost_params": ("Params (M)", 1e-6, 2),
        "cost_mean_bytes_client_up": ("Client up MB/rnd", 1 / 2**20, 1),
        "cost_mean_bytes_cloud": ("Cloud MB/rnd", 1 / 2**20, 1),
        "cost_mean_local_seconds_max": ("Train s/rnd (par.)", 1, 2),
        "cost_mean_local_seconds_sum": ("Train s/rnd (seq.)", 1, 1),
        "cost_mean_t_group": ("Group ms/rnd", 1e3, 2),
        "cost_mean_t_aggregate": ("Agg. ms/rnd", 1e3, 2),
        "sample_seconds_per_window": ("Infer. s/window", 1, 3),
        "cost_peak_gpu_bytes": ("Peak GPU MB", 1 / 2**20, 0),
    }
    rows = []
    for (ds, lab), g in d.groupby(["dataset", "method_label"]):
        row = {"dataset": ds, "method": lab}
        for c, (_, scale, _) in cols.items():
            row[c] = float(g[c].mean()) * scale if c in g and g[c].notna().any() else math.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Communication and computation cost per communication round (measured with one worker per "
             "A100, 10 rounds). Client up: total client-to-aggregator upload; Cloud: traffic crossing the cloud "
             "server; Train (par.): sum over edge rounds of the slowest client, i.e. wall-clock with parallel "
             "clients; (seq.): simulated sequential total. Inference: 100 samples per window.}", "\\label{tab:cost}",
             "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{ll" + "c" * len(cols) + "}", "\\toprule",
             "Dataset & Method & " + " & ".join(v[0] for v in cols.values()) + " \\\\", "\\midrule"]
    order = MAIN_ORDER + ["fedts_hdiff_k2"]
    for ds in DATASET_NAMES:
        s = t[t.dataset == ds].set_index("method")
        for k, lab in enumerate([l for l in order if l in s.index]):
            r = s.loc[lab]
            cells = [("--" if math.isnan(r[c]) else f"{r[c]:.{dg}f}") for c, (_, _, dg) in cols.items()]
            name = METHOD_NAMES.get(lab, "FedTS-HDiff, $\\kappa{=}2$" if lab == "fedts_hdiff_k2" else lab)
            lines.append(f"{DATASET_NAMES[ds] if k == 0 else ''} & {name} & " + " & ".join(cells) + " \\\\")
        if len(s):
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_cost", "\n".join(lines), t)


def scalability_table(df: pd.DataFrame, out: Path, metric: str = "crps") -> None:
    d = df[(df["group"] == "scalability") & (df["status"] == "complete")].copy()
    if d.empty:
        return
    summ = seed_summary(d, ["label"], [metric, "dtw_mean", "cost_mean_bytes_cloud", "cost_mean_local_seconds_max"])
    _write(out, "tab_scalability", _scalability_latex(summ, metric), summ)


def _scalability_latex(summ: pd.DataFrame, metric: str) -> str:
    import re

    s = summ.set_index("label")
    Ns = sorted({int(m.group(1)) for l in s.index if (m := re.fullmatch(r"scale_\w+?_N(\d+)(_K\d+)?", l))})
    Ks = sorted({int(m.group(1)) for l in s.index if (m := re.search(r"_K(\d+)$", l))})
    kappa = sorted(l for l in s.index if l.endswith("_kappa2"))
    k_kappa = re.search(r"_K(\d+)_kappa2", kappa[0]).group(1) if kappa else "?"
    head = ["FedAvg"] + [f"$K{{=}}{k}$" for k in Ks] + [f"$K{{=}}{k_kappa},\\kappa{{=}}2$"]

    def cell(key: str) -> str:
        return _fmt(s.loc[key, f"{metric}_mean"], s.loc[key, f"{metric}_sd"]) if key in s.index else "--"

    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Scalability: {METRICS[metric][0]} (mean $\\pm$ SD over seeds) as the number of "
             "clients $N$ and edge nodes $K$ grows; $\\kappa{=}2$ uses two edge rounds per cloud round and half the "
             "cloud rounds (equal local computation, half the cloud traffic).}", "\\label{tab:scalability}",
             "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{l" + "c" * len(head) + "}", "\\toprule",
             "$N$ & " + " & ".join(head) + " \\\\", "\\midrule"]
    for N in Ns:
        keys = [f"scale_fedavg_N{N}"] + [f"scale_hdiff_N{N}_K{k}" for k in Ks] + [f"scale_hdiff_N{N}_K{k_kappa}_kappa2"]
        lines.append(f"{N} & " + " & ".join(cell(k) for k in keys) + " \\\\")
    qrows = sorted({(int(m.group(1)), m.group(2)) for l in s.index if (m := re.search(r"_N(\d+)_q([\d.]+)$", l))})
    if qrows:
        lines += ["\\midrule", "\\multicolumn{" + str(len(head) + 1) + "}{l}{Partial participation "
                  "(columns: FedAvg, FedTS-HDiff)} \\\\"]
        for n, q in qrows:
            lines.append(f"$N{{=}}{n}, q{{=}}{q}$ & {cell(f'scale_fedavg_N{n}_q{q}')} & {cell(f'scale_hdiff_N{n}_q{q}')} "
                         + "& " * (len(head) - 2) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def dispersion_table(diags: list[dict], out: Path) -> pd.DataFrame:
    """Within-group share of gradient dispersion under each partition rule, and the
    descriptor-gradient distance correlation (Mantel)."""
    rows = []
    for d in diags:
        if d.get("method") != "fedts_hdiff":
            continue
        glob = d["single_group"]["delta_global"]
        base = {"dataset": d["dataset"], "seed": d["seed"], "round": d["round"], "delta_global": glob}
        for name, v in d["partitions"].items():
            if name.endswith("_sd"):
                continue
            rows.append({**base, "partition": name, "within_share": v["delta_in"] / glob if glob else math.nan,
                         "delta_in": v["delta_in"], "delta_out": v["delta_out"], "agg_bias": v.get("agg_bias")})
        for kind, c in d["descriptor_gradient_correlation"].items():
            rows.append({**base, "partition": f"corr/{kind}", "spearman": c["spearman"], "mantel_p": c["mantel_p"],
                         "lipschitz_q95": c.get("lipschitz_q95")})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t.to_csv(out / "dispersion_long.csv", index=False)
    mid = t[~t["partition"].str.startswith("corr/")]
    agg = mid.groupby(["dataset", "round", "partition"])["within_share"].agg(["mean", "std", "count"]).reset_index()
    agg.to_csv(out / "tab_dispersion.csv", index=False)
    corr = t[t["partition"].str.startswith("corr/")].groupby(["dataset", "round", "partition"])[
        ["spearman", "mantel_p", "lipschitz_q95"]].mean().reset_index()
    corr.to_csv(out / "tab_descriptor_gradient_corr.csv", index=False)
    # LaTeX: the mid-training round of each dataset
    parts = ["random", "round_robin", "balance/label", "balance/signal", "similarity/label", "similarity/signal",
             "kmeans/label", "kmeans/signal", "actual"]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Share of the global gradient dissimilarity that lies \\emph{within} edge groups, "
             "$\\delta_{in}/\\delta_{global}$ (Lemma~1), for different partitions of the same clients at the same "
             "model (mid-training round, mean over seeds; lower = more homogeneous groups). Last columns: Spearman "
             "correlation between pairwise descriptor distance and pairwise gradient distance (Mantel $p$).}",
             "\\label{tab:dispersion}", "\\resizebox{\\columnwidth}{!}{",
             "\\begin{tabular}{l" + "c" * (len(parts) + 2) + "}", "\\toprule",
             "Dataset & " + " & ".join(p.replace("_", "-") for p in parts) + " & $\\rho$ (label) & $\\rho$ (signal) \\\\",
             "\\midrule"]
    for ds in DATASET_NAMES:
        a = agg[agg.dataset == ds]
        if a.empty:
            continue
        rounds = sorted(a["round"].unique())
        r = rounds[len(rounds) // 2]
        a = a[a["round"] == r].set_index("partition")
        c = corr[(corr.dataset == ds) & (corr["round"] == r)].set_index("partition")
        cells = [(f"{a.loc[p, 'mean']:.3f}" if p in a.index else "--") for p in parts]
        for kind in ("label", "signal"):
            key = f"corr/{kind}"
            cells.append(f"{c.loc[key, 'spearman']:.2f} ({c.loc[key, 'mantel_p']:.3f})" if key in c.index else "--")
        lines.append(f"{DATASET_NAMES[ds]} (r={r}) & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_dispersion", "\n".join(lines))
    return t


def convergence_table(curves: pd.DataFrame, out: Path, labels: tuple[str, ...] = ("fedavg", "fedprox", "feddyn",
                                                                                      "fednova", "fedts_hdiff"),
                      metric: str = "dtw_mean") -> None:
    """DTW (mean +- SD over seeds) at 0, R/4, R/2, 3R/4 and R for the main methods (replaces Tables 2-5)."""
    if curves.empty or metric not in curves:
        return
    c = curves[curves["groups"].str.contains("main")]
    rows = []
    for ds, g in c.groupby("dataset"):
        R = int(g["round"].max())
        marks = sorted({0, R // 4, R // 2, 3 * R // 4, R})
        for lab in labels:
            sub = g[g["labels"].str.split(",").apply(lambda ls, l=lab: l in ls)]
            for r in marks:
                at = sub[sub["round"] == min(sub["round"].unique(), key=lambda x: abs(x - r))] if len(sub) else sub
                if len(at):
                    rows.append({"dataset": ds, "label": lab, "round": int(at["round"].iloc[0]),
                                 "mean": float(at[metric].mean()), "sd": float(at[metric].std(ddof=1))})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{{METRICS[metric][0]} across communication rounds (mean $\\pm$ SD over seeds, curve "
             "evaluation set).}", "\\label{tab:convergence}", "\\resizebox{\\columnwidth}{!}{"]
    for ds in DATASET_NAMES:
        s = t[t.dataset == ds]
        if s.empty:
            continue
        rounds = sorted(s["round"].unique())
        labs = [l for l in labels if l in set(s["label"])]
        lines += [f"\\begin{{tabular}}{{c{'c' * len(labs)}}}", "\\toprule",
                  f"\\multicolumn{{{len(labs) + 1}}}{{l}}{{{DATASET_NAMES[ds]}}} \\\\",
                  "Round & " + " & ".join(METHOD_NAMES.get(l, l) for l in labs) + " \\\\", "\\midrule"]
        for r in rounds:
            cells = []
            for l in labs:
                x = s[(s.label == l) & (s["round"] == r)]
                cells.append(_fmt(x["mean"].iloc[0], x["sd"].iloc[0]) if len(x) else "--")
            lines.append(f"{r} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", "\\vspace{2mm}", ""]
    lines += ["}", "\\end{table}", ""]
    _write(out, "tab_convergence", "\n".join(lines), t)
