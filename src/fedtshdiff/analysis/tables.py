"""LaTeX (booktabs) and CSV tables for the revised manuscript.

Conventions: mean +- sample SD over seeds; the best *federated* value per dataset and metric
is bold (centralised upper bounds are excluded); a dagger marks a baseline that is
significantly worse than FedTS-HDiff and a double dagger one that is significantly better
(seed-paired t-test, Holm-adjusted over the baselines of that dataset and metric, p < 0.05).
"""

from __future__ import annotations

import json
import math
import re
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
    """Quality from the 5-seed main runs; seconds/window from the exclusive-GPU cost runs
    (main runs share their GPU with another worker, so their timings are not clean)."""
    labels = ddim["labels"].str.split(",")
    d = ddim[labels.apply(lambda ls: "fedts_hdiff" in ls)]
    timing = ddim[labels.apply(lambda ls: "cost_fedts_hdiff" in ls)]
    if d.empty:
        return
    clean = timing.groupby(["dataset", "ddim_steps"])["sample_seconds_per_window"].mean()
    metrics = ["dtw_mean", "crps", "wis_90", "picp_90", "mpiw_90"]
    rows = []
    for (ds, steps), g in d.groupby(["dataset", "ddim_steps"]):
        row = {"dataset": ds, "steps": int(steps), "n": len(g)}
        for m in metrics + ["sample_seconds_per_window"]:
            row[f"{m}_mean"], row[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1)
        row["sample_seconds_per_window_mean"] = clean.get((ds, steps), float("nan"))
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(["dataset", "steps"])
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Inference cost--quality trade-off of DDIM sampling for FedTS-HDiff (same trained models, "
             "100 samples per window; seconds per window measured alone on one A100).}", "\\label{tab:ddim}",
             "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{llcccccc}", "\\toprule",
             "Dataset & Steps & " + " & ".join(f"{METRICS[m][0]}" for m in metrics) + " & s/window \\\\", "\\midrule"]
    for ds, g in t.groupby("dataset", sort=False):
        for k, (_, r) in enumerate(g.iterrows()):
            lines.append(f"{DATASET_NAMES.get(ds, ds) if k == 0 else ''} & {r['steps']} & " +
                         " & ".join(_fmt(r[f'{m}_mean'], r[f'{m}_sd']) for m in metrics) +
                         " & " + ("--" if math.isnan(r['sample_seconds_per_window_mean'])
                                  else f"{r['sample_seconds_per_window_mean']:.3f}") + " \\\\")
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
    # no per-client timings here: client-batched runs only measure chunk averages (see tab_cost)
    summ = seed_summary(d, ["label"], [metric, "dtw_mean", "cost_mean_bytes_cloud"])
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


# Follow-up screen: grouping x edge rounds at equal computation, plus group personalisation.

SCREEN_KINDS = ("fedavg", "fedavg_comm", "hier", "pers")


def screen_parse(label: str) -> tuple[str, int | None, str]:
    """Screen label -> (configuration name, kappa, kind), kind in SCREEN_KINDS."""
    if label == "screen_fedavg":
        return "FedAvg ($R$ rounds)", None, "fedavg"
    if m := re.fullmatch(r"screen_fedavg_c(\d+)", label):
        return f"FedAvg ($R/{m.group(1)}$ rounds)", int(m.group(1)), "fedavg_comm"
    if m := re.fullmatch(r"screen_pers_m([\d.]+)_k(\d+)", label):
        return f"Group-personalised (mix {m.group(1)})", int(m.group(2)), "pers"
    if m := re.fullmatch(r"screen_([a-z]+)_k(\d+)", label):
        return f"{m.group(1).capitalize()} grouping", int(m.group(2)), "hier"
    raise ValueError(f"not a screen label: {label}")


def screen_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Complete screen rows (a task shared with another group may carry non-screen labels too)."""
    d = df[(df["group"] == "screen") & df["label"].str.startswith("screen_")]
    return d.drop_duplicates(subset=["task_id", "label"])


def _screen_key(label: str) -> tuple:
    name, k, kind = screen_parse(label)
    return SCREEN_KINDS.index(kind), name if kind in ("hier", "pers") else "", k or 0


def screen_table(df: pd.DataFrame, out: Path, metrics: tuple[str, ...] = ("crps", "dtw_mean")) -> pd.DataFrame:
    d = screen_runs(df)
    ok = d[d["status"] == "complete"]
    if ok.empty:
        return ok
    summ = seed_summary(d, ["dataset", "label"], list(metrics) + ["cost_total_bytes_cloud"])
    vs_comp = paired_tests(ok, "screen_fedavg", metrics)
    blocks = []  # equal cloud rounds: kappa-edge-round runs against FedAvg with R/kappa rounds
    for k in sorted({screen_parse(l)[1] for l in ok["label"] if screen_parse(l)[2] == "fedavg_comm"}):
        sub = ok[ok["label"].map(lambda l, k=k: screen_parse(l)[1] == k)]
        blocks.append(paired_tests(sub, f"screen_fedavg_c{k}", metrics))
    vs_comm = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()
    out.mkdir(parents=True, exist_ok=True)
    vs_comp.to_csv(out / "tab_screen_tests_equal_compute.csv", index=False)
    vs_comm.to_csv(out / "tab_screen_tests_equal_cloud.csv", index=False)

    def delta(tests: pd.DataFrame, ds: str, lab: str) -> str:
        t = tests[(tests["dataset"] == ds) & (tests["metric"] == metrics[0]) & (tests["label"] == lab)] \
            if len(tests) else tests
        if t.empty:
            return "--"
        r = t.iloc[0]
        return f"{r['mean_diff']:+.4f}" + ("$^\\dagger$" if r["p_t_holm"] < 0.05 else "")

    m0, m1 = METRICS[metrics[0]][0], METRICS[metrics[1]][0]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Hierarchical screen at equal local computation: $\\kappa$ edge rounds per cloud round and "
             "$R/\\kappa$ cloud rounds (mean $\\pm$ SD over seeds). $\\Delta$: seed-paired difference in "
             f"{m0} to FedAvg with the same computation ($R$ rounds) and with the same number of cloud rounds "
             "($R/\\kappa$); negative is better, $^\\dagger$ Holm-adjusted $p<0.05$. Cloud: total traffic through "
             "the cloud server over training.}", "\\label{tab:screen}", "\\resizebox{\\columnwidth}{!}{",
             "\\begin{tabular}{llccccccc}", "\\toprule",
             f"Dataset & Configuration & $\\kappa$ & Cloud (MB) & {m0} $\\downarrow$ & $\\Delta$ FedAvg($R$) & "
             f"$\\Delta$ FedAvg($R/\\kappa$) & {m1} $\\downarrow$ \\\\", "\\midrule"]
    for ds in DATASET_NAMES:
        s = summ[summ["dataset"] == ds].set_index("label")
        for j, lab in enumerate(sorted(s.index, key=_screen_key)):
            name, k, kind = screen_parse(lab)
            r = s.loc[lab]
            mb = r.get("cost_total_bytes_cloud_mean", math.nan) / 2**20
            comp = "--" if kind == "fedavg" else delta(vs_comp, ds, lab)
            comm = delta(vs_comm, ds, lab) if kind in ("hier", "pers") and k and k > 1 else "--"
            lines.append(f"{DATASET_NAMES[ds] if j == 0 else ''} & {name} & {'--' if k is None or kind == 'fedavg_comm' else k} & "
                         f"{'--' if math.isnan(mb) else f'{mb:,.0f}'} & "
                         f"{_fmt(r.get(f'{metrics[0]}_mean'), r.get(f'{metrics[0]}_sd'))} & {comp} & {comm} & "
                         f"{_fmt(r.get(f'{metrics[1]}_mean'), r.get(f'{metrics[1]}_sd'))} \\\\")
        if len(s):
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_screen", "\n".join(lines), summ)
    return summ


# Personalised hierarchy: client heads (and, from the phsel group, the same hierarchy with a shared head,
# the validation switch between the two, and the MIT-BIH ablation) against FedRep, clustered FL, FedAvg.

PERSONAL_GROUPS = ("personal", "phsel")
PERSONAL_ORDER = ["ph_fedavg", "ph_cfl", "ph_fedrep", "ph_off_k1", "ph_off_k5", "ph_client_k1", "ph_client_k5",
                  "ph_sel_k1", "ph_sel_k5"]
PERSONAL_NAMES = {"ph_fedavg": "FedAvg", "ph_cfl": "Clustered FL", "ph_fedrep": "FedRep",
                  "ph_off": "FedTS-HDiff, shared head", "ph_client": "FedTS-HDiff, personal heads",
                  "ph_sel": "FedTS-HDiff, heads chosen on validation"}
ABLATION_PH = {"ph_fedrep": "FedRep (flat)", "pha_beta0": "Similarity groups, data weights ($\\beta{=}0$)",
               "pha_nogroup": "No grouping ($K{=}1$), adaptive weights",
               "pha_balance": "Balance grouping, adaptive weights",
               "ph_client_k1": "Similarity groups, adaptive weights (full)"}


def personal_parse(label: str) -> tuple[str, int | None]:
    """ph label -> (display name, kappa); kappa is None for the flat references."""
    if m := re.fullmatch(r"(ph_\w+?)_k(\d+)", label):
        return PERSONAL_NAMES[m.group(1)], int(m.group(2))
    return PERSONAL_NAMES[label], None


def _personal_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["group"].isin(PERSONAL_GROUPS)].drop_duplicates(subset=["task_id", "label"])


def personal_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: validation CRPS with personal heads on / off (tuning seed) and the choice
    (the lower validation CRPS; a tie keeps the shared head)."""
    d = _personal_rows(df)
    d = d[(d["status"] == "complete") & d["label"].isin(["phval_on", "phval_off"])]
    rows = []
    for ds, g in d.groupby("dataset"):
        v = g.set_index("label")["crps"]
        if {"phval_on", "phval_off"} <= set(v.index):
            rows.append({"dataset": ds, "val_crps_on": float(v["phval_on"]), "val_crps_off": float(v["phval_off"]),
                         "personal_heads": bool(v["phval_on"] < v["phval_off"])})
    return pd.DataFrame(rows, columns=["dataset", "val_crps_on", "val_crps_off", "personal_heads"])


def personal_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Test rows of the personal comparison, plus ph_sel_k* = the validation-selected variant's rows."""
    d = _personal_rows(df)
    d = d[d["label"].isin(PERSONAL_ORDER)]
    sel = personal_selection(df)
    extra = []
    for r in sel.itertuples():
        src = "ph_client" if r.personal_heads else "ph_off"
        for k in (1, 5):
            rows = d[(d["dataset"] == r.dataset) & (d["label"] == f"{src}_k{k}")].copy()
            rows["label"] = f"ph_sel_k{k}"
            extra.append(rows)
    return pd.concat([d, *extra], ignore_index=True) if extra else d


def personal_table(df: pd.DataFrame, out: Path,
                   metrics: tuple[str, ...] = ("crps", "dtw_mean", "wis_90", "picp_90")) -> pd.DataFrame:
    d = personal_runs(df)
    ok = d[d["status"] == "complete"]
    if ok.empty:
        return ok
    summ = seed_summary(d, ["dataset", "label"], list(metrics) + ["cost_total_bytes_cloud"])
    vs_avg = paired_tests(ok, "ph_fedavg", metrics)
    vs_rep = paired_tests(ok, "ph_fedrep", metrics)
    out.mkdir(parents=True, exist_ok=True)
    vs_avg.to_csv(out / "tab_personal_tests_vs_fedavg.csv", index=False)
    vs_rep.to_csv(out / "tab_personal_tests_vs_fedrep.csv", index=False)
    sel = personal_selection(df)
    if len(sel):
        sel.to_csv(out / "tab_personal_selection.csv", index=False)

    def delta(tests: pd.DataFrame, ds: str, lab: str) -> str:
        t = tests[(tests["dataset"] == ds) & (tests["metric"] == "crps") & (tests["label"] == lab)] \
            if len(tests) else tests
        if t.empty:
            return "--"
        r = t.iloc[0]
        return f"{r['mean_diff']:+.4f}" + ("$^\\dagger$" if r["p_t_holm"] < 0.05 else "")

    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Personalised hierarchy: every client keeps its own output head and the body is aggregated "
             "through $\\kappa$ edge rounds per cloud round ($R/\\kappa$ cloud rounds, equal computation); "
             "shared head: the same hierarchy without personal heads; chosen on validation: personal heads on or "
             "off per dataset by validation CRPS. Mean $\\pm$ SD over seeds; $\\Delta$: seed-paired CRPS "
             "difference to FedAvg and to FedRep (negative is better, $^\\dagger$ Holm-adjusted $p<0.05$). "
             "Cloud: total traffic through the cloud server over training.}",
             "\\label{tab:personal}", "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{llcccccccc}",
             "\\toprule",
             "Dataset & Method & $\\kappa$ & Cloud (MB) & CRPS $\\downarrow$ & $\\Delta$ FedAvg & $\\Delta$ FedRep & "
             "DTW $\\downarrow$ & WIS$_{90}$ $\\downarrow$ & PICP$_{90}$ \\\\", "\\midrule"]
    for ds in DATASET_NAMES:
        s = summ[summ["dataset"] == ds].set_index("label")
        labs = [l for l in PERSONAL_ORDER if l in s.index]
        for j, lab in enumerate(labs):
            name, k = personal_parse(lab)
            if lab.startswith("ph_sel") and len(sel) and ds in set(sel["dataset"]):
                on = bool(sel.set_index("dataset").loc[ds, "personal_heads"])
                name += " (personal)" if on else " (shared)"
            r = s.loc[lab]
            mb = r.get("cost_total_bytes_cloud_mean", math.nan) / 2**20
            cells = [_fmt(r.get(f"{m}_mean"), r.get(f"{m}_sd"), 3 if m == "picp_90" else 4) for m in metrics]
            lines.append(f"{DATASET_NAMES[ds] if j == 0 else ''} & {name} & {k or '--'} & "
                         f"{'--' if math.isnan(mb) else f'{mb:,.0f}'} & {cells[0]} & "
                         f"{'--' if lab == 'ph_fedavg' else delta(vs_avg, ds, lab)} & "
                         f"{'--' if lab == 'ph_fedrep' else delta(vs_rep, ds, lab)} & " + " & ".join(cells[1:]) + " \\\\")
        if labs:
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}}", "\\end{table}", ""]
    _write(out, "tab_personal", "\n".join(lines), summ)
    return summ


def personal_ablation_table(df: pd.DataFrame, out: Path, dataset: str | None = None,
                            metrics: tuple[str, ...] = ("crps", "dtw_mean", "wis_90")) -> pd.DataFrame:
    """What makes client heads on the hierarchy differ from FedRep (kappa = 1).  Default dataset: MIT-BIH,
    else the first one with ablation runs."""
    d = _personal_rows(df)
    have = sorted(set(d.loc[d["label"].str.startswith("pha_"), "dataset"]))
    if not have:
        return pd.DataFrame()
    dataset = dataset or ("mitbih" if "mitbih" in have else have[0])
    d = d[(d["dataset"] == dataset) & d["label"].isin(ABLATION_PH)]
    summ = seed_summary(d, ["dataset", "label"], list(metrics))
    tests = paired_tests(d[d["status"] == "complete"], "ph_fedrep", metrics)
    out.mkdir(parents=True, exist_ok=True)
    tests.to_csv(out / "tab_personal_ablation_tests.csv", index=False)
    s = summ.set_index("label")
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Personal heads on the hierarchy ($\\kappa{{=}}1$) on {DATASET_NAMES[dataset]}: which "
             "component separates it from FedRep. With data-proportional weights and one edge round the hierarchy "
             "is algebraically FedRep. $\\Delta$: seed-paired CRPS difference to FedRep ($^\\dagger$ Holm $p<0.05$).}",
             "\\label{tab:personal_ablation}", "\\begin{tabular}{lcccc}", "\\toprule",
             "Variant & CRPS $\\downarrow$ & $\\Delta$ FedRep & DTW $\\downarrow$ & WIS$_{90}$ $\\downarrow$ \\\\",
             "\\midrule"]
    for lab, name in ABLATION_PH.items():
        if lab not in s.index:
            continue
        r = s.loc[lab]
        t = tests[(tests["metric"] == "crps") & (tests["label"] == lab)] if len(tests) else tests
        dl = "--" if t.empty else f"{t.iloc[0]['mean_diff']:+.4f}" + ("$^\\dagger$" if t.iloc[0]["p_t_holm"] < 0.05 else "")
        lines.append(f"{name} & {_fmt(r.get('crps_mean'), r.get('crps_sd'))} & {dl} & "
                     f"{_fmt(r.get('dtw_mean_mean'), r.get('dtw_mean_sd'))} & {_fmt(r.get('wis_90_mean'), r.get('wis_90_sd'))} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    _write(out, "tab_personal_ablation", "\n".join(lines), summ)
    return summ
