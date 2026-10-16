"""Figures regenerated from stored results (no retraining)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .tables import DATASET_NAMES, METHOD_NAMES  # noqa: E402

STYLE = {
    "fedavg": ("#4C72B0", "-"), "fedprox": ("#DD8452", "-"), "fednova": ("#8172B3", "-"),
    "feddyn": ("#C44E52", "-"), "fedrep": ("#937860", "-"), "fedtdd": ("#DA8BC3", "-"),
    "cfl": ("#8C8C8C", "-"), "fedts_hdiff": ("#2CA02C", "-"),
    "drift_dynamic": ("#2CA02C", "-"), "drift_static": ("#C44E52", "--"), "drift_random": ("#8172B3", ":"),
    "drift_fedavg": ("#4C72B0", "-."),
}


def _plain(label: str) -> str:
    return re.sub(r"[\\${}^]|textbf|ast", "", METHOD_NAMES.get(label, label)).strip()


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=160)
    plt.close(fig)


def convergence(curves: pd.DataFrame, out: Path, group: str = "main", labels: list[str] | None = None,
                metrics: tuple[str, ...] = ("dtw_mean", "crps"), name: str = "fig_convergence") -> None:
    if curves.empty:
        return
    c = curves[curves["groups"].str.contains(group)]
    labels = labels or [l for l in STYLE if not l.startswith("drift")]
    datasets = [d for d in DATASET_NAMES if d in set(c["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(len(metrics), len(datasets), figsize=(3.4 * len(datasets), 2.6 * len(metrics)),
                             squeeze=False)
    for j, ds in enumerate(datasets):
        for i, m in enumerate(metrics):
            ax = axes[i, j]
            for lab in labels:
                sub = c[(c["dataset"] == ds) & c["labels"].str.split(",").apply(lambda ls, l=lab: l in ls)]
                if sub.empty or m not in sub:
                    continue
                g = sub.groupby("round")[m]
                mu, sd = g.mean(), g.std(ddof=1).fillna(0)
                col, ls = STYLE.get(lab, ("k", "-"))
                ax.plot(mu.index, mu.values, color=col, ls=ls, lw=1.6 if lab.endswith("hdiff") else 1.1,
                        label=_plain(lab))
                ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
            ax.set_title(f"{DATASET_NAMES[ds]}", fontsize=9)
            ax.set_xlabel("Communication round", fontsize=8)
            ax.set_ylabel(m.replace("_", " ").upper(), fontsize=8)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6, loc="upper right", framealpha=0.8)
    _save(fig, out, name)


def reliability(runs: pd.DataFrame, out: Path, labels: list[str]) -> None:
    d = runs[(runs["group"] == "main") & (runs["status"] == "complete") & runs["label"].isin(labels)]
    if d.empty:
        return
    levels = np.arange(0.1, 1.0, 0.1)
    datasets = [x for x in DATASET_NAMES if x in set(d["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.0), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--")
        for lab in labels:
            sub = d[(d["dataset"] == ds) & (d["label"] == lab)]
            if sub.empty:
                continue
            cov = np.array([json.loads(s) for s in sub["coverage_levels"]])
            col, ls = STYLE.get(lab, ("#555555", "-"))
            ax.errorbar(levels, cov.mean(0), yerr=cov.std(0, ddof=1) if len(cov) > 1 else None, color=col, ls=ls,
                        marker="o", ms=2.5, lw=1, capsize=1.5, label=_plain(lab))
        ax.set_title(DATASET_NAMES[ds], fontsize=9)
        ax.set_xlabel("Nominal central-interval coverage", fontsize=8)
        ax.set_ylabel("Empirical coverage", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    _save(fig, out, "fig_reliability")


def pit_histograms(runs: pd.DataFrame, out: Path, labels: list[str]) -> None:
    d = runs[(runs["group"] == "main") & (runs["status"] == "complete") & runs["label"].isin(labels)]
    if d.empty:
        return
    datasets = [x for x in DATASET_NAMES if x in set(d["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(len(labels), len(datasets), figsize=(2.6 * len(datasets), 1.6 * len(labels)),
                             squeeze=False, sharey=True)
    for j, ds in enumerate(datasets):
        for i, lab in enumerate(labels):
            ax = axes[i, j]
            sub = d[(d["dataset"] == ds) & (d["label"] == lab)]
            if len(sub):
                h = np.array([json.loads(s) for s in sub["pit_hist"]]).mean(0)
                ax.bar(np.arange(10) / 10 + 0.05, h, width=0.095, color=STYLE.get(lab, ("#777777",))[0])
            ax.axhline(0.1, color="k", lw=0.8, ls="--")
            if i == 0:
                ax.set_title(DATASET_NAMES[ds], fontsize=9)
            if j == 0:
                ax.set_ylabel(_plain(lab), fontsize=7)
            ax.tick_params(labelsize=6)
    _save(fig, out, "fig_pit")


def scalability(runs: pd.DataFrame, out: Path, metric: str = "crps") -> None:
    d = runs[(runs["group"] == "scalability") & (runs["status"] == "complete")]
    if d.empty:
        return
    d = d.copy()
    d["N"] = d["label"].str.extract(r"_N(\d+)").astype(float)
    d["K"] = d["label"].str.extract(r"_K(\d+)$").astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.0))
    fa = d[d["label"].str.match(r"scale_fedavg_N\d+$")].groupby("N")[metric]
    ax[0].errorbar(fa.mean().index, fa.mean(), yerr=fa.std(ddof=1), color="k", marker="s", label="FedAvg")
    for k, col in zip(sorted(d["K"].dropna().unique()), ["#a1d99b", "#41ab5d", "#006d2c", "#00441b"]):
        g = d[(d["K"] == k) & ~d["label"].str.contains("kappa")].groupby("N")[metric]
        ax[0].errorbar(g.mean().index, g.mean(), yerr=g.std(ddof=1), color=col, marker="o", label=f"FedTS-HDiff K={int(k)}")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("Number of clients N")
    ax[0].set_ylabel(metric.upper())
    ax[0].legend(fontsize=6)
    ax[0].grid(alpha=0.3)
    curves = ((r"scale_fedavg_N\d+$", "k", "FedAvg"), (r"scale_hdiff_N\d+_K4$", "#41ab5d", "FedTS-HDiff K=4"),
              (r"_kappa2$", "#d95f02", "FedTS-HDiff K=4, kappa=2"))  # full participation only (no _q suffix)
    for lab, col, name in curves:
        g = d[d["label"].str.contains(lab)].groupby("N")["cost_mean_bytes_cloud"]
        if len(g):
            ax[1].plot(g.mean().index, g.mean() / 2**20, marker="o", color=col, label=name)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Number of clients N")
    ax[1].set_ylabel("Cloud traffic per round (MB)")
    ax[1].legend(fontsize=6)
    ax[1].grid(alpha=0.3)
    _save(fig, out, "fig_scalability")


def drift_curves(curves: pd.DataFrame, runs: pd.DataFrame, out: Path) -> None:
    if curves.empty:
        return
    labels = ["drift_dynamic", "drift_static", "drift_random", "drift_fedavg"]
    convergence(curves, out, group="drift", labels=labels, metrics=("crps", "dtw_mean"), name="fig_drift")


def dispersion_bars(long: pd.DataFrame, out: Path) -> None:
    if long is None or long.empty:
        return
    t = long[~long["partition"].str.startswith("corr/")]
    parts = ["random", "round_robin", "balance/label", "balance/signal", "similarity/label", "similarity/signal",
             "kmeans/label", "kmeans/signal"]
    datasets = [d for d in DATASET_NAMES if d in set(t["dataset"])]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.0), squeeze=False, sharey=True)
    for j, ds in enumerate(datasets):
        sub = t[t["dataset"] == ds]
        g = sub.groupby("partition")["within_share"]
        ps = [p for p in parts if p in g.groups]
        axes[0, j].bar(range(len(ps)), [g.mean()[p] for p in ps], yerr=[g.std(ddof=1).get(p, 0) for p in ps],
                       color=["#bbbbbb", "#999999"] + ["#2CA02C"] * (len(ps) - 2), capsize=2)
        axes[0, j].set_xticks(range(len(ps)))
        axes[0, j].set_xticklabels(ps, rotation=60, ha="right", fontsize=6)
        axes[0, j].set_title(DATASET_NAMES[ds], fontsize=9)
        axes[0, j].grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel(r"$\delta_{in}/\delta_{global}$")
    _save(fig, out, "fig_dispersion")


SCREEN_COLORS = {"Balance grouping": "#2CA02C", "Similarity grouping": "#1f77b4", "Kmeans grouping": "#9467bd"}


def screen(runs: pd.DataFrame, out: Path, metric: str = "crps") -> None:
    """Metric against total cloud traffic at equal computation: one line per grouping over kappa, FedAvg over
    R, R/2, R/5, R/10 rounds, group-personalised variants as stars."""
    from .tables import screen_parse, screen_runs

    d = screen_runs(runs)
    d = d[d["status"] == "complete"].copy()
    if d.empty:
        return
    parsed = [screen_parse(l) for l in d["label"]]
    d["config"], d["kappa"], d["kind"] = [p[0] for p in parsed], [p[1] or 1 for p in parsed], [p[2] for p in parsed]
    d["cloud_mb"] = d["cost_total_bytes_cloud"].astype(float) / 2**20
    datasets = [x for x in DATASET_NAMES if x in set(d["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False)
    for j, ds in enumerate(datasets):
        ax, sub = axes[0, j], d[d["dataset"] == ds]
        fa = sub[sub["kind"].isin(["fedavg", "fedavg_comm"])].groupby("label")[["cloud_mb", metric]]
        if len(fa):
            m, sd = fa.mean().sort_values("cloud_mb"), fa.std(ddof=1)
            ax.errorbar(m["cloud_mb"], m[metric], yerr=sd.loc[m.index, metric], color="k", marker="s", ms=3, lw=1,
                        capsize=2, label="FedAvg ($R/\\kappa$ rounds)")
        for name, col in SCREEN_COLORS.items():
            g = sub[(sub["kind"] == "hier") & (sub["config"] == name)].groupby("kappa")[["cloud_mb", metric]]
            if not len(g):
                continue
            m, sd = g.mean(), g.std(ddof=1)
            ax.errorbar(m["cloud_mb"], m[metric], yerr=sd[metric], color=col, marker="o", ms=3, lw=1, capsize=2,
                        label=name)
            for k, row in m.iterrows():
                ax.annotate(f"$\\kappa$={k}", (row["cloud_mb"], row[metric]), fontsize=5, xytext=(2, 2),
                            textcoords="offset points", color=col)
        pers = sub[sub["kind"] == "pers"].groupby("label")[["cloud_mb", metric]].mean()
        for lab, row in pers.iterrows():
            ax.scatter(row["cloud_mb"], row[metric], marker="*", s=40, color="#d62728", zorder=5)
            ax.annotate(lab.replace("screen_pers_", ""), (row["cloud_mb"], row[metric]), fontsize=5, xytext=(2, -6),
                        textcoords="offset points", color="#d62728")
        ax.set_xscale("log")
        ax.set_title(DATASET_NAMES[ds], fontsize=9)
        ax.set_xlabel("Total cloud traffic (MB)", fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6)
    _save(fig, out, "fig_screen")


PERSONAL_COLORS = {"FedTS-HDiff, personal heads": "#2CA02C", "FedTS-HDiff, shared head": "#1f77b4"}
PERSONAL_REFS = {"FedAvg": ("k", "s"), "Clustered FL": ("#8C8C8C", "D"), "FedRep": ("#d62728", "^")}


def personal(runs: pd.DataFrame, out: Path, metric: str = "crps") -> None:
    """Metric against total cloud traffic: flat references as single points, personal heads over kappa."""
    from .tables import personal_parse, personal_runs

    d = personal_runs(runs)
    d = d[(d["status"] == "complete") & ~d["label"].str.startswith("ph_sel")].copy()
    if d.empty:
        return
    parsed = [personal_parse(l) for l in d["label"]]
    d["config"], d["kappa"] = [p[0] for p in parsed], [p[1] or 1 for p in parsed]
    d["cloud_mb"] = d["cost_total_bytes_cloud"].astype(float) / 2**20
    datasets = [x for x in DATASET_NAMES if x in set(d["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2), squeeze=False)
    for j, ds in enumerate(datasets):
        ax, sub = axes[0, j], d[d["dataset"] == ds]
        for name, (col, mk) in PERSONAL_REFS.items():
            g = sub[sub["config"] == name]
            if len(g):
                ax.errorbar(g["cloud_mb"].mean(), g[metric].mean(), yerr=g[metric].std(ddof=1), color=col, marker=mk,
                            ms=4, capsize=2, ls="none", label=name)
        for name, col in PERSONAL_COLORS.items():
            g = sub[sub["config"] == name].groupby("kappa")[["cloud_mb", metric]]
            if not len(g):
                continue
            m, sd = g.mean(), g.std(ddof=1)
            ax.errorbar(m["cloud_mb"], m[metric], yerr=sd[metric], color=col, marker="o", ms=3, lw=1, capsize=2,
                        label=name.replace("FedTS-HDiff, ", ""))
            for k, row in m.iterrows():
                ax.annotate(f"$\\kappa$={k}", (row["cloud_mb"], row[metric]), fontsize=5, xytext=(2, 2),
                            textcoords="offset points", color=col)
        ax.set_xscale("log")
        ax.set_title(DATASET_NAMES[ds], fontsize=9)
        ax.set_xlabel("Total cloud traffic (MB)", fontsize=8)
        ax.set_ylabel(metric.upper(), fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0, -1].legend(fontsize=6)
    _save(fig, out, "fig_personal")
