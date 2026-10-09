"""Statistics for the revision: seed-level summaries, paired tests with Holm correction,
window-level paired tests, and Friedman ranks across datasets (Reviewer 2, comment 5)."""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

# direction: -1 lower is better, +1 higher is better, 0 = closeness to target (PICP -> 0.9)
METRICS: dict[str, tuple[str, int]] = {
    "dtw_mean": ("DTW", -1),
    "psd_l2": ("PSD-L2", -1),
    "div_dtw": ("Div-DTW", +1),
    "crps": ("CRPS", -1),
    "wis_90": ("WIS90", -1),
    "picp_90": ("PICP90", 0),
    "mpiw_90": ("MPIW90", -1),
    "mse": ("MSE", -1),
    "mae": ("MAE", -1),
    "dtw_point": ("DTW (mean fc.)", -1),
    "wis_multi": ("WIS", -1),
    "calib_error": ("Cal. err.", -1),
    "pit_tv": ("PIT TV", -1),
}
TARGET = {"picp_90": 0.9}


def badness(metric: str, values: np.ndarray | pd.Series) -> np.ndarray:
    """Transform a metric so that smaller is better (used for ranking and bolding)."""
    v = np.asarray(values, dtype=float)
    d = METRICS.get(metric, ("", -1))[1]
    if d == 0:
        return np.abs(v - TARGET[metric])
    return -v if d > 0 else v


def seed_summary(df: pd.DataFrame, by: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        ok = g[g["status"] == "complete"]
        row["n"] = len(ok)
        row["diverged"] = int((g["status"] == "diverged").sum())
        for m in metrics:
            if m in ok:
                vals = ok[m].astype(float).dropna()
                row[f"{m}_mean"] = float(vals.mean()) if len(vals) else math.nan
                row[f"{m}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def paired_tests(df: pd.DataFrame, reference: str, metrics: Iterable[str], label_col: str = "label") -> pd.DataFrame:
    """Seed-paired comparison of every label against ``reference`` within each dataset.

    Reports mean difference (other - reference, in the metric's own units; for PICP the
    difference of |PICP - 0.9|), its 95% t-interval, paired t-test, exact Wilcoxon
    signed-rank test, effect size d_z, and Holm-adjusted p-values over the comparisons
    of one (dataset, metric).
    """
    rows = []
    ok = df[df["status"] == "complete"]
    for ds, g in ok.groupby("dataset"):
        ref = g[g[label_col] == reference].set_index("seed")
        for metric in metrics:
            block = []
            for lab, h in g.groupby(label_col):
                if lab == reference or metric not in h:
                    continue
                other = h.set_index("seed")
                seeds = sorted(set(ref.index) & set(other.index))
                if len(seeds) < 2:
                    continue
                a = badness(metric, other.loc[seeds, metric].astype(float))
                b = badness(metric, ref.loc[seeds, metric].astype(float))
                diff = a - b  # > 0: reference is better
                n = len(diff)
                mean, sd = float(diff.mean()), float(diff.std(ddof=1))
                half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if sd > 0 else 0.0
                t_p = float(stats.ttest_rel(a, b).pvalue) if sd > 0 else (1.0 if mean == 0 else 0.0)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        w_p = float(stats.wilcoxon(a, b, method="exact").pvalue) if np.any(diff != 0) else 1.0
                except ValueError:
                    w_p = math.nan
                block.append({"dataset": ds, "metric": metric, "label": lab, "reference": reference, "n": n,
                              "mean_diff": mean, "ci_low": mean - half, "ci_high": mean + half,
                              "d_z": mean / sd if sd > 0 else math.nan, "p_t": t_p, "p_wilcoxon": w_p,
                              "reference_better_all_seeds": bool(np.all(diff > 0))})
            if block:
                adj = holm([r["p_t"] for r in block])
                for r, p in zip(block, adj):
                    r["p_t_holm"] = p
                rows += block
    return pd.DataFrame(rows)


def window_level_test(a: np.ndarray, b: np.ndarray, metric: str, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Paired comparison over evaluation windows (a, b: [seeds, windows] for the same windows).

    Per-window differences are averaged over seeds first, so windows are the unit.
    """
    d = (badness(metric, a) - badness(metric, b)).mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n_boot)])
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        p = math.nan
    return {"windows": int(d.size), "mean_diff": float(d.mean()), "boot_low": float(np.quantile(boots, 0.025)),
            "boot_high": float(np.quantile(boots, 0.975)), "p_wilcoxon": p,
            "share_reference_better": float(np.mean(d > 0))}


def friedman(summary: pd.DataFrame, metric: str, label_col: str = "label") -> dict[str, object]:
    """Average ranks across datasets (rank 1 = best) and the Friedman test."""
    piv = summary.pivot(index="dataset", columns=label_col, values=f"{metric}_mean").dropna(axis=1)
    if piv.shape[0] < 2 or piv.shape[1] < 3:
        return {"metric": metric, "ranks": {}, "p": math.nan}
    bad = piv.apply(lambda col: pd.Series(badness(metric, col), index=col.index))
    ranks = bad.rank(axis=1, method="average")
    stat, p = stats.friedmanchisquare(*[bad[c].values for c in bad.columns])
    return {"metric": metric, "ranks": ranks.mean().sort_values().to_dict(), "chi2": float(stat), "p": float(p),
            "datasets": int(piv.shape[0])}
