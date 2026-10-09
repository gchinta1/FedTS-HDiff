"""Client-partition statistics for reproducibility (Reviewer 2, comment 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import build_config
from ..data.federated import build_federated
from ..fl.grouping import compute_descriptors
from .tables import DATASET_NAMES


def heterogeneity(data) -> dict[str, float]:
    out: dict[str, float] = {}
    X = [c.train_X for c in data.clients]
    y = [c.train_y for c in data.clients]
    if data.n_classes:
        phi = compute_descriptors("label", X, y, data.n_classes)
        tv = 0.5 * np.abs(phi[:, None, :] - phi[None, :, :]).sum(-1)
        iu = np.triu_indices(len(phi), 1)
        out["label_tv_mean"] = float(tv[iu].mean())
        out["classes_per_client_mean"] = float((phi > 0).sum(1).mean())
        out["n_classes"] = data.n_classes
    psd = compute_descriptors("psd", X, y, data.n_classes)
    m = psd.mean(0)
    out["psd_tv_to_mean"] = float(0.5 * np.abs(psd - m).sum(1).mean())
    stats = np.stack([[x.mean(), x.std()] for x in X])
    out["client_mean_sd"] = float(stats[:, 0].std())
    out["client_std_cv"] = float(stats[:, 1].std() / max(stats[:, 1].mean(), 1e-12))
    return out


def dataset_table(data_root: str | Path, datasets=tuple(DATASET_NAMES), out: Path | None = None) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        try:
            cfg = build_config(dataset=ds, overrides={"data_root": str(data_root)})
            data = build_federated(cfg)
        except FileNotFoundError:
            continue
        s = data.summary()
        row = {"dataset": ds, "clients": s["clients"], "client_unit": data.meta.get("preprocessing", {}).get(
            "client_unit", {"mitbih": "record", "electricity": "meter", "metrla": "sensor"}.get(ds, "")),
            "past/future": f"{cfg.past_len}/{cfg.future_len}", "split": cfg.split,
            "train_total": s["train"]["total"], "val_total": s["val"]["total"], "test_total": s["test"]["total"],
            "train_per_client": f"{s['train']['min']}/{s['train']['median']:.0f}/{s['train']['max']}",
            "participation": cfg.participation, "edges": cfg.num_edges, "descriptor": cfg.descriptor,
            **heterogeneity(data)}
        rows.append(row)
    t = pd.DataFrame(rows)
    if out is not None and len(t):
        out.mkdir(parents=True, exist_ok=True)
        t.to_csv(out / "tab_data.csv", index=False)
        lines = ["\\begin{table}[t]", "\\centering",
                 "\\caption{Federated partitions. Windows per client are min/median/max of the training part; "
                 "label TV is the mean pairwise total-variation distance between client label distributions; "
                 "PSD TV is the mean distance of client spectra to the population spectrum.}", "\\label{tab:data}",
                 "\\resizebox{\\columnwidth}{!}{", "\\begin{tabular}{lcccccccccc}", "\\toprule",
                 "Dataset & Clients & Unit & Past/Fut. & Split & Train/Val/Test & Train/client & Part. & $K$ & "
                 "Label TV & PSD TV \\\\", "\\midrule"]
        for _, r in t.iterrows():
            ltv = f"{r['label_tv_mean']:.3f}" if "label_tv_mean" in r and pd.notna(r.get("label_tv_mean")) else "--"
            lines.append(f"{DATASET_NAMES[r['dataset']]} & {r['clients']} & {r['client_unit']} & {r['past/future']} & "
                         f"{r['split']} & {r['train_total']}/{r['val_total']}/{r['test_total']} & {r['train_per_client']} & "
                         f"{int(r['participation'] * 100)}\\% & {r['edges']} & {ltv} & {r['psd_tv_to_mean']:.3f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
        (out / "tab_data.tex").write_text("\n".join(lines), encoding="utf-8")
    return t
