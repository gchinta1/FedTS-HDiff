"""Builders that turn raw public datasets into prepared federated windows.

The preprocessing reproduces the original scripts and the paper's Section 5.2,
with every deviation stated next to the parameter that controls it:

* **MIT-BIH** (`mitdb`, 48 records, 360 Hz): one client per record, lead 0, a
  128-sample window centred on each annotated R-peak (64 past / 64 future),
  each window independently robust-z-scored and clipped to [-1, 1]; at most 1500
  windows per record, records with < 300 windows dropped; label = beat symbol.
  ``beats_only=True`` (default) keeps only annotations that mark heartbeats; the
  original also turned rhythm/noise markers ('+', '~', '|', ...) into windows
  that are not beat-centred.
* **PTB-XL** (100 Hz, 10 s records): one client per record (as in the original
  code; the paper text says "patient"), lead 0, record-level normalisation,
  sliding windows of 128 with stride 8, 50 <= windows <= 1024; label = the first
  (alphabetical) diagnostic superclass.  Records are scanned in sorted order, so
  ``max_clients=N`` at load time reproduces the original first-N selection.
* **Electricity** (UCI LD2011-2014, 370 meters, 15 min): one client per meter,
  leading inactive zeros trimmed, client-level normalisation, 96/96 windows with
  stride 16, the most recent <= 1000 windows, >= 100 windows required.
* **METR-LA** (207 sensors, 5 min): one client per sensor, zero readings treated as
  missing and forward/backward filled, sensors with more than
  ``max_missing_fraction`` missing dropped, client-level normalisation, 64/64
  windows with stride 32, <= 1000 windows, >= 100 windows required.
"""

from __future__ import annotations

import ast
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import sha256_file
from .store import ClientWindows, PreparedDataset, robust_zscore_to_minus1_1, save_prepared, sliding_windows

# Heartbeat annotation symbols of the MIT-BIH database (non-beat markers excluded).
MITBIH_BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

URLS = {
    "ptbxl": "https://physionet.org/static/published-projects/ptb-xl/"
             "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
    "electricity": "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
}


# ======================================================================================
# MIT-BIH
# ======================================================================================


def prepare_mitbih(raw_dir: Path, *, pre_r: int = 64, post_r: int = 64, lead: int = 0,
                   max_beats: int = 1500, min_beats: int = 300, beats_only: bool = True) -> PreparedDataset:
    import wfdb

    rec_ids = sorted({p.stem for p in raw_dir.glob("*.hea")})
    if not rec_ids:
        raise FileNotFoundError(f"{raw_dir}: no MIT-BIH .hea files (use --download or place mitdb here)")
    per_record: list[tuple[str, np.ndarray, list[str]]] = []
    skipped: dict[str, str] = {}
    for rid in rec_ids:
        rec = wfdb.rdrecord(str(raw_dir / rid))
        ann = wfdb.rdann(str(raw_dir / rid), "atr")
        sig = rec.p_signal[:, lead].astype(np.float32)
        X_list, y_list = [], []
        for rp, sym in zip(ann.sample, ann.symbol):
            if beats_only and sym not in MITBIH_BEAT_SYMBOLS:
                continue
            start, end = int(rp) - pre_r, int(rp) + post_r
            if start < 0 or end > len(sig):
                continue
            X_list.append(robust_zscore_to_minus1_1(sig[start:end]))
            y_list.append(sym)
            if len(X_list) >= max_beats:
                break
        if len(X_list) < min_beats:
            skipped[rid] = f"{len(X_list)} windows < {min_beats}"
            continue
        per_record.append((rid, np.stack(X_list), y_list))
    symbols = sorted({s for _, _, ys in per_record for s in ys})
    label_map = {s: i for i, s in enumerate(symbols)}
    clients = [ClientWindows(rid, X, np.array([label_map[s] for s in ys], dtype=np.int64))
               for rid, X, ys in per_record]
    meta = {
        "source": "MIT-BIH Arrhythmia Database (PhysioNet mitdb)",
        "preprocessing": {"pre_r": pre_r, "post_r": post_r, "lead": lead, "max_beats": max_beats,
                          "min_beats": min_beats, "beats_only": beats_only,
                          "normalisation": "per-window robust z-score / 3, clipped to [-1,1]"},
        "skipped_records": skipped,
        "default_split": "random",
        "raw_hashes": {f"{rid}.dat": sha256_file(raw_dir / f"{rid}.dat") for rid in rec_ids},
    }
    return PreparedDataset("mitbih", pre_r, post_r, symbols, clients, meta)


# ======================================================================================
# PTB-XL
# ======================================================================================


def _ptbxl_labels(root: Path) -> tuple[dict[str, int], list[str], dict[str, int]]:
    import pandas as pd

    df = pd.read_csv(root / "ptbxl_database.csv")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    code_to_class = {str(code): str(row["diagnostic_class"]).strip()
                     for code, row in scp.iterrows()
                     if isinstance(row.get("diagnostic_class"), str) and row["diagnostic_class"].strip()}
    rec_label: dict[str, str] = {}
    rec_patient: dict[str, int] = {}
    for _, row in df.iterrows():
        rel = str(row["filename_lr"]).replace("\\", "/")
        rel = rel[len("records100/"):] if rel.startswith("records100/") else rel
        codes = ast.literal_eval(row["scp_codes"]) if isinstance(row["scp_codes"], str) else {}
        classes = sorted({code_to_class[c] for c in codes if c in code_to_class})
        if classes:
            rec_label[rel] = classes[0]
            rec_patient[rel] = int(row["patient_id"]) if not np.isnan(row["patient_id"]) else -1
    names = sorted(set(rec_label.values()))
    label_map = {n: i for i, n in enumerate(names)}
    return {r: label_map[n] for r, n in rec_label.items()}, names, rec_patient


def prepare_ptbxl(raw_dir: Path, *, window_len: int = 128, stride: int = 8, lead: int = 0,
                  max_windows: int = 1024, min_windows: int = 50, max_clients: int = 2000) -> PreparedDataset:
    import wfdb

    root = raw_dir
    if not (root / "ptbxl_database.csv").exists():
        cands = [p.parent for p in raw_dir.rglob("ptbxl_database.csv")]
        if not cands:
            raise FileNotFoundError(f"{raw_dir}: ptbxl_database.csv not found (use --download)")
        root = cands[0]
    rec_label, names, rec_patient = _ptbxl_labels(root)
    rec_dir = root / "records100"
    rec_ids = sorted(p.relative_to(rec_dir).with_suffix("").as_posix() for p in rec_dir.rglob("*.hea"))
    clients: list[ClientWindows] = []
    patients: list[int] = []
    read_errors = 0
    for rid in rec_ids:
        if rid not in rec_label:
            continue
        try:
            sig = wfdb.rdrecord(str(rec_dir / rid)).p_signal
        except Exception:  # noqa: BLE001 - the original skipped unreadable records the same way
            read_errors += 1
            continue
        if sig is None or sig.ndim != 2:
            continue
        x = robust_zscore_to_minus1_1(sig[:, lead if lead < sig.shape[1] else 0])
        X = sliding_windows(x, window_len, stride, max_windows)
        if X.shape[0] < min_windows:
            continue
        clients.append(ClientWindows(rid, X, np.full(X.shape[0], rec_label[rid], dtype=np.int64)))
        patients.append(rec_patient.get(rid, -1))
        if len(clients) >= max_clients:
            break
    meta = {
        "source": "PTB-XL 1.0.3 (PhysioNet), records100",
        "preprocessing": {"window_len": window_len, "stride": stride, "lead": lead,
                          "max_windows": max_windows, "min_windows": min_windows,
                          "label": "first alphabetical diagnostic_class of the record",
                          "client_unit": "record",
                          "normalisation": "per-record robust z-score / 3, clipped to [-1,1]"},
        "patient_ids": patients,
        "read_errors": read_errors,
        "default_split": "time",
        "raw_hashes": {f: sha256_file(root / f) for f in ("ptbxl_database.csv", "scp_statements.csv")},
    }
    half = window_len // 2
    return PreparedDataset("ptbxl", half, window_len - half, names, clients, meta)


# ======================================================================================
# Electricity (UCI LD2011-2014)
# ======================================================================================


def _read_electricity(path: Path):
    import pandas as pd

    return pd.read_csv(path, sep=";", decimal=",", index_col=0, low_memory=False)


def prepare_electricity(raw_dir: Path, *, past_len: int = 96, future_len: int = 96, stride: int = 16,
                        max_windows: int = 1000, min_windows: int = 100) -> PreparedDataset:
    files = list(raw_dir.rglob("LD2011_2014.txt"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}: LD2011_2014.txt not found (use --download)")
    df = _read_electricity(files[0])
    clients: list[ClientWindows] = []
    trimmed: dict[str, int] = {}
    for col in df.columns:
        x = df[col].to_numpy(dtype=np.float64)
        nz = np.flatnonzero(x != 0)
        if nz.size == 0:
            continue
        x = x[nz[0]:]
        trimmed[str(col)] = int(nz[0])
        X = sliding_windows(robust_zscore_to_minus1_1(x), past_len + future_len, stride, max_windows,
                            from_end=True)
        if X.shape[0] < min_windows:
            continue
        clients.append(ClientWindows(str(col), X, np.full(X.shape[0], -1, dtype=np.int64)))
    meta = {
        "source": "UCI ElectricityLoadDiagrams20112014 (LD2011_2014.txt)",
        "preprocessing": {"past_len": past_len, "future_len": future_len, "stride": stride,
                          "max_windows": max_windows, "min_windows": min_windows,
                          "window_selection": "most recent windows after trimming leading zeros",
                          "normalisation": "per-client robust z-score / 3, clipped to [-1,1]"},
        "leading_zeros_trimmed": trimmed,
        "default_split": "time",
        "raw_hashes": {files[0].name: sha256_file(files[0])},
    }
    return PreparedDataset("electricity", past_len, future_len, [], clients, meta)


# ======================================================================================
# METR-LA
# ======================================================================================


def _read_metrla(path: Path) -> tuple[np.ndarray, list[str]]:
    try:
        import pandas as pd

        df = pd.read_hdf(path)
        return df.to_numpy(dtype=np.float64), [str(c) for c in df.columns]
    except Exception:  # noqa: BLE001 - pytables missing; read the pandas "fixed" layout directly
        import h5py

        with h5py.File(path, "r") as f:
            g = f["df"]
            values = np.asarray(g["block0_values"], dtype=np.float64)
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in g["axis0"][()]]
        return values, cols


def prepare_metrla(raw_dir: Path, *, past_len: int = 64, future_len: int = 64, stride: int = 32,
                   max_windows: int = 1000, min_windows: int = 100,
                   max_missing_fraction: float = 0.2) -> PreparedDataset:
    files = list(raw_dir.rglob("metr-la.h5"))
    if not files:
        raise FileNotFoundError(f"{raw_dir}: metr-la.h5 not found. Download it from the DCRNN "
                                "repository (https://github.com/liyaguang/DCRNN) and place it there.")
    values, cols = _read_metrla(files[0])
    clients: list[ClientWindows] = []
    dropped: dict[str, float] = {}
    for j, cid in enumerate(cols):
        x = values[:, j].copy()
        missing = (x == 0) | ~np.isfinite(x)
        if missing.mean() > max_missing_fraction:
            dropped[cid] = float(missing.mean())
            continue
        x[missing] = np.nan
        idx = np.where(~np.isnan(x), np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x = x[idx]  # forward fill
        if np.isnan(x[0]):
            first = np.flatnonzero(~np.isnan(x))[0]
            x[:first] = x[first]  # backward fill of the leading gap
        X = sliding_windows(robust_zscore_to_minus1_1(x), past_len + future_len, stride, max_windows)
        if X.shape[0] < min_windows:
            continue
        clients.append(ClientWindows(cid, X, np.full(X.shape[0], -1, dtype=np.int64)))
    meta = {
        "source": "METR-LA (DCRNN release, metr-la.h5)",
        "preprocessing": {"past_len": past_len, "future_len": future_len, "stride": stride,
                          "max_windows": max_windows, "min_windows": min_windows,
                          "missing": "zero readings treated as missing; forward/backward fill",
                          "max_missing_fraction": max_missing_fraction,
                          "normalisation": "per-client robust z-score / 3, clipped to [-1,1]"},
        "dropped_sensors": dropped,
        "default_split": "time",
        "raw_hashes": {files[0].name: sha256_file(files[0])},
    }
    return PreparedDataset("metrla", past_len, future_len, [], clients, meta)


# ======================================================================================
# Synthetic (tests and smoke runs; no download)
# ======================================================================================


def make_synthetic(n_clients: int = 12, windows_per_client: int = 80, past_len: int = 16, future_len: int = 16,
                   n_classes: int = 4, seed: int = 0, dirichlet_alpha: float = 0.5) -> PreparedDataset:
    """Label-skewed sinusoid mixtures: class c has frequency (c+1)/L; clients differ in label mix,
    amplitude and noise, so label and signal descriptors are both informative."""
    rng = np.random.default_rng(seed)
    L = past_len + future_len
    t = np.arange(L)
    clients = []
    for k in range(n_clients):
        mix = rng.dirichlet(np.full(n_classes, dirichlet_alpha))
        amp = rng.uniform(0.4, 0.9)
        noise = rng.uniform(0.02, 0.15)
        n = int(windows_per_client * rng.uniform(0.7, 1.3))
        y = rng.choice(n_classes, size=n, p=mix)
        phase = rng.uniform(0, 2 * np.pi, size=n)
        X = amp * np.sin(2 * np.pi * (y[:, None] + 1) * t[None, :] / L + phase[:, None])
        X = X + noise * rng.standard_normal(X.shape)
        clients.append(ClientWindows(f"syn{k:03d}", np.clip(X, -1, 1).astype(np.float32), y.astype(np.int64)))
    meta = {"source": "synthetic", "seed": seed, "default_split": "random"}
    return PreparedDataset("synthetic", past_len, future_len, [f"c{c}" for c in range(n_classes)], clients, meta)


# ======================================================================================
# Download and dispatch
# ======================================================================================


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:  # noqa: S310 - fixed URLs
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dest)
    return dest


def download_raw(name: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if name == "mitbih":
        import wfdb

        if not list(raw_dir.glob("*.hea")):
            wfdb.dl_database("mitdb", str(raw_dir))
    elif name in ("ptbxl", "electricity"):
        marker = "ptbxl_database.csv" if name == "ptbxl" else "LD2011_2014.txt"
        if not list(raw_dir.rglob(marker)):
            archive = _download(URLS[name], raw_dir / f"{name}.zip")
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    target = (raw_dir / member).resolve()
                    if not target.is_relative_to(raw_dir.resolve()):
                        raise ValueError(f"unsafe path in archive: {member}")
                z.extractall(raw_dir)
            for inner in raw_dir.rglob("*.zip"):
                if inner != archive:
                    with zipfile.ZipFile(inner) as z:
                        z.extractall(inner.parent)
    elif name == "metrla":
        if not list(raw_dir.rglob("metr-la.h5")):
            raise FileNotFoundError(
                f"METR-LA cannot be downloaded automatically. Place metr-la.h5 (DCRNN release) in {raw_dir}.")


BUILDERS = {
    "mitbih": prepare_mitbih,
    "ptbxl": prepare_ptbxl,
    "electricity": prepare_electricity,
    "metrla": prepare_metrla,
}


def prepare(name: str, *, raw_root: str | Path, data_root: str | Path, download: bool = False,
            options: dict[str, Any] | None = None) -> Path:
    if name == "synthetic":
        return save_prepared(make_synthetic(**(options or {})), data_root)
    raw_dir = Path(raw_root) / name
    if download:
        download_raw(name, raw_dir)
    ds = BUILDERS[name](raw_dir, **(options or {}))
    if not ds.clients:
        raise RuntimeError(f"{name}: no client survived preprocessing")
    return save_prepared(ds, data_root)
