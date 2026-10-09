"""On-disk format of prepared federated datasets.

``<data_root>/processed/<name>/``
    ``windows.npz``  ``X_<k>`` float32 [n_k, L] windows in temporal order, ``y_<k>`` int64 [n_k]
                     (label per window, -1 when the dataset is unlabeled), for k = 0..N-1
    ``meta.json``    client ids, window geometry, class names, preprocessing parameters,
                     raw-source hashes, per-client counts
    ``SHA256SUMS``   hash of ``windows.npz`` and ``meta.json``

Clients are stored in their canonical order (the order the original scripts iterate
them), so ``max_clients=N`` always selects the same nested prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import atomic_write_json, atomic_write_text, read_json, sha256_file


@dataclass
class ClientWindows:
    cid: str
    X: np.ndarray  # [n, L] float32
    y: np.ndarray  # [n] int64


@dataclass
class PreparedDataset:
    name: str
    past_len: int
    future_len: int
    class_names: list[str]
    clients: list[ClientWindows]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def window_len(self) -> int:
        return self.past_len + self.future_len

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def dataset_dir(data_root: str | Path, name: str) -> Path:
    return Path(data_root) / "processed" / name


def save_prepared(ds: PreparedDataset, data_root: str | Path) -> Path:
    out = dataset_dir(data_root, ds.name)
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for k, c in enumerate(ds.clients):
        if c.X.ndim != 2 or c.X.shape[1] != ds.window_len:
            raise ValueError(f"client {c.cid}: windows must be [n, {ds.window_len}], got {c.X.shape}")
        arrays[f"X_{k}"] = c.X.astype(np.float32, copy=False)
        arrays[f"y_{k}"] = c.y.astype(np.int64, copy=False)
    tmp = out / "windows.tmp.npz"
    np.savez(tmp, **arrays)
    tmp.replace(out / "windows.npz")
    meta = dict(ds.meta)
    meta.update(
        name=ds.name,
        past_len=ds.past_len,
        future_len=ds.future_len,
        class_names=list(ds.class_names),
        client_ids=[c.cid for c in ds.clients],
        client_windows=[int(c.X.shape[0]) for c in ds.clients],
        n_clients=len(ds.clients),
    )
    atomic_write_json(out / "meta.json", meta)
    sums = {f: sha256_file(out / f) for f in ("windows.npz", "meta.json")}
    atomic_write_text(out / "SHA256SUMS", "".join(f"{h}  {f}\n" for f, h in sums.items()))
    return out


def verify_prepared(data_root: str | Path, name: str) -> str:
    """Re-hash the stored files; returns the combined data hash or raises."""
    d = dataset_dir(data_root, name)
    sums_path = d / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(f"{d}: dataset not prepared (run `fedts prepare-data --dataset {name}`)")
    lines = [ln.split() for ln in sums_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for digest, fname in lines:
        actual = sha256_file(d / fname)
        if actual != digest:
            raise ValueError(f"{d / fname}: SHA256 mismatch ({actual} != {digest})")
    return data_hash(data_root, name)


def data_hash(data_root: str | Path, name: str) -> str:
    """Stable identity of a prepared dataset: the hash of its SHA256SUMS file."""
    return sha256_file(dataset_dir(data_root, name) / "SHA256SUMS")


def load_prepared(data_root: str | Path, name: str) -> PreparedDataset:
    d = dataset_dir(data_root, name)
    meta = read_json(d / "meta.json")
    clients: list[ClientWindows] = []
    with np.load(d / "windows.npz") as z:
        for k, cid in enumerate(meta["client_ids"]):
            clients.append(ClientWindows(cid=str(cid), X=z[f"X_{k}"], y=z[f"y_{k}"]))
    return PreparedDataset(
        name=meta["name"],
        past_len=int(meta["past_len"]),
        future_len=int(meta["future_len"]),
        class_names=list(meta.get("class_names", [])),
        clients=clients,
        meta=meta,
    )


def robust_zscore_to_minus1_1(x: np.ndarray) -> np.ndarray:
    """The original normalisation: (x - mean) / (3 std), clipped to [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    m = float(x.mean())
    s = max(float(x.std()), 1e-6)
    return np.clip((x - m) / (3.0 * s), -1.0, 1.0).astype(np.float32)


def sliding_windows(x: np.ndarray, window_len: int, stride: int, max_windows: int | None = None,
                    from_end: bool = False) -> np.ndarray:
    """Windows of ``x`` (1-D) in temporal order.  ``from_end`` keeps the most recent ones."""
    n = len(x)
    if n < window_len:
        return np.zeros((0, window_len), dtype=np.float32)
    starts = np.arange(0, n - window_len + 1, stride)
    if max_windows is not None and len(starts) > max_windows:
        starts = starts[-max_windows:] if from_end else starts[:max_windows]
    idx = starts[:, None] + np.arange(window_len)[None, :]
    return np.asarray(x, dtype=np.float32)[idx]
