"""Run-time federated view of a prepared dataset: client splits, descriptors, evaluation sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..config import Config
from .store import PreparedDataset, load_prepared, verify_prepared


@dataclass
class Client:
    index: int
    cid: str
    train_X: np.ndarray  # [n_train, L] float32
    train_y: np.ndarray
    val_X: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    # device copies (filled by FederatedData.to_device)
    train_t: torch.Tensor | None = None

    @property
    def n_train(self) -> int:
        return int(self.train_X.shape[0])


@dataclass
class EvalSet:
    X: torch.Tensor  # [m, L]
    client_index: np.ndarray  # [m] owning client of each window
    window_index: np.ndarray  # [m] row in that client's test split


@dataclass
class FederatedData:
    name: str
    past_len: int
    future_len: int
    class_names: list[str]
    clients: list[Client]
    data_hash: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def window_len(self) -> int:
        return self.past_len + self.future_len

    def to_device(self, device: torch.device) -> None:
        for c in self.clients:
            c.train_t = torch.from_numpy(np.ascontiguousarray(c.train_X)).to(device)

    def summary(self) -> dict[str, Any]:
        tr = np.array([c.n_train for c in self.clients])
        va = np.array([c.val_X.shape[0] for c in self.clients])
        te = np.array([c.test_X.shape[0] for c in self.clients])

        def stats(a: np.ndarray) -> dict[str, float]:
            return {"total": int(a.sum()), "min": int(a.min()), "median": float(np.median(a)),
                    "max": int(a.max()), "mean": float(a.mean())}

        return {"dataset": self.name, "clients": self.n_clients, "train": stats(tr), "val": stats(va),
                "test": stats(te), "past_len": self.past_len, "future_len": self.future_len,
                "classes": self.class_names, "data_hash": self.data_hash}


def _split_indices(n: int, cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-client train/val/test split.

    ``time``: earliest (1-test_ratio) windows for training, the rest for testing (as in the
    original PTB-XL code).  ``random``: a seeded permutation (as in the original MIT-BIH code).
    The validation part is the *last* ``val_ratio`` share of the training part, so it is
    temporally between train and test under the time split.
    """
    n_test = max(1, int(cfg.test_ratio * n))
    order = np.arange(n) if cfg.split == "time" else rng.permutation(n)
    test = np.sort(order[n - n_test:]) if cfg.split == "time" else np.sort(order[:n_test])
    rest = order[: n - n_test] if cfg.split == "time" else np.sort(order[n_test:])
    n_val = int(round(cfg.val_ratio * len(rest)))
    val = rest[len(rest) - n_val:] if n_val else rest[:0]
    train = rest[: len(rest) - n_val]
    return train, val, test


def build_federated(cfg: Config, prepared: PreparedDataset | None = None, data_hash: str | None = None) -> FederatedData:
    if prepared is None:
        data_hash = verify_prepared(cfg.data_root, cfg.dataset)
        prepared = load_prepared(cfg.data_root, cfg.dataset)
    if prepared.past_len + prepared.future_len != cfg.window_len:
        raise ValueError(f"{cfg.dataset}: prepared window length {prepared.window_len} != config "
                         f"past_len+future_len {cfg.window_len}")
    raw = prepared.clients
    if cfg.min_windows:
        raw = [c for c in raw if c.X.shape[0] >= cfg.min_windows]
    if cfg.max_clients:
        raw = raw[: cfg.max_clients]
    rng = np.random.default_rng(cfg.split_seed)
    clients: list[Client] = []
    for k, rc in enumerate(raw):
        X, y = rc.X, rc.y
        if cfg.max_windows and X.shape[0] > cfg.max_windows:
            X, y = X[: cfg.max_windows], y[: cfg.max_windows]
        tr, va, te = _split_indices(X.shape[0], cfg, rng)
        clients.append(Client(k, rc.cid, X[tr], y[tr], X[va], X[te], y[te]))
    return FederatedData(prepared.name, cfg.past_len, cfg.future_len, prepared.class_names, clients,
                         data_hash or "unhashed", prepared.meta)


def eval_set(data: FederatedData, n_windows: int, seed: int, device: torch.device,
             split: str = "test") -> EvalSet:
    """A fixed, client-stratified evaluation subset shared by every method and seed.

    Windows are drawn round-robin over clients (each client's own order is a seeded
    permutation), so every client contributes and the subset does not depend on the
    training seed.  ``n_windows`` larger than the pool returns the whole pool.
    """
    rng = np.random.default_rng(seed)
    pools = []
    for c in data.clients:
        n = c.test_X.shape[0] if split == "test" else c.val_X.shape[0]
        pools.append(list(rng.permutation(n)))
    picks: list[tuple[int, int]] = []
    total = sum(len(p) for p in pools)
    target = min(n_windows, total)
    depth = 0
    while len(picks) < target:
        for ci, pool in enumerate(pools):
            if depth < len(pool) and len(picks) < target:
                picks.append((ci, int(pool[depth])))
        depth += 1
    ci_arr = np.array([p[0] for p in picks], dtype=np.int64)
    wi_arr = np.array([p[1] for p in picks], dtype=np.int64)
    src = [c.test_X if split == "test" else c.val_X for c in data.clients]
    X = np.stack([src[ci][wi] for ci, wi in picks]) if picks else np.zeros((0, data.window_len), np.float32)
    return EvalSet(torch.from_numpy(X).to(device), ci_arr, wi_arr)
