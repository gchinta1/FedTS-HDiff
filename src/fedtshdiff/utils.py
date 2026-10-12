"""Small shared helpers: seeding, atomic files, hashing, timing, environment capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_numerics() -> None:
    """Full FP32: no TF32, no cuDNN autotuning (same policy as the FedStop campaigns)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) % (2**63 - 1))
    return g


def derive_seed(*parts: int | str) -> int:
    """Deterministic 63-bit seed from heterogeneous parts (stable across processes)."""
    blob = "/".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little") & ((1 << 63) - 1)


# --------------------------------------------------------------------------------------
# Atomic file output
# --------------------------------------------------------------------------------------


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:  # Windows: the target is open in another process (a reader)
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, default=_json_default) + "\n")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------


class Stopwatch:
    """Accumulating wall-clock timers; synchronises CUDA so GPU work is attributed correctly."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device
        self.totals: dict[str, float] = {}

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[key] = self.totals.get(key, 0.0) + time.perf_counter() - t0

    def pop(self) -> dict[str, float]:
        out, self.totals = self.totals, {}
        return out


# --------------------------------------------------------------------------------------
# Environment capture
# --------------------------------------------------------------------------------------


def code_digest() -> str:
    """SHA-256 over every .py and .yaml file of the installed package (sorted, path-tagged)."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts:
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            h.update(b"\0")
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
            h.update(b"\0")
    return h.hexdigest()


def runtime_record(cuda_devices: bool = True) -> dict[str, Any]:
    """Environment record.  ``cuda_devices=False`` avoids touching CUDA in this process (the
    scheduler driver) and lists GPUs through nvidia-smi only."""
    rec: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_digest": code_digest(),
    }
    if cuda_devices and torch.cuda.is_available():
        rec["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    try:
        rec["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        rec["nvidia_smi"] = None
    return rec
