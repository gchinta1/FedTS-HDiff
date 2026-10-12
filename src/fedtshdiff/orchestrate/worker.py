"""Subprocess entry point: ``python -m fedtshdiff.orchestrate.worker <task.json>``."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from ..config import Config
from ..fl.runner import run_safely
from ..utils import read_json


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m fedtshdiff.orchestrate.worker <task.json>", file=sys.stderr)
        return 2
    warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")
    import os

    forbid = {g.strip() for g in os.environ.get("FEDTS_FORBID_GPUS", "").split(",") if g.strip()}
    visible = {g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()}
    if forbid & visible:
        print(f"refusing to run on forbidden GPU(s) {sorted(forbid & visible)}", file=sys.stderr)
        return 3
    if os.environ.get("FEDTS_REQUIRE_CUDA") == "1":
        import torch

        if not torch.cuda.is_available():
            print(f"GPU slot but CUDA is unavailable (CUDA_VISIBLE_DEVICES={sorted(visible)})", file=sys.stderr)
            return 4
    task = read_json(argv[0])
    cfg = Config().with_updates(**task["config"])
    return run_safely(cfg, Path(task["out_dir"]), task.get("fingerprint"))


if __name__ == "__main__":
    raise SystemExit(main())
