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
    task = read_json(argv[0])
    cfg = Config().with_updates(**task["config"])
    return run_safely(cfg, Path(task["out_dir"]), task.get("fingerprint"))


if __name__ == "__main__":
    raise SystemExit(main())
