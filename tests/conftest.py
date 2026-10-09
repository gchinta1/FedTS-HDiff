import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    from fedtshdiff.data.prepare import prepare

    root = tmp_path_factory.mktemp("data")
    prepare("synthetic", raw_root=root / "raw", data_root=root)
    return root


@pytest.fixture
def syn_cfg(data_root):
    from fedtshdiff.config import build_config

    def make(**ov):
        base = {"dataset": "synthetic", "data_root": str(data_root), "device": "cpu"}
        base.update(ov)
        return build_config(base)

    return make
