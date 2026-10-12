#!/usr/bin/env bash
# Install a verified FedTS-HDiff revision release on the G-Core A100 host.
#
#   bash setup_gcore.sh /path/to/fedtshdiff-<digest>.zip
#
# Layout (mirrors the FedStop deployments on the same host):
#   $BASE/releases/<ver>/revision_code   verified source
#   $BASE/venvs/<ver>                    private Python (copy of the FedStop CUDA stack + wfdb)
#   $BASE/deployments/<ver>              receipts, test results, runtime record
#   $BASE/data, $BASE/raw                prepared / raw datasets (shared across versions)
#   $BASE/results/<ver>                  campaign roots
set -euo pipefail
archive=$(readlink -f "$1")
BASE=${FEDTS_BASE:-/home/ubuntu/FedTS-HDiff}
SOURCE_VENV=${FEDTS_SOURCE_VENV:-/home/ubuntu/FedStop/venvs/v4-20261003-2a07db0cbccc}
GPUS=${FEDTS_GPUS:-1,2,3,4,5,6,7}   # physical GPU 0 hosts the Demo2 model service

ver=$(basename "$archive" .zip | sed 's/^fedtshdiff-//')
release="$BASE/releases/$ver"
venv="$BASE/venvs/$ver"
dep="$BASE/deployments/$ver"
mkdir -p "$BASE" "$dep" "$BASE/data" "$BASE/raw" "$BASE/results"

if [ -f "$archive.sha256" ]; then (cd "$(dirname "$archive")" && sha256sum --check "$(basename "$archive").sha256"); fi
python3 - "$archive" "$release" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path
archive, release = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        if not name.startswith("revision_code/") or ".." in Path(name).parts:
            raise SystemExit(f"unsafe member {name}")
    if not release.exists():
        release.mkdir(parents=True)
        z.extractall(release)
root = release / "revision_code"
manifest = json.loads((root / "release/SOURCE_MANIFEST.json").read_text())
for row in manifest["files"]:
    p = root / row["path"]
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"], row["path"]
print(json.dumps({"release": str(root), "verified_files": len(manifest["files"]), "code_digest": manifest["code_digest"]}))
PY
# zipfile.extractall drops Unix modes
find "$release/revision_code" -name '*.sh' -exec chmod 755 {} +

if [ ! -e "$venv" ]; then
    python3 -m venv --without-pip "$venv"
    old_site=$("$SOURCE_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    new_site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    cp -a "$old_site/." "$new_site/"   # private copy of the known-good CUDA stack; FedStop is untouched
fi
"$venv/bin/python" -m pip uninstall -y fedstop >/dev/null 2>&1 || true
"$venv/bin/python" -m pip install --no-deps --no-build-isolation -e "$release/revision_code"
# only add what is missing; never upgrade the copied CUDA/numeric stack
"$venv/bin/python" -c 'import psutil' 2>/dev/null || "$venv/bin/python" -m pip install "psutil>=5.9"
"$venv/bin/python" -c 'import wfdb' 2>/dev/null || "$venv/bin/python" -m pip install --upgrade-strategy only-if-needed "wfdb>=4.1"
"$venv/bin/python" -c 'import torch, numpy, scipy, pandas, yaml, h5py, matplotlib, wfdb, psutil; print("imports ok", torch.__version__, numpy.__version__)'
"$venv/bin/python" -m pip check

cd "$release/revision_code"
CUDA_VISIBLE_DEVICES="" "$venv/bin/python" -m pytest -q -p no:cacheprovider --junitxml "$dep/tests.xml" | tee "$dep/tests.log"
export CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$venv/bin/python" - "$dep" <<'PY'
import json, sys
from pathlib import Path
import torch
from fedtshdiff.utils import runtime_record
from fedtshdiff.orchestrate.scheduler import parse_slots
assert torch.cuda.is_available(), "CUDA not available"
slots = parse_slots("auto", 1)
rec = {**runtime_record(), "worker_physical_gpus": [s.device for s in slots],
       "visible": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
Path(sys.argv[1], "runtime_verified.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps({k: rec[k] for k in ("torch", "cuda_runtime", "worker_physical_gpus", "code_digest")}, indent=2))
PY
"$venv/bin/python" -m pip freeze > "$dep/requirements-frozen.txt"
ln -sfn "$release/revision_code/deploy/run_gcore.sh" "$BASE/run_gcore.sh"
echo "$ver" > "$BASE/ACTIVE_VERSION"
echo "installed $ver; next: $BASE/run_gcore.sh prepare-data && $BASE/run_gcore.sh preflight"
