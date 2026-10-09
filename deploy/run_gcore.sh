#!/usr/bin/env bash
# Controller for the FedTS-HDiff revision campaign on G-Core (GPUs 1-7, one tmux session).
#
#   run_gcore.sh prepare-data          download + preprocess the four datasets (METR-LA: place metr-la.h5 in raw/metrla)
#   run_gcore.sh bench                 measure throughput at 1/2/4/8 workers per GPU (choose SLOTS)
#   run_gcore.sh preflight             two-round smoke of every variant on the real data
#   run_gcore.sh dry-run [groups]      print the plan
#   run_gcore.sh start [groups]        launch (or resume) the campaign in tmux session fedts-rev
#   run_gcore.sh status                progress summary
#   run_gcore.sh stop                  graceful stop (Ctrl-C to the driver)
#   run_gcore.sh analyse               rebuild tables/figures/report from stored results
#   run_gcore.sh export                tar.gz of analysis + sealed outputs (no checkpoints)
set -euo pipefail
BASE=${FEDTS_BASE:-/home/ubuntu/FedTS-HDiff}
VER=${FEDTS_VERSION:-$(cat "$BASE/ACTIVE_VERSION")}
PY="$BASE/venvs/$VER/bin/python"
ROOT=${FEDTS_ROOT:-$BASE/results/revision-$VER}
DATA="$BASE/data"
RAW="$BASE/raw"
SLOTS=${FEDTS_SLOTS:-2}   # client-batched runs saturate a GPU alone; 2 lets unbatched baselines overlap
SESSION=fedts-rev
export CUDA_VISIBLE_DEVICES=${FEDTS_GPUS:-1,2,3,4,5,6,7}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 COLUMNS=4096
only=${2:-}
only_flag=()
[ -n "$only" ] && only_flag=(--only "$only")

case "${1:-status}" in
  prepare-data)
    CUDA_VISIBLE_DEVICES="" "$PY" -m fedtshdiff.cli prepare-data --dataset mitbih ptbxl electricity \
        --raw-root "$RAW" --data-root "$DATA" --download
    CUDA_VISIBLE_DEVICES="" "$PY" -m fedtshdiff.cli prepare-data --dataset metrla --raw-root "$RAW" --data-root "$DATA"
    CUDA_VISIBLE_DEVICES="" "$PY" -m fedtshdiff.cli describe-data --data-root "$DATA" ;;
  bench)
    "$PY" "$BASE/releases/$VER/revision_code/deploy/bench_concurrency.py" --data-root "$DATA" \
        --out "$BASE/deployments/$VER/concurrency.json" ;;
  preflight)
    "$PY" -m fedtshdiff.cli preflight --root "$BASE/results/preflight-$VER" --data-root "$DATA" --gpus auto \
        --slots-per-gpu "$SLOTS" ;;
  dry-run)
    "$PY" -m fedtshdiff.cli campaign --root "$ROOT" --data-root "$DATA" --gpus auto --slots-per-gpu "$SLOTS" \
        "${only_flag[@]}" --dry-run ;;
  start|resume)
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "already running: tmux attach -t $SESSION"; exit 0
    fi
    for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
      used=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | wc -l)
      if [ "$used" -gt 0 ]; then echo "GPU $g is busy ($used processes); refusing to start"; exit 1; fi
    done
    mkdir -p "$ROOT"
    tmux new-session -d -s "$SESSION" \
      "$PY -m fedtshdiff.cli campaign --root $ROOT --data-root $DATA --gpus auto --slots-per-gpu $SLOTS ${only_flag[*]} >> $ROOT/campaign.out 2>&1"
    echo "started in tmux session $SESSION; results: $ROOT" ;;
  status)
    "$PY" -m fedtshdiff.cli status --root "$ROOT" ;;
  stop)
    tmux send-keys -t "$SESSION" C-c && echo "stop requested" ;;
  analyse)
    CUDA_VISIBLE_DEVICES="" "$PY" -m fedtshdiff.cli analyse --root "$ROOT" --data-root "$DATA" ;;
  export)
    out="$BASE/exports/revision-$VER-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    mkdir -p "$BASE/exports"
    (cd "$ROOT" && tar --exclude='model.pt' --exclude='tasks' -czf "$out" analysis campaign.json status.json env.json \
        selection runs driver.log 2>/dev/null || true)
    sha256sum "$out" | tee "$out.sha256" ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
