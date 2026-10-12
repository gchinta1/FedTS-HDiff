"""Measure training throughput with 1/2/4/8 concurrent workers on one GPU.

Each worker runs real client-batched local training (20 clients of the chosen dataset, paper
configuration) for a fixed wall-clock window and reports client optimiser steps per second.  The chosen slots-per-GPU is the
largest count whose total throughput improves by >= 10% over the next-smaller count (the
rule used for the FedStop campaign).  Run on an otherwise idle device.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHILD = r"""
import sys, time, torch, warnings
warnings.filterwarnings("ignore")
from fedtshdiff.config import build_config
from fedtshdiff.data.federated import build_federated
from fedtshdiff.fl.trainer import FederatedTrainer
from fedtshdiff.models import build_forecaster
from fedtshdiff.utils import set_numerics
set_numerics()
cfg = build_config(dataset=sys.argv[1], overrides={"data_root": sys.argv[2], "method": "fedavg", "max_clients": 20})
dev = torch.device("cuda")
data = build_federated(cfg); data.to_device(dev)
tr = FederatedTrainer(cfg, data, build_forecaster(cfg, dev), dev)
tr.current_round = 1
ids = list(range(len(tr.train_clients)))
tr.train_many(ids, [tr.global_vec] * len(ids))  # warm-up (client-batched, as in the campaign)
torch.cuda.synchronize()
deadline = time.time() + float(sys.argv[3]); steps = 0
t0 = time.time()
while time.time() < deadline:
    steps += sum(r.steps for r in tr.train_many(ids, [tr.global_vec] * len(ids)))
print(steps / (time.time() - t0))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--device", default=None, help="physical GPU id (default: first of CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--counts", default="1,2,4,8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    if not dev:
        raise SystemExit("pass --device or set CUDA_VISIBLE_DEVICES (never defaults to GPU 0)")
    if dev in {g.strip() for g in os.environ.get("FEDTS_FORBID_GPUS", "").split(",")}:
        raise SystemExit(f"GPU {dev} is forbidden (FEDTS_FORBID_GPUS)")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1"}
    results = {}
    for k in (int(c) for c in a.counts.split(",")):
        procs = [subprocess.Popen([sys.executable, "-c", CHILD, a.dataset, a.data_root, str(a.seconds)],
                                  stdout=subprocess.PIPE, text=True, env=env) for _ in range(k)]
        rates = [float(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        results[k] = {"per_worker_steps_per_s": rates, "total_steps_per_s": sum(rates)}
        print(k, results[k], flush=True)
    counts = sorted(results)
    chosen = counts[0]
    for prev, cur in zip(counts, counts[1:]):
        if results[cur]["total_steps_per_s"] >= 1.10 * results[prev]["total_steps_per_s"]:
            chosen = cur
        else:
            break
    rec = {"device": dev, "dataset": a.dataset, "seconds": a.seconds, "results": results,
           "rule": ">=10% total-throughput gain to increase workers per GPU", "slots_per_gpu": chosen,
           "time": time.time()}
    Path(a.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(f"recommended FEDTS_SLOTS={chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
