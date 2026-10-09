# FedTS-HDiff revision code base

This is the experiment code for the TKDD major revision of *FedTS-HDiff: Distribution-Aware
Hierarchical Federated Diffusion Learning for Time-Series Forecasting*. One command runs every
experiment the reviewers asked for, in parallel on a multi-GPU node, with resume, provenance and
automatic tables, figures and a report.

Read these first:

- [`docs/ORIGINAL_CODE_AUDIT.md`](docs/ORIGINAL_CODE_AUDIT.md) — what the submitted code actually
  did, and the findings that change what the revision can claim.
- [`docs/REVISION_PLAN.md`](docs/REVISION_PLAN.md) — each reviewer comment mapped to an experiment,
  an output table/figure and a text change, plus the draft proposition for R2-C4.
- [`docs/METHODS.md`](docs/METHODS.md) — the protocol, method and metric definitions for §5.

The layout follows the FedStop/PEARL conventions: a `src/` package with typed YAML configs and
`--set` overrides; one subprocess per run, each seeing one GPU; fingerprint-checked resume; atomic
outputs; and stored results that regenerate every table.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# install the CUDA build of torch that matches the driver first, then:
python -m pip install -e ".[dev]"
python -m pytest -q          # 44 CPU tests on synthetic data, a few minutes
```

## Data

```bash
fedts prepare-data --dataset mitbih ptbxl electricity --raw-root raw --data-root data --download
# METR-LA: put metr-la.h5 (DCRNN release) in raw/metrla/, then
fedts prepare-data --dataset metrla --raw-root raw --data-root data
fedts describe-data --data-root data            # partition table for Reviewer 2, comment 7
```

Prepared data goes to `data/processed/<name>/`, which holds `windows.npz`, `meta.json` and
`SHA256SUMS`. Every run verifies the hashes, and the data hash is part of each run's fingerprint.
Data can be prepared on another machine and copied over as whole directories.

## Run

```bash
# a single run
fedts run --config src/fedtshdiff/configs/ptbxl.yaml --set method=fedavg --set seed=2 --out results/adhoc/x

# the whole revision campaign (656 runs) on the GPUs visible to the driver, 4 workers per GPU
fedts campaign --root results/revision --data-root data --gpus auto --slots-per-gpu 4 --dry-run
fedts campaign --root results/revision --data-root data --gpus auto --slots-per-gpu 4
fedts campaign ... --only tune,main,ablation      # subsets; --no-tuning uses the paper's baseline values
fedts status --root results/revision --watch 10
fedts analyse --root results/revision --data-root data
```

Campaign groups, in scheduling priority order:

| Group | Runs | What it answers |
|---|---|---|
| `cost` | 44 | 10-round runs, one worker per idle GPU: bytes, time, inference cost (R1-1, R2-6) |
| `tune` | 48 | validation selection of FedProx μ, FedDyn α, FedTDD λ and FedTS-HDiff β (same 3-point grid each) |
| `main` | 200 | 10 methods × 4 datasets × 5 seeds, with gradient diagnostics and DDIM-step evaluation |
| `ablation` | 170 (20 shared with main) | 9 variants × 4 datasets × 5 seeds (R2-8) |
| `descriptors` | 20 | label vs signal vs spectral descriptors on the ECG data (R2-2) |
| `drift` | 40 | concept drift on METR-LA and Electricity |
| `generative` | 40 | Gaussian LSTM and WGAN-GP under FedAvg (R1-2, R1-5) |
| `scalability` | 90 | PTB-XL: N clients × K edges, κ = 2, partial participation (R2-10) |
| `reproduce` | 24 | the submitted protocol (legacy DDIM spacing, MIT-BIH linear forms, METR-LA random split) |

Tasks that need a tuned value wait until the tuning runs for that (dataset, method) are sealed.
The selection is then written to `selection/<dataset>__<method>.json`. Identical configurations
requested by several groups run once.

**Resume.** Rerun the same command. A run counts as done only if its `done.json` fingerprint
matches the resolved configuration, the prepared-data hash and the science version. Anything else
reruns from scratch (there are no mid-run checkpoints). Workers left by a killed driver are
terminated on restart, and a lock prevents two drivers on one root. After a worker failure no new
run starts unless `--continue-on-error` is given; divergence is a recorded outcome, not a failure.

## Speed: client-batched training

The U-Net is so small that training one client at a time is limited by kernel launches. For
the U-Net diffusion methods, `client_batch` clients (default 32) are trained in one pass: their
parameters are rows of a [C, P] matrix, and convolutions run as batched matrix products. Each client
keeps its own random stream, padded minibatch slots are masked, and AdamW keeps per-client step
counters, so results equal one-at-a-time training up to floating point (`tests/test_batched.py`;
relative model difference ~1e-6). On MIT-BIH this is 6.4x faster per round. `client_batch` is not
part of the fingerprint. Cost-measurement runs use `client_batch=1`, so per-client times are
realistic. Centralised baselines, the LSTM and the GAN train one model at a time.

## Outputs

```
results/revision/
  campaign.json  status.json  driver.log  env.json  selection/*.json
  runs/<task_id>/  resolved_config.yaml meta.json rounds.csv curve.csv diagnostics.jsonl
                   final_metrics.json per_window.npz cost.json model.pt worker.log done.json
  analysis/  report.md runs.csv friedman.json window_level_tests.csv
             tables/*.tex|*.csv   figures/*.pdf|*.png
```

## G-Core deployment (7 × A100, GPU 0 reserved for Demo2)

```bash
python deploy/package_release.py                 # local: dist/fedtshdiff-<digest>.zip + .sha256
# copy the zip and .sha256 to the host, then on the host:
bash setup_gcore.sh fedtshdiff-<digest>.zip      # verify, private venv (copy of the FedStop CUDA stack), tests
~/FedTS-HDiff/run_gcore.sh prepare-data
~/FedTS-HDiff/run_gcore.sh bench                 # workers per GPU: 1/2/4/8 throughput -> FEDTS_SLOTS
~/FedTS-HDiff/run_gcore.sh preflight             # 2-round smoke of every variant on the real data
FEDTS_SLOTS=2 ~/FedTS-HDiff/run_gcore.sh start   # tmux session fedts-rev; CUDA_VISIBLE_DEVICES=1..7
~/FedTS-HDiff/run_gcore.sh status
~/FedTS-HDiff/run_gcore.sh analyse && ~/FedTS-HDiff/run_gcore.sh export
```

## Code map

```
src/fedtshdiff/
  config.py              typed flat config, validation, overrides, fingerprint
  data/                  prepare (MIT-BIH, PTB-XL, Electricity, METR-LA, synthetic), store, federated view, drift
  models/                U-Net (unchanged), diffusion + DDIM/DDPM, CSDI, Gaussian LSTM, WGAN-GP, Forecaster API
  fl/flat.py             flat parameters/gradients + exact AdamW with per-client state
  fl/batched.py          client-batched U-Net training (C clients per pass, identical random draws)
  fl/grouping.py         descriptors, balance/similarity/k-means/random/round-robin assignment, weights
  fl/trainer.py          all federated methods, kappa edge rounds, cost accounting
  fl/diagnostics.py      client gradients, Lemma-1 decomposition per partition, Mantel test
  fl/runner.py           one run end to end, outputs and seal
  eval/                  batched metrics (identical to the original), evaluation driver
  orchestrate/           campaign definition, planner (selection, resume), GPU-slot scheduler, worker
  analysis/              collection, statistics, LaTeX/CSV tables, figures, report
deploy/                  release packaging, G-Core setup/controller, concurrency benchmark
tests/                   CPU tests (metrics vs original code, AdamW, grouping, FedAvg equivalence, planner)
```
