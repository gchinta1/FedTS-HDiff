# Methods, protocol and definitions (for §5 of the revised manuscript)

## 1. Shared protocol (identical for every method)

| Item | Value |
|---|---|
| Backbone | Conditional 1-D U-Net of the submission (0.504 M parameters; base 32 channels, time embedding 128, GroupNorm(8), SiLU, two down/up levels) |
| Diffusion | Linear β ∈ [1e-4, 0.02]; T = 500 (ECG) / 200 (Electricity, METR-LA); ε-prediction MSE on the future segment only |
| Sampling | DDIM, η = 0, 200 steps, trailing spacing (starts at t = T−1); output clamped to [−1, 1] |
| Local optimiser | AdamW, lr 1e-4, weight decay 0.01, β = (0.9, 0.999), ε = 1e-8; one persistent optimiser state per client across rounds (as in the submission) |
| Local work | E = 4 epochs per round; batch 64 (ECG) / 32 (others); every epoch is a seeded permutation of the client's training windows |
| Rounds | R = 200 (ECG) / 250 (others); full participation unless stated |
| Splits | Per client: test = last 20% (time split) or a seeded random 20% (MIT-BIH); validation = last 10% of the remaining training part |
| Seeds | Reporting seeds 1–5; tuning seed 101; evaluation seed 2024 (fixed across methods and seeds) |
| Randomness control | Every client's local stream is keyed by (seed, round, client, edge round). All methods therefore see the same minibatch orders and diffusion noise for a given client and round, and the result does not depend on training order. Model initialisation depends only on the seed. |
| Numerics | Full FP32 (TF32 and cuDNN autotuning disabled) |
| Execution | Clients of the U-Net methods are trained in client-batched passes (identical random draws to one-at-a-time training; agreement to floating-point precision) |
| Evaluation | Fixed client-stratified set of 1,024 test windows (all test windows if fewer), 100 samples per window, sampling noise fixed by the evaluation seed. Convergence curves: first 256 windows of the same set, 50 samples, every 10 rounds. |

## 2. Methods

* **FedAvg**: data-proportional averaging of client models.
* **FedProx**: adds μ(θ − θ_start) to the local gradient; μ selected on validation from {0.001, 0.01, 0.1}.
* **FedNova**: θ ← θ_g + τ_eff Σ_i p_i (θ_i − θ_g)/τ_i, with τ_i the local step count and τ_eff = Σ p_i τ_i.
* **FedDyn**: local gradient g − h_i + α(θ − θ_g); h_i ← h_i − α(θ_i − θ_g); server h ← h − (α/N) Σ_{i∈S}(θ_i − θ_g), θ_g = mean_{i∈S} θ_i − h/α. α selected from {0.001, 0.01, 0.1}.
* **FedRep**: body = every U-Net parameter except `out_conv` (the head). Each round: 1 epoch on the head (body frozen), then E − 1 epochs on the body (head frozen). Only the body is aggregated. Evaluation uses each client's own head.
* **FedTDD (adapted)**: FedAvg plus a pool of *fully synthetic* windows. After local training each client generates 32 windows as [f₁ | f₂], with f₁ = G(real past) and f₂ = G(f₁) (25 DDIM steps), so no real values leave the client. In the next round every client adds λ × (diffusion loss on a minibatch from the other clients' pool). λ is selected from {0.01, 0.1, 1}. The original FedTDD addresses feature misalignment between clients with a server-side distillation model; with univariate, aligned series that mechanism reduces to synthetic-data exchange, which is what is implemented. Synthetic-window traffic is counted in the cost table.
* **Clustered FL (CFL)**: one-shot k-means (K = number of edges) of the client descriptors. One model is trained per cluster with FedAvg, and each client is evaluated with its cluster's model.
* **Centralised (same U-Net)**: the same model and optimiser on the union of all training windows, with R × E epochs (identical number of optimiser steps per sample).
* **CSDI (centralised)**: the official CSDI architecture for one feature (4 residual layers, 64 channels, 8-head temporal transformer, diffusion embedding 128, side information = time embedding 128 + feature embedding 16 + conditional mask), its quadratic schedule (T = 50, β ∈ [1e-4, 0.5]) and ancestral sampling, trained on the pooled data with the same budget.
* **FedAvg + Gaussian LSTM**: 2-layer LSTM (64 units), Gaussian likelihood, teacher forcing on the future positions, ancestral sampling.
* **FedAvg + WGAN-GP**: conditional generator (convolutional past encoder + noise 32 → MLP → tanh future) and critic on [past | future]. One critic update per minibatch, one generator update every 5 minibatches, gradient penalty 10, Adam(0.5, 0.9) without weight decay. Both networks are averaged.
* **FedTS-HDiff**: per round, descriptors → sequential softmax assignment to K edges (a = 10, b = 0, τ = 1; size penalty (n_k + n_i)/n̄) → κ edge rounds (each: local training from the edge model, then edge aggregation with w_i ∝ n_i^α exp(β(μ_i − 1))) → cloud aggregation with q_k ∝ n_k^α exp(β(μ_k − 1)). α = 1; β is selected on validation from {0.5, 1, 2}; κ = 1 unless stated.

## 3. Ablation variants (all datasets, 5 seeds)

| Label | Change from the full model | Note |
|---|---|---|
| `group_std_agg` | β = 0 (data-proportional weights at both levels) | ≡ FedAvg when κ = 1 (exact; see the audit, A2) |
| `adaptive_only` | no grouping (K = 1), adaptive weights kept | the "w/o hierarchy" model |
| `random_group` | uniform random edge per client and round | |
| `round_robin` | client j → edge j mod K | "w/o similarity" |
| `static_group` | assignment of round 1 kept | |
| `similarity_group` | k-means centres + softmax closeness score (similar clients together) | |
| `popmean_ref` | balance towards the population label mix instead of uniform | labeled datasets only |
| `edge_rounds2` | κ = 2 edge rounds per cloud round, R/2 cloud rounds | same local computation, half the cloud traffic |
| DDIM 25/50/100 | evaluation of the full model with fewer steps | no retraining |

## 4. Descriptors

* **label**: client label histogram (MIT-BIH: beat symbols; PTB-XL: diagnostic superclass).
* **signal**: [mean, SD, min, max, lag-1 autocorrelation, share of spectral power in the upper half of the frequencies] of the client's training windows. Min–max scaled across clients and divided by √6.
* **psd**: mean normalised power spectrum in 8 equal-width bands (a distribution).

Descriptors are computed from training windows only; under drift they are recomputed after every drift event.

## 5. Metrics

The paper's metrics are reproduced exactly (unit-tested against the original code): DTW (mean over samples), PSD-L2, Div-DTW (200 fixed sample pairs), CRPS, WIS₉₀, PICP₉₀, MPIW₉₀. Additional metrics: MSE/MAE of the ensemble mean, MAE of the median, DTW of the mean forecast, multi-level WIS over 10–90% central intervals (Bracher et al., 2021), the mean absolute calibration error |coverage − nominal| over those intervals, and the PIT histogram with its total-variation distance from uniform.

## 6. Statistics

Mean ± sample SD over 5 seeds. Each baseline is compared with FedTS-HDiff by a seed-paired t-test and an exact Wilcoxon signed-rank test. The 95% t-interval of the paired difference and the effect size d_z are reported, and p-values are Holm-adjusted across the baselines of one (dataset, metric). With n = 5 seeds the smallest two-sided exact Wilcoxon p-value is 0.0625, so the t-test is the primary test. A secondary window-level analysis averages each window's paired difference over seeds and applies a Wilcoxon test over the 1,024 windows, with a bootstrap interval. Friedman average ranks across the four datasets are given for completeness (4 datasets give low power).

## 7. Gradient diagnostics

At rounds 1, 25, R/2 and after the last round, every client's gradient of its local objective is computed at the current global model. Each client uses ≤ 256 of its training windows, with common diffusion timesteps and noise across clients. From the Gram matrix, δ_global, δ_in and δ_out (Lemma 1) and the aggregation bias ‖Σ a_i g_i − Σ p_i g_i‖² are computed for many partitions of the same clients. The correlation between pairwise descriptor distances and gradient distances is reported as Spearman ρ with a Mantel permutation p-value (999 permutations), together with the 95th percentile of ‖g_i − g_j‖/‖φ_i − φ_j‖ (an empirical L_φ).

## 8. Cost accounting

Bytes are counted analytically from the parameter count (FP32): client links = upload + download of every participant per edge round (+ descriptor floats), and cloud = traffic that crosses the cloud server (flat methods: all clients; hierarchical: K edge models up and down). Wall-clock is measured with CUDA synchronisation: local training (sum, and per-edge-round maximum = time with parallel clients), grouping and aggregation per round, and sampling time per window (100 samples) at each DDIM step count. Peak GPU memory is also recorded. Cost runs are 10-round runs with one worker on an otherwise idle A100.
