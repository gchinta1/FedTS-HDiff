# Reviewer comments → experiments → outputs → manuscript changes

Every row names the campaign group that produces the evidence (`fedts campaign --only <group>`) and
the generated artefact under `<root>/analysis/`. "Text" lists changes that need no new runs.
Numbers must be taken from the generated tables; nothing below presumes an outcome.

## Reviewer 1

| # | Comment | Evidence (group → artefact) | Text |
|---|---|---|---|
| R1-1 | Communication/compute overhead of per-round regrouping vs flat FedAvg | `cost` → `tab_cost` (bytes on client links and across the cloud, per-round local training (parallel/sequential), grouping and aggregation ms, inference s/window, peak memory), `tab_ddim` | Replace "lightweight" claims with the measured numbers. Descriptor upload is d floats per client (d = #classes or 6). Grouping is O(N·K·d) per round. |
| R1-2 | Situate diffusion among LSTM / tree ensembles / hybrid ARIMA-ML | `generative` → `tab_generative` (Gaussian LSTM, WGAN-GP under FedAvg) | Extend §2.4. The requested citations [1–3,6,7] fit here. |
| R1-3 | Variance for Electricity/METR-LA | `main` → all tables mean ± SD over 5 seeds | Delete the "three runs on ECG only" paragraph. |
| R1-4 | Privacy of descriptors | — | Limitations: descriptors can be released through secure aggregation (Bonawitz et al., 2017) or perturbed with the Gaussian/Laplace mechanism (Dwork & Roth). Label histograms have L1 sensitivity 2/n_i. The label-free descriptors (R2-2) remove label leakage. |
| R1-5 | Diffusion vs GAN empirically | `generative` → `tab_generative` (accuracy, calibration, Div-DTW, seed SD, divergences), `fig_convergence` | §2.4 + §5: stability evidence, not only an argument. The requested citation [5] fits here. |
| R1-6 | Connect grouping to clustering / efficiency literature | `ablation` (`similarity_group`, `static_group`), `main` (`cfl`) | §2.3: k-means/CFL (Sattler; Briggs; Ghosh IFCA), balanced clustering, DEA-based clustering [8]. State that the rule is a sequential *balanced assignment* (FedPhD-style), not a clustering — see audit A1. |
| — | Requested references 1–8 | — | Add where relevant (the reviewer's integration examples are reasonable for 1–8). |

## Reviewer 2

| # | Comment | Evidence (group → artefact) | Text |
|---|---|---|---|
| R2-1 | Novelty vs clustered / hierarchical / federated-diffusion work | `main` (`cfl`), `ablation` (`edge_rounds2`), `scalability` | New paragraph contrasting with FedPhD (shared balancing score; no pruning; time-series conditioning), CFL/IFCA (one model per cluster vs one global model), HierFAVG (κ>1). Be explicit that the grouping score is FedPhD's. |
| R2-2 | Signal-only descriptors on ECG data | `descriptors` → `tab_descriptors` (label vs 6-statistic signal vs 8-band spectral) | §3.3 + §5. |
| R2-3 | Similarity ↔ within-group gradient dispersion | Diagnostics in `main` (rounds 1, 25, R/2, R+1) → `tab_dispersion` (δ_in/δ_global for random, round-robin, balance, similarity, k-means, population-balance, actual), `tab_descriptor_gradient_corr` (Spearman + Mantel p, Lipschitz q95), `fig_dispersion` | New subsection "Does descriptor similarity predict gradient similarity?". |
| R2-4 | Theory: does grouping reduce heterogeneity? | Empirical: `tab_dispersion`. Theory: Proposition below. | Add Proposition 2 + proof (below); restate the contribution as reducing the term the data show is reduced. |
| R2-5 | Multiple runs, mean ± SD, significance | `main`, `ablation`, `descriptors`, `drift`, `generative` (5 seeds) → `*_tests.csv` (paired t, exact Wilcoxon, Holm, d_z, 95% CI), `window_level_tests.csv`, `friedman.json` | Daggers in tables; a statistics paragraph in §5.3. |
| R2-6 | Communication, grouping/aggregation time, training and inference cost vs FedAvg/FedProx | `cost` → `tab_cost`, `tab_ddim` | New table. |
| R2-7 | Partition details | `fedts describe-data` → `tab_data` (clients, unit, windows train/val/test, per-client min/median/max, label TV, PSD TV, participation, K, split) | New table in §5.2. |
| R2-8 | Ablations on more datasets | `ablation` (all 4 datasets × 9 variants × 5 seeds) → `tab_ablation` | Replace the PTB-XL-only table. |
| R2-9 | Uncertainty calibration | `main` → `tab_calibration` (CRPS, multi-level WIS, calibration error over 10–90% intervals, PIT TV, coverage at 50/80/90), `fig_reliability`, `fig_pit` | New subsection. |
| R2-10 | Scalability: number of clients and edges; hierarchy benefit at scale | `scalability` (PTB-XL, N ∈ {100,200,400,800} × K ∈ {2,4,8,16}, κ = 2, participation q ∈ {0.1,0.25,0.5}, 3 seeds) → `tab_scalability`, `fig_scalability` (accuracy and cloud traffic vs N) | New subsection. Hierarchy's measurable benefit is cloud traffic (∝K rather than ∝N per round, and further reduced by κ). |
| R2-11 | Complete configuration details | `resolved_config.yaml` per run, `selections.csv`, `env.json`, `docs/METHODS.md` | Expand §5.3 with a table (optimizer, lr, wd, batch, epochs, rounds, T, schedule, DDIM, seeds, K, a/b/τ/α/β, participation, hardware). |

## Proposition for R2-4 (draft for the manuscript)

Let D_in(C) = Σ_k q_k Σ_{i∈C_k} p_{i|k} ‖φ_i − φ̄_k‖² and D_out(C) = Σ_k q_k ‖φ̄_k − φ̄‖², with
φ̄_k and φ̄ the p-weighted group and population descriptor means.

(a) *Descriptor-Lipschitz gradients.* If ‖∇F_i(θ) − ∇F_j(θ)‖ ≤ L_φ‖φ_i − φ_j‖ + ε for all θ, i, j,
then for every partition C: δ_in(C) ≤ 4 L_φ² D_in(C) + 2ε².

(b) *Affine-in-descriptor gradients.* If ∇F_i(θ) = Γ(θ)φ_i + r_i(θ) with ‖r_i(θ)‖ ≤ ε, then
δ_in(C) ≤ 2‖Γ‖²_op D_in(C) + 2ε² and δ_out(C) ≤ 2‖Γ‖²_op D_out(C) + 2ε².
Under pure label shift (shared class-conditionals), ∇F_i = Σ_c φ_i(c)∇F^c holds exactly
(ε = 0, Γ = [∇F^1 … ∇F^C]), and both bounds hold without the factor 2.

*Proof sketch.* (a) ∇F_i − ∇F_k = Σ_{j∈C_k} p_{j|k}(∇F_i − ∇F_j). Apply Jensen, then
(a+b)² ≤ 2a² + 2b², then Σ_{i,j} p_i p_j ‖φ_i − φ_j‖² = 2 Σ_i p_i ‖φ_i − φ̄‖² within each group.
(b) ∇F_i − ∇F_k = Γ(φ_i − φ̄_k) + (r_i − r̄_k) and ∇F_k − ∇F = Γ(φ̄_k − φ̄) + (r̄_k − r̄). Square,
apply (a+b)² ≤ 2a² + 2b², and bound the weighted residual dispersion by ε². Lipschitz continuity
alone does **not** bound δ_out by D_out, because the mean of the gradients is not the gradient at
the mean descriptor.

*Consequences.* (i) A rule that minimises D_in (similarity/k-means) controls δ_in. A rule that
minimises D_out (balancing towards φ̄) controls δ_out. The submitted "balance" rule targets the
uniform distribution and therefore bounds δ_out only if φ̄ is uniform (audit A8). (ii) L_φ and ε
can be estimated from the diagnostics (`lipschitz_q95`; Spearman/Mantel). (iii) With κ = 1 and
data-proportional weights, the update does not depend on the partition (Remark 3). The
partition therefore affects convergence only through B_w (adaptive weights) or through κ > 1,
where δ_in enters the edge-level drift term.

## Follow-up screen (after round 1; agreed with the co-authors, October 2026)

Round 1 showed FedTS-HDiff ≡ FedAvg up to the fourth decimal on three datasets (also under the submitted
protocol): with κ = 1 and near data-proportional weights the two-level aggregation is a weighted FedAvg
(Remark 3). The screen tests the two mechanisms that can make the hierarchy differ:

| Question | Evidence (group → artefact) |
|---|---|
| Does κ > 1 help at equal computation, and does the grouping rule matter then? | `screen` (`screen_{balance,similarity,kmeans}_k{1,2,5,10}`) → `tab_screen`, `fig_screen`, `tab_screen_tests_equal_compute.csv` |
| Accuracy per unit of cloud traffic | `screen_fedavg_c{κ}` → `tab_screen_tests_equal_cloud.csv`, `fig_screen` (CRPS vs total cloud MB) |
| Group-level personalisation (personalised baselines won round 1) | `screen_pers_m0.0_k1`, `screen_pers_m0.5_k{1,5}` → `tab_screen` |

Launch: `fedts campaign --only screen` (171 runs, MIT-BIH/METR-LA/PTB-XL × 3 seeds). Configurations
that beat both FedAvg references are then confirmed with 5 seeds (and on Electricity).

## Personalised hierarchy (after the screen, October 2026)

The screen showed that the hierarchy keeps FedAvg's accuracy at a fraction of the cloud traffic but does
not improve it, while client-level personalisation (FedRep) is the largest accuracy lever in round 1 and
the diagnostics show that groups capture only a small part of the client heterogeneity. The `personal`
group (100 runs) showed that personal heads on the hierarchy match FedRep on all four datasets at
120–375× less cloud traffic; they lose to FedAvg on PTB-XL. The `phsel` group adds the validation switch,
the shared-head hierarchy at the same settings and an ablation of the small gain over FedRep on MIT-BIH.

| Question | Evidence (group → artefact) |
|---|---|
| FedRep-level accuracy at hierarchical cloud traffic? | `ph_client_k{1,5}` vs `ph_fedrep` → `tab_personal`, `tab_personal_tests_vs_fedrep.csv`, `fig_personal` |
| Does choosing personal heads on validation avoid the PTB-XL loss? | `phval_*` → `tab_personal_selection.csv`; `ph_sel_k*` in `tab_personal` |
| Effect of personal heads within the same hierarchy | `ph_off_k*` vs `ph_client_k*` |
| Why personal heads on the hierarchy beat FedRep on MIT-BIH | `pha_*` → `tab_personal_ablation` |

## Suggested order of work
1. `preflight`, then `cost` + `tune` (cost needs idle GPUs; tuning gates the dependent groups).
2. `main` + `ablation` (largest; gives R1-3, R2-3, R2-5, R2-8, R2-9).
3. `descriptors`, `drift`, `generative`, `scalability`, `reproduce`.
4. `fedts analyse` → tables/figures → rewrite §3.3, §4, §5 to match the evidence.
