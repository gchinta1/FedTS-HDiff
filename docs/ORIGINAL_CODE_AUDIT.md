# Audit of the submitted code and paper (before the revision runs)

Scope: `src_original/FedTS-HDiff_Mit.py`, `src_original/FedTS-HDiff_PTB-XL.py`, and `paper/main.tex`
as submitted to TKDD. The Electricity, METR-LA and baseline scripts are **not** in `src_original/`,
so the results in Tables 6–9 cannot be traced to code. Every finding below that is marked
*verified* is covered by a unit test in `tests/`. Several findings change what the revision
can claim. Each one says what the new code does about it.

## A. Findings that affect the paper's claims

### A1. The grouping balances edges; it does not group similar clients *(verified)*
The paper says it groups "statistically similar clients" to reduce *within-group* gradient
dispersion (Abstract, Sec. 1, Sec. 3.3, Remark 1–2, Conclusion). The code does something else. It
assigns clients so that each edge's running descriptor moves toward a reference `u`: the uniform
label distribution for labeled data, or the mean descriptor otherwise. The score is
`2 - ||phi'_k - u||`, which the original source itself calls the "similarity score used in FedPHD".
This makes every edge *resemble the population*, so it deliberately mixes dissimilar clients.

By the paper's own Lemma 1, `delta_global = delta_in + delta_out` holds for every partition.
Making edge objectives close to the global objective therefore lowers **delta_out**
(between-group). For a fixed population it tends to *raise* **delta_in**.

*New code:* `grouping=balance` reproduces the original assignment exactly
(`test_balance_grouping_reproduces_original_loop`). `grouping=similarity|kmeans` actually
clusters similar clients. The diagnostics measure `delta_in`, `delta_out` and the aggregation
bias for every rule on the same gradients (`tab_dispersion`). The text and theory must be
re-aligned with whichever term the data show is reduced.

### A2. With one edge round per cloud round, hierarchy alone is FedAvg *(verified)*
With `RG = 1` (the only setting used), edge aggregation is followed immediately by cloud
aggregation. The result is a flat weighted average with effective weights
`a_i = q_k * w_{i|k}`. The paper's Remark 3 already concedes this. Its consequences for the
ablation are not drawn:

* "Grouping + standard aggregation" (`beta_agg = 0`) is **numerically identical to FedAvg**
  for any grouping (`test_hierarchy_with_data_proportional_weights_is_fedavg`).
* "w/o hierarchy" with data-proportional weights is also FedAvg.
* Therefore the PTB-XL ablation differences between FedAvg (0.2239), grouping + standard
  aggregation (0.2091) and w/o hierarchy (0.2335) are **run-to-run and evaluation noise** (see A4).

Grouping can only matter through (i) the adaptive weights (`beta_agg != 0`) or (ii) more than
one edge round per cloud round (`edge_rounds = kappa > 1`, now implemented). The second is
also the only setting in which hierarchy reduces cloud traffic.

### A3. MIT-BIH: FedTS-HDiff is effectively FedAvg *(verified by reading)*
The MIT-BIH script uses a different (linear) selection rule than the paper:
`score = max(0, a_sel * mu' - n' + b_sel)` with `a_sel = 10`. Here `mu' <= 2` and `n'` is a
raw sample count (hundreds to thousands), so every score is clipped to 0. Assignment is then
uniformly random. The aggregation weight `n_i + 20 * mu_i` differs from `n_i` by only a few
percent. The paper also states `alpha_agg = 20, beta_agg = 0` for MIT-BIH under the
*exponential* formula, which would give `w_i = n_i^20`. That is not what the code ran.

*New code:* every dataset uses the paper's softmax/exponential forms with
`alpha = beta = tau = 1`. The `reproduce` group reruns the original MIT-BIH forms.

### A4. Final metrics come from 20 random test windows *(verified by reading)*
`evaluate_forecasting_generator(..., num_cases=20)` draws 20 windows with a shuffled
DataLoader (10 for the convergence curves). Out of thousands of test windows, that sample is
much too small for differences in the third decimal place to be meaningful.

*New code:* every method and seed is scored on the **same fixed, client-stratified set of
1,024 test windows** (`eval_set`, seed independent of training). Each window gets 100 samples,
and the sampling noise is common to all methods. Per-window scores are stored, which enables
window-level paired tests.

### A5. DDIM-200 does not start from the noise level *(verified)*
`step_ratio = T // steps; range(0, T, step_ratio)[:steps]` gives, for T = 500 and 200 steps,
timesteps 398 → 0. Sampling therefore starts from pure noise treated as `x_398`
(`sqrt(alpha_bar) ≈ 0.2`). With 25/50/100 steps the first timestep is 480/490/495. The
DDIM-step ablation thus compares samplers with different start points, not only different
step counts.

*New code:* `ddim_spacing=trailing` always starts at `T-1`; `legacy` reproduces the original.

### A6. The model has 0.50 M parameters, not "≈1.2 M" *(verified)*
`UNet1D(base_ch=32, time_dim=128)` has 503,809 parameters (recorded as `params` in every run's `cost.json`).

### A7. PTB-XL: client homogeneity weights are a no-op *(verified by derivation)*
Each PTB-XL client is one *record* (the paper says "patient") and carries a single label. Its
descriptor is therefore one-hot, and `mu_i = 2 - ||e_c - u||` is identical for every client.
Client-level weights `exp(beta (mu_i - 1))` cancel in the normalisation, so within edges the
weights are exactly data-proportional. Only the edge-level weights `q_k` are adaptive.
Moreover, each client's label is constant, which makes the "label distribution" a class id.

### A8. Balancing towards *uniform* is not balancing towards the population
Under label shift (shared class-conditionals), a client gradient is linear in its label mix:
`grad F_i = sum_c phi_i(c) grad F^c`. So `||grad F_k - grad F|| <= ||Gamma|| ||phi_k - phi_bar||`,
where `phi_bar` is the *population* label distribution. The code instead targets the uniform
distribution, which differs from `phi_bar` on imbalanced data (PTB-XL is dominated by NORM).

*New code:* `homogeneity_ref=mean` targets the population mix. It is reported as an ablation
(`popmean_ref`) and in the diagnostics (`balance_popmean/label`).

## B. Protocol issues to correct in the text

| # | Issue | Revision |
|---|---|---|
| B1 | Electricity bounds differ between Sec. 5.2 (≤300 windows, ≤150 clients) and Sec. 5.3 (100–1000 windows, ≤60 clients). | Sec. 5.3 values are used; fix Sec. 5.2. |
| B2 | METR-LA uses a random split with stride 32 < window 128, so overlapping windows fall in both train and test (leakage). | Default is a time split; `reproduce` reruns the random split. |
| B3 | PTB-XL clients are records, not patients; `max_windows = 1024` never binds (10 s × 100 Hz gives 110 windows of 128 with stride 8). | Report "record"; train/val/test counts in `tab_data`. |
| B4 | No validation split; baseline hyperparameters were "selected" with no stated selection data. | 10% of each client's training part is held out; FedProx μ, FedDyn α, FedTDD λ and FedTS-HDiff β are selected on validation CRPS with the same 3-point grid each (`tune` group). |
| B5 | Tables bold FedTS-HDiff on metrics it does not win (e.g. MIT-BIH PSD-L2, Div-DTW, PICP). | Bold = best federated value only; daggers = significant after Holm correction. |
| B6 | Convergence-table intermediate values (e.g. 0.150, 0.100, 0.092) look read off the plots. | Curves and tables are generated from `curve.csv` (mean ± SD over seeds). |
| B7 | MIT-BIH windows include non-beat annotations ('+', '~', '|', …), which are not R-peak-centred. | `beats_only=True` (beat symbols only); the choice is documented. |
| B8 | Algorithm 1 computes descriptors after local training; the code assigns before training. | Equivalent (descriptors are data statistics); the text should say "before". |
| B9 | One persistent AdamW per client across rounds (state carried over), undocumented. | Kept for all methods (`client_opt_state=persistent`) and documented; `reset` is available. |

## C. What the original scripts did correctly and the revision keeps
The U-Net architecture, diffusion schedule, losses, normalisation and window construction are
unchanged. All seven paper metrics are numerically identical to the original implementation
(`test_paper_metrics_match_original_code`). The PTB-XL label extraction and record order are
kept, as are full participation, deterministic client order and the per-client AdamW protocol.
