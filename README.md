# federated-conditional-diffusion-time-series
Hierarchical federated conditional diffusion model for ECG time-series forecasting with uncertainty quantification (MIT-BIH dataset).

## Federated Conditional Diffusion Models for ECG Time-Series Forecasting

### Overview

This project studies conditional generative modeling for time-series forecasting under decentralized data constraints, using ECG signals from the MIT-BIH Arrhythmia Database as a heterogeneous real-world benchmark.

We develop a hierarchical federated learning framework for training a conditional diffusion model that learns the distribution:

$$p(x_{future} \mid x_{past})$$

Unlike deterministic regression models, this approach captures multi-modal future trajectories and provides uncertainty quantification through generative sampling.

### Research Motivation

Modern forecasting problems increasingly involve:

- Non-IID distributed datasets
- Privacy constraints preventing centralization
- Need for calibrated uncertainty
- Multi-modal future behavior

This project integrates:

- Denoising Diffusion Probabilistic Models (DDPM)
- DDIM-based fast sampling
- Hierarchical Federated Aggregation
- Distribution-aware client selection

The framework is general and extendable to biomedical, financial, and macroeconomic forecasting systems.

### Problem Formulation

Each ECG record is treated as a federated client.

For every detected heartbeat:

1. Extract 128-sample window centered at R-peak
2. Split into:
   - Past: 64 samples
   - Future: 64 samples

The model learns:

$$\epsilon_\theta(x_t, t, x_{past})$$

Training objective:

$$L = \mathbb{E}_{t,\epsilon} \|\epsilon - \epsilon_\theta(x_t, t, x_{past})\|_2$$

Only the future segment is diffused and predicted, making this a conditional generative forecasting framework.

### Methodology

#### 1. Conditional Diffusion Architecture

- 1D U-Net backbone
- Sinusoidal time embeddings
- Residual blocks with GroupNorm
- Linear beta schedule (T = 500)
- Noise prediction objective

**Inference:**

- DDIM sampling
- Monte Carlo generation (K samples)
- Mean trajectory = forecast
- Variance = uncertainty estimate

#### 2. Hierarchical Federated Learning

**Structure:**

Clients → Edge Aggregators → Global Server

**Properties:**

- Each ECG record = one client
- Non-IID label distributions
- Sh-score–based heterogeneity metric
- Distribution-aware client selection
- Weighted aggregation based on:
  - Client size
  - Distribution alignment
  - Edge-level diversity

This design reduces aggregation bias and improves robustness under heterogeneous data distributions.

### Evaluation Metrics

Forecasting quality is evaluated on the future segment only using:

- **DTW (Dynamic Time Warping)** — temporal alignment similarity
- **PSD-L2 Distance** — spectral distribution similarity
- **Diversity (Pairwise DTW)** — generative variability
- **CRPS (Continuous Ranked Probability Score)** — probabilistic forecast accuracy
- **WIS90 (Weighted Interval Score)** — prediction interval quality
- **PICP90 (Prediction Interval Coverage Probability)** — uncertainty calibration
- **MPIW90 (Mean Prediction Interval Width)** — prediction interval sharpness
- **Diffusion Loss** — training stability indicator
Metrics are averaged across multiple conditional cases.

### Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Past Length | 64 |
| Future Length | 64 |
| Diffusion Steps | 500 |
| DDIM Steps | 200 |
| Global Rounds | 200 |
| Local Epochs | 4 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |

### Installation

Create environment:

```bash
conda create -n fed_diffusion python=3.10
conda activate fed_diffusion
```

Install dependencies:

```bash
pip install torch numpy matplotlib wfdb
```

### Dataset Setup

Download the MIT-BIH Arrhythmia Database from PhysioNet.

Place the dataset in:

```
mit-bih-arrhythmia-database-1.0.0/
```

Ensure the directory contains `.hea`, `.dat`, and `.atr` files.

The script expects:

```python
MITBIH_DIR = "mit-bih-arrhythmia-database-1.0.0"
```

### Running the Experiment

```bash
python main_fedphd.py
```

Training will:

- Perform hierarchical federated rounds
- Periodically save forecast samples
- Compute conditional generative metrics
- Save performance curves

### Output Artifacts

Inside:

```
final_mitbih_fed_forecasting_ddim_64_64/
```

You will find:

- Sample forecast grids (`.png`)
- Raw forecast tensors (`.pt`)
- DTW curve
- PSD-L2 curve
- Diversity curve
- Diffusion loss curve

### Research Contributions

- Conditional diffusion for ECG forecasting
- Future-only diffusion formulation
- Hierarchical federated aggregation
- Distribution-aware client selection
- Monte Carlo uncertainty estimation
- Diversity-aware generative evaluation

### Potential Extensions

- Transformer-based backbone
- Personalization layers per client
- Adaptive edge clustering
- Multi-lead ECG forecasting
- Application to financial or macroeconomic time-series

### License

MIT License
