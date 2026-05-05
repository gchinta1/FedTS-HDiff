# FedTS-HDiff: Federated Time-Series Hierarchical Diffusion

Hierarchical federated conditional diffusion model for ECG time-series forecasting with uncertainty quantification using the MIT-BIH Arrhythmia Database.

---

## 📌 Overview

This project introduces **FedTS-HDiff (Federated Time-Series Hierarchical Diffusion)**, a hierarchical federated learning framework for conditional generative time-series forecasting under decentralized data constraints.

Using ECG signals from the MIT-BIH Arrhythmia Database as a heterogeneous real-world benchmark, the model learns the conditional distribution:

p(x_future | x_past)

Unlike deterministic forecasting approaches, **FedTS-HDiff** captures **multi-modal future trajectories** and provides **uncertainty quantification** via generative sampling.

---

## 🚀 Motivation

Modern forecasting systems increasingly face:

- Non-IID distributed datasets  
- Privacy constraints preventing centralization  
- Need for calibrated uncertainty  
- Multi-modal future behavior  

**FedTS-HDiff** integrates:

- Denoising Diffusion Probabilistic Models (DDPM)  
- DDIM-based fast sampling  
- Hierarchical Federated Aggregation  
- Distribution-aware client selection  

Applications include:

- Biomedical signals  
- Financial time-series  
- Macroeconomic forecasting  

---

## 🧠 Problem Formulation

Each ECG record is treated as a federated client.

For every detected heartbeat:

1. Extract a 128-sample window centered at the R-peak  
2. Split into:
   - Past: 64 samples  
   - Future: 64 samples  

The model learns:

εθ(x_t, t, x_past)

Training objective:

L = E ||ε - εθ(x_t, t, x_past)||²

Only the **future segment** is diffused and predicted.

---

## ⚙️ Methodology

### 1. Conditional Diffusion Model

- 1D U-Net backbone  
- Sinusoidal time embeddings  
- Residual blocks + GroupNorm  
- Linear beta schedule (T = 500)  
- Noise prediction objective  

**Inference:**

- DDIM sampling  
- Monte Carlo sampling (K trajectories)  
- Mean → forecast  
- Variance → uncertainty  

---

### 2. Hierarchical Federated Learning

**Architecture:**

Clients → Edge Aggregators → Global Server  

**Key Features:**

- Each ECG record = one client  
- Non-IID distributions  
- Distribution-aware client selection  
- Weighted aggregation using:
  - Client size  
  - Distribution similarity  
  - Edge diversity  

---

## 📊 Evaluation Metrics

- DTW (Dynamic Time Warping)  
- PSD-L2 Distance  
- Diversity (Pairwise DTW)  
- CRPS (Continuous Ranked Probability Score)  
- WIS90  
- PICP90  
- MPIW90  
- Diffusion Loss  

---

## 🔬 Experimental Setup

| Parameter | Value |
|----------|------|
| Past Length | 64 |
| Future Length | 64 |
| Diffusion Steps | 500 |
| DDIM Steps | 200 |
| Global Rounds | 200 |
| Local Epochs | 4 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |

---

## 🛠️ Installation

```bash
conda create -n fedts_hdiff python=3.10
conda activate fedts_hdiff
pip install torch numpy matplotlib wfdb
