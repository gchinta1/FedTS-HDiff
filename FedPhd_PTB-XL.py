"""
PTB-XL (WFDB) + Federated FedPHD + Conditional Diffusion Forecasting with DDIM

Requirements:
  pip install wfdb pandas numpy torch matplotlib
"""

import os
import math
import random
import ast
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt


# CONFIG

PTBXL_ROOT = r"ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DATA_DIR = os.path.join(PTBXL_ROOT, "records100")  # 100 Hz

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Forecasting window: 64 past + 64 future = 128
PAST_LEN = 64
FUTURE_LEN = 64
WINDOW_LEN = PAST_LEN + FUTURE_LEN

USE_LEAD = 0

# Sliding-window extraction
MAX_WINDOWS_PER_RECORD = 1024
MIN_WINDOWS_PER_CLIENT = 50
STRIDE = 8

MAX_CLIENTS = 300
MAX_SCAN = 4000

# -----------------------------
# FedPHD settings (FIXED)
# -----------------------------
NUM_EDGES = 4
R = 200
LOCAL_EPOCHS = 4
BATCH_SIZE = 64

# Selection hyperparams
A_SEL = 10.0        
B_SEL = 0.0          
TAU_SEL = 1.0         

# Aggregation hyperparams (multiplicative)
ALPHA_AGG = 1.0       
BETA_AGG = 1.0        

# Global aggregation frequency (keep same behavior)
RG = 1

SAVE_EVERY = 25

# Diffusion schedule
T_TRAIN = 500
BETA_START = 1e-4
BETA_END = 0.02

DDIM_STEPS = 200
DDIM_ETA = 0.0

LR = 1e-4

OUT_DIR = "ptbxl_fedphd_fixed_forecasting_ddim_64_64"
os.makedirs(OUT_DIR, exist_ok=True)


# PTB-XL LABEL LOADING (diagnostic_class)

def load_ptbxl_diagnostic_class_labels(ptbxl_root: str, sampling_rate: int = 100):
    """
    Uses scp_statements.csv column: diagnostic_class
    Returns:
      recid_to_label: dict[rec_id -> int]
      label_map: dict[label_name -> int]
    """
    db_path = os.path.join(ptbxl_root, "ptbxl_database.csv")
    scp_path = os.path.join(ptbxl_root, "scp_statements.csv")

    df = pd.read_csv(db_path)
    scp = pd.read_csv(scp_path, index_col=0)

    if "diagnostic_class" not in scp.columns:
        raise ValueError("scp_statements.csv does not have 'diagnostic_class' column.")

    df["scp_codes"] = df["scp_codes"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})

    fn_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if fn_col not in df.columns:
        raise ValueError(f"Column {fn_col} not found in ptbxl_database.csv")

    code_to_class = {}
    for code, row in scp.iterrows():
        val = row.get("diagnostic_class", None)
        if isinstance(val, str) and val.strip():
            code_to_class[str(code)] = val.strip()

    def to_rec_id(relpath: str):
        rel = str(relpath).replace("\\", "/")
        if sampling_rate == 100 and rel.startswith("records100/"):
            rel = rel[len("records100/"):]
        if sampling_rate == 500 and rel.startswith("records500/"):
            rel = rel[len("records500/"):]
        rel = rel.replace("/", os.sep).replace("\\", os.sep)
        return os.path.normpath(rel)

    recid_to_labelname = {}
    skipped = 0

    for _, row in df.iterrows():
        rec_id = to_rec_id(row[fn_col])
        scp_codes = row["scp_codes"] or {}

        classes = set()
        for code in scp_codes.keys():
            c = code_to_class.get(str(code), None)
            if c is not None:
                classes.add(c)

        if not classes:
            skipped += 1
            continue

        chosen = sorted(list(classes))[0]  # deterministic single-label
        recid_to_labelname[rec_id] = chosen

    uniq = sorted(set(recid_to_labelname.values()))
    label_map = {name: i for i, name in enumerate(uniq)}
    recid_to_label = {rid: label_map[name] for rid, name in recid_to_labelname.items()}

    print(f"[Labels] source=diagnostic_class | #labeled_records={len(recid_to_label)} | #skipped={skipped}")
    print(f"[Labels] #classes={len(label_map)} | classes={list(label_map.keys())}")
    return recid_to_label, label_map


# WFDB RECORD LISTING + WINDOW EXTRACTION

def list_record_ids(root_dir: str) -> List[str]:
    recs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".hea"):
                base = fn[:-4]
                rel_dir = os.path.relpath(dirpath, root_dir)
                rec_id = base if rel_dir == "." else os.path.join(rel_dir, base)
                recs.append(os.path.normpath(rec_id))
    return sorted(set(recs))


def robust_zscore_to_minus1_1(x: np.ndarray) -> np.ndarray:
    m = float(x.mean())
    s = float(x.std())
    s = max(s, 1e-6)
    x = (x - m) / (3.0 * s)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def extract_windows_for_record(
    root_dir: str,
    rec_id: str,
    use_lead: int,
    window_len: int = 128,
    max_windows: int | None = 1024,
    stride: int = 32,
):
    import wfdb

    rec_path = os.path.join(root_dir, rec_id)
    record = wfdb.rdrecord(rec_path)

    sig = record.p_signal
    if sig is None or sig.ndim != 2:
        return np.zeros((0, 1, window_len), dtype=np.float32)

    if use_lead >= sig.shape[1]:
        use_lead = 0

    x = sig[:, use_lead].astype(np.float32)
    if len(x) < window_len:
        return np.zeros((0, 1, window_len), dtype=np.float32)

    x = robust_zscore_to_minus1_1(x)

    X_list = []
    for start in range(0, len(x) - window_len + 1, stride):
        win = x[start:start + window_len]
        X_list.append(win[None, :])
        if max_windows is not None and len(X_list) >= max_windows:
            break

    if len(X_list) == 0:
        return np.zeros((0, 1, window_len), dtype=np.float32)

    X = np.stack(X_list, axis=0).astype(np.float32)
    return X


def build_federated_clients_from_wfdb_records(
    root_dir: str,
    recid_to_label: Dict[str, int],
    label_map: Dict[str, int],
):
    rec_ids = list_record_ids(root_dir)
    print(f"Found {len(rec_ids)} records (.hea files) under records100/")

    client_X_list = []
    client_y_list = []

    skip_count = 0
    windows_counts = []
    scanned = 0
    labeled_seen = 0

    for rid in rec_ids:
        scanned += 1
        if scanned > MAX_SCAN:
            break

        if rid not in recid_to_label:
            continue
        labeled_seen += 1

        try:
            X = extract_windows_for_record(
                root_dir=root_dir,
                rec_id=rid,
                use_lead=USE_LEAD,
                window_len=WINDOW_LEN,
                max_windows=MAX_WINDOWS_PER_RECORD,
                stride=STRIDE,
            )
        except Exception:
            skip_count += 1
            continue

        windows_counts.append(int(X.shape[0]))
        if X.shape[0] < MIN_WINDOWS_PER_CLIENT:
            continue

        lab = int(recid_to_label[rid])
        y = np.full((X.shape[0],), lab, dtype=np.int64)

        client_X_list.append(X)
        client_y_list.append(y)

        if len(client_X_list) >= MAX_CLIENTS:
            break

    print("Total labeled records encountered:", labeled_seen)
    print("Total skipped records (read errors):", skip_count)
    if windows_counts:
        print("Windows stats:",
              "min=", min(windows_counts),
              "max=", max(windows_counts),
              "mean=", sum(windows_counts) / len(windows_counts))
    print("Total kept clients:", len(client_X_list))

    if len(client_X_list) == 0:
        raise RuntimeError("No usable labeled records found. Check paths/stride/windows/MIN_WINDOWS_PER_CLIENT.")

    print("Num classes:", len(label_map))
    return client_X_list, client_y_list, label_map


# DATASET + SPLIT (TIME-BASED)

class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()  # (N,1,128)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], int(self.y[idx])


def make_global_testset_timebased(client_X_list, client_y_list, test_ratio=0.2):
    """
    No shuffle. Uses tail windows as test to reduce leakage from overlap.
    """
    test_X, test_y = [], []
    client_train_indices = []

    for X, y in zip(client_X_list, client_y_list):
        n = X.shape[0]
        n_test = int(test_ratio * n)
        n_test = max(1, n_test)

        train_idx = np.arange(0, n - n_test)
        test_idx = np.arange(n - n_test, n)

        test_X.append(X[test_idx])
        test_y.append(y[test_idx])
        client_train_indices.append(train_idx.tolist())

    test_X = np.concatenate(test_X, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    global_test = WindowsDataset(test_X, test_y)
    return client_train_indices, global_test


# FedPHD UTILITIES (same structure, but FIXED scoring usage)

def sh_score(q: Dict[int, float], target: Dict[int, float]) -> float:
    s = 0.0
    for k in target.keys():
        diff = q.get(k, 0.0) - target.get(k, 0.0)
        s += diff * diff
    return 2.0 - math.sqrt(s)


def compute_label_distribution_from_local_y(local_y: np.ndarray, indices: List[int], num_classes: int):
    cnt = Counter(int(local_y[i]) for i in indices)
    total = len(indices)
    if total <= 0:
        return {c: 0.0 for c in range(num_classes)}
    return {c: cnt.get(c, 0) / total for c in range(num_classes)}


def update_edge_distribution(qe, ne, qn, nn, keys):
    new_ne = ne + nn
    if new_ne <= 0:
        return qe, ne
    new_qe = {}
    for k in keys:
        edge_part = qe.get(k, 0.0) * ne
        client_part = qn.get(k, 0.0) * nn
        new_qe[k] = (edge_part + client_part) / new_ne
    return new_qe, new_ne


def stable_softmax(scores: List[float], tau: float = 1.0) -> List[float]:
    s = np.array(scores, dtype=np.float64)
    s = (s - s.max()) / max(tau, 1e-8)
    p = np.exp(s)
    p = p / (p.sum() + 1e-12)
    return p.tolist()


# DIFFUSION SCHEDULE

betas = torch.linspace(BETA_START, BETA_END, T_TRAIN, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    out = a.gather(-1, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ac = extract(sqrt_alphas_cumprod, t, x0.shape)
    sqrt_om = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
    x_t = sqrt_ac * x0 + sqrt_om * noise
    return x_t, noise


# MODEL (UNet1D) - same as your original

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_factor)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = F.silu(self.lin1(emb))
        emb = self.lin2(emb)
        return emb


class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb).view(t_emb.size(0), -1, 1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet1D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base_ch=32, time_dim=128):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)

        self.in_conv = nn.Conv1d(in_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock1D(base_ch, base_ch * 2, time_dim)
        self.down2 = ResBlock1D(base_ch * 2, base_ch * 4, time_dim)

        self.pool = nn.MaxPool1d(2)

        self.mid1 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.up1 = ResBlock1D(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim)
        self.up2 = ResBlock1D(base_ch * 2 + base_ch * 2, base_ch, time_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv1d(base_ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_emb(t)

        x0 = self.in_conv(x)
        d1 = self.down1(x0, t_emb)
        p1 = self.pool(d1)

        d2 = self.down2(p1, t_emb)
        p2 = self.pool(d2)

        m = self.mid2(self.mid1(p2, t_emb), t_emb)

        u1 = self.up(m)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.up(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.up2(u2, t_emb)

        return self.out_conv(u2)


def get_diffusion_model():
    return UNet1D(in_ch=2, out_ch=1, base_ch=32, time_dim=128).to(device)


def split_past_future(x_full: torch.Tensor):
    past = x_full[:, :, :PAST_LEN]
    future0 = x_full[:, :, PAST_LEN:]
    return past, future0


# LOSS / TRAINING

def diffusion_loss(model, x_full):
    past, future0 = split_past_future(x_full)

    b = x_full.size(0)
    t = torch.randint(0, T_TRAIN, (b,), device=device).long()

    noise = torch.randn_like(future0)
    future_t, noise = q_sample(future0, t, noise)

    model_in = torch.cat([future_t, past], dim=1)  # (B,2,64)
    noise_pred = model(model_in, t)                # (B,1,64)
    return F.mse_loss(noise_pred, noise)


def train_local_one_round(model, opt, loader, epochs=1):
    model.train()
    for _ in range(epochs):
        for x_full, _ in loader:
            x_full = x_full.to(device, non_blocking=True)
            loss = diffusion_loss(model, x_full)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def evaluate_loss(model, test_dataset, num_batches=10):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=0)
    total, count = 0.0, 0
    for i, (x_full, _) in enumerate(loader):
        if i >= num_batches:
            break
        x_full = x_full.to(device, non_blocking=True)
        total += diffusion_loss(model, x_full).item()
        count += 1
    return total / max(count, 1)


# DDIM SAMPLING

@torch.no_grad()
def ddim_sample_future(model, past, num_samples=200, ddim_steps=100, eta=0.0):
    model.eval()

    B = num_samples
    past = past.to(device)

    if past.size(0) == 1 and B > 1:
        past = past.repeat(B, 1, 1)
    elif past.size(0) != B:
        raise ValueError("past batch size must be 1 or num_samples")

    step_ratio = max(1, T_TRAIN // ddim_steps)
    ddim_timesteps = list(range(0, T_TRAIN, step_ratio))[:ddim_steps]
    ddim_timesteps = list(reversed(ddim_timesteps))

    x = torch.randn(B, 1, FUTURE_LEN, device=device)

    for i, t_int in enumerate(ddim_timesteps):
        t = torch.full((B,), t_int, device=device, dtype=torch.long)
        eps = model(torch.cat([x, past], dim=1), t)  # (B,1,64)

        alpha_bar_t = alphas_cumprod[t_int]
        alpha_bar_prev = (
            alphas_cumprod[ddim_timesteps[i + 1]]
            if (i + 1) < len(ddim_timesteps)
            else torch.tensor(1.0, device=device)
        )

        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        if eta == 0.0:
            sigma_t = 0.0
        else:
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev)
                / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps
        noise = torch.randn_like(x) if sigma_t > 0 else 0.0
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise

    return x.clamp(-1, 1)


# SAVE SAMPLES

def _save_plot_grid(samples_full: torch.Tensor, png_path: str, ncol: int = 4):
    samples = samples_full.detach().cpu().numpy()
    N = samples.shape[0]
    nrow = int(math.ceil(N / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 2))
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = np.array([axes])
    elif ncol == 1:
        axes = axes.reshape(-1, 1)

    idx = 0
    for r in range(nrow):
        for c in range(ncol):
            ax = axes[r, c]
            ax.axis("off")
            if idx < N:
                ax.axis("on")
                ax.plot(samples[idx])
                ax.set_xticks([])
                ax.set_yticks([])
            idx += 1

    plt.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_samples_forecasting(model, global_testset, tag: str, n: int = 16, K: int = 100):
    loader = DataLoader(global_testset, batch_size=n, shuffle=True, num_workers=0)
    x_full, _ = next(iter(loader))
    x_full = x_full.to(device)

    past, _ = split_past_future(x_full)

    mean_futures = []
    for i in range(n):
        past_i = past[i:i + 1]
        futures_K = ddim_sample_future(model, past_i, num_samples=K, ddim_steps=DDIM_STEPS, eta=DDIM_ETA)
        mean_i = futures_K.mean(dim=0, keepdim=True)
        mean_futures.append(mean_i)

    mean_future = torch.cat(mean_futures, dim=0)
    stitched = torch.cat([past, mean_future], dim=2).squeeze(1)

    pt_path = os.path.join(OUT_DIR, f"{tag}.pt")
    png_path = os.path.join(OUT_DIR, f"{tag}.png")
    torch.save(stitched.detach().cpu(), pt_path)
    _save_plot_grid(stitched, png_path, ncol=int(math.sqrt(n)) or 4)

    print("Saved:", pt_path)
    print("Saved:", png_path)


# METRICS

def dtw_distance(a, b):
    L = len(a)
    D = np.full((L + 1, L + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, L + 1):
        ai = a[i - 1]
        for j in range(1, L + 1):
            cost = (ai - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[L, L] / L))


def psd_features_rfft(x, nfft=256):
    n, L = x.shape
    nfft = max(nfft, L)
    X = np.fft.rfft(x, n=nfft, axis=1)
    P = (X.real**2 + X.imag**2).astype(np.float64)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return P.astype(np.float32)


def psd_l2(real, fake, nfft=256):
    fr = psd_features_rfft(real, nfft=nfft)
    ff = psd_features_rfft(fake, nfft=nfft)
    mr = fr.mean(axis=0)
    mf = ff.mean(axis=0)
    return float(np.sqrt(np.mean((mr - mf) ** 2)))


def diversity_dtw(fake, pairs=200, seed=43):
    rng = np.random.default_rng(seed)
    n = fake.shape[0]
    d = []
    for _ in range(pairs):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            j = (j + 1) % n
        d.append(dtw_distance(fake[i], fake[j]))
    return float(np.mean(d)), float(np.std(d))


@torch.no_grad()
def forecast_with_uncertainty(model, x_full_1, K=200):
    x_full_1 = x_full_1.to(device)
    past, future0 = split_past_future(x_full_1)

    true_future = future0.detach().cpu().numpy().squeeze()

    futures = ddim_sample_future(model, past, num_samples=K, ddim_steps=DDIM_STEPS, eta=DDIM_ETA)
    futures_np = futures.detach().cpu().numpy().squeeze(1)

    mean = futures_np.mean(axis=0)
    std = futures_np.std(axis=0)
    return true_future, mean, std, futures_np


def conditional_future_metrics(true_future, future_samples):
    dtws = [dtw_distance(true_future, future_samples[k]) for k in range(future_samples.shape[0])]
    dtw_mean = float(np.mean(dtws))
    dtw_std = float(np.std(dtws))
    psd_dist = psd_l2(true_future[None, :], future_samples, nfft=FUTURE_LEN)
    div_mean, div_std = diversity_dtw(future_samples, pairs=200)
    return {
        "dtw_mean": dtw_mean,
        "dtw_std": dtw_std,
        "psd_l2": psd_dist,
        "div_dtw_mean": div_mean,
        "div_dtw_std": div_std,
    }


@torch.no_grad()
def evaluate_forecasting_generator(model, global_testset, num_cases=10, K=200):
    loader = DataLoader(global_testset, batch_size=1, shuffle=True, num_workers=0)
    results = []
    for i, (x_full, _) in enumerate(loader):
        if i >= num_cases:
            break
        true_future, _mean, _std, samples = forecast_with_uncertainty(model, x_full, K=K)
        results.append(conditional_future_metrics(true_future, samples))

    out = {}
    for k in results[0].keys():
        out[k] = float(np.mean([r[k] for r in results]))

    print("\n=== CONDITIONAL (PAST->FUTURE) METRICS AVERAGED OVER CASES ===")
    for k, v in out.items():
        print(f"{k}: {v:.6f}")
    return out


# STATE HELPERS

def get_model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_model_state(model, state_dict):
    model.load_state_dict(state_dict)


def average_states(states, weights):
    avg = {}
    for k in states[0].keys():
        avg[k] = sum(w * s[k] for s, w in zip(states, weights))
    return avg


# MAIN (FedPHD FIXED)

def main():
    # Load labels
    recid_to_label, label_map = load_ptbxl_diagnostic_class_labels(PTBXL_ROOT, sampling_rate=100)

    # Build clients
    client_X_list, client_y_list, label_map = build_federated_clients_from_wfdb_records(
        DATA_DIR, recid_to_label, label_map
    )
    NUM_CLIENTS = len(client_X_list)
    num_classes = len(label_map)
    print(f"Clients: {NUM_CLIENTS}, classes: {num_classes}")

    # Datasets + time-based split
    client_datasets = [WindowsDataset(X, y) for X, y in zip(client_X_list, client_y_list)]
    client_train_indices, global_testset = make_global_testset_timebased(client_X_list, client_y_list, test_ratio=0.2)

    # label distribution stuff
    keys = list(range(num_classes))
    target = {c: 1.0 / num_classes for c in range(num_classes)}

    client_sizes = [len(idxs) for idxs in client_train_indices]
    avg_n = float(np.mean([s for s in client_sizes if s > 0])) if any(s > 0 for s in client_sizes) else 1.0

    client_dists = [
        compute_label_distribution_from_local_y(client_y_list[i], client_train_indices[i], num_classes)
        for i in range(NUM_CLIENTS)
    ]
    client_mu = [sh_score(q, target) for q in client_dists]

    global_model = get_diffusion_model()
    global_state = get_model_state(global_model)

    # initial save + metrics
    save_samples_forecasting(global_model, global_testset, tag="round_0000", n=16, K=100)
    _ = evaluate_forecasting_generator(global_model, global_testset, num_cases=10, K=200)

    # FedPHD edge stats
    qe = [{k: 0.0 for k in keys} for _ in range(NUM_EDGES)]
    ne = [0 for _ in range(NUM_EDGES)]

    # client models + opts
    client_models = [get_diffusion_model() for _ in range(NUM_CLIENTS)]
    client_opts = [torch.optim.AdamW(client_models[i].parameters(), lr=LR) for i in range(NUM_CLIENTS)]

    for r in range(1, R + 1):
        print(f"\n=== Global Round {r}/{R} (FedPHD FIXED) ===")

        # broadcast global -> clients
        for m in client_models:
            set_model_state(m, global_state)

        # -------- FIXED client -> edge assignment ----------
        client_to_edge = []
        for cid in range(NUM_CLIENTS):
            scores = []
            for e in range(NUM_EDGES):
                q_prime, n_prime = update_edge_distribution(
                    qe[e], ne[e],
                    client_dists[cid], client_sizes[cid],
                    keys
                )
                mu_prime = sh_score(q_prime, target)

                # FIX 1: normalize n_prime so it doesn't dominate
                n_norm = float(n_prime) / max(avg_n, 1.0)

                # score on comparable scales
                score = A_SEL * mu_prime - n_norm + B_SEL
                scores.append(score)

            # FIX 2: stable softmax (no all-zero collapse)
            probs = stable_softmax(scores, tau=TAU_SEL)

            rnd = random.random()
            cum = 0.0
            chosen = 0
            for e, p in enumerate(probs):
                cum += p
                if rnd <= cum:
                    chosen = e
                    break

            client_to_edge.append(chosen)
            qe[chosen], ne[chosen] = update_edge_distribution(
                qe[chosen], ne[chosen],
                client_dists[cid], client_sizes[cid],
                keys
            )
        # ---------------------------------------------------

        # local training
        client_states = []
        for cid in range(NUM_CLIENTS):
            idxs = client_train_indices[cid]
            if len(idxs) == 0:
                client_states.append(global_state)
                continue

            loader = DataLoader(
                Subset(client_datasets[cid], idxs),
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            train_local_one_round(client_models[cid], client_opts[cid], loader, epochs=LOCAL_EPOCHS)
            client_states.append(get_model_state(client_models[cid]))

        # edge aggregation (FIX 3: multiplicative weights)
        edge_states = []
        for e in range(NUM_EDGES):
            members = [cid for cid in range(NUM_CLIENTS) if client_to_edge[cid] == e]
            if not members:
                edge_states.append(global_state)
                continue

            raw_w = []
            for cid in members:
                size = max(1, client_sizes[cid])
                mu = client_mu[cid]
                w = (size ** ALPHA_AGG) * math.exp(BETA_AGG * (mu - 1.0))
                raw_w.append(float(w))

            wsum = sum(raw_w) or 1.0
            weights = [w / wsum for w in raw_w]
            member_states = [client_states[cid] for cid in members]
            edge_states.append(average_states(member_states, weights))

        # global aggregation (FIX 3 also)
        if (r % RG) == 0:
            edge_mu = [sh_score(qe[e], target) for e in range(NUM_EDGES)]
            raw_w = []
            for e in range(NUM_EDGES):
                size = max(1, ne[e])
                mu = edge_mu[e]
                w = (size ** ALPHA_AGG) * math.exp(BETA_AGG * (mu - 1.0))
                raw_w.append(float(w))

            wsum = sum(raw_w) or 1.0
            edge_weights = [w / wsum for w in raw_w]

            global_state = average_states(edge_states, edge_weights)
            set_model_state(global_model, global_state)

            # reset edge stats each global aggregation (same as your original)
            qe = [{k: 0.0 for k in keys} for _ in range(NUM_EDGES)]
            ne = [0 for _ in range(NUM_EDGES)]

        if r % 10 == 0:
            loss = evaluate_loss(global_model, global_testset, num_batches=10)
            print(f"[Eval] diffusion loss: {loss:.4f}")

        if r % SAVE_EVERY == 0:
            save_samples_forecasting(global_model, global_testset, tag=f"round_{r:04d}", n=16, K=100)
            _ = evaluate_forecasting_generator(global_model, global_testset, num_cases=10, K=200)

    print("\nTraining finished.")
    save_samples_forecasting(global_model, global_testset, tag="final", n=16, K=100)
    final_metrics = evaluate_forecasting_generator(global_model, global_testset, num_cases=20, K=200)
    print("\nFINAL METRICS DICT:", final_metrics)


if __name__ == "__main__":
    main()
