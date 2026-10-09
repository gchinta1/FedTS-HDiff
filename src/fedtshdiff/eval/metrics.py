"""Batched forecasting metrics.

The paper's metrics are reproduced exactly as defined in the original code, but computed
for whole batches of windows on the device:

dtw_mean   mean over the K samples of DTW(sample, truth), DTW = sqrt(D[L,L] / L) with squared cost
psd_l2     RMSE between the truth's normalised PSD and the samples' mean normalised PSD (nfft = F)
div_dtw    mean DTW over 200 fixed random sample pairs (rng seed 43, as in the original)
crps       ensemble CRPS (V-statistic with K^2 denominator, identical to the pairwise formula)
wis_90, picp_90, mpiw_90   central 90% interval from linear-interpolated sample quantiles

Additional metrics for the revision: point errors of the mean/median forecast, DTW of the
mean forecast, multi-level weighted interval score (Bracher et al., 2021), central-interval
coverage at 10..90 % with its mean absolute calibration error, and PIT histograms.
"""

from __future__ import annotations

import numpy as np
import torch

LEVELS = tuple(round(0.1 * k, 1) for k in range(1, 10))  # nominal central-interval coverages
PIT_BINS = 10


def dtw_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """DTW of the original code for each row pair; a, b: [n, L] -> [n] (float64).

    Row recursion: D[i, j] = S[j] + cummin_k(b_k - S[k-1]) with b_k = min(D[i-1,k-1], D[i-1,k])
    and S the prefix sum of row i's costs; equivalent to the O(L^2) double loop.
    """
    a = a.double()
    b = b.double()
    n, L = a.shape
    inf = torch.tensor(float("inf"), dtype=torch.float64, device=a.device)
    prev = torch.full((n, L + 1), float("inf"), dtype=torch.float64, device=a.device)
    prev[:, 0] = 0.0
    for i in range(L):
        cost = (a[:, i:i + 1] - b) ** 2  # [n, L]
        best_prev = torch.minimum(prev[:, :-1], prev[:, 1:])  # b_k for k = 1..L
        S = torch.cumsum(cost, dim=1)
        S_shift = torch.cat([torch.zeros((n, 1), dtype=torch.float64, device=a.device), S[:, :-1]], dim=1)
        row = S + torch.cummin(best_prev - S_shift, dim=1).values
        prev = torch.cat([inf.expand(n, 1), row], dim=1)
    return torch.sqrt(prev[:, L] / L)


def _quantiles(samples: torch.Tensor, qs: list[float]) -> torch.Tensor:
    """samples [M, K, F] -> [len(qs), M, F] (linear interpolation, as numpy's default)."""
    q = torch.tensor(qs, dtype=samples.dtype, device=samples.device)
    return torch.quantile(samples, q, dim=1)


def _psd(x: torch.Tensor, nfft: int) -> torch.Tensor:
    X = torch.fft.rfft(x.double(), n=nfft, dim=-1)
    P = X.real ** 2 + X.imag ** 2
    return P / (P.sum(-1, keepdim=True) + 1e-12)


_PAIRS_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def diversity_pairs(K: int, pairs: int = 200, seed: int = 43) -> tuple[np.ndarray, np.ndarray]:
    key = (K, pairs)
    if key not in _PAIRS_CACHE:
        rng = np.random.default_rng(seed)
        ii, jj = [], []
        for _ in range(pairs):
            i, j = rng.integers(0, K, size=2)
            if i == j:
                j = (j + 1) % K
            ii.append(int(i))
            jj.append(int(j))
        _PAIRS_CACHE[key] = (np.array(ii), np.array(jj))
    return _PAIRS_CACHE[key]


@torch.no_grad()
def window_metrics(truth: torch.Tensor, samples: torch.Tensor, gen: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """truth [M, F], samples [M, K, F] -> per-window metric tensors [M] (+ coverage [9, M], pit [M, F])."""
    M, K, Fl = samples.shape
    y = truth.double()
    s = samples.double()
    out: dict[str, torch.Tensor] = {}

    dtw = dtw_batch(s.reshape(M * K, Fl), y.repeat_interleave(K, 0)).view(M, K)
    out["dtw_mean"] = dtw.mean(1)
    out["dtw_std"] = dtw.std(1, unbiased=False)
    mean_fc = s.mean(1)
    median_fc = s.median(1).values
    out["dtw_point"] = dtw_batch(mean_fc, y)

    out["psd_l2"] = torch.sqrt(((_psd(y, Fl) - _psd(s, Fl).mean(1)) ** 2).mean(-1))

    pi, pj = diversity_pairs(K)
    pi_t = torch.as_tensor(pi, device=s.device)
    pj_t = torch.as_tensor(pj, device=s.device)
    div = dtw_batch(s[:, pi_t].reshape(-1, Fl), s[:, pj_t].reshape(-1, Fl)).view(M, len(pi))
    out["div_dtw"] = div.mean(1)

    srt = torch.sort(s, dim=1).values
    coef = (2 * torch.arange(1, K + 1, device=s.device, dtype=torch.float64) - K - 1).view(1, K, 1)
    term1 = (s - y[:, None, :]).abs().mean(1)
    term2 = (coef * srt).sum(1) / (K * K)
    out["crps"] = (term1 - term2).mean(-1)

    out["mse"] = ((mean_fc - y) ** 2).mean(-1)
    out["mae"] = (mean_fc - y).abs().mean(-1)
    out["mae_median"] = (median_fc - y).abs().mean(-1)

    qs = sorted({0.5} | {round(0.5 - lv / 2, 4) for lv in LEVELS} | {round(0.5 + lv / 2, 4) for lv in LEVELS})
    Q = _quantiles(s, qs)
    qi = {q: k for k, q in enumerate(qs)}
    cover = []
    wis_terms = 0.5 * (Q[qi[0.5]] - y).abs()
    for lv in LEVELS:
        alpha = round(1 - lv, 4)
        lo, hi = Q[qi[round(0.5 - lv / 2, 4)]], Q[qi[round(0.5 + lv / 2, 4)]]
        inside = ((y >= lo) & (y <= hi)).double()
        cover.append(inside.mean(-1))
        interval = (hi - lo) + (2 / alpha) * (torch.clamp(lo - y, min=0) + torch.clamp(y - hi, min=0))
        wis_terms = wis_terms + (alpha / 2) * interval
        if lv == 0.9:
            out["picp_90"] = inside.mean(-1)
            out["mpiw_90"] = (hi - lo).mean(-1)
            out["wis_90"] = interval.mean(-1)
    out["wis_multi"] = (wis_terms / (len(LEVELS) + 0.5)).mean(-1)
    out["coverage"] = torch.stack(cover)  # [9, M]

    below = (s < y[:, None, :]).double().sum(1)
    ties = (s == y[:, None, :]).double().sum(1)
    u = torch.rand(y.shape, device=s.device, generator=gen, dtype=torch.float64) if gen is not None else 0.5
    out["pit"] = (below + u * ties) / K  # [M, F]
    return out


def summarise(per_window: dict[str, np.ndarray]) -> dict[str, float | list[float]]:
    """Averages over windows plus calibration summaries."""
    out: dict[str, float | list[float]] = {}
    for k, v in per_window.items():
        if k in ("coverage", "pit"):
            continue
        out[k] = float(np.mean(v))
    cov = per_window["coverage"].mean(axis=1)
    out["coverage_levels"] = [float(x) for x in cov]
    out["calib_error"] = float(np.mean(np.abs(cov - np.array(LEVELS))))
    hist, _ = np.histogram(per_window["pit"].reshape(-1), bins=PIT_BINS, range=(0.0, 1.0))
    hist = hist / max(hist.sum(), 1)
    out["pit_hist"] = [float(x) for x in hist]
    out["pit_tv"] = float(0.5 * np.abs(hist - 1.0 / PIT_BINS).sum())
    return out
