"""Verbatim metric functions of the original FedTS-HDiff scripts (reference for tests)."""
import numpy as np


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
    return float(np.sqrt(np.mean((fr.mean(axis=0) - ff.mean(axis=0)) ** 2)))


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


def crps_ensemble(y_true, samples):
    term1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


def picp(y_true, samples, alpha=0.1):
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mpiw(samples, alpha=0.1):
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean(upper - lower))


def wis(y_true, samples, alpha=0.1):
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    width = upper - lower
    below = np.maximum(0.0, lower - y_true)
    above = np.maximum(0.0, y_true - upper)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def conditional_future_metrics(true_future, future_samples):
    dtws = [dtw_distance(true_future, future_samples[k]) for k in range(future_samples.shape[0])]
    div_mean, _ = diversity_dtw(future_samples, pairs=200)
    return {
        "dtw_mean": float(np.mean(dtws)),
        "dtw_std": float(np.std(dtws)),
        "psd_l2": psd_l2(true_future[None, :], future_samples, nfft=len(true_future)),
        "div_dtw": div_mean,
        "crps": crps_ensemble(true_future, future_samples),
        "wis_90": wis(true_future, future_samples, alpha=0.1),
        "picp_90": picp(true_future, future_samples, alpha=0.1),
        "mpiw_90": mpiw(future_samples, alpha=0.1),
    }
