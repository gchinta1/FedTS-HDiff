import numpy as np
import torch

from fedtshdiff.eval.metrics import LEVELS, dtw_batch, summarise, window_metrics
from reference_original import conditional_future_metrics, dtw_distance


def test_dtw_matches_double_loop():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (20, 33))
    b = rng.uniform(-1, 1, (20, 33))
    got = dtw_batch(torch.tensor(a), torch.tensor(b)).numpy()
    ref = np.array([dtw_distance(x, y) for x, y in zip(a, b)])
    assert np.allclose(got, ref, atol=1e-12)


def test_paper_metrics_match_original_code():
    rng = np.random.default_rng(0)
    M, K, F = 5, 17, 64
    truth = rng.uniform(-1, 1, (M, F)).astype(np.float32)
    samples = np.clip(truth[:, None, :] + 0.3 * rng.standard_normal((M, K, F)), -1, 1).astype(np.float32)
    out = window_metrics(torch.tensor(truth), torch.tensor(samples))
    for m in range(M):
        ref = conditional_future_metrics(truth[m].astype(np.float64), samples[m].astype(np.float64))
        for key, val in ref.items():
            assert abs(out[key][m].item() - val) <= 1e-5 * max(1.0, abs(val)), key


def test_calibration_of_a_perfect_forecaster():
    # truth and samples drawn from the same distribution -> coverage ~ nominal, PIT ~ uniform
    g = torch.Generator().manual_seed(0)
    M, K, F = 400, 200, 16
    truth = torch.randn(M, F, generator=g)
    samples = torch.randn(M, K, F, generator=g)
    per = {k: v.numpy() for k, v in window_metrics(truth, samples, g).items()}
    s = summarise(per)
    assert np.allclose(s["coverage_levels"], LEVELS, atol=0.03)
    assert s["calib_error"] < 0.02
    assert s["pit_tv"] < 0.03
