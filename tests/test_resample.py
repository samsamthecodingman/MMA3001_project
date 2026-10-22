"""Tests for resampling irregular series onto a uniform grid."""
import numpy as np
import pandas as pd
import pytest

from occupancy.resample import holdout_error, to_uniform


def irregular(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(300, n))
    idx = pd.Timestamp("2024-05-06") + pd.to_timedelta(t, unit="s")
    y = 600 + 200 * np.sin(2 * np.pi * t / 86400)
    return pd.Series(y, index=idx, name="co2_ppm"), t


@pytest.mark.parametrize("method", ["linear", "pchip", "cubic", "nearest"])
def test_uniform_grid_and_accuracy(method):
    s, _ = irregular()
    u = to_uniform(s, "15min", method, max_gap="3h")
    assert (np.diff(u.index.asi8) == 15 * 60 * 1e9).all()
    t = (u.index - pd.Timestamp("2024-05-06")).total_seconds().to_numpy()
    truth = 600 + 200 * np.sin(2 * np.pi * t / 86400)
    tol = 25 if method == "nearest" else 2
    assert np.nanmax(np.abs(u.to_numpy() - truth)) < tol


def test_long_gap_masked():
    s, _ = irregular()
    gap = (s.index > s.index[200]) & (s.index < s.index[200] + pd.Timedelta("4h"))
    u = to_uniform(s[~gap], "15min", "linear", max_gap="45min")
    inside = (u.index > s.index[200] + pd.Timedelta("1h")) & (u.index < s.index[200] + pd.Timedelta("3h"))
    assert u[inside].isna().all()


def test_holdout_error_smooth_signal_small():
    s, _ = irregular(2000)
    e_lin = holdout_error(s, "linear")
    e_near = holdout_error(s, "nearest")
    assert e_lin["mae"] < e_near["mae"]
    assert e_lin["n"] > 100


def test_bad_inputs():
    with pytest.raises(ValueError):
        to_uniform(pd.Series([1.0, 2.0]), "15min")
    s, _ = irregular(5)
    with pytest.raises(ValueError):
        to_uniform(s.iloc[:1], "15min")
    with pytest.raises(ValueError):
        to_uniform(s, "15min", method="quintic")
