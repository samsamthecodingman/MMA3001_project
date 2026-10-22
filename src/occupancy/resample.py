"""Resampling irregular sensor time series onto a uniform grid.

Finite-difference derivatives and most ML models need uniformly spaced
samples. This module interpolates irregular reports onto a fixed grid
and leaves long gaps as NaN rather than inventing data across them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

METHODS = ("linear", "pchip", "cubic", "nearest")


def _interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray, method: str) -> np.ndarray:
    if method == "linear":
        return np.interp(xq, x, y)
    if method == "pchip":
        return PchipInterpolator(x, y, extrapolate=False)(xq)
    if method == "cubic":
        return CubicSpline(x, y, bc_type="natural", extrapolate=False)(xq)
    if method == "nearest":
        i = np.clip(np.searchsorted(x, xq), 1, len(x) - 1)
        left_closer = (xq - x[i - 1]) <= (x[i] - xq)
        return np.where(left_closer, y[i - 1], y[i])
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


def to_uniform(
    s: pd.Series,
    freq: str = "15min",
    method: str = "linear",
    max_gap: str = "45min",
) -> pd.Series:
    """Interpolate an irregular series onto a uniform time grid.

    Args:
        s: Series with a DatetimeIndex (need not be sorted/unique). NaNs
            are dropped first.
        freq: Target grid spacing (pandas offset string).
        method: One of ``linear``, ``pchip`` (shape-preserving cubic,
            no overshoot), ``cubic`` (natural spline) or ``nearest``.
        max_gap: Grid points further than this from the nearest real
            observation on *either* side of a gap are set to NaN.

    Returns:
        Series on the uniform grid spanning the data.

    Raises:
        ValueError: if ``s`` has fewer than 2 valid points or the index
            is not datetime-like.
    """
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")
    s = s.dropna()
    s = s[~s.index.duplicated()].sort_index()
    if len(s) < 2:
        raise ValueError("need at least two valid observations")
    grid = pd.date_range(s.index[0].ceil(freq), s.index[-1].floor(freq), freq=freq)
    x = s.index.asi8 / 1e9
    xq = grid.asi8 / 1e9
    yq = _interp(x, s.to_numpy(float), xq, method)
    # mask grid points inside gaps longer than max_gap
    j = np.searchsorted(x, xq)
    jl, jr = np.clip(j - 1, 0, len(x) - 1), np.clip(j, 0, len(x) - 1)
    gap = x[jr] - x[jl]
    exact = x[jr] == xq
    yq[(gap > pd.Timedelta(max_gap).total_seconds()) & ~exact] = np.nan
    return pd.Series(yq, index=grid, name=s.name)


def holdout_error(
    s: pd.Series, method: str, frac: float = 0.2, seed: int = 0
) -> dict[str, float]:
    """Interpolation error measured on randomly withheld observations.

    A fraction of the real observations is removed, the remainder is
    interpolated at the withheld times and compared to the truth. This
    validates the resampling step on real data without needing a model.

    Returns:
        Dict with ``mae``, ``rmse`` and ``max`` absolute error, and ``n``.
    """
    s = s.dropna()
    s = s[~s.index.duplicated()].sort_index()
    rng = np.random.default_rng(seed)
    hold = rng.random(len(s)) < frac
    hold[[0, -1]] = False  # keep end points so there is no extrapolation
    x = s.index.asi8 / 1e9
    y = s.to_numpy(float)
    pred = _interp(x[~hold], y[~hold], x[hold], method)
    err = pred - y[hold]
    err = err[np.isfinite(err)]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max": float(np.max(np.abs(err))),
        "n": int(err.size),
    }
