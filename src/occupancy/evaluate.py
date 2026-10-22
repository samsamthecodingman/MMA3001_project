"""Metrics and leakage-free time-series splitting."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def metrics(y_true: np.ndarray, y_pred: np.ndarray, occ_threshold: float = 0.5) -> dict[str, float]:
    """Regression and occupied/vacant classification metrics.

    Args:
        y_true: True occupancy [persons].
        y_pred: Predicted occupancy [persons].
        occ_threshold: A room counts as occupied when occupancy exceeds
            this value (applied to both truth and prediction).

    Returns:
        Dict with ``mae``, ``rmse``, ``r2``, ``bias`` (mean error),
        ``f1``/``accuracy`` for the occupied state, and ``n``.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0:
        raise ValueError("no valid (finite) pairs to evaluate")
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    yt, yp = y > occ_threshold, p > occ_threshold
    tp = np.sum(yt & yp)
    prec = tp / max(yp.sum(), 1)
    rec = tp / max(yt.sum(), 1)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "r2": float(1 - np.sum(e**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(e)),
        "f1": float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0,
        "accuracy": float(np.mean(yt == yp)),
        "n": int(y.size),
    }


def chrono_split(
    index: pd.DatetimeIndex, train: float = 0.6, val: float = 0.2, gap: str = "1D"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/validation/test masks with gaps between them.

    Random splits leak information in autocorrelated time series (a test
    sample's neighbours are in training). Splitting by time, with a gap
    so the lagged features never straddle a boundary, avoids this.

    Returns:
        Boolean masks ``(train, val, test)`` aligned with ``index``.
    """
    t0, t1 = index.min(), index.max()
    span = t1 - t0
    a = t0 + span * train
    b = t0 + span * (train + val)
    g = pd.Timedelta(gap)
    idx = index
    return (
        np.asarray(idx < a),
        np.asarray((idx >= a + g) & (idx < b)),
        np.asarray(idx >= b + g),
    )


def blocked_folds(
    index: pd.DatetimeIndex, n_folds: int = 4, gap: str = "1D"
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window ("forward chaining") cross-validation folds.

    Fold k trains on the first k blocks and validates on block k+1,
    leaving ``gap`` between them.

    Yields:
        ``(train_mask, val_mask)`` boolean arrays.
    """
    edges = pd.date_range(index.min(), index.max(), periods=n_folds + 2)
    g = pd.Timedelta(gap)
    for k in range(1, n_folds + 1):
        tr = np.asarray(index < edges[k])
        va = np.asarray((index >= edges[k] + g) & (index < edges[k + 1]))
        if tr.any() and va.any():
            yield tr, va
