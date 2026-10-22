"""Feature construction for the data-driven occupancy models.

All features are *causal* by default (use only the current and past
samples) so the models could run in real time on a building system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENV_COLUMNS = ("co2_ppm", "temp_c", "rh_pct")


def build_features(df: pd.DataFrame, lags: int = 3, causal: bool = True) -> pd.DataFrame:
    """Build a feature matrix from a uniformly sampled room DataFrame.

    Args:
        df: DataFrame on a uniform DatetimeIndex with at least
            ``co2_ppm``; ``temp_c`` and ``rh_pct`` are used if present.
        lags: Number of past CO2 samples (and differences) to include.
        causal: If True only past/current values are used. If False the
            next CO2 difference is also included (adds one grid step of
            latency but gives a central-difference derivative).

    Returns:
        Feature DataFrame with the same index. Rows with insufficient
        history contain NaN and should be dropped by the caller.

    Raises:
        KeyError: if ``co2_ppm`` is missing.
    """
    if "co2_ppm" not in df:
        raise KeyError("df must contain a 'co2_ppm' column")
    c = df["co2_ppm"].astype(float)
    X = pd.DataFrame(index=df.index)
    X["co2"] = c
    X["dco2"] = c.diff()
    for k in range(1, lags + 1):
        X[f"co2_lag{k}"] = c.shift(k)
        X[f"dco2_lag{k}"] = c.diff().shift(k)
    if not causal:
        X["dco2_next"] = c.diff().shift(-1)
    # running baseline: low percentile of the past 24 h approximates C_out
    X["co2_above_base"] = c - c.rolling("24h", min_periods=1).quantile(0.05)
    for col in ("temp_c", "rh_pct"):
        if col in df:
            v = df[col].astype(float)
            X[col] = v
            X[f"d_{col}"] = v.diff()
    h = df.index.hour + df.index.minute / 60
    X["hour_sin"] = np.sin(2 * np.pi * h / 24)
    X["hour_cos"] = np.cos(2 * np.pi * h / 24)
    X["weekend"] = (df.index.day_of_week >= 5).astype(float)
    return X
