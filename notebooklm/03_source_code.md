# MMA3001 project source code (src/occupancy, scripts, tests, Makefile)

## FILE: src/occupancy/evaluate.py

```python
"""Metrics and leakage-free time-series splitting.

* :func:`metrics` -- MAE, RMSE, R^2, bias and occupied/vacant F1/accuracy.
* :func:`chrono_split` -- chronological train/validation/test masks.
* :func:`blocked_folds` -- expanding-window cross-validation folds.
"""
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
        ``f1``/``accuracy``/``balanced_accuracy`` for the occupied state, and ``n``. Pairs where
        either value is not finite are ignored; ``r2`` is NaN if the truth
        is constant.

    Raises:
        ValueError: if there are no valid (finite) pairs.
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
        # mean of the true-positive and true-negative rates: unlike F1 and
        # accuracy it scores 0.5 for a constant prediction, so it shows skill
        # even when most intervals are occupied
        "balanced_accuracy": float(0.5 * (rec + np.sum(~yt & ~yp) / max((~yt).sum(), 1))),
        "n": int(y.size),
    }


def chrono_split(
    index: pd.DatetimeIndex, train: float = 0.6, val: float = 0.2, gap: str = "1D"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/validation/test masks with gaps between them.

    Random splits leak information in autocorrelated time series (a test
    sample's neighbours are in training). Splitting by time, with a gap
    so the lagged features never straddle a boundary, avoids this.

    Args:
        index: Timestamps of the samples.
        train: Fraction of the time span used for training.
        val: Fraction of the time span used for validation (following
            training).
        gap: Buffer removed after the training and validation periods.

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

    Args:
        index: Timestamps of the samples.
        n_folds: Number of folds requested (folds with an empty training
            or validation set are skipped).
        gap: Buffer between the training and validation periods.

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
```

## FILE: src/occupancy/features.py

```python
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
```

## FILE: src/occupancy/io.py

```python
"""Loading and tidying the raw Monash Smart Infrastructure (MSI) exports.

Raw inputs (see ``data/README.md`` for how to obtain them):

* **Environmental sensors** — one CSV row per sensor report. The
  ``jsondata`` column holds a list of
  ``{"values": [{"time": <unix s>, "value": x}], "variable": {"name", "unit"}}``
  objects (CO2, temperature, humidity, pressure, ...).
* **Occupancy sensors** — one CSV row per event per floor-space:
  ``floorspaceid``, ``headcount`` (people), ``occupancystatus`` and a
  local timestamp with UTC offset (``+10``/``+11`` across DST).

Both are converted to tidy, typed tables with timezone-aware
(Australia/Melbourne) timestamps and cached as parquet.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .timeutil import unix_seconds

TZ = "Australia/Melbourne"

#: Mapping from the vendor's variable names to tidy column names + units.
ENV_VARIABLES = {
    "Carbon dioxide": "co2_ppm",
    "Temperature": "temp_c",
    "Humidity": "rh_pct",
    "Pressure": "pressure_mb",
    "Light Volatile Organic Compounds": "lvoc_ppb",
    "Formaldehyde": "hcho_ugm3",
    "Battery": "battery_v",
}

#: Physically plausible ranges used to flag sensor faults (inclusive).
VALID_RANGES = {
    "co2_ppm": (350.0, 5000.0),
    "temp_c": (5.0, 45.0),
    "rh_pct": (0.0, 100.0),
    "pressure_mb": (900.0, 1100.0),
}


def parse_env_json(js: str) -> dict[str, tuple[float, float]]:
    """Parse one ``jsondata`` cell.

    Args:
        js: JSON string from the export.

    Returns:
        ``{column_name: (unix_time, value)}`` for recognised variables.
        Unrecognised variables and malformed entries are skipped; an
        unparseable cell returns an empty dict.
    """
    try:
        items = json.loads(js)
    except (TypeError, ValueError):
        return {}
    out = {}
    for it in items:
        try:
            col = ENV_VARIABLES.get(it["variable"]["name"])
            v = it["values"][0]
            if col is not None:
                out[col] = (float(v["time"]), float(v["value"]))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


def load_env(path: str | pathlib.Path) -> pd.DataFrame:
    """Load the environmental-sensor export into a tidy wide table.

    Args:
        path: CSV path.

    Returns:
        DataFrame with columns ``sensorid``, ``time`` (tz-aware), and one
        column per variable in :data:`ENV_VARIABLES` (NaN when absent).
        The measurement time inside the JSON is used (not the database
        ``createdate``), taking the CO2 timestamp where available.
    """
    raw = pd.read_csv(path, usecols=["sensorid", "jsondata"], dtype={"sensorid": str})
    recs = []
    for sid, js in zip(raw["sensorid"].to_numpy(), raw["jsondata"].to_numpy()):
        d = parse_env_json(js)
        if not d:
            continue
        t = d.get("co2_ppm", next(iter(d.values())))[0]
        row = {"sensorid": sid, "unix": t}
        row.update({k: v for k, (_, v) in d.items()})
        recs.append(row)
    df = pd.DataFrame.from_records(recs)
    df["time"] = pd.to_datetime(df.pop("unix"), unit="s", utc=True).dt.tz_convert(TZ)
    for c in ENV_VARIABLES.values():
        if c not in df:
            df[c] = np.nan
    return df.sort_values(["sensorid", "time"]).reset_index(drop=True)


def load_occupancy(path: str | pathlib.Path) -> pd.DataFrame:
    """Load the occupancy-event export.

    Args:
        path: CSV path.

    Returns:
        DataFrame with ``floorspaceid`` (category), ``time`` (tz-aware),
        ``headcount`` (float32) and ``status`` (category), sorted by
        space and time.
    """
    df = pd.read_csv(
        path,
        usecols=["floorspaceid", "occupancystatus", "headcount", "collecteddate"],
        dtype={"floorspaceid": "category", "occupancystatus": "category", "headcount": "float32"},
    )
    # timestamps look like "2024-05-01 09:00:00+10"; pandas needs "+10:00"
    ts = df.pop("collecteddate").str.replace(r"([+-]\d\d)$", r"\1:00", regex=True)
    df["time"] = pd.to_datetime(ts, utc=True).dt.tz_convert(TZ)
    df = df.rename(columns={"occupancystatus": "status"})
    return df.sort_values(["floorspaceid", "time"]).reset_index(drop=True)


def flag_invalid(env: pd.DataFrame) -> pd.DataFrame:
    """Replace physically implausible readings with NaN.

    Ranges are in :data:`VALID_RANGES`. Columns not listed there are left
    untouched.

    Args:
        env: Wide env table (e.g. from :func:`load_env`).

    Returns:
        A copy of ``env`` with out-of-range values set to NaN. The number
        of values removed per column is stored in ``df.attrs["n_flagged"]``.
    """
    env = env.copy()
    counts = {}
    for col, (lo, hi) in VALID_RANGES.items():
        if col in env:
            bad = env[col].notna() & ~env[col].between(lo, hi)
            counts[col] = int(bad.sum())
            env.loc[bad, col] = np.nan
    env.attrs["n_flagged"] = counts
    return env


def occupancy_on_grid_loop(events: pd.DataFrame, freq: str = "15min", max_hold: str = "30min") -> pd.Series:
    """Convert headcount *events* for one space to a time-weighted mean per interval.

    Each event's headcount holds until the next event (zero-order hold),
    but not longer than ``max_hold``: if the counter goes silent for
    longer, the interval is treated as missing rather than assumed
    constant (a silent counter may be offline).

    Args:
        events: Rows of one space with ``time`` and ``headcount``.
        freq: Grid spacing.
        max_hold: Longest time an event's value is carried forward.

    Returns:
        Series of mean headcount per interval (NaN where unobserved),
        labelled by interval start.

    Note:
        Reference (pure-Python loop) implementation kept for testing and
        for the performance comparison; use :func:`occupancy_on_grid`.
    """
    e = events[["time", "headcount"]].dropna().sort_values("time")
    tz = e["time"].dt.tz
    ts_abs = unix_seconds(e["time"])
    h = e["headcount"].to_numpy(float)
    hold = pd.Timedelta(max_hold).total_seconds()
    dur = np.append(np.diff(ts_abs), 0.0)
    valid = dur <= hold
    width = pd.Timedelta(freq).total_seconds()
    g0 = np.floor(ts_abs[0] / width) * width  # grid start, aligned to freq in UTC
    n = int(np.ceil((ts_abs[-1] - g0) / 60.0)) + 1
    grid = pd.to_datetime(g0 + 60.0 * np.arange(n), unit="s", utc=True).tz_convert(tz)
    ts = ts_abs - g0
    num = np.zeros(n)
    den = np.zeros(n)
    # distribute each segment [ts, ts+dur) over the 60-s bins it covers
    for a, d, v in zip(ts[valid], dur[valid], h[valid]):
        if d <= 0:
            continue
        b0, b1 = a, a + d
        i0, i1 = int(b0 // 60), int(min(b1 // 60, n - 1))
        for i in range(i0, i1 + 1):
            lo, hi = max(b0, i * 60), min(b1, (i + 1) * 60)
            if hi > lo:
                num[i] += v * (hi - lo)
                den[i] += hi - lo
    fine = pd.DataFrame({"num": num, "den": den}, index=grid)
    agg = fine.resample(freq).sum()
    cov = agg["den"] / pd.Timedelta(freq).total_seconds()
    out = (agg["num"] / agg["den"]).where(cov >= 0.5)  # need >=50% coverage
    out.name = "occupancy"
    return out


def occupancy_on_grid(events: pd.DataFrame, freq: str = "15min", max_hold: str = "30min",
                      min_coverage: float = 0.5) -> pd.Series:
    """Vectorised time-weighted mean headcount per interval.

    Same semantics as :func:`occupancy_on_grid_loop` (zero-order hold,
    capped at ``max_hold``) but computed with a cumulative integral:
    with segment k holding value ``h_k`` from ``t_k`` for ``d_k`` seconds
    (``valid_k = d_k <= max_hold``), the running integral at any time x
    in segment k is ``S_k + valid_k h_k (x - t_k)``, where ``S_k`` is the
    cumulative sum over earlier segments. Interval sums are differences
    of this integral at the grid edges, so the cost is O(n_events +
    n_intervals) in compiled NumPy code instead of a Python loop.

    Args:
        events: Rows of one space with ``time`` and ``headcount``.
        freq: Grid spacing.
        max_hold: Longest time an event's value is carried forward.
        min_coverage: Minimum fraction of an interval that must be
            covered by valid segments, else the result is NaN.

    Returns:
        Series of mean headcount per interval, labelled by interval start
        (NaN where the interval is not sufficiently covered).

    Raises:
        ValueError: if fewer than two valid events are given.
    """
    e = events[["time", "headcount"]].dropna().sort_values("time")
    if len(e) < 2:
        raise ValueError("need at least two events")
    tz = e["time"].dt.tz
    t = unix_seconds(e["time"])
    h = e["headcount"].to_numpy(float)
    d = np.append(np.diff(t), 0.0)
    w = (d <= pd.Timedelta(max_hold).total_seconds()).astype(float)
    S = np.concatenate([[0.0], np.cumsum(w * h * d)])  # value integral before segment k
    T = np.concatenate([[0.0], np.cumsum(w * d)])  # covered-time integral
    edges = pd.date_range(pd.Timestamp(t[0], unit="s", tz="UTC").floor(freq),
                          pd.Timestamp(t[-1], unit="s", tz="UTC").ceil(freq), freq=freq)
    x = unix_seconds(edges)
    k = np.clip(np.searchsorted(t, x, side="right") - 1, 0, len(t) - 1)
    dx = np.clip(x - t[k], 0, d[k])  # time into segment k (0 before first event)
    before = x < t[0]
    Ix = np.where(before, 0.0, S[k] + w[k] * h[k] * dx)
    Cx = np.where(before, 0.0, T[k] + w[k] * dx)
    num, cov = np.diff(Ix), np.diff(Cx)
    width = pd.Timedelta(freq).total_seconds()
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(cov >= min_coverage * width, num / cov, np.nan)
    return pd.Series(val, index=edges[:-1].tz_convert(tz), name="occupancy")
```

## FILE: src/occupancy/models.py

```python
"""Occupancy estimators with a common ``fit``/``predict`` interface.

* :class:`TimeOfWeekBaseline` — mean occupancy per weekday-hour slot
  (uses no sensor data at all; any useful model must beat it).
* :class:`PhysicsEstimator` — calibrated CO2 mass-balance inversion.
* scikit-learn regressors (linear, tree, random forest, SVR, MLP) built
  by :func:`make_ml_model` on the features from
  :mod:`occupancy.features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .physics import G_SEDENTARY, RoomParams, calibrate, estimate_occupancy


class TimeOfWeekBaseline:
    """Predict the training-set mean occupancy for each (weekday, hour).

    Attributes:
        table_: Mean occupancy per (day of week, hour) slot (after ``fit``).
        default_: Global training mean, used for unseen slots.
    """

    def fit(self, index: pd.DatetimeIndex, y: np.ndarray) -> "TimeOfWeekBaseline":
        """Learn slot means.

        Args:
            index: Timestamp of each target value.
            y: Observed occupancy [persons] (NaNs are ignored in the slot means).

        Returns:
            ``self``.
        """
        s = pd.Series(np.asarray(y, float), index=index)
        self.table_ = s.groupby([index.day_of_week, index.hour]).mean()
        self.default_ = float(np.nanmean(y))
        return self

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Look up the slot mean for each timestamp (global mean if unseen).

        Args:
            index: Timestamps to predict.

        Returns:
            Predicted occupancy [persons], one value per timestamp.
        """
        keys = list(zip(index.day_of_week, index.hour))
        return np.array([self.table_.get(k, self.default_) for k in keys], float)


class PhysicsEstimator:
    """CO2 mass-balance inversion with parameters calibrated on training data.

    Args:
        dt: Grid spacing [s].
        scheme: Finite-difference scheme for dC/dt.
        window: Savitzky-Golay window if ``scheme="savgol"``.
        g: Per-person CO2 generation [m^3/s].
        params: If given, skip calibration and use these parameters
            (e.g. nominal design values).

    Attributes:
        params_: Room parameters (``None`` until ``fit`` unless supplied).
    """

    def __init__(self, dt: float, scheme: str = "savgol", window: int = 5,
                 g: float = G_SEDENTARY, params: RoomParams | None = None):
        """Store the configuration; see the class docstring for the arguments."""
        self.dt, self.scheme, self.window, self.g = dt, scheme, window, g
        self.params_ = params

    def fit(self, co2: np.ndarray, y: np.ndarray) -> "PhysicsEstimator":
        """Calibrate V, Q, C_out by least squares (see :func:`~occupancy.physics.calibrate`).

        Args:
            co2: Uniformly sampled CO2 [ppm]; NaN rows are ignored.
            y: Measured occupancy [persons] on the same grid.

        Returns:
            ``self``.

        Raises:
            ValueError: if the calibration is non-physical or has too
                few valid samples.
        """
        if self.params_ is None:
            self.params_ = calibrate(co2, y, self.dt, self.g, self.scheme, self.window)
        return self

    def predict(self, co2: np.ndarray) -> np.ndarray:
        """Estimate occupancy from a contiguous uniformly sampled CO2 array.

        Args:
            co2: CO2 [ppm] on the grid spacing ``dt`` given at construction.

        Returns:
            Estimated occupancy [persons], same shape as ``co2``.
        """
        return estimate_occupancy(co2, self.dt, self.params_, self.scheme, self.window)


def make_ml_model(name: str, seed: int = 0):
    """Construct a scikit-learn regressor by name.

    Feature scaling is included for the scale-sensitive models (ridge,
    SVR, MLP). Hyper-parameters are sensible defaults; see the tuning
    script for the selection procedure.

    Args:
        name: ``linear``, ``tree``, ``forest``, ``svr`` or ``mlp``.
        seed: Random seed for stochastic models.

    Returns:
        An unfitted scikit-learn estimator (``fit(X, y)``/``predict(X)``).

    Raises:
        ValueError: for an unknown name.
    """
    if name == "linear":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "tree":
        return DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=seed)
    if name == "forest":
        return RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=10,
                                     n_jobs=-1, random_state=seed)
    if name == "svr":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.2, gamma="scale"))
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                         max_iter=500, early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=20, random_state=seed),
        )
    raise ValueError(f"unknown model {name!r}")


ML_MODELS = ("linear", "tree", "forest", "svr", "mlp")
```

## FILE: src/occupancy/physics.py

```python
"""Well-mixed CO2 mass-balance model of a room.

The room is treated as a single well-mixed control volume. Occupants
generate CO2 and ventilation exchanges room air with supply air:

    V dC/dt = 1e6 * g * N(t) - Q (C - C_out)

where

* ``C``      room CO2 concentration [ppm]
* ``C_out``  supply / outdoor CO2 concentration [ppm]
* ``V``      effective room volume [m^3]
* ``Q``      ventilation (outdoor-air) flow rate [m^3/s]
* ``g``      CO2 generation per person [m^3/s per person]
* ``N(t)``   number of occupants [persons]

The factor 1e6 converts a volume fraction to ppm.

This module provides

* the forward problem (given ``N(t)`` predict ``C(t)``) with explicit
  Euler, classical RK4 and the exact solution for piecewise-constant ``N``;
* the inverse problem (given measured ``C(t)`` estimate ``N(t)``) using
  finite-difference derivatives;
* calibration of the unknown room parameters by linear least squares.

Assumptions (see report for discussion): perfect mixing, constant
``Q`` and ``C_out`` within a calibration window, constant per-person
generation rate, no other CO2 sources/sinks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

PPM = 1e6
"""Conversion factor from volume fraction to ppm."""

G_SEDENTARY = 5.2e-6
"""CO2 generation of a sedentary adult office worker [m^3/s/person]
(~0.0052 L/s, ASHRAE 62.1 / Persily & de Jonge 2017)."""


@dataclass(frozen=True)
class RoomParams:
    """Physical parameters of the single-zone CO2 model.

    Attributes:
        V: Effective well-mixed room volume [m^3]. Must be > 0.
        Q: Outdoor-air ventilation flow rate [m^3/s]. Must be > 0.
        C_out: Supply/outdoor CO2 concentration [ppm].
        g: CO2 generation per occupant [m^3/s/person].
    """

    V: float
    Q: float
    C_out: float = 420.0
    g: float = G_SEDENTARY

    def __post_init__(self) -> None:
        """Validate that V, Q and g are strictly positive."""
        if self.V <= 0 or self.Q <= 0 or self.g <= 0:
            raise ValueError("V, Q and g must be strictly positive")

    @property
    def tau(self) -> float:
        """Air-exchange time constant V/Q [s]."""
        return self.V / self.Q

    @property
    def ach(self) -> float:
        """Air changes per hour, 3600 Q / V [1/h]."""
        return 3600.0 * self.Q / self.V

    def steady_state(self, N: float | np.ndarray) -> float | np.ndarray:
        """Steady-state CO2 for constant occupancy N [ppm]."""
        return self.C_out + PPM * self.g * np.asarray(N) / self.Q


def rhs(C: float | np.ndarray, N: float | np.ndarray, p: RoomParams) -> float | np.ndarray:
    """Right-hand side dC/dt of the mass balance [ppm/s].

    Args:
        C: Room CO2 concentration [ppm].
        N: Occupant count [persons].
        p: Room parameters.

    Returns:
        Time derivative of concentration [ppm/s].
    """
    return (PPM * p.g * N - p.Q * (C - p.C_out)) / p.V


def simulate(
    t: np.ndarray,
    N: np.ndarray,
    p: RoomParams,
    C0: float | None = None,
    method: str = "rk4",
) -> np.ndarray:
    """Forward-simulate room CO2 for a given occupancy history.

    Occupancy is treated as piecewise constant: ``N[k]`` holds over
    ``[t[k], t[k+1])`` (zero-order hold, as a people counter reports).

    Args:
        t: Strictly increasing sample times [s], shape (n,).
        N: Occupancy at each sample time [persons], shape (n,).
        p: Room parameters.
        C0: Initial concentration [ppm]. Defaults to ``p.C_out``.
        method: ``"euler"`` (explicit, 1st order), ``"rk4"`` (classical
            Runge-Kutta, 4th order) or ``"exact"`` (analytical solution of
            the linear ODE over each constant-N interval).

    Returns:
        Concentration at each time in ``t`` [ppm], shape (n,).

    Raises:
        ValueError: on mismatched shapes, non-increasing ``t`` or an
            unknown method.
    """
    t = np.asarray(t, dtype=float)
    N = np.asarray(N, dtype=float)
    if t.shape != N.shape or t.ndim != 1:
        raise ValueError("t and N must be 1-D arrays of equal length")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    C = np.empty_like(t)
    C[0] = p.C_out if C0 is None else C0
    dts = np.diff(t)
    if method == "exact":
        # C(t+h) = Css + (C - Css) exp(-h/tau) for constant N over the step
        decay = np.exp(-dts / p.tau)
        css = p.steady_state(N[:-1])
        for k, (d, s) in enumerate(zip(decay, css)):
            C[k + 1] = s + (C[k] - s) * d
    elif method == "euler":
        for k, h in enumerate(dts):
            C[k + 1] = C[k] + h * rhs(C[k], N[k], p)
    elif method == "rk4":
        for k, h in enumerate(dts):
            n = N[k]
            k1 = rhs(C[k], n, p)
            k2 = rhs(C[k] + 0.5 * h * k1, n, p)
            k3 = rhs(C[k] + 0.5 * h * k2, n, p)
            k4 = rhs(C[k] + h * k3, n, p)
            C[k + 1] = C[k] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    else:
        raise ValueError(f"unknown method {method!r}")
    return C


def simulate_uniform_fast(N: np.ndarray, dt: float, p: RoomParams, C0: float | None = None) -> np.ndarray:
    """Vectorised exact simulation on a uniform grid.

    With constant step ``dt`` the exact update
    ``C[k+1] = a C[k] + (1 - a) Css[k]``, ``a = exp(-dt/tau)``, is a
    first-order linear recurrence, i.e. an IIR filter, so it can be
    evaluated in compiled code by :func:`scipy.signal.lfilter` instead
    of a Python loop. Gives identical results to
    ``simulate(..., method="exact")`` (to round-off).

    Args:
        N: Occupancy per step [persons], shape (n,).
        dt: Uniform step [s].
        p: Room parameters.
        C0: Initial concentration [ppm]; defaults to ``p.C_out``.

    Returns:
        Concentration [ppm], shape (n,).
    """
    from scipy.signal import lfilter

    N = np.asarray(N, dtype=float)
    a = np.exp(-dt / p.tau)
    css = p.steady_state(N)
    c0 = p.C_out if C0 is None else C0
    # y[k] = a*y[k-1] + (1-a)*css[k-1]; feed css delayed by one sample
    x = np.concatenate([[0.0], css[:-1]])
    # transposed direct form: y[0] = b0*x[0] + zi = c0 since x[0] = 0
    y, _ = lfilter([1 - a], [1, -a], x, zi=[c0])
    return y


def euler_stable_step(p: RoomParams) -> float:
    """Largest step for which explicit Euler is stable [s].

    For dC/dt = -C/tau + f, the amplification factor is |1 - h/tau|,
    so explicit Euler is stable for h < 2 tau (non-oscillatory for h < tau).
    """
    return 2.0 * p.tau


def derivative(C: np.ndarray, dt: float, scheme: str = "central", window: int = 5) -> np.ndarray:
    """Finite-difference estimate of dC/dt on a uniform grid [ppm/s].

    Args:
        C: Concentration samples on a uniform grid [ppm]. NaNs propagate.
        dt: Grid spacing [s].
        scheme: ``"backward"`` (1st order, causal), ``"central"`` (2nd
            order, one-sided at ends) or ``"savgol"`` (Savitzky-Golay
            local quadratic fit, smooths sensor noise).
        window: Window length (odd, samples) for ``"savgol"``.

    Returns:
        Derivative array, same shape as ``C``.
    """
    C = np.asarray(C, dtype=float)
    if scheme == "backward":
        d = np.empty_like(C)
        d[0] = np.nan
        d[1:] = np.diff(C) / dt
        return d
    if scheme == "central":
        return np.gradient(C, dt)
    if scheme == "savgol":
        if window % 2 == 0 or window < 3:
            raise ValueError("savgol window must be odd and >= 3")
        # filter each contiguous run of finite samples separately so gaps
        # (NaN) don't contaminate their neighbours; short runs stay NaN
        d = np.full_like(C, np.nan)
        for s, e in _finite_runs(C):
            if e - s >= window:
                d[s:e] = savgol_filter(C[s:e], window, polyorder=2, deriv=1, delta=dt, mode="interp")
        return d
    raise ValueError(f"unknown scheme {scheme!r}")


def _finite_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) indices of contiguous finite runs in ``x``."""
    f = np.concatenate([[False], np.isfinite(x), [False]])
    edges = np.flatnonzero(np.diff(f.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def estimate_occupancy(
    C: np.ndarray,
    dt: float,
    p: RoomParams,
    scheme: str = "central",
    window: int = 5,
    clip: bool = True,
) -> np.ndarray:
    """Invert the mass balance to estimate occupancy from measured CO2.

    ``N = (V dC/dt + Q (C - C_out)) / (1e6 g)``

    Args:
        C: Uniformly sampled concentration [ppm].
        dt: Sample spacing [s].
        p: Room parameters (calibrated or assumed).
        scheme: Derivative scheme, see :func:`derivative`.
        window: Savitzky-Golay window (samples) if ``scheme="savgol"``.
        clip: If True, negative estimates are set to 0 (occupancy is
            non-negative; negatives arise from noise and model error).

    Returns:
        Estimated occupancy [persons] (real-valued), same shape as ``C``.
    """
    dCdt = derivative(C, dt, scheme, window)
    N = (p.V * dCdt + p.Q * (np.asarray(C) - p.C_out)) / (PPM * p.g)
    return np.clip(N, 0, None) if clip else N


def calibrate(
    C: np.ndarray,
    N: np.ndarray,
    dt: float,
    g: float = G_SEDENTARY,
    scheme: str = "central",
    window: int = 5,
) -> RoomParams:
    """Calibrate V, Q and C_out from paired CO2 and occupancy data.

    The mass balance is linear in the unknowns, so with
    ``a = V/(1e6 g)``, ``b = Q/(1e6 g)``, ``c = -Q C_out/(1e6 g)``:

        N = a dC/dt + b C + c

    which is solved by ordinary least squares (an "equation-error"
    fit). Rows containing NaN are dropped.

    Args:
        C: Uniformly sampled concentration [ppm].
        N: Measured occupancy on the same grid [persons].
        dt: Sample spacing [s].
        g: Assumed per-person generation rate [m^3/s].
        scheme: Derivative scheme used to form dC/dt.
        window: Savitzky-Golay window if applicable.

    Returns:
        Calibrated :class:`RoomParams`.

    Raises:
        ValueError: if the fit gives non-physical (non-positive) V or Q,
            which indicates the data do not excite the dynamics (e.g. a
            room that is never occupied) or are not single-zone.
    """
    dCdt = derivative(C, dt, scheme, window)
    A = np.column_stack([dCdt, C, np.ones_like(C)])
    ok = np.all(np.isfinite(A), axis=1) & np.isfinite(N)
    if ok.sum() < 10:
        raise ValueError("not enough valid samples to calibrate")
    (a, b, c), *_ = np.linalg.lstsq(A[ok], np.asarray(N)[ok], rcond=None)
    V, Q = a * PPM * g, b * PPM * g
    if V <= 0 or Q <= 0:
        raise ValueError(f"non-physical calibration: V={V:.3g} m^3, Q={Q:.3g} m^3/s")
    return RoomParams(V=V, Q=Q, C_out=-c / b, g=g)
```

## FILE: src/occupancy/pipeline.py

```python
"""End-to-end experiment: prepare rooms, fit all estimators, evaluate.

A *room table* is a DataFrame on a uniform DatetimeIndex with columns
``co2_ppm`` (required), optionally ``temp_c``/``rh_pct``, and the
target ``occupancy`` (mean occupant count over each grid interval).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluate import chrono_split, metrics
from .features import build_features
from .models import ML_MODELS, PhysicsEstimator, TimeOfWeekBaseline, make_ml_model
from .resample import to_uniform


def prepare_room(
    env: pd.DataFrame,
    occ: pd.Series,
    freq: str = "15min",
    method: str = "linear",
    max_gap: str = "45min",
    center: bool = True,
    min_occ_coverage: float = 0.5,
) -> pd.DataFrame:
    """Align irregular env readings and occupancy counts on one grid.

    Env channels are interpolated (they are continuous physical fields);
    occupancy is *averaged* over each interval (it is a count, so
    interpolating it would be meaningless).

    With ``center=True`` the occupancy target at grid time t is the mean
    over [t - dt/2, t + dt/2), i.e. centred on the instantaneous CO2
    value it is compared with. Labelling the mean over [t, t + dt) at t
    instead makes occupancy lead CO2 by dt/2, which biased the calibrated
    room volume by about +40% at dt = 15 min in the synthetic study.
    ``occ`` should then be finer than ``freq`` (e.g. 1-min) for the shift
    to be meaningful.

    Args:
        env: Irregular env readings with DatetimeIndex.
        occ: Occupancy counts with DatetimeIndex (any sampling).
        freq: Grid spacing.
        method: Interpolation method for env channels.
        max_gap: Longest gap bridged by interpolation.
        center: If True, centre the occupancy target on the grid time
            (see above); if False, label the interval mean at its start.
        min_occ_coverage: Minimum fraction of the expected occupancy samples
            (inferred from the median spacing of ``occ``) that an interval
            must contain; otherwise its target is NaN.

    Returns:
        Room table (see module docstring); rows lacking CO2 or occupancy
        are kept as NaN so the time grid stays uniform.
    """
    cols = {}
    for c in ("co2_ppm", "temp_c", "rh_pct"):
        if c in env and env[c].notna().sum() >= 2:
            cols[c] = to_uniform(env[c], freq, method, max_gap)
    table = pd.DataFrame(cols)
    if center:
        occ = occ.shift(freq=pd.Timedelta(freq) / 2)
    r = occ.resample(freq, label="left", closed="left")
    o, n = r.mean(), r.count()
    # require >= min_occ_coverage of the expected samples in each interval,
    # so a bin with a single valid minute is not treated as an interval mean
    step = occ.index.to_series().diff().median()
    expected = pd.Timedelta(freq) / step if pd.notna(step) and step > pd.Timedelta(0) else 1
    o = o.where(n >= min_occ_coverage * expected)
    table["occupancy"] = o.reindex(table.index)
    return table


def trim_to_paired(table: pd.DataFrame) -> pd.DataFrame:
    """Restrict a room table to the span where CO2 and occupancy overlap.

    The env and occupancy records only partly overlap; splitting over the
    full CO2 record would put training periods where no labels exist.

    Args:
        table: Room table with ``co2_ppm`` and ``occupancy`` columns.

    Returns:
        The slice of ``table`` from the first to the last row where both
        CO2 and occupancy are present.

    Raises:
        ValueError: if there is less than one day of paired data.
    """
    paired = table["co2_ppm"].notna() & table["occupancy"].notna()
    if paired.sum() < 96:
        raise ValueError("less than one day of paired CO2/occupancy data")
    first, last = paired[paired].index[[0, -1]]
    return table.loc[first:last]


def fit_predict(table: pd.DataFrame, dt: float, fit_mask: np.ndarray, test_mask: np.ndarray,
                models=ML_MODELS, seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, tuple], object]:
    """Fit every estimator on ``fit_mask`` rows and predict ``test_mask`` rows.

    Args:
        table: Room table (see module docstring).
        dt: Grid spacing [s].
        fit_mask: Boolean mask of rows available for fitting.
        test_mask: Boolean mask of rows to predict.
        models: Names of the scikit-learn models to fit (see
            :func:`~occupancy.models.make_ml_model`); the time-of-week
            baseline and the physics model are always included.
        seed: Random seed for stochastic models.

    Returns:
        ``(preds, timings, phys_params)``: predictions per model for the
        test rows (NaN where features are unavailable), ``(fit_s,
        predict_s)`` per model, and the calibrated physics parameters
        (``None`` if calibration was non-physical).
    """
    y = table["occupancy"].to_numpy(float)
    co2 = table["co2_ppm"].to_numpy(float)
    idx_te = table.index[test_mask]
    preds, timings = {}, {}

    # 1) naive baseline: time-of-week mean (uses no sensor data)
    t0 = time.perf_counter()
    ok = fit_mask & np.isfinite(y)
    base = TimeOfWeekBaseline().fit(table.index[ok], y[ok])
    t1 = time.perf_counter()
    preds["time_of_week"] = base.predict(idx_te)
    timings["time_of_week"] = (t1 - t0, time.perf_counter() - t1)

    # 2) physics: calibrated on fit rows only, applied to the full series
    #    (the derivative needs neighbouring samples), scored on test rows
    phys_full = np.full(len(table), np.nan)
    phys_params = None
    try:
        t0 = time.perf_counter()
        ph = PhysicsEstimator(dt).fit(np.where(fit_mask, co2, np.nan), np.where(fit_mask, y, np.nan))
        t1 = time.perf_counter()
        phys_full = ph.predict(co2)
        preds["physics"] = phys_full[test_mask]
        timings["physics"] = (t1 - t0, time.perf_counter() - t1)
        phys_params = ph.params_
    except ValueError as err:  # non-physical calibration -> report, don't crash
        print(f"  physics calibration failed: {err}")

    # 3) data-driven models on causal features (+ hybrids with the physics estimate)
    X = build_features(table)
    X_h = X.assign(phys=phys_full)
    for name in models:
        for feats, label in ((X, name), (X_h, f"{name}+phys")):
            if label.endswith("+phys") and (name not in ("forest", "mlp") or phys_params is None):
                continue  # hybrid only for the two strongest learners
            F = feats.to_numpy()
            finite = np.isfinite(F).all(axis=1)
            fm = fit_mask & finite & np.isfinite(y)
            tm = test_mask & finite
            mdl = make_ml_model(name, seed)
            t0 = time.perf_counter()
            mdl.fit(F[fm], y[fm])
            t1 = time.perf_counter()
            yhat = np.full(test_mask.sum(), np.nan)
            yhat[tm[test_mask]] = np.clip(mdl.predict(F[tm]), 0, None)
            preds[label] = yhat
            timings[label] = (t1 - t0, time.perf_counter() - t1)
    return preds, timings, phys_params


def _score(y_te: np.ndarray, preds: dict, timings: dict) -> pd.DataFrame:
    """Score each model's predictions and attach its fit/predict timings."""
    # score every model on the same rows (where all models have a prediction)
    common = np.isfinite(y_te)
    for yhat in preds.values():
        common &= np.isfinite(yhat)
    rows = []
    for name, yhat in preds.items():
        m = metrics(y_te[common], np.asarray(yhat)[common])
        m.update(model=name, fit_s=timings[name][0], predict_s=timings[name][1])
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def run_room(table: pd.DataFrame, dt: float, models=ML_MODELS, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Chronological evaluation ("deploy on the future").

    Split is chronological 60/20/20 with 1-day gaps over the paired span.
    Models are trained on train+val and scored once on the final test
    period. This tests extrapolation forward in time, including any
    change of operating regime (e.g. teaching -> exam period).

    Args:
        table: Room table (see module docstring).
        dt: Grid spacing [s].
        models: Names of the scikit-learn models to evaluate.
        seed: Random seed for stochastic models.

    Returns:
        ``(scores, extra)``: scores DataFrame (one row per model) and a
        dict with test predictions, truth and physics parameters.
    """
    table = trim_to_paired(table)
    tr, va, te = chrono_split(table.index)
    preds, timings, params = fit_predict(table, dt, tr | va, te, models, seed)
    y = table["occupancy"].to_numpy(float)
    idx = table.index[te]
    return _score(y[te], preds, timings), {
        "preds": {k: pd.Series(v, index=idx) for k, v in preds.items()},
        "y_test": pd.Series(y[te], index=idx), "params": params}


def run_room_cv(table: pd.DataFrame, dt: float, block: str = "7D", gap: str = "1D",
                models=ML_MODELS, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocked leave-one-block-out cross-validation.

    The paired span is cut into contiguous blocks (default one week).
    Each block is predicted by models trained on all *other* blocks,
    excluding a ``gap`` buffer either side so lagged/rolling features and
    autocorrelation cannot leak across the boundary. Out-of-fold
    predictions are pooled and scored once, so every regime in the record
    (teaching weeks, breaks) appears in the test data.

    Args:
        table: Room table (see module docstring).
        dt: Grid spacing [s].
        block: Block length (pandas offset string).
        gap: Buffer excluded from training on either side of a test block.
        models: Names of the scikit-learn models to evaluate.
        seed: Random seed for stochastic models.

    Returns:
        ``(scores, oof)``: pooled scores per model and the out-of-fold
        prediction table (one column per model plus ``y_true`` and ``fold``).
    """
    table = trim_to_paired(table)
    idx = table.index
    edges = pd.date_range(idx[0].floor("D"), idx[-1].ceil("D") + pd.Timedelta(block), freq=block)
    y = table["occupancy"].to_numpy(float)
    oof = pd.DataFrame(index=idx)
    oof["y_true"] = y
    oof["fold"] = -1
    tim: dict[str, list] = {}
    g = pd.Timedelta(gap)
    for k in range(len(edges) - 1):
        te = np.asarray((idx >= edges[k]) & (idx < edges[k + 1]))
        if not (te & np.isfinite(y)).any():
            continue
        fit = np.asarray((idx < edges[k] - g) | (idx >= edges[k + 1] + g))
        if (fit & np.isfinite(y)).sum() < 96 * 3:
            continue
        preds, timings, _ = fit_predict(table, dt, fit, te, models, seed)
        for name, v in preds.items():
            if name not in oof:
                oof[name] = np.nan
            oof.loc[te, name] = v
            tim.setdefault(name, []).append(timings[name])
        oof.loc[te, "fold"] = k
    names = [c for c in oof.columns if c not in ("y_true", "fold")]
    timings = {n: tuple(np.mean(tim[n], axis=0)) for n in names}
    scored = oof[oof["fold"] >= 0]
    scores = _score(scored["y_true"].to_numpy(), {n: scored[n].to_numpy() for n in names}, timings)
    scores["folds"] = int(scored["fold"].nunique())
    return scores, oof
```

## FILE: src/occupancy/resample.py

```python
"""Resampling irregular sensor time series onto a uniform grid.

Finite-difference derivatives and most ML models need uniformly spaced
samples. This module interpolates irregular reports onto a fixed grid
and leaves long gaps as NaN rather than inventing data across them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from .timeutil import unix_seconds

METHODS = ("linear", "pchip", "cubic", "nearest")


def _interp(x: np.ndarray, y: np.ndarray, xq: np.ndarray, method: str) -> np.ndarray:
    """Interpolate ``(x, y)`` at ``xq`` with the named method (NaN outside the data for spline types).

    Raises:
        ValueError: for an unknown method.
    """
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
    # round in UTC: rounding a local time can land in a DST gap/overlap
    tz = s.index.tz
    i0, i1 = (s.index[[0, -1]].tz_convert("UTC") if tz is not None else s.index[[0, -1]])
    grid = pd.date_range(i0.ceil(freq), i1.floor(freq), freq=freq)
    if tz is not None:
        grid = grid.tz_convert(tz)
    x = unix_seconds(s.index)
    xq = unix_seconds(grid)
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

    Args:
        s: Series with a DatetimeIndex.
        method: Interpolation method, as in :func:`to_uniform`.
        frac: Fraction of observations to withhold.
        seed: RNG seed for the choice of withheld points.

    Returns:
        Dict with ``mae``, ``rmse`` and ``max`` absolute error, and ``n``.
    """
    s = s.dropna()
    s = s[~s.index.duplicated()].sort_index()
    rng = np.random.default_rng(seed)
    hold = rng.random(len(s)) < frac
    hold[[0, -1]] = False  # keep end points so there is no extrapolation
    x = unix_seconds(s.index)
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
```

## FILE: src/occupancy/synthetic.py

```python
"""Synthetic room data generated from the CO2 mass-balance model.

Used for (i) verifying the numerical code against a known truth
("manufactured" data) and (ii) developing the pipeline before/without
access to the real building data. The generator mimics the real data's
key defects: irregular sampling, sensor noise and quantisation, and
dropouts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import RoomParams, simulate


def occupancy_schedule(
    start: str = "2024-05-06",
    days: int = 14,
    freq_s: int = 60,
    capacity: int = 12,
    seed: int = 0,
) -> pd.Series:
    """Random but realistic weekday occupancy schedule.

    Weekdays have a base load between ~9:00 and ~17:00 plus random
    meetings; weekends are mostly empty.

    Args:
        start: Start date (Monday recommended).
        days: Number of days to generate.
        freq_s: Time step of the schedule [s].
        capacity: Maximum occupancy of the room [persons].
        seed: RNG seed for reproducibility.

    Returns:
        Integer occupancy series indexed by timestamp.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 86400 // freq_s, freq=f"{freq_s}s")
    N = np.zeros(len(idx), dtype=int)
    hours = idx.hour + idx.minute / 60
    for d in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        in_day = (idx >= day) & (idx < day + pd.Timedelta(days=1))
        if day.day_of_week >= 5:
            if rng.random() < 0.2:  # occasional weekend visit
                a = rng.uniform(10, 14)
                N[in_day & (hours >= a) & (hours < a + rng.uniform(1, 3))] = rng.integers(1, 3)
            continue
        arrive, leave = rng.normal(9, 0.4), rng.normal(17, 0.5)
        base = rng.integers(2, max(3, capacity // 2))
        N[in_day & (hours >= arrive) & (hours < leave)] = base
        lunch = rng.uniform(12, 13)
        N[in_day & (hours >= lunch) & (hours < lunch + 0.75)] //= 2
        for _ in range(rng.integers(0, 4)):  # meetings
            m0 = rng.uniform(arrive, leave - 1)
            sel = in_day & (hours >= m0) & (hours < m0 + rng.uniform(0.5, 1.5))
            N[sel] = np.minimum(capacity, N[sel] + rng.integers(2, capacity // 2 + 1))
    return pd.Series(N, index=idx, name="occupancy")


def make_room(
    params: RoomParams | None = None,
    days: int = 14,
    seed: int = 0,
    sample_mean_s: float = 600.0,
    noise_ppm: float = 15.0,
    dropout: float = 0.03,
    capacity: int = 12,
) -> tuple[pd.DataFrame, pd.Series, RoomParams]:
    """Generate one synthetic room.

    Args:
        params: Room parameters; default is a 150 m^3 meeting room at
            ~3 air changes per hour.
        days: Length of record [days].
        seed: RNG seed.
        sample_mean_s: Mean interval between env-sensor reports [s];
            actual intervals are exponentially jittered (irregular).
        noise_ppm: Std. dev. of additive CO2 sensor noise [ppm].
        dropout: Fraction of reports randomly removed.
        capacity: Max occupancy.

    Returns:
        ``(env, occ, params)`` where ``env`` is an irregularly sampled
        DataFrame with column ``co2_ppm`` (+ ``temp_c``), ``occ`` is the
        1-minute true occupancy, and ``params`` the true parameters.
    """
    rng = np.random.default_rng(seed)
    p = params or RoomParams(V=150.0, Q=150.0 * 3 / 3600, C_out=420.0)
    occ = occupancy_schedule(days=days, seed=seed, capacity=capacity)
    t = (occ.index - occ.index[0]).total_seconds().to_numpy()
    C = simulate(t, occ.to_numpy(float), p, method="exact")
    # irregular sample times
    gaps = rng.exponential(sample_mean_s, size=int(1.5 * t[-1] / sample_mean_s))
    ts = np.cumsum(gaps)
    ts = ts[ts < t[-1]]
    ts = ts[rng.random(len(ts)) > dropout]
    co2 = np.interp(ts, t, C) + rng.normal(0, noise_ppm, len(ts))
    temp = 21 + 0.08 * np.interp(ts, t, occ.to_numpy(float)) + rng.normal(0, 0.1, len(ts))
    env = pd.DataFrame(
        {"co2_ppm": np.round(co2), "temp_c": np.round(temp, 1)},
        index=occ.index[0] + pd.to_timedelta(ts, unit="s"),
    )
    env.index.name = "time"
    return env, occ, p
```

## FILE: src/occupancy/timeutil.py

```python
"""Time helpers robust to pandas datetime resolution (s/ms/us/ns)."""
from __future__ import annotations

import numpy as np
import pandas as pd

def unix_seconds(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Seconds since the Unix epoch as float64, independent of the
    datetime resolution pandas chose (s/ms/us/ns) and of the timezone."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return ((t - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(1, "s")).to_numpy(float)

```

## FILE: scripts/ingest.py

```python
"""Ingest and audit the raw MSI exports, then build per-room tables.

Steps
1. Parse the environmental export (JSON per row) -> tidy table; flag
   physically implausible values; cache ``data/processed/env.parquet``.
2. Parse the occupancy event export (~9M rows) -> per-space 1-min mean
   headcount; cache ``data/processed/occ_1min.parquet``.
3. Write an audit (row counts, sampling intervals, gaps, flagged values,
   coverage) to ``reports/results/audit/``.
4. Associate env sensors with occupancy spaces. The location sheet does
   not link any *working* env sensor to a counted space, so association is
   inferred from data: Pearson correlation between CO2 and headcount after
   removing each series' time-of-week mean (so shared working-hours
   patterns alone cannot create a match). Pairs with residual correlation
   >= ``MIN_RESID_R`` and >= ``MIN_DAYS`` of overlap become room tables in
   ``data/processed/rooms.parquet``; weaker pairs are kept as controls.

Usage:
    python scripts/ingest.py
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd

from occupancy.io import flag_invalid, load_env, load_occupancy, occupancy_on_grid
from occupancy.pipeline import prepare_room

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
AUDIT = ROOT / "reports" / "results" / "audit"
FREQ = "15min"
MIN_DAYS = 14
MIN_RESID_R = 0.5   # primary (co-located) pairs
CONTROL_R = 0.3     # weaker pairs kept as "not co-located" controls
FROZEN_MIN_UNIQUE = 100  # sensors with fewer unique timestamps are frozen

#: Occupancy floor-space IDs that the sensor sheet ties to a room (the
#: sheet's OCC tab is flagged as unreliable; only these two IDs appear in it).
SPACE_ROOM = {
    "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b": "60-G.20",
    "227f1068-9e41-469f-bef0-275f27a0bdba": "60-G.25",
}


def env_sensors_by_room(sheet: pd.DataFrame) -> dict[str, list[str]]:
    """Env sensor IDs per ``<building>-<room>`` key from the ENV sheet."""
    s = sheet.copy()
    s["key"] = s["Building"].astype(str) + "-" + s["Floor/Room"].astype(str)
    return s.groupby("key")["ID"].apply(lambda x: [str(i) for i in x]).to_dict()


def main() -> None:
    """Run the ingest, audit and association steps described in the module docstring."""
    PROC.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    timings = {}

    # ---- environmental data
    t0 = time.perf_counter()
    env_path = PROC / "env.parquet"
    if env_path.exists():
        env = pd.read_parquet(env_path)
    else:
        env = load_env(next((RAW / "env").glob("*.csv")))
        env.to_parquet(env_path)
    timings["env_load_s"] = time.perf_counter() - t0
    env_f = flag_invalid(env)
    flagged = env_f.attrs["n_flagged"]

    per_sensor = env_f.groupby("sensorid").agg(
        n=("time", "size"), start=("time", "min"), end=("time", "max"),
        co2_valid=("co2_ppm", lambda s: s.notna().mean()),
        co2_median=("co2_ppm", "median"),
    )
    dt = env_f.groupby("sensorid")["time"].diff().dt.total_seconds()
    per_sensor["median_interval_s"] = dt.groupby(env_f["sensorid"]).median()
    per_sensor["p95_interval_s"] = dt.groupby(env_f["sensorid"]).quantile(0.95)
    per_sensor["gaps_over_1h"] = (dt > 3600).groupby(env_f["sensorid"]).sum()
    per_sensor.to_csv(AUDIT / "env_per_sensor.csv")
    dup = env_f.duplicated(["sensorid", "time"]).sum()

    # ---- occupancy data
    t0 = time.perf_counter()
    occ_path = PROC / "occ_1min.parquet"
    if occ_path.exists():
        occ15 = pd.read_parquet(occ_path)
        occ_audit = pd.read_csv(AUDIT / "occ_per_space.csv", index_col=0)
    else:
        occ = load_occupancy(next((RAW / "occ").glob("*.csv")))
        timings["occ_load_s"] = time.perf_counter() - t0
        rows, series = [], {}
        for sid, g in occ.groupby("floorspaceid", observed=True):
            d = g["time"].diff().dt.total_seconds()
            rows.append({
                "floorspaceid": sid, "room": SPACE_ROOM.get(sid, "unknown"),
                "events": len(g), "start": g["time"].min(), "end": g["time"].max(),
                "headcount_mean": g["headcount"].mean(), "headcount_p99": g["headcount"].quantile(0.99),
                "headcount_max": g["headcount"].max(), "median_event_gap_s": d.median(),
                "silences_over_30min": int((d > 1800).sum()),
                "non_integer_headcount": int((g["headcount"] % 1 != 0).sum()),
            })
            t1 = time.perf_counter()
            series[sid] = occupancy_on_grid(g, "1min")  # fine grid; centred 15-min means formed later
            timings[f"aggregate_{sid[:8]}_s"] = time.perf_counter() - t1
        occ_audit = pd.DataFrame(rows).set_index("floorspaceid")
        occ_audit.to_csv(AUDIT / "occ_per_space.csv")
        occ15 = pd.DataFrame(series)
        occ15.to_parquet(occ_path)

    # ---- drop exact re-sent payloads and frozen sensors
    env_u = env_f.drop_duplicates(["sensorid", "time"])
    uniq = env_u.groupby("sensorid")["time"].nunique()
    frozen = sorted(uniq[uniq < FROZEN_MIN_UNIQUE].index)
    live = env_u[~env_u["sensorid"].isin(frozen)]

    # ---- data-driven sensor/space association
    def resid(x: pd.Series) -> pd.Series:
        """Remove the time-of-week (weekday, hour) mean from a series."""
        return x - x.groupby([x.index.day_of_week, x.index.hour]).transform("mean")

    rows, tables = [], []
    for sens, e in live.groupby("sensorid"):
        e = e.set_index("time")
        for sid in occ15.columns:
            tab = prepare_room(e, occ15[sid].dropna(), FREQ)
            both = tab["co2_ppm"].notna() & tab["occupancy"].notna()
            n = int(both.sum())
            r = rr = np.nan
            if n > 96:
                c, o = tab.loc[both, "co2_ppm"], tab.loc[both, "occupancy"]
                r, rr = np.corrcoef(c, o)[0, 1], np.corrcoef(resid(c), resid(o))[0, 1]
            role = ("primary" if n >= 96 * MIN_DAYS and rr >= MIN_RESID_R else
                    "control" if n >= 96 * MIN_DAYS and rr >= CONTROL_R else "unused")
            rows.append({"sensorid": sens, "floorspaceid": sid, "space_room": SPACE_ROOM.get(sid, "unknown"),
                         "overlap_days": n / 96, "r": r, "r_resid": rr, "role": role})
            if role != "unused":
                tables.append(tab.assign(room=f"{sens}|{sid[:8]}|{role}"))
    pairs = pd.DataFrame(rows).sort_values("r_resid", ascending=False)
    pairs.to_csv(AUDIT / "sensor_space_association.csv", index=False)
    if tables:
        pd.concat(tables).to_parquet(PROC / "rooms.parquet")

    summary = {
        "env_rows": len(env), "env_sensors": env["sensorid"].nunique(),
        "env_time_range": f"{env['time'].min()} .. {env['time'].max()}",
        "env_duplicates": int(dup), "env_flagged": flagged, "frozen_sensors": frozen,
        "occ_spaces": len(occ_audit), "room_tables": len(tables), **timings,
    }
    pd.Series(summary).to_csv(AUDIT / "summary.csv", header=False)
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(pairs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
```

## FILE: scripts/make_figures.py

```python
"""Build the report figures for the REAL data from already-computed files.

Nothing is re-fitted here: the script only reads the processed tables and the
result files written by ``scripts/ingest.py`` and ``scripts/run_experiment.py``
(plus a cheap interpolation hold-out experiment for figure 7) and writes PNGs
at 150 dpi, prefixed ``real_``, to ``reports/figures/``.

Figures:
    1. ``real_audit_timeline.png``   unique env reports per day + occupancy coverage
    2. ``real_association.png``      sensor-space residual-correlation heatmap
    3. ``real_cv_bars.png``          pooled blocked-CV MAE and R^2 per model and room
    4. ``real_cv_week.png``          one teaching week, out-of-fold predictions + CO2
    5. ``real_scatter.png``          predicted vs true headcount, physics and forest
    6. ``real_chrono_regime.png``    daily means across the chronological test split
    7. ``real_resample_holdout.png`` interpolation hold-out error (also writes
       ``reports/results/real/resample_holdout.csv``)

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from occupancy.resample import holdout_error  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"

DPI = 150
PRIMARY_SENSOR = "6012002000869"
LIVE_SENSORS = ("6012002000869", "6012002000326")
FROZEN_SENSORS = ("6012002000227", "6012002000241", "6012002000777")

#: One colour per model, reused in every figure (Okabe-Ito based, colour-blind safe).
MODEL_COLORS = {
    "time_of_week": "#7f7f7f",
    "physics": "#D55E00",
    "linear": "#CC79A7",
    "tree": "#E69F00",
    "forest": "#009E73",
    "forest+phys": "#0072B2",
    "svr": "#56B4E9",
    "mlp": "#6a3d9a",
    "mlp+phys": "#8c564b",
}
TRUTH_COLOR = "black"
#: Colours for the three rooms in the bar chart (kept apart from the model colours).
ROOM_COLORS = {"primary": "#1f3b63", "0f3f37f8": "#7aa6d6", "941f0352": "#bdbdbd"}
ROOM_HATCH = {"primary": "", "0f3f37f8": "///", "941f0352": "..."}
#: Short labels for the five occupancy floor spaces (keyed by 8-char id prefix).
SPACE_LABELS = {
    "3e5a14df": "3e5a14df (primary)",
    "0f3f37f8": "0f3f37f8 (control A)",
    "941f0352": "941f0352 (control B)",
    "227f1068": "227f1068 (60-G.25)",
    "cca1ac3e": "cca1ac3e (60-G.20)",
}
SENSOR_COLORS = {"6012002000869": "#0072B2", "6012002000326": "#009E73"}
FROZEN_MARKERS = {"6012002000227": "s", "6012002000241": "^", "6012002000777": "D"}
FROZEN_COLOR = "#D55E00"


def set_style() -> None:
    """Apply global matplotlib settings (>= 9 pt text, 150 dpi, light grid)."""
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": DPI, "savefig.dpi": DPI, "axes.grid": True,
        "grid.alpha": 0.3, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.constrained_layout.use": True,
    })


def room_label(room: str) -> str:
    """Human-readable label for a room key like ``6012002000869|3e5a14df|primary``."""
    _, space, role = room.split("|")
    if role == "primary":
        return "primary (co-located)"
    return f"control {'A' if space == '0f3f37f8' else 'B'} ({space})"


def room_key(room: str) -> str:
    """Return the key used in :data:`ROOM_COLORS` for a room string."""
    _, space, role = room.split("|")
    return "primary" if role == "primary" else space


def save(fig: plt.Figure, name: str) -> None:
    """Save ``fig`` into the figures directory at 150 dpi and close it."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote reports/figures/{name}")


def naive_day(t: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar day as a timezone-naive timestamp (for plotting)."""
    idx = pd.DatetimeIndex(t)
    return idx.tz_localize(None).floor("D")


def load_env_unique() -> pd.DataFrame:
    """Environmental table with the re-sent duplicate rows removed."""
    env = pd.read_parquet(PROC / "env.parquet")
    return env.drop_duplicates(["sensorid", "time"]).sort_values(["sensorid", "time"])


def fig_audit_timeline() -> None:
    """Figure 1: unique env reports per day (top) and occupancy coverage (bottom)."""
    raw = pd.read_parquet(PROC / "env.parquet", columns=["sensorid", "time"])
    n_rows = raw.groupby("sensorid").size()
    uniq = raw.drop_duplicates(["sensorid", "time"])
    uniq = uniq.assign(day=naive_day(uniq["time"]))
    n_uniq = uniq.groupby("sensorid").size()
    per_day = uniq.groupby(["sensorid", "day"]).size()

    occ = pd.read_parquet(PROC / "occ_1min.parquet")
    cov = (occ.notna().groupby(occ.index.tz_localize(None).floor("D")).sum() / 1440.0).clip(upper=1.0)
    cov.columns = [SPACE_LABELS[c[:8]] for c in cov.columns]
    order = [SPACE_LABELS[k] for k in ("3e5a14df", "0f3f37f8", "941f0352", "227f1068", "cca1ac3e")]
    cov = cov[order]
    full = pd.date_range(cov.index.min(), cov.index.max(), freq="D")
    cov = cov.reindex(full).fillna(0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 6.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    for s in LIVE_SENSORS:
        d = per_day[s]
        live = d[d >= 100]
        extra = f" from {live.index.min():%b %Y}" if s != PRIMARY_SENSOR else ""
        ax1.plot(d.index, d.values, "o", ms=3, color=SENSOR_COLORS[s],
                 label=f"...{s[-4:]} live{extra} ({n_uniq[s]:,} unique ts)")
    for s in FROZEN_SENSORS:
        d = per_day[s]
        ax1.scatter(d.index, d.values, s=55, marker=FROZEN_MARKERS[s], color=FROZEN_COLOR,
                    edgecolor="black", linewidth=0.6, zorder=5,
                    label=f"...{s[-4:]} frozen ({n_uniq[s]} unique ts, {n_rows[s]:,} rows)")
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 400)
    ax1.set_ylabel("Unique reports per day\n(count, log scale)")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False)
    ax1.set_title("Environmental sensors: only two report live data")

    x_edges = np.append(full.values, full[-1].to_datetime64() + np.timedelta64(1, "D"))
    mesh = ax2.pcolormesh(x_edges, np.arange(len(order) + 1), cov.to_numpy().T,
                          cmap="viridis", vmin=0, vmax=1, shading="flat")
    ax2.set_yticks(np.arange(len(order)) + 0.5)
    ax2.set_yticklabels(order)
    ax2.invert_yaxis()
    ax2.grid(False)
    ax2.set_title("Occupancy data: daily coverage per floor space")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.set_xlabel("Date")
    fig.colorbar(mesh, ax=[ax1, ax2], location="right", fraction=0.04, pad=0.02,
                 label="Fraction of minutes with data")
    save(fig, "real_audit_timeline.png")


def fig_association() -> None:
    """Figure 2: heatmap of residual correlation between sensors and spaces."""
    df = pd.read_csv(RES / "audit" / "sensor_space_association.csv", dtype={"sensorid": str})
    df["sp"] = df["floorspaceid"].str[:8]
    cols = ["3e5a14df", "0f3f37f8", "941f0352", "227f1068", "cca1ac3e"]
    rows = list(LIVE_SENSORS)
    r = np.full((len(rows), len(cols)), np.nan)
    days = np.zeros_like(r)
    for i, s in enumerate(rows):
        for j, c in enumerate(cols):
            m = df[(df["sensorid"] == s) & (df["sp"] == c)]
            if len(m):
                r[i, j] = m["r_resid"].iloc[0]
                days[i, j] = m["overlap_days"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    im = ax.imshow(np.ma.masked_invalid(r), cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(r[i, j]):
                txt, col = f"n/a\n{days[i, j]:.0f} d", "black"
            else:
                txt = f"{r[i, j]:.2f}\n{days[i, j]:.0f} d"
                col = "white" if abs(r[i, j]) > 0.4 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=10)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SPACE_LABELS[c].replace(" (", "\n(") for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"sensor ...{s[-4:]}" for s in rows])
    ax.set_xlabel("Occupancy floor space")
    ax.set_ylabel("Env sensor")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, label="Residual correlation $r_{resid}$", fraction=0.04, pad=0.02)
    ax.set_title("CO$_2$-headcount association (value; days of overlap)", fontsize=10)
    save(fig, "real_association.png")


def fig_cv_bars() -> None:
    """Figure 3: grouped horizontal bars of pooled CV MAE and R^2 per model and room."""
    cv = pd.read_csv(RES / "real" / "cv_scores_by_room.csv")
    rooms = sorted(cv["room"].unique(), key=lambda r: (room_key(r) != "primary", r))
    prim = [r for r in rooms if room_key(r) == "primary"][0]
    models = cv[cv["room"] == prim].sort_values("mae")["model"].tolist()
    nm, nr = len(models), len(rooms)
    h = 0.8 / nr
    y0 = np.arange(nm)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.6), sharey=True)
    for ax, metric, xl in zip(axes, ("mae", "r2"),
                              ("Pooled CV MAE (people)", "Pooled CV $R^2$ (-)")):
        for k, room in enumerate(rooms):
            sub = cv[cv["room"] == room].set_index("model").loc[models]
            key = room_key(room)
            nf = int(sub["folds"].iloc[0])
            ax.barh(y0 + (k - (nr - 1) / 2) * h, sub[metric].to_numpy(), height=h * 0.92,
                    color=ROOM_COLORS[key], hatch=ROOM_HATCH[key], edgecolor="black",
                    linewidth=0.5, label=f"{room_label(room)}, {nf} weeks")
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y0)
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=1, frameon=False)
    fig.suptitle("Blocked leave-one-week-out CV (models sorted by primary-room MAE)", fontsize=10)
    save(fig, "real_cv_bars.png")


def pick_teaching_week(oof: pd.DataFrame) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Pick the May-2024 CV fold with the highest occupancy variance.

    Args:
        oof: Out-of-fold predictions of the primary room (``fold``, ``y_true``).

    Returns:
        ``(fold, start, end)`` of the chosen fold.
    """
    best = None
    for f, g in oof[oof["fold"] >= 0].groupby("fold"):
        if not (g.index[0].year == 2024 and g.index[0].month == 5):
            continue
        v = float(np.nanvar(g["y_true"].to_numpy()))
        if best is None or v > best[0]:
            best = (v, int(f), g.index[0].floor("D"), g.index[-1])
    assert best is not None, "no May-2024 fold found"
    print(f"chosen teaching week: fold {best[1]} {best[2].date()}..{best[3].date()} var={best[0]:.2f}")
    return best[1], best[2], best[3]


def load_oof(primary: bool = True) -> pd.DataFrame:
    """Out-of-fold CV predictions, restricted to the primary room."""
    oof = pd.read_parquet(RES / "real" / "cv_oof_predictions.parquet")
    return oof[oof["room"].str.endswith("|primary") == primary]


def fig_cv_week() -> None:
    """Figure 4: one May-2024 week, truth vs out-of-fold predictions and CO2."""
    oof = load_oof()
    fold, t0, t1 = pick_teaching_week(oof)
    wk = oof[oof["fold"] == fold]
    env = load_env_unique()
    co2 = env[(env["sensorid"] == PRIMARY_SENSOR) & (env["time"] >= t0) & (env["time"] <= t1 + pd.Timedelta("15min"))]
    tz = wk.index.tz

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(wk.index, wk["y_true"], color=TRUTH_COLOR, lw=1.8, label="truth (headcount)")
    for m in ("time_of_week", "physics", "forest"):
        ax1.plot(wk.index, wk[m], color=MODEL_COLORS[m], lw=1.1, label=m)
    ax1.set_ylabel("Occupancy (people)")
    ax1.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), framealpha=0.9,
               columnspacing=1.0, handlelength=1.4)
    ax1.set_title(f"Primary room, out-of-fold predictions, week of {t0:%d %b %Y} (CV fold {fold})")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.12)
    ax2.plot(co2["time"], co2["co2_ppm"], color="#333333", lw=1.0)
    ax2.set_ylabel("CO$_2$ (ppm)")
    ax2.xaxis.set_major_locator(mdates.DayLocator(tz=tz))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=tz))
    ax2.set_xlabel(f"Local time, sensor ...{PRIMARY_SENSOR[-4:]}")
    ax2.set_xlim(t0, t1 + pd.Timedelta("1h"))
    save(fig, "real_cv_week.png")


def scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """MAE and R^2 of predictions ``p`` against ``y``."""
    mae = float(np.mean(np.abs(p - y)))
    r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
    return mae, r2


def fig_scatter() -> None:
    """Figure 5: predicted vs true headcount for physics and forest (pooled CV)."""
    oof = load_oof()
    sub = oof[oof["fold"] >= 0]
    lim = float(np.ceil(max(sub["y_true"].max(), sub[["physics", "forest"]].max().max()) / 5) * 5)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.0), sharex=True, sharey=True)
    for ax, m in zip(axes, ("physics", "forest")):
        d = sub[["y_true", m]].dropna()
        mae, r2 = scores(d["y_true"].to_numpy(), d[m].to_numpy())
        ax.scatter(d["y_true"], d[m], s=9, alpha=0.25, color=MODEL_COLORS[m], linewidths=0)
        ax.plot([0, lim], [0, lim], color="black", lw=1.0, ls="--")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("True occupancy (people)")
        ax.set_title(m)
        ax.text(0.04, 0.96, f"MAE = {mae:.2f}\n$R^2$ = {r2:.2f}\nn = {len(d):,}", transform=ax.transAxes,
                va="top", fontsize=9.5, bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
    axes[0].set_ylabel("Predicted occupancy (people)")
    fig.suptitle("Primary room, out-of-fold predictions (dashed: y = x)", fontsize=10)
    save(fig, "real_scatter.png")


def fig_chrono_regime() -> None:
    """Figure 6: daily mean headcount over the paired span with the test period shaded."""
    oof = load_oof()
    oof = oof[oof["y_true"].notna()]
    truth = oof["y_true"].groupby(naive_day(oof.index)).mean()
    tp = pd.read_parquet(RES / "real" / "test_predictions.parquet")
    tp = tp[tp["room"].str.endswith("|primary")]
    t_start = naive_day(tp.index).min()
    t_end = naive_day(tp.index).max() + pd.Timedelta("1D")
    tp = tp[tp["y_true"].notna()]
    day = naive_day(tp.index)
    daily = tp[["y_true", "time_of_week", "physics", "forest"]].groupby(day).mean()

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.axvspan(t_start, t_end, color="#f2c14e", alpha=0.3, lw=0)
    ax.text(t_start + pd.Timedelta("0.5D"), 0.96, "chronological\ntest period", transform=ax.get_xaxis_transform(),
            va="top", ha="left", fontsize=9)
    ax.plot(truth.index, truth.values, "o-", color=TRUTH_COLOR, lw=1.6, ms=3.5, label="truth")
    for m in ("time_of_week", "physics", "forest"):
        ax.plot(daily.index, daily[m], "o-", color=MODEL_COLORS[m], lw=1.2, ms=3.5, label=m)
    ax.set_ylabel("Daily mean occupancy (people)")
    ax.set_xlabel("Date (2024)")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlim(truth.index.min() - pd.Timedelta("1D"), t_end)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), framealpha=0.9, columnspacing=1.0)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.set_title("Primary room: end-of-semester regime shift")
    save(fig, "real_chrono_regime.png")


def run_holdout() -> pd.DataFrame:
    """Interpolation hold-out error for the primary sensor's CO2.

    Returns:
        Table with mean and sd (over seeds 0..4) of MAE, RMSE and max error
        for every interpolation method and withheld fraction.
    """
    env = load_env_unique()
    s = env[env["sensorid"] == PRIMARY_SENSOR].set_index("time")["co2_ppm"].dropna()
    s = s[~s.index.duplicated()].sort_index()
    rows = []
    for method in ("linear", "pchip", "cubic", "nearest"):
        for frac in (0.1, 0.3, 0.5):
            for seed in range(5):
                rows.append({"method": method, "frac": frac, "seed": seed,
                             **holdout_error(s, method, frac, seed)})
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["method", "frac"], sort=False).agg(
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"), rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"), max_mean=("max", "mean"), n_mean=("n", "mean")).reset_index()
    agg.insert(0, "n_obs", len(s))
    out = RES / "real" / "resample_holdout.csv"
    agg.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}")
    return agg


def fig_resample_holdout() -> None:
    """Figure 7: mean hold-out MAE (ppm) per interpolation method and fraction."""
    agg = run_holdout()
    methods = list(dict.fromkeys(agg["method"]))
    fracs = sorted(agg["frac"].unique())
    cols = {"linear": "#4c78a8", "pchip": "#72b7b2", "cubic": "#f58518", "nearest": "#9d9da1"}
    hatch = {"linear": "", "pchip": "//", "cubic": "xx", "nearest": ".."}
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    for k, m in enumerate(methods):
        d = agg[agg["method"] == m].set_index("frac").loc[fracs]
        ax.bar(np.arange(len(fracs)) + (k - (len(methods) - 1) / 2) * w, d["mae_mean"], width=w * 0.92,
               yerr=d["mae_sd"], capsize=3, color=cols[m], hatch=hatch[m], edgecolor="black",
               linewidth=0.5, label=m, error_kw={"lw": 1.0})
    ax.set_xticks(np.arange(len(fracs)))
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_xlabel("Fraction of CO$_2$ observations withheld")
    ax.set_ylabel("Hold-out MAE (ppm)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", framealpha=0.9, columnspacing=1.0, handlelength=1.4)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
    ax.set_title(f"Interpolation hold-out, sensor ...{PRIMARY_SENSOR[-4:]} (mean $\\pm$ sd, 5 seeds)", fontsize=10)
    save(fig, "real_resample_holdout.png")


def main() -> None:
    """Build all figures."""
    set_style()
    fig_audit_timeline()
    fig_association()
    fig_cv_bars()
    fig_cv_week()
    fig_scatter()
    fig_chrono_regime()
    fig_resample_holdout()


if __name__ == "__main__":
    main()
```

## FILE: scripts/profile_performance.py

```python
"""Performance analysis: timing, scaling, FLOP and arithmetic-intensity estimates.

Produces ``reports/results/performance.csv``, ``reports/results/flops.csv``,
``reports/results/profile_pipeline.txt`` and ``reports/figures/fig_scaling.png``.

Usage:
    python scripts/profile_performance.py
"""
from __future__ import annotations

import cProfile
import io
import pathlib
import pstats
import timeit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.physics import RoomParams, estimate_occupancy, simulate, simulate_uniform_fast

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
FIG = ROOT / "reports" / "figures"
P = RoomParams(V=150.0, Q=0.125)


def best_time(fn, repeat: int = 3) -> float:
    """Best-of-``repeat`` wall time of one call [s] (min filters OS noise)."""
    n = 1
    while timeit.timeit(fn, number=n) < 0.2 and n < 1000:
        n *= 4
    return min(timeit.repeat(fn, number=n, repeat=repeat)) / n


def scaling() -> pd.DataFrame:
    """Time forward solvers and the inverse estimator vs problem size."""
    rows = []
    rng = np.random.default_rng(0)
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        t = np.arange(n) * 60.0
        N = rng.integers(0, 12, n).astype(float)
        C = simulate_uniform_fast(N, 60.0, P)
        cases = {
            "rk4_loop": lambda: simulate(t, N, P, method="rk4"),
            "exact_loop": lambda: simulate(t, N, P, method="exact"),
            "exact_lfilter": lambda: simulate_uniform_fast(N, 60.0, P),
            "inverse_central": lambda: estimate_occupancy(C, 60.0, P, "central"),
            "inverse_savgol": lambda: estimate_occupancy(C, 60.0, P, "savgol"),
        }
        for name, fn in cases.items():
            if n == 1_000_000 and name.endswith("loop"):
                continue  # too slow; extrapolate from linear scaling
            s = best_time(fn, repeat=3)
            rows.append({"case": name, "n": n, "seconds": s, "ns_per_sample": 1e9 * s / n})
    return pd.DataFrame(rows)


def flop_table(n_features: int = 24) -> pd.DataFrame:
    """Hand-counted FLOPs and bytes per sample for each estimator.

    Counts multiply/add/divide/exp as 1 FLOP each. Bytes assume float64
    (8 B) and count compulsory reads/writes of the per-sample arrays only.
    """
    h1, h2 = 64, 32  # MLP hidden layers (see models.make_ml_model)
    mlp_flops = 2 * (n_features * h1 + h1 * h2 + h2) + h1 + h2  # MACs*2 + ReLU
    rows = [
        # rhs = (k*N - Q*(C - Cout))/V : 1 mul, 1 sub, 1 mul, 1 sub, 1 div = 5
        {"method": "Euler step", "flops": 5 + 2, "bytes": 8 * 3},
        {"method": "RK4 step", "flops": 4 * 5 + 3 * 2 + 6, "bytes": 8 * 3},
        {"method": "exact step (lfilter)", "flops": 3 + 3, "bytes": 8 * 3},
        {"method": "inverse, central diff", "flops": 3 + 5, "bytes": 8 * 2},
        {"method": "inverse, Savitzky-Golay(5)", "flops": 2 * 5 + 5, "bytes": 8 * 2},
        {"method": f"MLP inference ({n_features}-64-32-1)", "flops": mlp_flops,
         "bytes": 8 * (n_features + 1)},
        {"method": "random forest (200 trees, depth<=12)", "flops": 200 * 12,
         "bytes": 8 * n_features},
    ]
    df = pd.DataFrame(rows)
    df["arith_intensity_flop_per_byte"] = df["flops"] / df["bytes"]
    return df


def profile_pipeline() -> str:
    """cProfile the full synthetic experiment and return the top entries."""
    import runpy

    pr = cProfile.Profile()
    pr.enable()
    import sys

    argv, sys.argv = sys.argv, ["run_experiment.py", "--synthetic"]
    try:
        runpy.run_path(str(ROOT / "scripts" / "run_experiment.py"), run_name="__main__")
    finally:
        sys.argv = argv
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    return s.getvalue()


def main() -> None:
    """Run all timing, scaling and FLOP analyses and write the result files."""
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    perf = scaling()
    perf.to_csv(RES / "performance.csv", index=False)
    print(perf.pivot(index="n", columns="case", values="ns_per_sample").round(1))
    flops = flop_table()
    flops.to_csv(RES / "flops.csv", index=False)
    print(flops.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for case, g in perf.groupby("case"):
        ax.loglog(g["n"], g["seconds"], "o-", label=case)
    ax.set_xlabel("number of samples n [-]")
    ax.set_ylabel("wall time [s]")
    ax.set_title("Solver / estimator scaling (best of 3)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_scaling.png", dpi=150)

    (RES / "profile_pipeline.txt").write_text(profile_pipeline())
    print("profile written to", RES / "profile_pipeline.txt")


if __name__ == "__main__":
    main()
```

## FILE: scripts/run_experiment.py

```python
"""Run the full model comparison on synthetic or real rooms.

Usage:
    python scripts/run_experiment.py --synthetic          # manufactured rooms
    python scripts/run_experiment.py --data data/processed/rooms.parquet

Outputs go to ``reports/results/<tag>/``: per-room scores, pooled summary
and test-period predictions.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from occupancy.physics import RoomParams
from occupancy.pipeline import prepare_room, run_room, run_room_cv
from occupancy.synthetic import make_room

ROOT = pathlib.Path(__file__).resolve().parents[1]


def synthetic_rooms(n: int = 4, days: int = 42) -> dict[str, pd.DataFrame]:
    """A few manufactured rooms with different sizes and ventilation."""
    specs = [(150, 3.0, 12), (60, 2.0, 6), (400, 4.0, 30), (90, 1.5, 8)][:n]
    rooms = {}
    for i, (V, ach, cap) in enumerate(specs):
        env, occ, _ = make_room(RoomParams(V=V, Q=V * ach / 3600), days=days, seed=i, capacity=cap)
        rooms[f"synthetic_{i}"] = prepare_room(env, occ)
    return rooms


def main() -> None:
    """Parse command-line arguments, evaluate all models per room and write results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", type=pathlib.Path)
    ap.add_argument("--freq", default="15min")
    args = ap.parse_args()
    if args.synthetic:
        rooms, tag = synthetic_rooms(), "synthetic"
    elif args.data:
        df = pd.read_parquet(args.data)
        rooms = {r: g.drop(columns="room").asfreq(args.freq) for r, g in df.groupby("room")}
        tag = "real"
    else:
        ap.error("give --synthetic or --data")
    out = ROOT / "reports" / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    dt = pd.Timedelta(args.freq).total_seconds()
    all_scores, all_preds, params, cv_scores, cv_oof = [], [], [], [], []
    for name, table in rooms.items():
        print(f"room {name}: {len(table)} rows, occupancy coverage "
              f"{table['occupancy'].notna().mean():.0%}")
        scores, extra = run_room(table, dt)
        cvs, oof = run_room_cv(table, dt)
        cv_scores.append(cvs.assign(room=name))
        cv_oof.append(oof.assign(room=name))
        all_scores.append(scores.assign(room=name))
        p = extra["preds"]
        all_preds.append(pd.DataFrame(p).assign(y_true=extra["y_test"], room=name))
        if extra["params"] is not None:
            q = extra["params"]
            params.append({"room": name, "V_m3": q.V, "Q_m3s": q.Q, "ACH": q.ach, "C_out_ppm": q.C_out})
    scores = pd.concat(all_scores)
    scores.to_csv(out / "scores_by_room.csv")
    pd.concat(all_preds).to_parquet(out / "test_predictions.parquet")
    pd.DataFrame(params).to_csv(out / "physics_params.csv", index=False)
    summary = scores.groupby(level=0)[["mae", "rmse", "r2", "f1", "fit_s", "predict_s"]].median()
    summary = summary.sort_values("mae")
    summary.to_csv(out / "summary_median.csv")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary)
    cv = pd.concat(cv_scores)
    cv.to_csv(out / "cv_scores_by_room.csv")
    pd.concat(cv_oof).to_parquet(out / "cv_oof_predictions.parquet")
    print("\nBlocked leave-one-week-out CV (pooled out-of-fold):")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(cv.reset_index().pivot(index="model", columns="room", values="mae"))
        print(cv.reset_index().pivot(index="model", columns="room", values="r2"))
    if params:
        print(pd.DataFrame(params).round(3).to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
```

## FILE: scripts/sensitivity_real.py

```python
"""Sensitivity, model-selection and data-requirement study on the real primary pair.

All numbers use the same blocked leave-one-week-out protocol as
``run_experiment.py`` (1-day buffers, pooled out-of-fold scoring) on the
co-located sensor/space pair.

Studies
1. Physics estimator vs derivative scheme and grid spacing.
2. Invariance of calibrated predictions to the assumed per-person CO2
   generation rate g (only the physical interpretation of V, Q changes).
3. Hyper-parameter sensitivity of the forest and MLP (model selection).
4. Learning curve: accuracy vs number of labelled training days for the
   physics model (3 parameters) vs the forest and linear models.

Usage:
    python scripts/sensitivity_real.py
Outputs: ``reports/results/real/sens_*.csv`` and ``reports/figures/real_sens_*.png``.
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.evaluate import metrics
from occupancy.features import build_features
from occupancy.models import PhysicsEstimator, make_ml_model
from occupancy.pipeline import prepare_room, trim_to_paired

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results" / "real"
FIG = ROOT / "reports" / "figures"
SENSOR, SPACE = "6012002000869", "3e5a14df-4b14-4ea5-bb9d-46577c6dc9c8"


def load_pair(freq: str) -> pd.DataFrame:
    """Rebuild the primary room table at grid spacing ``freq``."""
    env = pd.read_parquet(ROOT / "data/processed/env.parquet")
    env = env[env["sensorid"] == SENSOR].drop_duplicates("time").set_index("time")
    occ = pd.read_parquet(ROOT / "data/processed/occ_1min.parquet")[SPACE].dropna()
    return trim_to_paired(prepare_room(env, occ, freq))


def week_folds(idx: pd.DatetimeIndex, gap: str = "1D"):
    """Yield (fit_mask, test_mask) for leave-one-week-out with buffers."""
    edges = pd.date_range(idx[0].floor("D"), idx[-1].ceil("D") + pd.Timedelta("7D"), freq="7D")
    g = pd.Timedelta(gap)
    for k in range(len(edges) - 1):
        te = np.asarray((idx >= edges[k]) & (idx < edges[k + 1]))
        fit = np.asarray((idx < edges[k] - g) | (idx >= edges[k + 1] + g))
        if te.any() and fit.sum() > 96 * 3:
            yield fit, te


def cv_physics(t: pd.DataFrame, dt: float, scheme: str, window: int = 5, g: float = 5.2e-6):
    """Pooled out-of-fold physics predictions; returns (metrics, params list)."""
    y, c = t["occupancy"].to_numpy(float), t["co2_ppm"].to_numpy(float)
    oof = np.full(len(t), np.nan)
    params = []
    for fit, te in week_folds(t.index):
        try:
            ph = PhysicsEstimator(dt, scheme, window, g).fit(np.where(fit, c, np.nan), np.where(fit, y, np.nan))
        except ValueError:
            continue
        oof[te] = ph.predict(c)[te]
        params.append(ph.params_)
    return metrics(y, oof), params, oof


def cv_ml(t: pd.DataFrame, model_factory, train_days: int | None = None, seed: int = 0):
    """Pooled out-of-fold predictions for an sklearn model on causal features."""
    X = build_features(t).to_numpy()
    y = t["occupancy"].to_numpy(float)
    ok = np.isfinite(X).all(axis=1)
    oof = np.full(len(t), np.nan)
    rng = np.random.default_rng(seed)
    for fit, te in week_folds(t.index):
        fm = fit & ok & np.isfinite(y)
        if train_days is not None:
            fm &= subsample_days(t.index, fm, train_days, rng)
        if fm.sum() < 20:
            continue
        m = model_factory()
        m.fit(X[fm], y[fm])
        tm = te & ok
        oof[tm] = np.clip(m.predict(X[tm]), 0, None)
    return metrics(y, oof)


def subsample_days(idx: pd.DatetimeIndex, fit: np.ndarray, n_days: int, rng) -> np.ndarray:
    """Mask keeping only ``n_days`` randomly chosen whole days of ``fit``."""
    days = pd.Index(idx.normalize())
    cand = days[fit].unique()
    keep = rng.choice(cand, size=min(n_days, len(cand)), replace=False)
    return np.asarray(days.isin(keep))


def main() -> None:
    """Run the four real-data studies and write their CSV files and figures."""
    RES.mkdir(parents=True, exist_ok=True)
    tables = {f: load_pair(f) for f in ("10min", "15min", "30min")}

    # 1) derivative scheme x grid spacing
    rows = []
    for f, t in tables.items():
        dt = pd.Timedelta(f).total_seconds()
        for scheme, w in [("backward", 5), ("central", 5), ("savgol", 3), ("savgol", 5), ("savgol", 7)]:
            if scheme == "savgol" and w == 3:
                scheme_lbl = "savgol3"
            else:
                scheme_lbl = scheme if scheme != "savgol" else f"savgol{w}"
            m, params, _ = cv_physics(t, dt, scheme, w)
            V = np.median([p.V for p in params]) if params else np.nan
            Q = np.median([p.Q for p in params]) if params else np.nan
            rows.append({"grid": f, "scheme": scheme_lbl, "mae": m["mae"], "r2": m["r2"], "f1": m["f1"],
                         "V_median": V, "ACH_median": 3600 * Q / V if params else np.nan,
                         "folds_ok": len(params)})
    s1 = pd.DataFrame(rows)
    s1.to_csv(RES / "sens_physics_scheme_grid.csv", index=False)
    print(s1.round(3).to_string(index=False))

    # 2) g invariance
    t15 = tables["15min"]
    rows, preds = [], []
    for g in (3.5e-6, 5.2e-6, 7.0e-6):
        m, params, oof = cv_physics(t15, 900.0, "savgol", 5, g)
        preds.append(oof)
        rows.append({"g_m3s": g, "mae": m["mae"], "r2": m["r2"],
                     "V_median": np.median([p.V for p in params]),
                     "Q_median": np.median([p.Q for p in params])})
    s2 = pd.DataFrame(rows)
    s2["max_abs_pred_diff_vs_nominal"] = [np.nanmax(np.abs(p - preds[1])) for p in preds]
    s2.to_csv(RES / "sens_g_invariance.csv", index=False)
    print(s2.to_string(index=False))

    # 3) hyper-parameters (forest depth / leaf size, MLP width / L2)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rows = []
    for depth in (4, 8, 12, None):
        for leaf in (5, 20, 50):
            m = cv_ml(t15, lambda: RandomForestRegressor(200, max_depth=depth, min_samples_leaf=leaf,
                                                         n_jobs=-1, random_state=0))
            rows.append({"model": "forest", "setting": f"depth={depth},leaf={leaf}", "mae": m["mae"], "r2": m["r2"]})
    for hidden in ((16,), (32, 16), (64, 32)):
        for alpha in (1e-4, 1e-2, 1.0):
            m = cv_ml(t15, lambda: make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=hidden, alpha=alpha, max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=0)))
            rows.append({"model": "mlp", "setting": f"hidden={hidden},alpha={alpha}", "mae": m["mae"], "r2": m["r2"]})
    s3 = pd.DataFrame(rows)
    s3.to_csv(RES / "sens_hyperparameters.csv", index=False)
    print(s3.round(3).to_string(index=False))

    # 4) learning curve: labelled days available for training
    rows = []
    for nd in (1, 2, 4, 7, 14):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            # physics with subsampled calibration days
            y, c = t15["occupancy"].to_numpy(float), t15["co2_ppm"].to_numpy(float)
            oof = np.full(len(t15), np.nan)
            for fit, te in week_folds(t15.index):
                fm = fit & subsample_days(t15.index, fit & np.isfinite(y), nd, rng)
                try:
                    ph = PhysicsEstimator(900.0).fit(np.where(fm, c, np.nan), np.where(fm, y, np.nan))
                    oof[te] = ph.predict(c)[te]
                except ValueError:
                    pass
            mp = metrics(y, oof)
            rows.append({"days": nd, "seed": seed, "model": "physics", "mae": mp["mae"], "r2": mp["r2"],
                         "coverage": float(np.isfinite(oof).mean())})
            for name in ("forest", "linear"):
                m = cv_ml(t15, lambda: make_ml_model(name), train_days=nd, seed=seed)
                rows.append({"days": nd, "seed": seed, "model": name, "mae": m["mae"], "r2": m["r2"],
                             "coverage": 1.0})
    s4 = pd.DataFrame(rows)
    s4.to_csv(RES / "sens_learning_curve.csv", index=False)
    lc = s4.groupby(["model", "days"])[["mae", "r2"]].agg(["mean", "std"])
    print(lc.round(3).to_string())

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for scheme, g in s1.groupby("scheme"):
        ax[0].plot([pd.Timedelta(x).total_seconds() / 60 for x in g["grid"]], g["mae"], "o-", label=scheme)
    ax[0].set_xlabel("grid spacing [min]")
    ax[0].set_ylabel("CV MAE [persons]")
    ax[0].set_title("Physics estimator: derivative scheme")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for name, g in s4.groupby("model"):
        st = g.groupby("days")["r2"].agg(["mean", "std"])
        ax[1].errorbar(st.index, st["mean"], yerr=st["std"], marker="o", capsize=3, label=name)
    ax[1].set_xscale("log")
    ax[1].set_xticks([1, 2, 4, 7, 14], ["1", "2", "4", "7", "14"])
    ax[1].set_xlabel("labelled training days [d]")
    ax[1].set_ylabel("CV R$^2$ [-]")
    ax[1].set_title("Data requirement (learning curve)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "real_sens_physics_learning.png", dpi=150)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
```

## FILE: scripts/sensitivity_synthetic.py

```python
"""Verification and sensitivity study on synthetic rooms (known truth).

Because the synthetic generator returns the true room parameters and the
true occupancy, the numerical code and the calibration/estimation chain
can be *verified* (does the code converge at the theoretical order?) and
its *sensitivity* to analyst choices (grid spacing, derivative scheme) and
to data quality (CO2 sensor noise, label alignment) can be quantified.

Sections
--------
1. Convergence: global error at t = 1 h of explicit Euler and RK4 against
   the exact solution, constant N = 10, V = 150 m^3, Q = 0.125 m^3/s.
2. Explicit-Euler stability: decay from 1000 ppm with N = 0 for
   h = 0.5, 1.5, 2.2 time constants (tau = V/Q).
3. Sensitivity of calibration + occupancy estimation to grid spacing,
   derivative scheme and sensor noise.
4. Alignment check of the occupancy target (interval mean over
   ``[t, t+dt)``) against the instantaneous CO2 at ``t``.

Design choices
--------------
* ``make_room`` reports CO2 at irregular intervals with a default mean of
  600 s. A 5-min grid would then be *finer than the data* (linear
  interpolation between sparse samples, whose slope is piecewise constant),
  so every run of the grid-spacing sweep (and of the alignment check) uses
  ``sample_mean_s = 120`` s, giving ~5 reports per 5-min cell. The sweep is
  additionally repeated with the default 600 s for comparison
  (``sample_mean_s`` column of ``sensitivity_grid.csv``). The noise sweep
  uses the default 600 s room.
* Calibration uses the first 70 % of the record (by sample count) and the
  occupancy estimate is evaluated on the last 30 %.
* Savitzky-Golay window is fixed at 5 *samples* (so its time span grows
  with the grid spacing).
* Seed 0 is the "default" room. Seeds 0-4 are run for every setting
  (different random schedule and noise, same true parameters) so that
  differences can be compared with seed-to-seed spread; figures show the
  mean +/- one standard deviation over seeds.

Outputs: ``reports/figures/*.png`` (150 dpi) and
``reports/results/synthetic/*.csv``.

Usage:
    python scripts/sensitivity_synthetic.py
"""
from __future__ import annotations

import pathlib
import time
import warnings
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from occupancy.evaluate import metrics  # noqa: E402
from occupancy.physics import (  # noqa: E402
    RoomParams,
    calibrate,
    estimate_occupancy,
    simulate,
)
from occupancy.pipeline import prepare_room  # noqa: E402
from occupancy.synthetic import make_room  # noqa: E402

# src/occupancy/synthetic.py uses Timestamp.day_of_week, deprecated in pandas 3/4
# (harmless, reported separately); silence only that message.
warnings.filterwarnings("ignore", message="Timestamp.day_of_week is deprecated")

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results" / "synthetic"

SEEDS = (0, 1, 2, 3, 4)
FREQS = ("5min", "10min", "15min", "30min")
SCHEMES = ("backward", "central", "savgol")
NOISES = (0.0, 5.0, 15.0, 30.0, 50.0)
DAYS = 28
TRAIN_FRAC = 0.7
SAVGOL_WINDOW = 5
FINE_SAMPLE_S = 120.0
DEFAULT_SAMPLE_S = 600.0

# colour-blind-safe (Okabe-Ito) palette
COLORS = {"backward": "#D55E00", "central": "#0072B2", "savgol": "#009E73"}
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 100,
        "savefig.dpi": 150,
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to ``reports/figures`` at 150 dpi and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def _save_csv(df: pd.DataFrame, name: str) -> None:
    """Write a table to ``reports/results/synthetic``."""
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES_DIR / name, index=False)
    print(f"  wrote reports/results/synthetic/{name}")


# --------------------------------------------------------------------------
# 1. Convergence
# --------------------------------------------------------------------------
def convergence() -> pd.DataFrame:
    """Measure the global error of Euler and RK4 at t = 1 h vs step size.

    Constant N = 10, C(0) = C_out. The reference is the exact solution of
    the linear ODE (``method="exact"``). Observed order is the slope of the
    least-squares line through log(error) vs log(h), and also the pairwise
    order between consecutive step sizes.

    Returns:
        Table of errors and observed orders (also written to
        ``convergence.csv`` and plotted in ``fig_convergence.png``).
    """
    p = RoomParams(V=150.0, Q=0.125)
    hs = np.array([37.5, 75.0, 150.0, 300.0, 600.0])
    T = 3600.0
    rows = []
    errs: dict[str, np.ndarray] = {}
    for method in ("euler", "rk4"):
        e = []
        for h in hs:
            n = int(round(T / h))
            t = np.arange(n + 1) * h
            N = np.full(n + 1, 10.0)
            ref = simulate(t, N, p, method="exact")[-1]
            e.append(abs(simulate(t, N, p, method=method)[-1] - ref))
        errs[method] = np.array(e)
        slope = np.polyfit(np.log(hs), np.log(errs[method]), 1)[0]
        pair = np.diff(np.log(errs[method])) / np.diff(np.log(hs))
        for i, h in enumerate(hs):
            rows.append(
                {
                    "method": method,
                    "h_s": h,
                    "error_ppm": errs[method][i],
                    "pairwise_order": np.nan if i == 0 else pair[i - 1],
                    "fitted_order": slope,
                }
            )
    df = pd.DataFrame(rows)
    _save_csv(df, "convergence.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    mk = {"euler": ("o-", "#D55E00", "Explicit Euler"), "rk4": ("s-", "#0072B2", "RK4")}
    for m, (fmt, c, lab) in mk.items():
        order = df.loc[df.method == m, "fitted_order"].iloc[0]
        ax.loglog(hs, errs[m], fmt, color=c, lw=2, ms=7, label=f"{lab} (observed order {order:.2f})")
    for order, m, ls in ((1, "euler", "--"), (4, "rk4", ":")):
        anchor = errs[m][2]  # pass through the h = 150 s point
        ax.loglog(hs, anchor * (hs / hs[2]) ** order, ls, color="gray", lw=1.5, label=f"slope {order} reference")
    ax.set_xlabel("Step size $h$ [s]")
    ax.set_ylabel("Global error at $t$ = 1 h [ppm]")
    ax.set_title("Convergence vs exact solution\n($N$=10, $V$=150 m$^3$, $Q$=0.125 m$^3$/s)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "fig_convergence.png")
    return df


# --------------------------------------------------------------------------
# 2. Euler stability
# --------------------------------------------------------------------------
def euler_stability() -> None:
    """Illustrate the stability limit h < 2 tau of explicit Euler.

    Pure decay (N = 0) from 1000 ppm to C_out = 420 ppm, solved with step
    sizes h = 0.5 tau, 1.5 tau and 2.2 tau and compared with the exact
    solution. The amplification factor is (1 - h/tau): monotone decay for
    h < tau, decaying oscillation for tau < h < 2 tau and growing
    oscillation for h > 2 tau. Saved to ``fig_euler_stability.png``.
    """
    p = RoomParams(V=150.0, Q=0.125)
    tau = p.tau
    T = 14 * tau
    tf = np.linspace(0, T, 600)
    exact = simulate(tf, np.zeros_like(tf), p, C0=1000.0, method="exact")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, mult in zip(axes, (0.5, 1.5, 2.2)):
        h = mult * tau
        n = int(np.ceil(T / h))
        t = np.arange(n + 1) * h
        C = simulate(t, np.zeros(n + 1), p, C0=1000.0, method="euler")
        ax.plot(tf / 3600, exact, "-", color="gray", lw=2, label="Exact")
        ax.plot(t / 3600, C, "o-", color="#D55E00", lw=1.5, ms=5, label="Explicit Euler")
        ax.axhline(p.C_out, color="k", lw=0.8, ls=":")
        regime = "monotone decay" if mult < 1 else ("damped oscillation" if mult < 2 else "unstable: diverges")
        ax.set_title(f"$h$ = {mult:g} $\\tau$ = {h:.0f} s\n({regime}, |1-h/$\\tau$| = {abs(1 - mult):.1f})")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("CO$_2$ concentration [ppm]")
        ax.legend(loc="best")
    fig.suptitle(f"Explicit Euler stability, $N$=0, $\\tau$ = V/Q = {tau:.0f} s, stable only for $h$ < 2$\\tau$ = {2 * tau:.0f} s", y=1.03)
    fig.tight_layout()
    _save(fig, "fig_euler_stability.png")


# --------------------------------------------------------------------------
# 3. Sensitivity sweep
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _room(seed: int, sample_mean_s: float, noise_ppm: float):
    """Cached synthetic room (env, occ, true params), 28 days."""
    return make_room(days=DAYS, seed=seed, sample_mean_s=sample_mean_s, noise_ppm=noise_ppm)


def run_case(
    env: pd.DataFrame,
    occ: pd.Series,
    truth: RoomParams,
    freq: str,
    scheme: str,
    occ_shift: pd.Timedelta | None = None,
    center: bool = True,
) -> dict[str, float]:
    """Calibrate on the first 70 % and estimate occupancy on the last 30 %.

    Args:
        env: Irregular environment readings.
        occ: 1-minute true occupancy.
        truth: True room parameters.
        freq: Grid spacing, pandas offset string.
        scheme: Derivative scheme.
        occ_shift: If given, the true occupancy index is shifted by this
            amount before the interval mean is taken (alignment test).
        center: Passed to ``prepare_room``; the alignment study sets it
            False and applies its own explicit shifts.

    Returns:
        Dict with signed relative errors ``V_relerr``, ``Q_relerr``
        (calibrated/true - 1), ``Cout_abs_err`` [ppm], ``Cout_hat``, and
        occupancy ``mae``/``rmse``/``bias`` [persons] on the test block.
        All NaN when the calibration is non-physical.
    """
    dt = pd.Timedelta(freq).total_seconds()
    occ_in = occ if occ_shift is None else occ.shift(freq=occ_shift)
    tab = prepare_room(env, occ_in, freq=freq, center=center)
    C = tab["co2_ppm"].to_numpy(float)
    N = tab["occupancy"].to_numpy(float)
    cut = int(TRAIN_FRAC * len(tab))
    nan = dict(V_relerr=np.nan, Q_relerr=np.nan, Cout_abs_err=np.nan, Cout_hat=np.nan, mae=np.nan, rmse=np.nan, bias=np.nan)
    try:
        est = calibrate(C[:cut], N[:cut], dt, g=truth.g, scheme=scheme, window=SAVGOL_WINDOW)
    except ValueError:
        return nan
    Nhat = estimate_occupancy(C[cut:], dt, est, scheme=scheme, window=SAVGOL_WINDOW)
    m = metrics(N[cut:], Nhat)
    return dict(
        V_relerr=est.V / truth.V - 1,
        Q_relerr=est.Q / truth.Q - 1,
        Cout_abs_err=abs(est.C_out - truth.C_out),
        Cout_hat=est.C_out,
        mae=m["mae"],
        rmse=m["rmse"],
        bias=m["bias"],
    )


def sensitivity() -> pd.DataFrame:
    """Sweep grid spacing x derivative scheme x sample rate, and sensor noise.

    Sweep ``dt``: freq in 5/10/15/30 min x scheme x seed, for both
    ``sample_mean_s`` = 120 s (primary) and 600 s (default), noise 15 ppm.
    Sweep ``noise``: noise in 0..50 ppm at 15 min with Savitzky-Golay,
    ``sample_mean_s`` = 600 s.

    Returns:
        Long-format results table (``sensitivity_grid.csv``). Figures
        ``fig_sensitivity_dt.png`` and ``fig_sensitivity_noise.png`` are
        produced from it.
    """
    rows = []
    for seed in SEEDS:
        for sm in (FINE_SAMPLE_S, DEFAULT_SAMPLE_S):
            env, occ, truth = _room(seed, sm, 15.0)
            for freq in FREQS:
                for scheme in SCHEMES:
                    r = run_case(env, occ, truth, freq, scheme)
                    rows.append(dict(sweep="dt", seed=seed, sample_mean_s=sm, noise_ppm=15.0, freq=freq,
                                     dt_min=pd.Timedelta(freq).total_seconds() / 60, scheme=scheme, **r))
        for noise in NOISES:
            env, occ, truth = _room(seed, DEFAULT_SAMPLE_S, noise)
            r = run_case(env, occ, truth, "15min", "savgol")
            rows.append(dict(sweep="noise", seed=seed, sample_mean_s=DEFAULT_SAMPLE_S, noise_ppm=noise, freq="15min",
                             dt_min=15.0, scheme="savgol", **r))
    df = pd.DataFrame(rows)
    _save_csv(df, "sensitivity_grid.csv")
    plot_dt(df)
    plot_noise(df)
    return df


def _band(ax: plt.Axes, x, g: pd.core.groupby.DataFrameGroupBy, col: str, **kw) -> None:
    """Plot the across-seed mean of ``col`` with +/- 1 sd error bars."""
    mean = g[col].mean()
    sd = g[col].std()
    ax.errorbar(x, mean, yerr=sd, capsize=3, lw=2, marker="o", ms=6, **kw)


def plot_dt(df: pd.DataFrame) -> None:
    """Plot occupancy MAE and V/Q relative error vs grid spacing.

    Primary panel row: ``sample_mean_s`` = 120 s. One line per scheme.
    """
    d = df[(df.sweep == "dt") & (df.sample_mean_s == FINE_SAMPLE_S)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for scheme in SCHEMES:
        s = d[d.scheme == scheme]
        g = s.groupby("dt_min")
        x = sorted(s.dt_min.unique())
        for ax, col in zip(axes, ("mae", "V_relerr", "Q_relerr")):
            _band(ax, x, g, col, color=COLORS[scheme], label=scheme)
    axes[0].set_ylabel("Occupancy MAE, test block [persons]")
    axes[1].set_ylabel("Relative error of $V$ [%]")
    axes[2].set_ylabel("Relative error of $Q$ [%]")
    for ax in axes[1:]:
        ax.axhline(0, color="k", lw=0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    for ax in axes:
        ax.set_xlabel("Grid spacing $\\Delta t$ [min]")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 15, 30])
        ax.set_xticklabels(["5", "10", "15", "30"])
        ax.minorticks_off()
    axes[0].legend(title="Derivative scheme")
    fig.suptitle(
        f"Sensitivity to grid spacing (28-day room, noise 15 ppm, env. report interval ~{FINE_SAMPLE_S:.0f} s; "
        f"mean $\\pm$ 1 sd over {len(SEEDS)} seeds)",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig_sensitivity_dt.png")


def plot_noise(df: pd.DataFrame) -> None:
    """Plot occupancy MAE and V/Q relative error vs CO2 sensor noise."""
    d = df[df.sweep == "noise"]
    g = d.groupby("noise_ppm")
    x = sorted(d.noise_ppm.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    _band(axes[0], x, g, "mae", color=COLORS["savgol"], label="Savitzky-Golay, 15 min")
    _band(axes[1], x, g, "V_relerr", color="#0072B2", label="$V$")
    _band(axes[1], x, g, "Q_relerr", color="#D55E00", label="$Q$")
    axes[0].set_ylabel("Occupancy MAE, test block [persons]")
    axes[1].set_ylabel("Relative error of calibrated parameter [%]")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}"))
    for ax in axes:
        ax.set_xlabel("CO$_2$ sensor noise, std. dev. [ppm]")
        ax.legend()
    fig.suptitle(f"Sensitivity to sensor noise (15-min grid, report interval ~{DEFAULT_SAMPLE_S:.0f} s; mean $\\pm$ 1 sd over {len(SEEDS)} seeds)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig_sensitivity_noise.png")


# --------------------------------------------------------------------------
# 4. Alignment check
# --------------------------------------------------------------------------
def alignment() -> pd.DataFrame:
    """Test whether re-centring the occupancy target changes V/Q bias.

    With ``center=False``, ``prepare_room`` labels the occupancy mean of ``[t, t+dt)`` at ``t``
    (centre of mass ``t + dt/2``) while CO2 is the instantaneous value at
    ``t``. With the 1-minute truth ``occ`` we build:

    * ``interval_mean`` - the pipeline default, mean over ``[t, t+dt)``;
    * ``centred`` - mean over ``[t-dt/2, t+dt/2)``, obtained by shifting
      the occupancy index *forward* by ``dt/2`` (``occ.shift(freq=+dt/2)``)
      before the resample (a value at time s moves to s+dt/2, so cell
      ``[t, t+dt)`` collects original times ``[t-dt/2, t+dt/2)``);
    * ``anti_centred`` - the wrong direction (shift by ``-dt/2``,
      mean over ``[t+dt/2, t+3dt/2)``) as a negative control: if the
      centring logic is right this should be worse than the default.

    Savitzky-Golay is used for all rows. Runs over all grid spacings, two
    report intervals and all seeds; written to ``alignment.csv``.

    Returns:
        Long-format table of results.
    """
    rows = []
    for seed in SEEDS:
        for sm in (DEFAULT_SAMPLE_S, FINE_SAMPLE_S):
            env, occ, truth = _room(seed, sm, 15.0)
            for freq in FREQS:
                half = pd.Timedelta(freq) / 2
                for variant, shift in (("interval_mean", None), ("centred", half), ("anti_centred", -half)):
                    r = run_case(env, occ, truth, freq, "savgol", occ_shift=shift, center=False)
                    rows.append(dict(seed=seed, sample_mean_s=sm, freq=freq, variant=variant, **r))
    df = pd.DataFrame(rows)
    _save_csv(df, "alignment.csv")

    summ = (
        df.groupby(["sample_mean_s", "freq", "variant"])[["V_relerr", "Q_relerr", "Cout_abs_err", "mae", "rmse"]]
        .mean()
        .reset_index()
    )
    summ["freq"] = pd.Categorical(summ["freq"], FREQS, ordered=True)
    summ = summ.sort_values(["sample_mean_s", "freq", "variant"])
    pd.set_option("display.width", 200)
    print("\nAlignment check (mean over seeds, savgol):")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    d15 = df[(df.freq == "15min") & (df.sample_mean_s == DEFAULT_SAMPLE_S)]
    print("\n15min, default room, seed 0:")
    print(d15[d15.seed == 0][["variant", "V_relerr", "Q_relerr", "Cout_abs_err", "mae", "rmse"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return df


# --------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> None:
    """Print compact mean-over-seeds summaries of the sensitivity sweep."""
    pd.set_option("display.width", 200)
    cols = ["V_relerr", "Q_relerr", "Cout_abs_err", "mae", "rmse"]
    fmt = lambda v: f"{v:.4f}"  # noqa: E731
    for sm in (FINE_SAMPLE_S, DEFAULT_SAMPLE_S):
        d = df[(df.sweep == "dt") & (df.sample_mean_s == sm)]
        print(f"\nGrid sweep, report interval ~{sm:.0f} s (mean over seeds):")
        print(d.groupby(["dt_min", "scheme"])[cols].mean().to_string(float_format=fmt))
    print("\nSeed 0, report interval 120 s:")
    d = df[(df.sweep == "dt") & (df.sample_mean_s == FINE_SAMPLE_S) & (df.seed == 0)]
    print(d[["freq", "scheme"] + cols].to_string(index=False, float_format=fmt))
    print("\nNoise sweep (mean over seeds):")
    print(df[df.sweep == "noise"].groupby("noise_ppm")[cols].mean().to_string(float_format=fmt))
    print("\nNoise sweep seed 0:")
    print(df[(df.sweep == "noise") & (df.seed == 0)][["noise_ppm"] + cols].to_string(index=False, float_format=fmt))
    print(f"\nNaN (failed calibrations): {int(df['mae'].isna().sum())} of {len(df)} runs")


def main() -> None:
    """Run all four sections and report timing."""
    t0 = time.perf_counter()
    print("1. convergence")
    conv = convergence()
    print(conv.groupby("method")["fitted_order"].first().to_string())
    print(conv.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("2. Euler stability")
    euler_stability()
    print("3. sensitivity sweep")
    df = sensitivity()
    summarise(df)
    print("4. alignment")
    alignment()
    print(f"\ntotal runtime {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
```

## FILE: tests/test_evaluate_pipeline.py

```python
"""Tests for metrics, time-series splits and room-table preparation."""
import numpy as np
import pandas as pd
import pytest

from occupancy.evaluate import blocked_folds, chrono_split, metrics
from occupancy.pipeline import prepare_room
from occupancy.resample import to_uniform


def test_balanced_accuracy_constant_predictor_has_no_skill():
    y = np.array([0, 0, 3, 4, 5, 6, 2, 1, 0, 7], float)
    m = metrics(y, np.full_like(y, 3.0))  # always "occupied"
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics(y, y)["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics(y, y)["r2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics([np.nan], [1.0])


def test_splits_are_disjoint_and_ordered():
    idx = pd.date_range("2024-05-01", periods=96 * 40, freq="15min", tz="Australia/Melbourne")
    tr, va, te = chrono_split(idx)
    assert not (tr & va).any() and not (va & te).any()
    assert idx[tr].max() < idx[va].min() < idx[va].max() < idx[te].min()
    for fit, val in blocked_folds(idx):
        assert idx[fit].max() + pd.Timedelta("1D") <= idx[val].min()


def test_resample_handles_dst_boundaries():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(
        ["2024-04-06 16:07Z", "2024-04-06 16:40Z", "2024-04-06 17:20Z"]).tz_convert("Australia/Melbourne"))
    u = to_uniform(s, "15min", max_gap="2h")
    assert u.index.is_monotonic_increasing and u.notna().all()


def test_prepare_room_coverage_floor_and_centring():
    tz = "Australia/Melbourne"
    t1 = pd.date_range("2024-05-06 09:00", "2024-05-06 12:00", freq="1min", tz=tz)
    occ = pd.Series(4.0, index=t1)
    occ[(occ.index >= "2024-05-06 10:00") & (occ.index < "2024-05-06 10:14")] = np.nan  # 1 valid min left
    env = pd.DataFrame({"co2_ppm": np.linspace(500, 600, 19)},
                       index=pd.date_range("2024-05-06 09:00", periods=19, freq="10min", tz=tz))
    tab = prepare_room(env, occ, "15min", center=False)
    assert np.isnan(tab.loc["2024-05-06 10:00", "occupancy"])
    assert tab.loc["2024-05-06 09:15", "occupancy"] == pytest.approx(4.0)
```

## FILE: tests/test_io.py

```python
"""Tests for raw-data parsing and event aggregation."""
import json

import numpy as np
import pandas as pd
import pytest

from occupancy.io import flag_invalid, occupancy_on_grid, occupancy_on_grid_loop, parse_env_json


def test_parse_env_json():
    js = json.dumps([
        {"values": [{"time": 1700000000, "value": 612}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
        {"values": [{"time": 1700000000, "value": 22.5}], "variable": {"name": "Temperature", "unit": "°C"}},
        {"values": [{"time": 1700000000, "value": 1}], "variable": {"name": "Unknown", "unit": "x"}},
        {"values": [], "variable": {"name": "Humidity", "unit": "%"}},
    ])
    d = parse_env_json(js)
    assert d == {"co2_ppm": (1700000000.0, 612.0), "temp_c": (1700000000.0, 22.5)}
    assert parse_env_json("not json") == {}
    assert parse_env_json(None) == {}


def test_flag_invalid():
    env = pd.DataFrame({"co2_ppm": [400, 65535, 100, np.nan], "temp_c": [20, 21, -40, 22]})
    out = flag_invalid(env)
    assert out["co2_ppm"].isna().sum() == 3
    assert out.attrs["n_flagged"] == {"co2_ppm": 2, "temp_c": 1}


def events(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(30, n)
    gaps[rng.random(n) < 0.002] = 7200  # occasional long silences
    t = pd.Timestamp("2024-05-06 08:00", tz="Australia/Melbourne") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    return pd.DataFrame({"time": t, "headcount": rng.integers(0, 10, n).astype(float)})


def test_vectorised_matches_loop():
    ev = events()
    a = occupancy_on_grid(ev)
    b = occupancy_on_grid_loop(ev)
    both = a.index.intersection(b.index)
    assert len(both) > 50
    np.testing.assert_allclose(a[both].to_numpy(), b[both].to_numpy(), rtol=1e-9, equal_nan=True)


def test_constant_headcount_and_gap():
    t = pd.date_range("2024-05-06 09:00", periods=121, freq="30s", tz="Australia/Melbourne")
    ev = pd.DataFrame({"time": t, "headcount": 4.0})
    # 2 h silence then resume: the gap must be NaN, not 4
    t2 = t[-1] + pd.Timedelta("2h") + pd.to_timedelta(np.arange(61) * 30, unit="s")
    ev = pd.concat([ev, pd.DataFrame({"time": t2, "headcount": 2.0})])
    g = occupancy_on_grid(ev, "15min")
    assert g["2024-05-06 09:00":"2024-05-06 09:45"].eq(4.0).all()
    assert g["2024-05-06 10:15":"2024-05-06 11:45"].isna().all()
    with pytest.raises(ValueError):
        occupancy_on_grid(ev.iloc[:1])


@pytest.mark.parametrize("fn", [occupancy_on_grid, occupancy_on_grid_loop])
def test_hand_calculated_interval_means(fn):
    """0 people 09:00-09:05, 6 people 09:05-09:20, then 0.

    09:00 bin: (0*5 + 6*10)/15 = 4;  09:15 bin: (6*5 + 0*10)/15 = 2.
    """
    tz = "Australia/Melbourne"
    t = [pd.Timestamp("2024-05-06 09:00", tz=tz), pd.Timestamp("2024-05-06 09:05", tz=tz)]
    t += list(pd.date_range("2024-05-06 09:20", periods=21, freq="30s", tz=tz))
    ev = pd.DataFrame({"time": t, "headcount": np.array([0, 6] + [0] * 21, float)})
    g = fn(ev, "15min")
    assert g[pd.Timestamp("2024-05-06 09:00", tz=tz)] == pytest.approx(4.0)
    assert g[pd.Timestamp("2024-05-06 09:15", tz=tz)] == pytest.approx(2.0)
```

## FILE: tests/test_physics.py

```python
"""Verification tests for the CO2 mass-balance model.

These use manufactured data where the true answer is known exactly, so
they check the *implementation* (verification), not whether the model
describes real rooms (validation, done in the analysis scripts).
"""
import numpy as np
import pytest

from occupancy.physics import (
    RoomParams,
    calibrate,
    derivative,
    estimate_occupancy,
    euler_stable_step,
    simulate,
)

P = RoomParams(V=150.0, Q=0.125, C_out=420.0)  # tau = 1200 s, 3 ACH


def step_input(n=200, h=60.0, level=10):
    t = np.arange(n) * h
    N = np.where(t >= 0, level, 0).astype(float)
    return t, N


def test_steady_state_and_ach():
    assert P.ach == pytest.approx(3.0)
    # 10 people, Q=0.125 m3/s -> 1e6*5.2e-6*10/0.125 = 416 ppm above outdoor
    assert P.steady_state(10) == pytest.approx(420 + 416)


def test_exact_matches_analytical():
    t, N = step_input()
    C = simulate(t, N, P, method="exact")
    css = P.steady_state(10)
    expected = css + (P.C_out - css) * np.exp(-t / P.tau)
    np.testing.assert_allclose(C, expected, rtol=1e-12)


@pytest.mark.parametrize("method,order", [("euler", 1), ("rk4", 4)])
def test_convergence_order(method, order):
    """Global error should fall as h^order (observed order within 0.3)."""
    T = 3600.0
    errs = []
    hs = [600.0, 300.0, 150.0]
    for h in hs:
        t = np.arange(0, T + h / 2, h)
        N = np.full_like(t, 10.0)
        C = simulate(t, N, P, method=method)
        exact = simulate(t, N, P, method="exact")
        errs.append(abs(C[-1] - exact[-1]))
    observed = np.log(errs[0] / errs[-1]) / np.log(hs[0] / hs[-1])
    assert observed == pytest.approx(order, abs=0.3)


def test_euler_instability_beyond_limit():
    h = 1.2 * euler_stable_step(P)  # beyond 2*tau -> diverges
    t = np.arange(30) * h
    C = simulate(t, np.zeros_like(t), P, C0=1000.0, method="euler")
    assert abs(C[-1] - P.C_out) > abs(1000.0 - P.C_out)
    # RK4 at a stable step decays as it should
    t2 = np.arange(30) * 300.0
    C2 = simulate(t2, np.zeros_like(t2), P, C0=1000.0, method="rk4")
    assert abs(C2[-1] - P.C_out) < 1.0


def test_inverse_recovers_constant_occupancy():
    t = np.arange(0, 6 * 3600, 60.0)
    N = np.full_like(t, 7.0)
    C = simulate(t, N, P, method="exact")
    Nhat = estimate_occupancy(C, 60.0, P, scheme="central")
    np.testing.assert_allclose(Nhat[2:-2], 7.0, atol=0.05)


def test_derivative_schemes_on_polynomial():
    dt = 10.0
    x = np.arange(50) * dt
    C = 3e-4 * x**2 + 2 * x + 400
    exact = 6e-4 * x + 2
    np.testing.assert_allclose(derivative(C, dt, "central")[1:-1], exact[1:-1], rtol=1e-10)
    np.testing.assert_allclose(derivative(C, dt, "savgol", 5), exact, rtol=1e-8)
    with pytest.raises(ValueError):
        derivative(C, dt, "savgol", 4)


def test_calibration_recovers_parameters():
    rng = np.random.default_rng(1)
    t = np.arange(0, 3 * 86400, 60.0)
    N = np.repeat(rng.integers(0, 12, len(t) // 60 + 1), 60)[: len(t)].astype(float)
    C = simulate(t, N, P, method="exact")
    est = calibrate(C, N, 60.0, scheme="central")
    assert est.V == pytest.approx(P.V, rel=0.05)
    assert est.Q == pytest.approx(P.Q, rel=0.05)
    assert est.C_out == pytest.approx(P.C_out, abs=5)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        RoomParams(V=-1, Q=0.1)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 0.0]), np.array([1.0, 1.0]), P)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 1.0]), np.array([1.0]), P)
    with pytest.raises(ValueError):
        simulate(np.array([0.0, 1.0]), np.array([1.0, 1.0]), P, method="bogus")
    with pytest.raises(ValueError):
        calibrate(np.full(100, 420.0), np.zeros(100), 60.0)  # unexcited room


def test_fast_simulation_matches_loop():
    from occupancy.physics import simulate_uniform_fast

    rng = np.random.default_rng(3)
    t = np.arange(0, 86400, 60.0)
    N = rng.integers(0, 15, len(t)).astype(float)
    ref = simulate(t, N, P, C0=800.0, method="exact")
    fast = simulate_uniform_fast(N, 60.0, P, C0=800.0)
    np.testing.assert_allclose(fast, ref, rtol=1e-10)
```

## FILE: tests/test_resample.py

```python
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
    assert (np.diff(u.index) == pd.Timedelta("15min")).all()
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
```

## FILE: Makefile

```make
# Virtual occupancy sensing (MMA3001) -- reproducible workflow.
#
# The data-dependent targets (ingest, experiment, sensitivity, figures, all)
# need the raw MSI exports unzipped under data/raw/ -- see data/README.md.
# `make test`, `make test-report`, `make docs` and `make profile` need no data.

PY      ?= .venv/bin/python
VENV    ?= .venv
PYTHON3 ?= python3

.PHONY: all install test test-report ingest experiment experiment-synthetic \
        sensitivity profile figures docs clean-docs help

help:
	@echo "Targets: install test test-report ingest experiment sensitivity profile figures docs all"

# Create the virtual environment (if absent) and install the package + dev tools.
install:
	test -x $(PY) || $(PYTHON3) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

# HTML test report (self-contained) and JUnit XML for the submission.
test-report:
	$(PY) -m pytest -q --html=reports/test_report.html --self-contained-html \
	    --junitxml=reports/test_results.xml

# Raw CSV/XLSX -> data/processed/*.parquet and reports/results/audit/*.csv
ingest:
	$(PY) scripts/ingest.py

# Model comparison on manufactured rooms (no data needed), then on real rooms.
experiment-synthetic:
	$(PY) scripts/run_experiment.py --synthetic

experiment: experiment-synthetic
	$(PY) scripts/run_experiment.py --data data/processed/rooms.parquet

sensitivity:
	$(PY) scripts/sensitivity_synthetic.py
	$(PY) scripts/sensitivity_real.py

profile:
	$(PY) scripts/profile_performance.py

figures:
	$(PY) scripts/make_figures.py

# API documentation (Google-style docstrings, LaTeX maths) -> docs/index.html
docs:
	$(PY) -m pdoc -d google --math -o docs occupancy

clean-docs:
	rm -rf docs/*

# Full pipeline from raw data to documentation.
all: ingest experiment sensitivity profile figures test-report docs
```
