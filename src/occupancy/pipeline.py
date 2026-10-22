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
    o = occ.resample(freq, label="left", closed="left").mean()
    table["occupancy"] = o.reindex(table.index)
    return table


def trim_to_paired(table: pd.DataFrame) -> pd.DataFrame:
    """Restrict a room table to the span where CO2 and occupancy overlap.

    The env and occupancy records only partly overlap; splitting over the
    full CO2 record would put training periods where no labels exist.

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
    rows = []
    for name, yhat in preds.items():
        m = metrics(y_te, yhat)
        m.update(model=name, fit_s=timings[name][0], predict_s=timings[name][1])
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def run_room(table: pd.DataFrame, dt: float, models=ML_MODELS, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Chronological evaluation ("deploy on the future").

    Split is chronological 60/20/20 with 1-day gaps over the paired span.
    Models are trained on train+val and scored once on the final test
    period. This tests extrapolation forward in time, including any
    change of operating regime (e.g. teaching -> exam period).

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
