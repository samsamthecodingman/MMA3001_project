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
) -> pd.DataFrame:
    """Align irregular env readings and occupancy counts on one grid.

    Env channels are interpolated (they are continuous physical fields);
    occupancy is *averaged* over each interval (it is a count, so
    interpolating it would be meaningless).

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
    o = occ.resample(freq, label="left", closed="left").mean()
    table["occupancy"] = o.reindex(table.index)
    return table


def run_room(table: pd.DataFrame, dt: float, models=ML_MODELS, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Fit every estimator on one room and score it on the held-out test period.

    Split is chronological 60/20/20 with 1-day gaps. The validation block
    is reserved for model/hyper-parameter selection (see tuning script);
    here models are trained on train+val and scored once on test.

    Returns:
        ``(scores, preds)``: scores DataFrame (one row per model) and a
        dict of test-period prediction Series for plotting.
    """
    tr, va, te = chrono_split(table.index)
    fit_mask = tr | va
    y = table["occupancy"].to_numpy(float)
    co2 = table["co2_ppm"].to_numpy(float)
    rows, preds = [], {}
    idx_te = table.index[te]

    def record(name, yhat_te, fit_s, pred_s):
        m = metrics(y[te], yhat_te)
        m.update(model=name, fit_s=fit_s, predict_s=pred_s)
        rows.append(m)
        preds[name] = pd.Series(yhat_te, index=idx_te)

    # 1) naive baseline
    t0 = time.perf_counter()
    ok = fit_mask & np.isfinite(y)
    base = TimeOfWeekBaseline().fit(table.index[ok], y[ok])
    t1 = time.perf_counter()
    record("time_of_week", base.predict(idx_te), t1 - t0, time.perf_counter() - t1)

    # 2) physics: calibrated on the fit period, applied to the full series
    #    (derivative needs neighbours) then evaluated on the test period
    phys_full = np.full(len(table), np.nan)
    try:
        t0 = time.perf_counter()
        ph = PhysicsEstimator(dt).fit(np.where(fit_mask, co2, np.nan), np.where(fit_mask, y, np.nan))
        t1 = time.perf_counter()
        phys_full = ph.predict(co2)
        record("physics", phys_full[te], t1 - t0, time.perf_counter() - t1)
        phys_params = ph.params_
    except ValueError as err:  # non-physical calibration -> report, don't crash
        phys_params = None
        print(f"  physics calibration failed: {err}")

    # 3) data-driven models on causal features
    X = build_features(table)
    X_h = X.assign(phys=phys_full)
    valid = X.notna().all(axis=1).to_numpy() & np.isfinite(y)
    for name in models:
        for feats, label in ((X, name), (X_h, f"{name}+phys")):
            if label.endswith("+phys") and (name not in ("forest", "mlp") or phys_params is None):
                continue  # hybrid only for the two strongest learners
            fm = fit_mask & valid & np.isfinite(feats.to_numpy()).all(axis=1)
            tm = te & np.isfinite(feats.to_numpy()).all(axis=1)
            mdl = make_ml_model(name, seed)
            t0 = time.perf_counter()
            mdl.fit(feats[fm].to_numpy(), y[fm])
            t1 = time.perf_counter()
            yhat = np.full(te.sum(), np.nan)
            yhat[tm[te]] = np.clip(mdl.predict(feats[tm].to_numpy()), 0, None)
            record(label, yhat, t1 - t0, time.perf_counter() - t1)
    scores = pd.DataFrame(rows).set_index("model")
    return scores, {"preds": preds, "y_test": pd.Series(y[te], index=idx_te), "params": phys_params}
