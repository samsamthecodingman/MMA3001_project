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
