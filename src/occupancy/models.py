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
