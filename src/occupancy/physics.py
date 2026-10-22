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
