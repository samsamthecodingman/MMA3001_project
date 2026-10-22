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
