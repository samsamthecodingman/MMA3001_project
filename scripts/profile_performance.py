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
