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
