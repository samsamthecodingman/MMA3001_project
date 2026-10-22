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

    Ranges are in :data:`VALID_RANGES`. Returns a copy; the number of
    values removed per column is stored in ``df.attrs["n_flagged"]``.
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
        Series of mean headcount per interval, labelled by interval start.
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
