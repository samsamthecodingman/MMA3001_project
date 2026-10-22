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
