"""Run the full model comparison on synthetic or real rooms.

Usage:
    python scripts/run_experiment.py --synthetic          # manufactured rooms
    python scripts/run_experiment.py --data data/processed/rooms.parquet

Outputs go to ``reports/results/<tag>/``: per-room scores, pooled summary
and test-period predictions.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

from occupancy.physics import RoomParams
from occupancy.pipeline import prepare_room, run_room, run_room_cv
from occupancy.synthetic import make_room

ROOT = pathlib.Path(__file__).resolve().parents[1]


def synthetic_rooms(n: int = 4, days: int = 42) -> dict[str, pd.DataFrame]:
    """A few manufactured rooms with different sizes and ventilation."""
    specs = [(150, 3.0, 12), (60, 2.0, 6), (400, 4.0, 30), (90, 1.5, 8)][:n]
    rooms = {}
    for i, (V, ach, cap) in enumerate(specs):
        env, occ, _ = make_room(RoomParams(V=V, Q=V * ach / 3600), days=days, seed=i, capacity=cap)
        rooms[f"synthetic_{i}"] = prepare_room(env, occ)
    return rooms


def main() -> None:
    """Parse command-line arguments, evaluate all models per room and write results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data", type=pathlib.Path)
    ap.add_argument("--freq", default="15min")
    args = ap.parse_args()
    if args.synthetic:
        rooms, tag = synthetic_rooms(), "synthetic"
    elif args.data:
        df = pd.read_parquet(args.data)
        rooms = {r: g.drop(columns="room").asfreq(args.freq) for r, g in df.groupby("room")}
        tag = "real"
    else:
        ap.error("give --synthetic or --data")
    out = ROOT / "reports" / "results" / tag
    out.mkdir(parents=True, exist_ok=True)
    dt = pd.Timedelta(args.freq).total_seconds()
    all_scores, all_preds, params, cv_scores, cv_oof = [], [], [], [], []
    for name, table in rooms.items():
        print(f"room {name}: {len(table)} rows, occupancy coverage "
              f"{table['occupancy'].notna().mean():.0%}")
        scores, extra = run_room(table, dt)
        cvs, oof = run_room_cv(table, dt)
        cv_scores.append(cvs.assign(room=name))
        cv_oof.append(oof.assign(room=name))
        all_scores.append(scores.assign(room=name))
        p = extra["preds"]
        all_preds.append(pd.DataFrame(p).assign(y_true=extra["y_test"], room=name))
        if extra["params"] is not None:
            q = extra["params"]
            params.append({"room": name, "V_m3": q.V, "Q_m3s": q.Q, "ACH": q.ach, "C_out_ppm": q.C_out})
    scores = pd.concat(all_scores)
    scores.to_csv(out / "scores_by_room.csv")
    pd.concat(all_preds).to_parquet(out / "test_predictions.parquet")
    pd.DataFrame(params).to_csv(out / "physics_params.csv", index=False)
    summary = scores.groupby(level=0)[["mae", "rmse", "r2", "f1", "fit_s", "predict_s"]].median()
    summary = summary.sort_values("mae")
    summary.to_csv(out / "summary_median.csv")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary)
    cv = pd.concat(cv_scores)
    cv.to_csv(out / "cv_scores_by_room.csv")
    pd.concat(cv_oof).to_parquet(out / "cv_oof_predictions.parquet")
    print("\nBlocked leave-one-week-out CV (pooled out-of-fold):")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(cv.reset_index().pivot(index="model", columns="room", values="mae"))
        print(cv.reset_index().pivot(index="model", columns="room", values="r2"))
    if params:
        print(pd.DataFrame(params).round(3).to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
