"""Tests for metrics, time-series splits and room-table preparation."""
import numpy as np
import pandas as pd
import pytest

from occupancy.evaluate import blocked_folds, chrono_split, metrics
from occupancy.pipeline import prepare_room
from occupancy.resample import to_uniform


def test_balanced_accuracy_constant_predictor_has_no_skill():
    y = np.array([0, 0, 3, 4, 5, 6, 2, 1, 0, 7], float)
    m = metrics(y, np.full_like(y, 3.0))  # always "occupied"
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics(y, y)["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics(y, y)["r2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics([np.nan], [1.0])


def test_splits_are_disjoint_and_ordered():
    idx = pd.date_range("2024-05-01", periods=96 * 40, freq="15min", tz="Australia/Melbourne")
    tr, va, te = chrono_split(idx)
    assert not (tr & va).any() and not (va & te).any()
    assert idx[tr].max() < idx[va].min() < idx[va].max() < idx[te].min()
    for fit, val in blocked_folds(idx):
        assert idx[fit].max() + pd.Timedelta("1D") <= idx[val].min()


def test_resample_handles_dst_boundaries():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(
        ["2024-04-06 16:07Z", "2024-04-06 16:40Z", "2024-04-06 17:20Z"]).tz_convert("Australia/Melbourne"))
    u = to_uniform(s, "15min", max_gap="2h")
    assert u.index.is_monotonic_increasing and u.notna().all()


def test_prepare_room_coverage_floor_and_centring():
    tz = "Australia/Melbourne"
    t1 = pd.date_range("2024-05-06 09:00", "2024-05-06 12:00", freq="1min", tz=tz)
    occ = pd.Series(4.0, index=t1)
    occ[(occ.index >= "2024-05-06 10:00") & (occ.index < "2024-05-06 10:14")] = np.nan  # 1 valid min left
    env = pd.DataFrame({"co2_ppm": np.linspace(500, 600, 19)},
                       index=pd.date_range("2024-05-06 09:00", periods=19, freq="10min", tz=tz))
    tab = prepare_room(env, occ, "15min", center=False)
    assert np.isnan(tab.loc["2024-05-06 10:00", "occupancy"])
    assert tab.loc["2024-05-06 09:15", "occupancy"] == pytest.approx(4.0)
