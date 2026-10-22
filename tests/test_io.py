"""Tests for raw-data parsing and event aggregation."""
import json

import numpy as np
import pandas as pd
import pytest

from occupancy.io import flag_invalid, occupancy_on_grid, occupancy_on_grid_loop, parse_env_json


def test_parse_env_json():
    js = json.dumps([
        {"values": [{"time": 1700000000, "value": 612}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
        {"values": [{"time": 1700000000, "value": 22.5}], "variable": {"name": "Temperature", "unit": "°C"}},
        {"values": [{"time": 1700000000, "value": 1}], "variable": {"name": "Unknown", "unit": "x"}},
        {"values": [], "variable": {"name": "Humidity", "unit": "%"}},
    ])
    d = parse_env_json(js)
    assert d == {"co2_ppm": (1700000000.0, 612.0), "temp_c": (1700000000.0, 22.5)}
    assert parse_env_json("not json") == {}
    assert parse_env_json(None) == {}


def test_flag_invalid():
    env = pd.DataFrame({"co2_ppm": [400, 65535, 100, np.nan], "temp_c": [20, 21, -40, 22]})
    out = flag_invalid(env)
    assert out["co2_ppm"].isna().sum() == 3
    assert out.attrs["n_flagged"] == {"co2_ppm": 2, "temp_c": 1}


def events(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(30, n)
    gaps[rng.random(n) < 0.002] = 7200  # occasional long silences
    t = pd.Timestamp("2024-05-06 08:00", tz="Australia/Melbourne") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    return pd.DataFrame({"time": t, "headcount": rng.integers(0, 10, n).astype(float)})


def test_vectorised_matches_loop():
    ev = events()
    a = occupancy_on_grid(ev)
    b = occupancy_on_grid_loop(ev)
    both = a.index.intersection(b.index)
    assert len(both) > 50
    np.testing.assert_allclose(a[both].to_numpy(), b[both].to_numpy(), rtol=1e-9, equal_nan=True)


def test_constant_headcount_and_gap():
    t = pd.date_range("2024-05-06 09:00", periods=121, freq="30s", tz="Australia/Melbourne")
    ev = pd.DataFrame({"time": t, "headcount": 4.0})
    # 2 h silence then resume: the gap must be NaN, not 4
    t2 = t[-1] + pd.Timedelta("2h") + pd.to_timedelta(np.arange(61) * 30, unit="s")
    ev = pd.concat([ev, pd.DataFrame({"time": t2, "headcount": 2.0})])
    g = occupancy_on_grid(ev, "15min")
    assert g["2024-05-06 09:00":"2024-05-06 09:45"].eq(4.0).all()
    assert g["2024-05-06 10:15":"2024-05-06 11:45"].isna().all()
    with pytest.raises(ValueError):
        occupancy_on_grid(ev.iloc[:1])


@pytest.mark.parametrize("fn", [occupancy_on_grid, occupancy_on_grid_loop])
def test_hand_calculated_interval_means(fn):
    """0 people 09:00-09:05, 6 people 09:05-09:20, then 0.

    09:00 bin: (0*5 + 6*10)/15 = 4;  09:15 bin: (6*5 + 0*10)/15 = 2.
    """
    tz = "Australia/Melbourne"
    t = [pd.Timestamp("2024-05-06 09:00", tz=tz), pd.Timestamp("2024-05-06 09:05", tz=tz)]
    t += list(pd.date_range("2024-05-06 09:20", periods=21, freq="30s", tz=tz))
    ev = pd.DataFrame({"time": t, "headcount": np.array([0, 6] + [0] * 21, float)})
    g = fn(ev, "15min")
    assert g[pd.Timestamp("2024-05-06 09:00", tz=tz)] == pytest.approx(4.0)
    assert g[pd.Timestamp("2024-05-06 09:15", tz=tz)] == pytest.approx(2.0)
