"""Synthetic room data generated from the CO2 mass-balance model.

Used for (i) verifying the numerical code against a known truth
("manufactured" data) and (ii) developing the pipeline before/without
access to the real building data. The generator mimics the real data's
key defects: irregular sampling, sensor noise and quantisation, and
dropouts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import RoomParams, simulate


def occupancy_schedule(
    start: str = "2024-05-06",
    days: int = 14,
    freq_s: int = 60,
    capacity: int = 12,
    seed: int = 0,
) -> pd.Series:
    """Random but realistic weekday occupancy schedule.

    Weekdays have a base load between ~9:00 and ~17:00 plus random
    meetings; weekends are mostly empty.

    Args:
        start: Start date (Monday recommended).
        days: Number of days to generate.
        freq_s: Time step of the schedule [s].
        capacity: Maximum occupancy of the room [persons].
        seed: RNG seed for reproducibility.

    Returns:
        Integer occupancy series indexed by timestamp.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 86400 // freq_s, freq=f"{freq_s}s")
    N = np.zeros(len(idx), dtype=int)
    hours = idx.hour + idx.minute / 60
    for d in range(days):
        day = pd.Timestamp(start) + pd.Timedelta(days=d)
        in_day = (idx >= day) & (idx < day + pd.Timedelta(days=1))
        if day.day_of_week >= 5:
            if rng.random() < 0.2:  # occasional weekend visit
                a = rng.uniform(10, 14)
                N[in_day & (hours >= a) & (hours < a + rng.uniform(1, 3))] = rng.integers(1, 3)
            continue
        arrive, leave = rng.normal(9, 0.4), rng.normal(17, 0.5)
        base = rng.integers(2, max(3, capacity // 2))
        N[in_day & (hours >= arrive) & (hours < leave)] = base
        lunch = rng.uniform(12, 13)
        N[in_day & (hours >= lunch) & (hours < lunch + 0.75)] //= 2
        for _ in range(rng.integers(0, 4)):  # meetings
            m0 = rng.uniform(arrive, leave - 1)
            sel = in_day & (hours >= m0) & (hours < m0 + rng.uniform(0.5, 1.5))
            N[sel] = np.minimum(capacity, N[sel] + rng.integers(2, capacity // 2 + 1))
    return pd.Series(N, index=idx, name="occupancy")


def make_room(
    params: RoomParams | None = None,
    days: int = 14,
    seed: int = 0,
    sample_mean_s: float = 600.0,
    noise_ppm: float = 15.0,
    dropout: float = 0.03,
    capacity: int = 12,
) -> tuple[pd.DataFrame, pd.Series, RoomParams]:
    """Generate one synthetic room.

    Args:
        params: Room parameters; default is a 150 m^3 meeting room at
            ~3 air changes per hour.
        days: Length of record [days].
        seed: RNG seed.
        sample_mean_s: Mean interval between env-sensor reports [s];
            actual intervals are exponentially jittered (irregular).
        noise_ppm: Std. dev. of additive CO2 sensor noise [ppm].
        dropout: Fraction of reports randomly removed.
        capacity: Max occupancy.

    Returns:
        ``(env, occ, params)`` where ``env`` is an irregularly sampled
        DataFrame with column ``co2_ppm`` (+ ``temp_c``), ``occ`` is the
        1-minute true occupancy, and ``params`` the true parameters.
    """
    rng = np.random.default_rng(seed)
    p = params or RoomParams(V=150.0, Q=150.0 * 3 / 3600, C_out=420.0)
    occ = occupancy_schedule(days=days, seed=seed, capacity=capacity)
    t = (occ.index - occ.index[0]).total_seconds().to_numpy()
    C = simulate(t, occ.to_numpy(float), p, method="exact")
    # irregular sample times
    gaps = rng.exponential(sample_mean_s, size=int(1.5 * t[-1] / sample_mean_s))
    ts = np.cumsum(gaps)
    ts = ts[ts < t[-1]]
    ts = ts[rng.random(len(ts)) > dropout]
    co2 = np.interp(ts, t, C) + rng.normal(0, noise_ppm, len(ts))
    temp = 21 + 0.08 * np.interp(ts, t, occ.to_numpy(float)) + rng.normal(0, 0.1, len(ts))
    env = pd.DataFrame(
        {"co2_ppm": np.round(co2), "temp_c": np.round(temp, 1)},
        index=occ.index[0] + pd.to_timedelta(ts, unit="s"),
    )
    env.index.name = "time"
    return env, occ, p
