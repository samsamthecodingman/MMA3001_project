"""Time helpers robust to pandas datetime resolution (s/ms/us/ns)."""
from __future__ import annotations

import numpy as np
import pandas as pd

def unix_seconds(times: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Seconds since the Unix epoch as float64, independent of the
    datetime resolution pandas chose (s/ms/us/ns) and of the timezone."""
    t = pd.DatetimeIndex(times)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return ((t - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(1, "s")).to_numpy(float)

