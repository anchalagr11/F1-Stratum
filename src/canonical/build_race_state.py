"""
STRATUM-F1 — Canonical Race State Builder

Transforms raw session data into a canonical race state DataFrame
with one row per (driver, lap). Computes positional gaps, pit events,
and rolling pace averages.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns in the output canonical race state
CANONICAL_COLUMNS = [
    "race_id",
    "lap",
    "driver",
    "position",
    "lap_time",
    "compound",
    "tyre_age",
    "gap_ahead",
    "gap_behind",
    "pit_this_lap",
    "track_status",
    "rolling_3lap_mean",
    "rolling_5lap_mean",
]


def build_race_state(
    session_data: dict[str, pd.DataFrame],
    race_id: Optional[str] = None,
) -> pd.DataFrame:
    """Build a canonical race state DataFrame from raw session data.

    Produces one row per (driver, lap) with position, pace, tyre info,
    gaps to surrounding cars, pit indicators, and rolling averages.

    Args:
        session_data: Dictionary returned by ``load_race_session`` with
            at minimum a ``"laps"`` key.
        race_id: Optional identifier for this race. If ``None``, defaults
            to ``"unknown_race"``.

    Returns:
        Clean DataFrame conforming to ``CANONICAL_COLUMNS``.

    Raises:
        ValueError: If the laps DataFrame is missing or empty.
    """
    if race_id is None:
        race_id = "unknown_race"

    laps = session_data.get("laps")
    if laps is None or laps.empty:
        raise ValueError("session_data must contain a non-empty 'laps' DataFrame.")

    logger.info("Building canonical race state for race_id=%s", race_id)

    df = _build_base_frame(laps, race_id)
    df = _compute_gaps(df)
    df = _compute_rolling_averages(df)

    # Enforce column order and fill remaining NaNs
    df = df[CANONICAL_COLUMNS].copy()
    df["lap_time"] = df["lap_time"].astype(float)
    df["rolling_3lap_mean"] = df["rolling_3lap_mean"].astype(float)
    df["rolling_5lap_mean"] = df["rolling_5lap_mean"].astype(float)

    logger.info(
        "Canonical race state built — %d rows, %d drivers, %d laps",
        len(df),
        df["driver"].nunique(),
        df["lap"].nunique(),
    )
    return df


def _build_base_frame(laps: pd.DataFrame, race_id: str) -> pd.DataFrame:
    """Extract and rename core fields from the raw laps DataFrame.

    Args:
        laps: Raw laps DataFrame from FastF1.
        race_id: Identifier string for this race.

    Returns:
        DataFrame with base canonical columns populated.
    """
    df = pd.DataFrame(index=laps.index)
    df["lap"] = laps["LapNumber"].astype(int)
    df["driver"] = laps["Driver"].astype(str)
    df["race_id"] = race_id

    # Position (may contain NaN for incomplete laps)
    df["position"] = (
        laps["Position"].astype(float) if "Position" in laps.columns
        else np.nan
    )

    # Lap time in seconds
    df["lap_time"] = _to_seconds(laps, "LapTime")

    # Tyre compound and age
    df["compound"] = (
        laps["Compound"].astype(str) if "Compound" in laps.columns
        else "UNKNOWN"
    )
    df["tyre_age"] = (
        laps["TyreLife"].astype(float) if "TyreLife" in laps.columns
        else np.nan
    )

    # Pit indicator: True if the driver pitted on this lap
    df["pit_this_lap"] = _detect_pit(laps)

    # Track status (first character is the status code)
    df["track_status"] = (
        laps["TrackStatus"].astype(str) if "TrackStatus" in laps.columns
        else "1"  # '1' means green/all-clear
    )

    return df


def _to_seconds(laps: pd.DataFrame, col: str) -> pd.Series:
    """Convert a column to float seconds.

    Handles both timedelta and already-numeric representations.

    Args:
        laps: Laps DataFrame.
        col: Column name to convert.

    Returns:
        Series of float seconds (NaN where conversion fails).
    """
    if col not in laps.columns:
        return pd.Series(np.nan, index=laps.index)

    series = laps[col]
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _detect_pit(laps: pd.DataFrame) -> pd.Series:
    """Determine whether the driver pitted on each lap.

    Uses ``PitInTime`` or ``PitOutTime`` presence as indicators.

    Args:
        laps: Laps DataFrame.

    Returns:
        Boolean Series — ``True`` if a pit event occurred on that lap.
    """
    pit_in = pd.Series(False, index=laps.index)
    pit_out = pd.Series(False, index=laps.index)

    if "PitInTime" in laps.columns:
        pit_in = laps["PitInTime"].notna()
    if "PitOutTime" in laps.columns:
        pit_out = laps["PitOutTime"].notna()

    return pit_in | pit_out


def _compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gap to car ahead and car behind (in seconds) per lap.

    Gaps are calculated from cumulative lap times within each lap's
    position ordering.

    Args:
        df: Base canonical DataFrame with ``lap``, ``driver``, ``position``,
            and ``lap_time``.

    Returns:
        DataFrame with ``gap_ahead`` and ``gap_behind`` columns added.
    """
    # Compute cumulative time per driver across laps
    df = df.sort_values(["driver", "lap"]).copy()
    df["cum_time"] = df.groupby("driver")["lap_time"].cumsum()

    gap_ahead_list: list[float] = []
    gap_behind_list: list[float] = []

    for lap_num, group in df.groupby("lap"):
        sorted_group = group.sort_values("position")
        cum_times = sorted_group["cum_time"].values
        positions = sorted_group["position"].values

        ga = np.full(len(sorted_group), np.nan)
        gb = np.full(len(sorted_group), np.nan)

        for i in range(len(sorted_group)):
            if i > 0 and not np.isnan(cum_times[i]) and not np.isnan(cum_times[i - 1]):
                ga[i] = cum_times[i] - cum_times[i - 1]
            if i < len(sorted_group) - 1 and not np.isnan(cum_times[i]) and not np.isnan(cum_times[i + 1]):
                gb[i] = cum_times[i + 1] - cum_times[i]

        # Map back via the sorted index
        for idx, (g_a, g_b) in zip(sorted_group.index, zip(ga, gb)):
            gap_ahead_list.append((idx, g_a))
            gap_behind_list.append((idx, g_b))

    # Apply to original DataFrame via index
    ga_series = pd.Series(
        {idx: val for idx, val in gap_ahead_list},
        dtype=float,
    )
    gb_series = pd.Series(
        {idx: val for idx, val in gap_behind_list},
        dtype=float,
    )
    df["gap_ahead"] = ga_series
    df["gap_behind"] = gb_series

    df.drop(columns=["cum_time"], inplace=True)
    return df


def _compute_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling 3-lap and 5-lap mean lap times per driver.

    Args:
        df: Canonical DataFrame with ``driver``, ``lap``, ``lap_time``.

    Returns:
        DataFrame with ``rolling_3lap_mean`` and ``rolling_5lap_mean`` added.
    """
    df = df.sort_values(["driver", "lap"]).copy()

    df["rolling_3lap_mean"] = (
        df.groupby("driver")["lap_time"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    )
    df["rolling_5lap_mean"] = (
        df.groupby("driver")["lap_time"]
        .transform(lambda s: s.rolling(window=5, min_periods=1).mean())
    )

    return df
