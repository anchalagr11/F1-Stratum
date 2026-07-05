"""
STRATUM-F1 — Strategy Feature Engineering

Adds strategy-relevant derived features to the canonical race state:
traffic penalty, undercut estimate, and pit loss estimate.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configurable constants (sensible defaults; override per-track)
# ──────────────────────────────────────────────────────────────

# If gap_ahead < this threshold (seconds), driver is in "dirty air"
TRAFFIC_THRESHOLD_SEC: float = 1.5

# Penalty added per lap while in traffic (seconds)
TRAFFIC_PENALTY_PER_LAP: float = 0.3

# Base pit-stop time loss (pit lane Δ vs a normal lap), seconds
BASE_PIT_LOSS_SEC: float = 22.0

# Per-track adjustment factor applied to pit loss
# Default 1.0; street circuits ≈ 1.15, short tracks ≈ 0.90
TRACK_PIT_FACTOR: float = 1.0

# Fresh-tyre advantage used in undercut heuristic (seconds/lap)
FRESH_TYRE_ADVANTAGE_SEC: float = 1.0

# Number of laps over which the undercut advantage is summed
UNDERCUT_WINDOW_LAPS: int = 3


def add_strategy_features(
    df: pd.DataFrame,
    base_pit_loss: float = BASE_PIT_LOSS_SEC,
    track_factor: float = TRACK_PIT_FACTOR,
    traffic_threshold: float = TRAFFIC_THRESHOLD_SEC,
    traffic_penalty: float = TRAFFIC_PENALTY_PER_LAP,
    fresh_tyre_adv: float = FRESH_TYRE_ADVANTAGE_SEC,
    undercut_window: int = UNDERCUT_WINDOW_LAPS,
) -> pd.DataFrame:
    """Augment the canonical race state with strategy features.

    All parameters have sensible defaults and can be overridden
    per-track or per-analysis.

    Args:
        df: Canonical race state DataFrame (one row per driver × lap).
        base_pit_loss: Base time lost during a pit stop (seconds).
        track_factor: Multiplier for track-specific pit loss.
        traffic_threshold: Gap-ahead threshold (s) below which traffic
            penalty applies.
        traffic_penalty: Seconds added per lap when in traffic.
        fresh_tyre_adv: Assumed pace advantage on fresh tyres (s/lap).
        undercut_window: Number of laps the undercut advantage persists.

    Returns:
        DataFrame with additional columns:
        ``traffic_penalty``, ``undercut_estimate``, ``pit_loss_estimate``.
    """
    logger.info("Adding strategy features (%d rows)", len(df))

    df = df.copy()
    df["traffic_penalty"] = _compute_traffic_penalty(
        df, threshold=traffic_threshold, penalty=traffic_penalty,
    )
    df["undercut_estimate"] = _compute_undercut_estimate(
        df, fresh_tyre_adv=fresh_tyre_adv, window=undercut_window,
    )
    df["pit_loss_estimate"] = _compute_pit_loss(
        base_pit_loss=base_pit_loss, track_factor=track_factor, n_rows=len(df),
    )

    logger.info("Strategy features added successfully.")
    return df


def _compute_traffic_penalty(
    df: pd.DataFrame,
    threshold: float,
    penalty: float,
) -> pd.Series:
    """Estimate time lost due to running in traffic (dirty air).

    A non-zero penalty is assigned when ``gap_ahead`` is below the
    configured threshold, indicating aerodynamic disadvantage.

    Args:
        df: Race state DataFrame with ``gap_ahead`` column.
        threshold: Gap (seconds) below which traffic penalty kicks in.
        penalty: Penalty value (seconds) when in traffic.

    Returns:
        Series of per-row traffic penalty values.
    """
    gap = df["gap_ahead"].fillna(np.inf)
    return np.where(gap < threshold, penalty, 0.0)


def _compute_undercut_estimate(
    df: pd.DataFrame,
    fresh_tyre_adv: float,
    window: int,
) -> pd.Series:
    """Heuristic estimate of the undercut potential (seconds gained).

    The undercut benefit is modeled as:
        ``fresh_tyre_advantage × window − tyre_age_penalty``

    where ``tyre_age_penalty`` scales with current tyre age,
    reflecting that older tyres amplify the undercut benefit.

    Args:
        df: Race state DataFrame with ``tyre_age`` column.
        fresh_tyre_adv: Assumed seconds-per-lap gained on fresh rubber.
        window: Number of laps over which the advantage accumulates.

    Returns:
        Series of undercut estimates (positive = favourable to pit).
    """
    tyre_age = df["tyre_age"].fillna(0.0)
    # Older tyres → larger undercut benefit
    age_factor = 1.0 + (tyre_age / 30.0)  # normalized ramp
    return fresh_tyre_adv * window * age_factor


def _compute_pit_loss(
    base_pit_loss: float,
    track_factor: float,
    n_rows: int,
) -> pd.Series:
    """Estimate total time lost during a pit stop.

    Currently a constant per-row value; future versions can
    incorporate dynamic factors (fuel load, tyre-change speed).

    Args:
        base_pit_loss: Base pit lane time loss (seconds).
        track_factor: Track-specific multiplier.
        n_rows: Number of rows in the DataFrame.

    Returns:
        Constant Series of pit-loss estimates.
    """
    return pd.Series(base_pit_loss * track_factor, index=range(n_rows))
