"""
STRATUM-F1 — Shared Test Fixtures

Provides reusable synthetic DataFrames for unit tests, avoiding
any dependency on FastF1 or network access.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_laps_df() -> pd.DataFrame:
    """Synthetic raw laps DataFrame mimicking FastF1 output."""
    drivers = ["VER", "NOR", "LEC"]
    laps_per_driver = 10
    rows = []
    for drv in drivers:
        for lap in range(1, laps_per_driver + 1):
            base_time = 92.0 + np.random.default_rng(42 + abs(hash(drv)) % 1000).normal(0, 0.5)
            rows.append({
                "Driver": drv,
                "DriverNumber": {"VER": "1", "NOR": "4", "LEC": "16"}[drv],
                "LapNumber": lap,
                "LapTime": base_time + lap * 0.04,
                "Sector1Time": 28.0,
                "Sector2Time": 32.0,
                "Sector3Time": 32.0,
                "Compound": "MEDIUM" if lap <= 5 else "HARD",
                "TyreLife": float(lap if lap <= 5 else lap - 5),
                "FreshTyre": lap == 1 or lap == 6,
                "Stint": 1 if lap <= 5 else 2,
                "Position": float(drivers.index(drv) + 1),
                "PitInTime": 150.0 if lap == 5 else None,
                "PitOutTime": 175.0 if lap == 6 else None,
                "IsPersonalBest": False,
                "Team": {"VER": "Red Bull", "NOR": "McLaren", "LEC": "Ferrari"}[drv],
                "TrackStatus": "1",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def session_data(raw_laps_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Synthetic session_data dict as returned by load_race_session."""
    return {
        "laps": raw_laps_df,
        "weather": pd.DataFrame({
            "AirTemp": [30.0, 30.5],
            "TrackTemp": [45.0, 46.0],
        }),
        "track_status": pd.DataFrame({
            "Status": ["1", "2"],
            "Message": ["AllClear", "YellowFlag"],
        }),
        "results": pd.DataFrame({
            "Position": [1, 2, 3],
            "Driver": ["VER", "NOR", "LEC"],
        }),
    }


@pytest.fixture
def canonical_race_state(session_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Canonical race state built from synthetic session data."""
    from src.canonical.build_race_state import build_race_state
    return build_race_state(session_data, race_id="test_2024_race")


@pytest.fixture
def enriched_race_state(canonical_race_state: pd.DataFrame) -> pd.DataFrame:
    """Canonical race state with strategy features added."""
    from src.features.strategy_features import add_strategy_features
    return add_strategy_features(canonical_race_state)
