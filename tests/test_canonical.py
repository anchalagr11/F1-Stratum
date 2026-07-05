"""
Tests for src.canonical.build_race_state
"""

import numpy as np
import pandas as pd
import pytest

from src.canonical.build_race_state import build_race_state, CANONICAL_COLUMNS


class TestBuildRaceState:
    """Tests for the canonical race state builder."""

    def test_output_shape(self, session_data: dict) -> None:
        """Output has expected number of rows (drivers × laps)."""
        df = build_race_state(session_data, race_id="test")
        # 3 drivers × 10 laps = 30 rows
        assert len(df) == 30

    def test_canonical_columns_present(self, session_data: dict) -> None:
        """All canonical columns are present in output."""
        df = build_race_state(session_data, race_id="test")
        for col in CANONICAL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_race_id_assigned(self, session_data: dict) -> None:
        """race_id is correctly assigned to all rows."""
        df = build_race_state(session_data, race_id="2024_singapore")
        assert (df["race_id"] == "2024_singapore").all()

    def test_default_race_id(self, session_data: dict) -> None:
        """Defaults to 'unknown_race' when no race_id is provided."""
        df = build_race_state(session_data)
        assert (df["race_id"] == "unknown_race").all()

    def test_drivers_present(self, session_data: dict) -> None:
        """All three synthetic drivers are present."""
        df = build_race_state(session_data, race_id="test")
        drivers = set(df["driver"].unique())
        assert drivers == {"VER", "NOR", "LEC"}

    def test_lap_numbers_correct(self, session_data: dict) -> None:
        """Lap numbers range from 1 to 10 for each driver."""
        df = build_race_state(session_data, race_id="test")
        for drv in df["driver"].unique():
            laps = sorted(df[df["driver"] == drv]["lap"].tolist())
            assert laps == list(range(1, 11))

    def test_lap_time_positive(self, session_data: dict) -> None:
        """All lap times are positive floats."""
        df = build_race_state(session_data, race_id="test")
        valid = df["lap_time"].dropna()
        assert (valid > 0).all()

    def test_pit_detection(self, session_data: dict) -> None:
        """Pit events are correctly detected."""
        df = build_race_state(session_data, race_id="test")
        ver_pits = df[(df["driver"] == "VER") & (df["pit_this_lap"] == True)]
        # Driver pits on lap 5 (PitInTime) and lap 6 (PitOutTime)
        assert 5 in ver_pits["lap"].values or 6 in ver_pits["lap"].values

    def test_rolling_averages_computed(self, session_data: dict) -> None:
        """Rolling 3-lap and 5-lap means are not all NaN."""
        df = build_race_state(session_data, race_id="test")
        assert df["rolling_3lap_mean"].notna().any()
        assert df["rolling_5lap_mean"].notna().any()

    def test_rolling_3lap_correctness(self, session_data: dict) -> None:
        """Rolling 3-lap mean is correct for a specific driver."""
        df = build_race_state(session_data, race_id="test")
        ver = df[df["driver"] == "VER"].sort_values("lap")
        times = ver["lap_time"].values
        # At lap 3, rolling 3-lap mean should be mean of laps 1-3
        expected = np.mean(times[:3])
        actual = ver["rolling_3lap_mean"].iloc[2]
        assert abs(actual - expected) < 0.01

    def test_empty_laps_raises(self) -> None:
        """Raises ValueError when laps DataFrame is empty."""
        with pytest.raises(ValueError, match="non-empty"):
            build_race_state({"laps": pd.DataFrame()})

    def test_missing_laps_raises(self) -> None:
        """Raises ValueError when 'laps' key is missing."""
        with pytest.raises(ValueError):
            build_race_state({"weather": pd.DataFrame()})

    def test_compound_column(self, session_data: dict) -> None:
        """Compound column reflects tyre changes."""
        df = build_race_state(session_data, race_id="test")
        ver = df[df["driver"] == "VER"].sort_values("lap")
        # First 5 laps should be MEDIUM, last 5 HARD
        assert ver.iloc[0]["compound"] == "MEDIUM"
        assert ver.iloc[9]["compound"] == "HARD"
