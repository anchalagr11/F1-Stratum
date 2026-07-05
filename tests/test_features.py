"""
Tests for src.features.strategy_features
"""

import numpy as np
import pandas as pd
import pytest

from src.features.strategy_features import add_strategy_features


class TestStrategyFeatures:
    """Tests for the strategy feature engineering module."""

    def test_output_columns(self, canonical_race_state: pd.DataFrame) -> None:
        """Output contains the three new feature columns."""
        df = add_strategy_features(canonical_race_state)
        assert "traffic_penalty" in df.columns
        assert "undercut_estimate" in df.columns
        assert "pit_loss_estimate" in df.columns

    def test_row_count_preserved(self, canonical_race_state: pd.DataFrame) -> None:
        """Number of rows is preserved after adding features."""
        original_len = len(canonical_race_state)
        df = add_strategy_features(canonical_race_state)
        assert len(df) == original_len

    def test_original_columns_preserved(self, canonical_race_state: pd.DataFrame) -> None:
        """Original columns are not dropped."""
        original_cols = set(canonical_race_state.columns)
        df = add_strategy_features(canonical_race_state)
        assert original_cols.issubset(set(df.columns))

    def test_does_not_modify_input(self, canonical_race_state: pd.DataFrame) -> None:
        """Input DataFrame is not modified in place."""
        original_cols = list(canonical_race_state.columns)
        add_strategy_features(canonical_race_state)
        assert list(canonical_race_state.columns) == original_cols

    def test_traffic_penalty_values(self, canonical_race_state: pd.DataFrame) -> None:
        """Traffic penalty is either 0 or the configured penalty value."""
        df = add_strategy_features(canonical_race_state, traffic_penalty=0.5)
        unique_vals = set(df["traffic_penalty"].unique())
        assert unique_vals.issubset({0.0, 0.5})

    def test_traffic_penalty_with_small_gap(self) -> None:
        """Traffic penalty applies when gap_ahead < threshold."""
        df = pd.DataFrame({
            "gap_ahead": [0.5, 2.0, 1.0, np.nan],
            "tyre_age": [5, 5, 5, 5],
        })
        result = add_strategy_features(df, traffic_threshold=1.5, traffic_penalty=0.3)
        assert result["traffic_penalty"].iloc[0] == 0.3  # 0.5 < 1.5
        assert result["traffic_penalty"].iloc[1] == 0.0  # 2.0 >= 1.5
        assert result["traffic_penalty"].iloc[2] == 0.3  # 1.0 < 1.5
        assert result["traffic_penalty"].iloc[3] == 0.0  # NaN → inf >= 1.5

    def test_undercut_increases_with_tyre_age(self) -> None:
        """Undercut estimate grows with tyre age."""
        df = pd.DataFrame({
            "gap_ahead": [2.0, 2.0, 2.0],
            "tyre_age": [5.0, 15.0, 30.0],
        })
        result = add_strategy_features(df)
        vals = result["undercut_estimate"].values
        assert vals[0] < vals[1] < vals[2]

    def test_pit_loss_constant(self, canonical_race_state: pd.DataFrame) -> None:
        """Pit loss is constant across all rows."""
        df = add_strategy_features(canonical_race_state, base_pit_loss=20.0, track_factor=1.1)
        expected = 20.0 * 1.1
        assert (df["pit_loss_estimate"] == expected).all()

    def test_custom_track_factor(self) -> None:
        """Track factor correctly scales pit loss."""
        df = pd.DataFrame({
            "gap_ahead": [2.0],
            "tyre_age": [10.0],
        })
        result = add_strategy_features(df, base_pit_loss=22.0, track_factor=1.15)
        assert abs(result["pit_loss_estimate"].iloc[0] - 25.3) < 0.01
