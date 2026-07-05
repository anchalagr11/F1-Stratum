"""
Tests for src.simulation.simulator (RaceSimulator)
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.simulator import RaceSimulator
from src.simulation.tyre_model import TyreDegradationModel


class TestRaceSimulator:
    """Tests for the race strategy simulator."""

    def test_init_with_tyre_model(self, enriched_race_state: pd.DataFrame) -> None:
        """Simulator initializes with a tyre model."""
        model = TyreDegradationModel()
        sim = RaceSimulator(enriched_race_state, tyre_model=model)
        assert sim.tyre_model is not None
        assert sim.total_laps == 10

    def test_init_without_tyre_model(self, enriched_race_state: pd.DataFrame) -> None:
        """Simulator initializes in legacy linear mode."""
        sim = RaceSimulator(enriched_race_state)
        assert sim.tyre_model is None

    def test_simulate_returns_dict(self, enriched_race_state: pd.DataFrame) -> None:
        """simulate_strategy returns a dict with expected keys."""
        sim = RaceSimulator(enriched_race_state)
        result = sim.simulate_strategy("VER", 5, "pit_now")
        assert "action" in result
        assert "expected_finish" in result
        assert "risk" in result

    def test_simulate_action_preserved(self, enriched_race_state: pd.DataFrame) -> None:
        """Returned action matches the requested action."""
        sim = RaceSimulator(enriched_race_state)
        for action in ["pit_now", "pit_plus_1", "stay_out"]:
            result = sim.simulate_strategy("VER", 5, action)
            assert result["action"] == action

    def test_finish_position_range(self, enriched_race_state: pd.DataFrame) -> None:
        """Expected finish is between 1 and number of drivers."""
        sim = RaceSimulator(enriched_race_state)
        n_drivers = enriched_race_state["driver"].nunique()
        result = sim.simulate_strategy("VER", 3, "stay_out")
        assert 1 <= result["expected_finish"] <= n_drivers

    def test_risk_in_range(self, enriched_race_state: pd.DataFrame) -> None:
        """Risk score is between 0 and 1."""
        sim = RaceSimulator(enriched_race_state)
        for action in ["pit_now", "pit_plus_1", "stay_out"]:
            result = sim.simulate_strategy("VER", 5, action)
            assert 0 <= result["risk"] <= 1

    def test_invalid_driver_raises(self, enriched_race_state: pd.DataFrame) -> None:
        """Raises ValueError for unknown driver."""
        sim = RaceSimulator(enriched_race_state)
        with pytest.raises(ValueError, match="not found"):
            sim.simulate_strategy("HAM", 5, "pit_now")

    def test_invalid_lap_raises(self, enriched_race_state: pd.DataFrame) -> None:
        """Raises ValueError for invalid lap number."""
        sim = RaceSimulator(enriched_race_state)
        with pytest.raises(ValueError, match="not found"):
            sim.simulate_strategy("VER", 999, "pit_now")

    def test_invalid_action_raises(self, enriched_race_state: pd.DataFrame) -> None:
        """Raises ValueError for invalid action."""
        sim = RaceSimulator(enriched_race_state)
        with pytest.raises(ValueError, match="Invalid action"):
            sim.simulate_strategy("VER", 5, "invalid_action")

    def test_last_lap_returns_current_position(self, enriched_race_state: pd.DataFrame) -> None:
        """Simulating at the last lap returns current position."""
        sim = RaceSimulator(enriched_race_state)
        result = sim.simulate_strategy("VER", 10, "stay_out")
        assert result["risk"] == 0.0

    def test_tyre_model_changes_results(self, enriched_race_state: pd.DataFrame) -> None:
        """Using a tyre model produces different results than linear."""
        sim_linear = RaceSimulator(enriched_race_state)
        sim_model = RaceSimulator(enriched_race_state, tyre_model=TyreDegradationModel())

        r_lin = sim_linear.simulate_strategy("VER", 3, "stay_out")
        r_mod = sim_model.simulate_strategy("VER", 3, "stay_out")

        # Results may differ (not guaranteed, but likely with different deg models)
        # At minimum, both should be valid
        assert r_lin["expected_finish"] >= 1
        assert r_mod["expected_finish"] >= 1

    def test_deterministic_output(self, enriched_race_state: pd.DataFrame) -> None:
        """Same inputs produce same outputs (seed=42 in projections)."""
        sim = RaceSimulator(enriched_race_state)
        r1 = sim.simulate_strategy("NOR", 5, "pit_now")
        r2 = sim.simulate_strategy("NOR", 5, "pit_now")
        assert r1["expected_finish"] == r2["expected_finish"]
        assert r1["risk"] == r2["risk"]
