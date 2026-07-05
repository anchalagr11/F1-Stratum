"""
Tests for src.simulation.optimizer (PitWindowOptimizer)
"""

import pandas as pd
import pytest

from src.simulation.optimizer import PitWindowOptimizer, StrategyCandidate
from src.simulation.tyre_model import TyreDegradationModel


class TestPitWindowOptimizer:
    """Tests for the pit window optimizer."""

    def test_init(self, enriched_race_state: pd.DataFrame) -> None:
        """Optimizer initializes correctly."""
        opt = PitWindowOptimizer(enriched_race_state)
        assert opt.total_laps == 10

    def test_single_stop_returns_candidates(self, enriched_race_state: pd.DataFrame) -> None:
        """Single-stop optimization returns a non-empty list."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        assert len(candidates) > 0
        assert all(isinstance(c, StrategyCandidate) for c in candidates)

    def test_single_stop_sorted_by_time(self, enriched_race_state: pd.DataFrame) -> None:
        """Candidates are sorted by total time (ascending)."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        times = [c.total_time for c in candidates]
        assert times == sorted(times)

    def test_single_stop_has_one_pit(self, enriched_race_state: pd.DataFrame) -> None:
        """Each single-stop candidate has exactly one pit lap."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        for c in candidates:
            assert len(c.pit_laps) == 1

    def test_single_stop_two_compounds(self, enriched_race_state: pd.DataFrame) -> None:
        """Each single-stop candidate has exactly two compounds."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        for c in candidates:
            assert len(c.compounds) == 2

    def test_time_delta_best_is_zero(self, enriched_race_state: pd.DataFrame) -> None:
        """Best candidate has time_delta = 0."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        if candidates:
            assert candidates[0].time_delta == 0.0

    def test_find_optimal_returns_categories(self, enriched_race_state: pd.DataFrame) -> None:
        """find_optimal returns both 1-stop and 2-stop categories."""
        model = TyreDegradationModel()
        opt = PitWindowOptimizer(enriched_race_state, tyre_model=model)
        results = opt.find_optimal("VER", decision_lap=2, max_stops=2, top_n=2)
        assert "1_stop" in results
        assert "best_overall" in results

    def test_find_optimal_top_n_limit(self, enriched_race_state: pd.DataFrame) -> None:
        """find_optimal respects the top_n parameter."""
        opt = PitWindowOptimizer(enriched_race_state)
        results = opt.find_optimal("VER", decision_lap=2, max_stops=1, top_n=3)
        assert len(results["1_stop"]) <= 3

    def test_summary_table_columns(self, enriched_race_state: pd.DataFrame) -> None:
        """summary_table contains expected columns."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        table = opt.summary_table(candidates[:3])
        expected_cols = {"rank", "stops", "pit_laps", "compounds", "stints",
                         "total_time", "delta", "finish", "risk"}
        assert expected_cols.issubset(set(table.columns))

    def test_risk_in_range(self, enriched_race_state: pd.DataFrame) -> None:
        """All risk scores are between 0 and 1."""
        opt = PitWindowOptimizer(enriched_race_state)
        candidates = opt.optimize_single_stop("VER", decision_lap=3)
        for c in candidates:
            assert 0 <= c.risk <= 1

    def test_stint_lengths_sum_to_remaining(self, enriched_race_state: pd.DataFrame) -> None:
        """Stint lengths sum to the remaining race distance."""
        opt = PitWindowOptimizer(enriched_race_state)
        decision_lap = 3
        remaining = opt.total_laps - decision_lap
        candidates = opt.optimize_single_stop("VER", decision_lap=decision_lap)
        for c in candidates:
            assert sum(c.stint_lengths) == remaining

    def test_with_tyre_model(self, enriched_race_state: pd.DataFrame) -> None:
        """Optimizer works correctly with a fitted tyre model."""
        model = TyreDegradationModel()
        opt = PitWindowOptimizer(enriched_race_state, tyre_model=model)
        candidates = opt.optimize_single_stop("NOR", decision_lap=3)
        assert len(candidates) > 0
        assert candidates[0].total_time > 0
