"""
Regression tests for the Sprint 2 model calibration.

These lock in the behaviour that fixed the validation-backtest defects:
- degenerate tyre fits fall back to physical defaults (no "magic tyre"),
- compound ordering SOFT >= MEDIUM >= HARD is enforced,
- rivals are projected with a stop so finishing positions spread out,
- the track-position penalty discourages marginal extra stops.
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.tyre_model import TyreDegradationModel, DEFAULT_COMPOUND_PARAMS
from src.simulation.optimizer import PitWindowOptimizer


def _synthetic_race(n_drivers: int = 4, n_laps: int = 40, seed: int = 0) -> pd.DataFrame:
    """Build a minimal canonical race state for optimizer tests."""
    rng = np.random.default_rng(seed)
    drivers = [f"D{i}" for i in range(n_drivers)]
    rows = []
    for di, drv in enumerate(drivers):
        base = 90.0 + di * 0.5  # each driver a bit slower than the last
        for lap in range(1, n_laps + 1):
            age = lap
            deg = 0.05 * age  # gentle real degradation
            lt = base + deg + rng.normal(0, 0.05)
            rows.append({
                "driver": drv, "lap": lap, "position": di + 1,
                "lap_time": lt, "compound": "MEDIUM", "tyre_age": float(age),
                "rolling_3lap_mean": lt, "rolling_5lap_mean": lt,
                "pit_this_lap": False, "track_status": "1",
            })
    return pd.DataFrame(rows)


class TestTyreFitGuard:
    def test_low_r2_keeps_default(self) -> None:
        """A near-flat, noisy compound must not collapse to ~zero degradation."""
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({
            "driver": ["X"] * n,
            "lap": range(1, n + 1),
            "tyre_age": np.arange(1, n + 1, dtype=float),
            # pure noise vs age -> low R^2 -> fit should be rejected
            "lap_time": 90.0 + rng.normal(0, 0.3, n),
            "compound": ["SOFT"] * n,
            "pit_this_lap": [False] * n,
            "track_status": ["1"] * n,
        })
        model = TyreDegradationModel()
        model.fit_from_race_data(df, min_samples=10)
        # SOFT should retain a realistic (non-trivial) degradation, not ~0.
        assert model.degradation("SOFT", 15) >= 0.5 * DEFAULT_COMPOUND_PARAMS["SOFT"].a * 15

    def test_compound_ordering_enforced(self) -> None:
        """After fitting, SOFT must degrade at least as fast as HARD."""
        model = TyreDegradationModel()
        # start from a deliberately inverted state
        model.params["SOFT"] = DEFAULT_COMPOUND_PARAMS["HARD"]  # too durable
        model._enforce_compound_ordering()
        assert model.degradation("SOFT", 15) >= model.degradation("HARD", 15)


class TestPositionSpread:
    def test_finish_positions_spread(self) -> None:
        """Estimated finish should differentiate fast from slow drivers."""
        race = _synthetic_race()
        opt = PitWindowOptimizer(race, tyre_model=TyreDegradationModel())
        fast = opt.find_optimal("D0", decision_lap=5, max_stops=1, top_n=1)["best_overall"][0]
        slow = opt.find_optimal("D3", decision_lap=5, max_stops=1, top_n=1)["best_overall"][0]
        # Not everyone collapses to P1; the slower car is projected lower.
        assert fast.expected_finish <= slow.expected_finish
        assert slow.expected_finish > 1.0


class TestStopPenalty:
    def test_penalty_reduces_stops(self) -> None:
        """A track-position penalty should not increase the chosen stop count."""
        race = _synthetic_race(n_laps=50)
        model = TyreDegradationModel()
        pure = PitWindowOptimizer(race, tyre_model=model, stop_penalty=0.0)
        realistic = PitWindowOptimizer(race, tyre_model=model, stop_penalty=25.0)
        b_pure = pure.find_optimal("D0", decision_lap=5, max_stops=2, top_n=1)["best_overall"][0]
        b_real = realistic.find_optimal("D0", decision_lap=5, max_stops=2, top_n=1)["best_overall"][0]
        assert len(b_real.pit_laps) <= len(b_pure.pit_laps)
