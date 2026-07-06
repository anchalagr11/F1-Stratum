"""
Tests for Safety-Car / VSC awareness and rival-relative undercut analysis.
"""

import pandas as pd
import pytest

from tests.test_calibration import _synthetic_race
from src.simulation.optimizer import (
    PitWindowOptimizer, SC_PIT_LOSS, VSC_PIT_LOSS, DEFAULT_PIT_LOSS,
)
from src.simulation.tyre_model import TyreDegradationModel


def _race_with_caution() -> pd.DataFrame:
    race = _synthetic_race(n_drivers=6, n_laps=50)
    race.loc[race["lap"].isin([20, 21, 22]), "track_status"] = "124"  # Safety Car
    race.loc[race["lap"].isin([35, 36]), "track_status"] = "6"        # VSC
    return race


class TestSafetyCarAwareness:
    def test_pit_cost_detects_caution(self) -> None:
        opt = PitWindowOptimizer(_race_with_caution(), tyre_model=TyreDegradationModel())
        assert opt._pit_cost_for_lap(10) == (DEFAULT_PIT_LOSS, False)   # green
        assert opt._pit_cost_for_lap(21) == (SC_PIT_LOSS, True)         # Safety Car
        assert opt._pit_cost_for_lap(35) == (VSC_PIT_LOSS, True)        # VSC

    def test_sc_aware_flag_disables_discount(self) -> None:
        opt = PitWindowOptimizer(
            _race_with_caution(), tyre_model=TyreDegradationModel(), sc_aware=False
        )
        assert opt._pit_cost_for_lap(21) == (DEFAULT_PIT_LOSS, False)

    def test_caution_savings_positive_under_sc(self) -> None:
        opt = PitWindowOptimizer(_race_with_caution(), tyre_model=TyreDegradationModel())
        assert opt._caution_savings([21]) > 0        # SC lap saves time
        assert opt._caution_savings([10]) == 0.0     # green lap saves nothing

    def test_optimizer_still_returns_valid_candidates(self) -> None:
        opt = PitWindowOptimizer(_race_with_caution(), tyre_model=TyreDegradationModel())
        cands = opt.optimize_single_stop("D0", decision_lap=5)
        assert cands and all(c.total_time > 0 for c in cands)


class TestUndercut:
    def test_undercut_returns_verdict(self) -> None:
        opt = PitWindowOptimizer(_synthetic_race(n_drivers=6), tyre_model=TyreDegradationModel())
        result = opt.evaluate_undercut("D3", decision_lap=10)
        assert result["verdict"] in {"UNDERCUT", "HOLD"}
        assert "undercut_gain_s" in result and "net_s" in result

    def test_leader_has_no_undercut_target(self) -> None:
        opt = PitWindowOptimizer(_synthetic_race(n_drivers=6), tyre_model=TyreDegradationModel())
        assert opt.evaluate_undercut("D0", decision_lap=10)["verdict"] == "N/A"

    def test_rival_behind_is_not_a_target(self) -> None:
        opt = PitWindowOptimizer(_synthetic_race(n_drivers=6), tyre_model=TyreDegradationModel())
        # D0 leads; D3 is behind, so undercutting D3 is nonsensical
        assert opt.evaluate_undercut("D0", decision_lap=10, rival="D3")["verdict"] == "N/A"
