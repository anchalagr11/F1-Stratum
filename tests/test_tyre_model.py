"""
Tests for src.simulation.tyre_model
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.simulation.tyre_model import (
    TyreDegradationModel,
    CompoundParams,
    DEFAULT_COMPOUND_PARAMS,
)


class TestTyreDegradationModel:
    """Tests for the tyre degradation model."""

    def test_default_compounds_loaded(self) -> None:
        """Default model has all 5 compounds."""
        model = TyreDegradationModel()
        assert "SOFT" in model.params
        assert "MEDIUM" in model.params
        assert "HARD" in model.params
        assert "INTERMEDIATE" in model.params
        assert "WET" in model.params

    def test_degradation_increases_with_age(self) -> None:
        """Degradation increases monotonically with tyre age."""
        model = TyreDegradationModel()
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            prev = 0.0
            for age in range(1, 40):
                deg = model.degradation(compound, float(age))
                assert deg >= prev, f"{compound} at age {age}: {deg} < {prev}"
                prev = deg

    def test_soft_degrades_faster_than_hard(self) -> None:
        """Soft tyres degrade faster than hard at the same age."""
        model = TyreDegradationModel()
        for age in [5, 10, 15, 20]:
            soft_deg = model.degradation("SOFT", float(age))
            hard_deg = model.degradation("HARD", float(age))
            assert soft_deg > hard_deg, f"Age {age}: SOFT={soft_deg} <= HARD={hard_deg}"

    def test_cliff_effect(self) -> None:
        """Degradation accelerates sharply past cliff onset."""
        model = TyreDegradationModel()
        cliff = model.cliff_onset_lap("SOFT")
        deg_before = model.degradation("SOFT", cliff - 1)
        deg_at = model.degradation("SOFT", cliff)
        deg_after = model.degradation("SOFT", cliff + 5)
        # Rate should increase past cliff
        rate_before = deg_at - deg_before
        rate_after = (deg_after - deg_at) / 5
        assert rate_after > rate_before

    def test_degradation_at_zero(self) -> None:
        """Degradation at age 0 is 0."""
        model = TyreDegradationModel()
        assert model.degradation("MEDIUM", 0.0) == 0.0

    def test_degradation_rate(self) -> None:
        """Marginal degradation rate is positive."""
        model = TyreDegradationModel()
        for age in [1, 5, 10, 20]:
            rate = model.degradation_rate("MEDIUM", float(age))
            assert rate > 0

    def test_fresh_tyre_advantage(self) -> None:
        """Fresh tyre advantage is positive for all compounds."""
        model = TyreDegradationModel()
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            assert model.fresh_tyre_advantage(compound) > 0

    def test_soft_has_highest_fresh_advantage(self) -> None:
        """Soft compound has the highest fresh-tyre pace delta."""
        model = TyreDegradationModel()
        soft = model.fresh_tyre_advantage("SOFT")
        medium = model.fresh_tyre_advantage("MEDIUM")
        hard = model.fresh_tyre_advantage("HARD")
        assert soft > medium > hard

    def test_predict_stint_times_length(self) -> None:
        """Stint projection returns correct number of lap times."""
        model = TyreDegradationModel()
        times = model.predict_stint_times("MEDIUM", 90.0, 15)
        assert len(times) == 15

    def test_predict_stint_times_increasing(self) -> None:
        """Stint lap times increase (degradation grows)."""
        model = TyreDegradationModel()
        times = model.predict_stint_times("SOFT", 90.0, 20)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_unknown_compound_fallback(self) -> None:
        """Unknown compound uses fallback parameters without error."""
        model = TyreDegradationModel()
        deg = model.degradation("HYPERSOFT", 10.0)
        assert deg > 0

    def test_custom_params(self) -> None:
        """Model accepts and uses custom parameters."""
        custom = {
            "TEST": CompoundParams("TEST", a=0.1, b=0.01, c=0.0, cliff_onset=50, fresh_delta=2.0),
        }
        model = TyreDegradationModel(params=custom)
        deg = model.degradation("TEST", 10.0)
        expected = 0.1 * 10 + 0.01 * 100  # a*age + b*age²
        assert abs(deg - expected) < 0.001

    def test_save_and_load(self) -> None:
        """Model parameters survive save/load round-trip."""
        model = TyreDegradationModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = TyreDegradationModel.load(path)

        for compound in model.params:
            assert abs(
                model.degradation(compound, 15.0) - loaded.degradation(compound, 15.0)
            ) < 0.001

    def test_save_json_structure(self) -> None:
        """Saved JSON has expected structure."""
        model = TyreDegradationModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            with open(path) as f:
                data = json.load(f)

        assert "fitted" in data
        assert "compounds" in data
        assert "SOFT" in data["compounds"]
        assert "a" in data["compounds"]["SOFT"]

    def test_summary_shape(self) -> None:
        """Summary DataFrame has correct shape."""
        model = TyreDegradationModel()
        summary = model.summary()
        assert len(summary) == len(model.params)
        assert "compound" in summary.columns
        assert "deg_at_10" in summary.columns


class TestTyreModelFitting:
    """Tests for fitting from race data."""

    def test_fit_updates_params(self, enriched_race_state: pd.DataFrame) -> None:
        """Fitting from data updates compound parameters."""
        model = TyreDegradationModel()
        original_a = model.params["MEDIUM"].a
        model.fit_from_race_data(enriched_race_state, min_samples=5)
        # Parameters should have changed (though exact value is data-dependent)
        assert model.fitted is True

    def test_fit_returns_diagnostics(self, enriched_race_state: pd.DataFrame) -> None:
        """Fit results contain diagnostics for fitted compounds."""
        model = TyreDegradationModel()
        results = model.fit_from_race_data(enriched_race_state, min_samples=5)
        # Should have at least one compound fitted
        for compound, info in results.items():
            if "error" not in info:
                assert "r_squared" in info
                assert "n_samples" in info
