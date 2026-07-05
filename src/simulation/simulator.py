"""
STRATUM-F1 — Race Strategy Simulator

Simulates counterfactual pit-stop strategies by projecting future
lap times under different actions (pit now, pit +1, stay out) and
estimating finishing positions.

Supports pluggable tyre degradation models for per-compound
non-linear degradation curves.
"""

import logging
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .tyre_model import TyreDegradationModel

logger = logging.getLogger(__name__)

# Type alias for valid actions
Action = Literal["pit_now", "pit_plus_1", "stay_out"]

# ──────────────────────────────────────────────────────────
# Default simulation parameters
# ──────────────────────────────────────────────────────────

# Linear tyre degradation per lap (seconds)
DEFAULT_TYRE_DEG_PER_LAP: float = 0.05

# Standard deviation of per-lap noise (seconds)
DEFAULT_LAP_NOISE_STD: float = 0.15

# Time lost in the pit lane (seconds)
DEFAULT_PIT_LOSS: float = 22.0

# Fresh-tyre pace advantage on the out-lap (seconds)
DEFAULT_FRESH_TYRE_GAIN: float = 1.0


class RaceSimulator:
    """Race strategy simulator with pluggable tyre degradation.

    Projects future lap times for a given driver starting from a
    specific lap, under three possible actions. Estimates expected
    finishing position and associated risk.

    When a ``TyreDegradationModel`` is provided, uses its per-compound
    non-linear curves. Otherwise falls back to a flat linear rate.

    Attributes:
        race_state: Canonical race state DataFrame.
        total_laps: Total number of laps in the race.
        tyre_model: Per-compound degradation model (if provided).
        tyre_deg: Legacy linear degradation rate (s/lap), used only
            when no tyre_model is supplied.
        lap_noise_std: Std-dev of Gaussian lap-time noise.
        pit_loss: Total pit-stop time loss (seconds).
        fresh_tyre_gain: Pace advantage on fresh tyres (s/lap).
    """

    def __init__(
        self,
        race_state: pd.DataFrame,
        tyre_model: Optional[TyreDegradationModel] = None,
        tyre_deg: float = DEFAULT_TYRE_DEG_PER_LAP,
        lap_noise_std: float = DEFAULT_LAP_NOISE_STD,
        pit_loss: float = DEFAULT_PIT_LOSS,
        fresh_tyre_gain: float = DEFAULT_FRESH_TYRE_GAIN,
    ) -> None:
        """Initialize the simulator with a canonical race state.

        Args:
            race_state: DataFrame with columns ``driver``, ``lap``,
                ``lap_time``, ``position``, ``tyre_age``, ``compound``,
                ``rolling_3lap_mean``, ``rolling_5lap_mean``.
            tyre_model: Optional per-compound degradation model.
                If provided, overrides the flat ``tyre_deg`` parameter.
            tyre_deg: Legacy linear degradation rate (s/lap).
            lap_noise_std: Standard deviation for lap-time noise.
            pit_loss: Pit-stop time loss (seconds).
            fresh_tyre_gain: Time gained per lap on fresh tyres.
        """
        self.race_state = race_state.copy()
        self.total_laps = int(race_state["lap"].max())
        self.tyre_model = tyre_model
        self.tyre_deg = tyre_deg
        self.lap_noise_std = lap_noise_std
        self.pit_loss = pit_loss
        self.fresh_tyre_gain = fresh_tyre_gain

        mode = "tyre model" if tyre_model else "linear fallback"
        logger.info(
            "RaceSimulator initialized — %d total laps, %d drivers, deg=%s",
            self.total_laps,
            race_state["driver"].nunique(),
            mode,
        )

    def simulate_strategy(
        self,
        driver: str,
        lap: int,
        action: Action,
    ) -> dict[str, float | str]:
        """Simulate a counterfactual strategy for a given driver and lap.

        Projects the remaining lap times under the chosen action,
        compares against other drivers' projected times, and
        estimates expected finishing position plus risk.

        Args:
            driver: Three-letter driver abbreviation (e.g. ``"VER"``).
            lap: Lap number at which the decision is made.
            action: One of ``"pit_now"``, ``"pit_plus_1"``, ``"stay_out"``.

        Returns:
            Dictionary with keys:
            - ``action`` (str): The action evaluated.
            - ``expected_finish`` (float): Projected finishing position.
            - ``risk`` (float): Risk score in [0, 1], higher = riskier.

        Raises:
            ValueError: If the driver or lap is not found in the race state.
        """
        self._validate_inputs(driver, lap, action)

        driver_state = self._get_driver_state(driver, lap)
        baseline_pace = self._get_baseline_pace(driver, lap)
        current_tyre_age = float(driver_state["tyre_age"].iloc[0])
        current_compound = str(driver_state["compound"].iloc[0]) if "compound" in driver_state.columns else "MEDIUM"
        remaining_laps = self.total_laps - lap

        if remaining_laps <= 0:
            return {"action": action, "expected_finish": float(driver_state["position"].iloc[0]), "risk": 0.0}

        # Project lap times for the target driver under the chosen action
        projected_times = self._project_lap_times(
            baseline_pace=baseline_pace,
            tyre_age=current_tyre_age,
            compound=current_compound,
            remaining_laps=remaining_laps,
            action=action,
            pit_lap_offset=0 if action == "pit_now" else (1 if action == "pit_plus_1" else remaining_laps),
        )
        driver_total = float(np.sum(projected_times))

        # Project totals for all other drivers (assume they stay out)
        other_totals = self._project_other_drivers(driver, lap, remaining_laps)

        # Estimate finishing position
        expected_finish = 1.0 + sum(1 for t in other_totals if t < driver_total)

        # Risk: combination of tyre age risk and noise-based uncertainty
        risk = self._compute_risk(
            action=action,
            tyre_age=current_tyre_age,
            remaining_laps=remaining_laps,
        )

        result = {
            "action": action,
            "expected_finish": round(expected_finish, 2),
            "risk": round(risk, 4),
        }
        logger.info("Simulation result for %s on lap %d: %s", driver, lap, result)
        return result

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _validate_inputs(self, driver: str, lap: int, action: str) -> None:
        """Validate that driver, lap, and action are valid."""
        valid_actions = {"pit_now", "pit_plus_1", "stay_out"}
        if action not in valid_actions:
            raise ValueError(f"Invalid action '{action}'. Must be one of {valid_actions}.")

        drivers = self.race_state["driver"].unique()
        if driver not in drivers:
            raise ValueError(f"Driver '{driver}' not found. Available: {list(drivers)}")

        laps = self.race_state["lap"].unique()
        if lap not in laps:
            raise ValueError(f"Lap {lap} not found. Available range: {int(min(laps))}–{int(max(laps))}")

    def _get_driver_state(self, driver: str, lap: int) -> pd.DataFrame:
        """Retrieve the driver's state at a specific lap."""
        mask = (self.race_state["driver"] == driver) & (self.race_state["lap"] == lap)
        state = self.race_state.loc[mask]
        if state.empty:
            raise ValueError(f"No data for driver={driver}, lap={lap}")
        return state

    def _get_baseline_pace(self, driver: str, lap: int) -> float:
        """Get baseline pace from rolling averages.

        Prefers the 5-lap rolling mean; falls back to 3-lap, then raw lap time.

        Args:
            driver: Driver abbreviation.
            lap: Current lap number.

        Returns:
            Baseline lap time in seconds.
        """
        state = self._get_driver_state(driver, lap)
        rolling_5 = state["rolling_5lap_mean"].iloc[0]
        rolling_3 = state["rolling_3lap_mean"].iloc[0]
        raw = state["lap_time"].iloc[0]

        for candidate in [rolling_5, rolling_3, raw]:
            if pd.notna(candidate) and candidate > 0:
                return float(candidate)

        # Last resort: median lap time for this driver
        driver_times = self.race_state.loc[
            self.race_state["driver"] == driver, "lap_time"
        ].dropna()
        return float(driver_times.median()) if not driver_times.empty else 90.0

    def _get_degradation(self, compound: str, tyre_age: float) -> float:
        """Get degradation penalty using the tyre model or linear fallback.

        Args:
            compound: Tyre compound name.
            tyre_age: Current tyre age in laps.

        Returns:
            Degradation penalty in seconds.
        """
        if self.tyre_model is not None:
            return self.tyre_model.degradation(compound, tyre_age)
        return self.tyre_deg * tyre_age

    def _get_fresh_bonus(self, compound: str, laps_since_pit: float) -> float:
        """Get fresh-tyre pace advantage, decaying over laps.

        Args:
            compound: Tyre compound name.
            laps_since_pit: Laps since the pit stop.

        Returns:
            Fresh-tyre bonus in seconds (positive = faster).
        """
        if self.tyre_model is not None:
            base_advantage = self.tyre_model.fresh_tyre_advantage(compound)
        else:
            base_advantage = self.fresh_tyre_gain
        return max(0.0, base_advantage * (1.0 - laps_since_pit / 10.0))

    def _project_lap_times(
        self,
        baseline_pace: float,
        tyre_age: float,
        compound: str,
        remaining_laps: int,
        action: str,
        pit_lap_offset: int,
        post_pit_compound: str = "MEDIUM",
    ) -> np.ndarray:
        """Project future lap times given a strategy action.

        Uses the tyre degradation model (if available) for compound-
        aware non-linear degradation; otherwise falls back to linear.

        Model:
            lap_time = baseline + deg(compound, age) + ε

        On the pit lap, ``pit_loss`` is added and tyre age resets.
        Post-pit laps use the new compound's degradation curve.

        Args:
            baseline_pace: Base lap time (seconds).
            tyre_age: Current tyre age (laps).
            compound: Current tyre compound.
            remaining_laps: Laps remaining in the race.
            action: Strategy action string.
            pit_lap_offset: Number of laps from now until the pit stop.
            post_pit_compound: Compound fitted after pit (default MEDIUM).

        Returns:
            Array of projected lap times.
        """
        rng = np.random.default_rng(seed=42)  # deterministic for reproducibility
        projected = np.zeros(remaining_laps)
        current_age = tyre_age
        current_compound = compound
        pitted = False

        for i in range(remaining_laps):
            noise = rng.normal(0, self.lap_noise_std)
            deg = self._get_degradation(current_compound, current_age)

            if i == pit_lap_offset and not pitted:
                # Pit stop on this lap
                projected[i] = baseline_pace + deg + noise + self.pit_loss
                current_age = 1.0  # fresh tyres
                current_compound = post_pit_compound
                pitted = True
            elif pitted:
                # Post-pit: fresh tyre advantage decays over laps
                fresh_bonus = self._get_fresh_bonus(current_compound, current_age)
                projected[i] = baseline_pace + deg + noise - fresh_bonus
                current_age += 1.0
            else:
                projected[i] = baseline_pace + deg + noise
                current_age += 1.0

        return projected

    def _project_other_drivers(
        self,
        exclude_driver: str,
        lap: int,
        remaining_laps: int,
    ) -> list[float]:
        """Project total remaining time for all other drivers (stay-out).

        Args:
            exclude_driver: Driver to exclude from projections.
            lap: Current lap number.
            remaining_laps: Laps remaining.

        Returns:
            List of projected total remaining times for each other driver.
        """
        totals: list[float] = []
        other_drivers = [
            d for d in self.race_state["driver"].unique()
            if d != exclude_driver
        ]

        for drv in other_drivers:
            try:
                pace = self._get_baseline_pace(drv, lap)
                state = self._get_driver_state(drv, lap)
                age = float(state["tyre_age"].iloc[0]) if pd.notna(state["tyre_age"].iloc[0]) else 10.0
                compound = str(state["compound"].iloc[0]) if "compound" in state.columns else "MEDIUM"
            except (ValueError, IndexError):
                continue

            times = self._project_lap_times(
                baseline_pace=pace,
                tyre_age=age,
                compound=compound,
                remaining_laps=remaining_laps,
                action="stay_out",
                pit_lap_offset=remaining_laps,  # no pit
            )
            totals.append(float(np.sum(times)))

        return totals

    def _compute_risk(
        self,
        action: str,
        tyre_age: float,
        remaining_laps: int,
    ) -> float:
        """Compute a risk score ∈ [0, 1] for the given strategy.

        Risk factors:
        - Staying out on old tyres increases risk of pace cliff.
        - Pitting introduces variability from pit-stop execution.
        - More remaining laps amplify uncertainty.

        Args:
            action: Strategy action.
            tyre_age: Current tyre age.
            remaining_laps: Laps remaining.

        Returns:
            Risk score between 0 (low risk) and 1 (high risk).
        """
        # Base risk from tyre age (older → higher risk of cliff)
        age_risk = min(1.0, tyre_age / 40.0)

        # Pit execution risk
        pit_risk = 0.1 if action in ("pit_now", "pit_plus_1") else 0.0

        # Race-length uncertainty
        length_risk = min(1.0, remaining_laps / self.total_laps) * 0.3

        if action == "stay_out":
            # Staying out on old tyres is risky
            raw_risk = age_risk * 0.6 + length_risk
        else:
            # Pitting reduces tyre risk but adds pit execution risk
            raw_risk = age_risk * 0.2 + pit_risk + length_risk

        return min(1.0, max(0.0, raw_risk))
