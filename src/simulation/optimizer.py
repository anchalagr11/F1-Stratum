"""
STRATUM-F1 — Pit Window Optimizer

Finds optimal pit-stop timing by evaluating all feasible pit laps
for a given driver. Supports single-stop and two-stop strategies
with compound selection.

Approach:
- Brute-force sweep over candidate pit laps within a configurable window
- For each candidate, project total race time using the simulator's
  degradation model (or linear fallback)
- Score each option by projected total time, estimated finishing
  position, and risk
- Return ranked strategies sorted by expected performance
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .tyre_model import TyreDegradationModel

logger = logging.getLogger(__name__)

# Available dry compounds for pit strategy evaluation
DRY_COMPOUNDS: list[str] = ["SOFT", "MEDIUM", "HARD"]

# Minimum stint length (laps) — too short is unrealistic
MIN_STINT_LAPS: int = 5

# Default pit-stop time loss (seconds)
DEFAULT_PIT_LOSS: float = 22.0


@dataclass
class StrategyCandidate:
    """A single pit strategy option with projected outcomes.

    Attributes:
        pit_laps: List of lap numbers at which pits occur.
        compounds: List of compounds for each stint
            (len = len(pit_laps) + 1).
        total_time: Projected total race time (seconds).
        expected_finish: Estimated finishing position.
        risk: Risk score ∈ [0, 1].
        stint_lengths: Duration of each stint in laps.
        time_delta: Time difference vs. the best strategy (seconds).
    """

    pit_laps: list[int]
    compounds: list[str]
    total_time: float
    expected_finish: float
    risk: float
    stint_lengths: list[int]
    time_delta: float = 0.0


class PitWindowOptimizer:
    """Finds optimal pit-stop timing for a given driver.

    Sweeps over all feasible single-stop and two-stop pit lap
    combinations, projects total race time using the tyre degradation
    model, and ranks strategies by expected performance.

    Attributes:
        race_state: Canonical race state DataFrame.
        tyre_model: Per-compound degradation model.
        total_laps: Total laps in the race.
        pit_loss: Time lost per pit stop (seconds).
        noise_std: Lap-time noise standard deviation.
    """

    def __init__(
        self,
        race_state: pd.DataFrame,
        tyre_model: Optional[TyreDegradationModel] = None,
        pit_loss: float = DEFAULT_PIT_LOSS,
        noise_std: float = 0.0,
    ) -> None:
        """Initialize the optimizer.

        Args:
            race_state: Canonical race state with ``driver``, ``lap``,
                ``lap_time``, ``compound``, ``tyre_age`` columns.
            tyre_model: Tyre degradation model for projections.
                Uses defaults if ``None``.
            pit_loss: Time lost per pit stop (seconds).
            noise_std: Noise for projections (0 = deterministic).
        """
        self.race_state = race_state.copy()
        self.total_laps = int(race_state["lap"].max())
        self.tyre_model = tyre_model or TyreDegradationModel()
        self.pit_loss = pit_loss
        self.noise_std = noise_std
        logger.info(
            "PitWindowOptimizer initialized — %d laps, pit_loss=%.1fs",
            self.total_laps, self.pit_loss,
        )

    def optimize_single_stop(
        self,
        driver: str,
        decision_lap: int,
        compounds: Optional[list[str]] = None,
        window: Optional[tuple[int, int]] = None,
    ) -> list[StrategyCandidate]:
        """Find the best single pit-stop strategy.

        Sweeps all feasible pit laps within the window and evaluates
        each (pit_lap, post-pit compound) combination.

        Args:
            driver: Three-letter driver abbreviation.
            decision_lap: Lap at which the decision is made.
            compounds: Compounds to consider for the second stint.
                Defaults to all dry compounds.
            window: (earliest_lap, latest_lap) to consider for pitting.
                Defaults to ``(decision_lap, total_laps - MIN_STINT_LAPS)``.

        Returns:
            List of ``StrategyCandidate``s sorted by total time (best first).
        """
        if compounds is None:
            compounds = DRY_COMPOUNDS

        baseline_pace = self._get_baseline_pace(driver, decision_lap)
        current_state = self._get_driver_state(driver, decision_lap)
        current_compound = str(current_state["compound"].iloc[0])
        current_age = float(current_state["tyre_age"].iloc[0])

        # Define search window
        earliest = window[0] if window else decision_lap
        latest = window[1] if window else (self.total_laps - MIN_STINT_LAPS)
        earliest = max(earliest, decision_lap)
        latest = min(latest, self.total_laps - MIN_STINT_LAPS)

        logger.info(
            "Optimizing single-stop for %s from lap %d (window: %d–%d)",
            driver, decision_lap, earliest, latest,
        )

        candidates: list[StrategyCandidate] = []
        other_totals = self._project_others_total(driver, decision_lap)

        for pit_lap in range(earliest, latest + 1):
            for post_compound in compounds:
                # Skip same compound on same compound (regulation usually requires change)
                # but allow it — the user can filter downstream
                total_time = self._project_strategy_time(
                    baseline_pace=baseline_pace,
                    current_age=current_age,
                    current_compound=current_compound,
                    decision_lap=decision_lap,
                    pit_laps=[pit_lap],
                    stint_compounds=[current_compound, post_compound],
                )
                stint_1 = pit_lap - decision_lap
                stint_2 = self.total_laps - pit_lap
                expected_finish = self._estimate_position(total_time, other_totals)
                risk = self._compute_risk(
                    stint_lengths=[stint_1, stint_2],
                    compounds=[current_compound, post_compound],
                )

                candidates.append(StrategyCandidate(
                    pit_laps=[pit_lap],
                    compounds=[current_compound, post_compound],
                    total_time=round(total_time, 3),
                    expected_finish=expected_finish,
                    risk=round(risk, 4),
                    stint_lengths=[stint_1, stint_2],
                ))

        # Sort by total time and compute deltas
        candidates.sort(key=lambda c: c.total_time)
        if candidates:
            best_time = candidates[0].total_time
            for c in candidates:
                c.time_delta = round(c.total_time - best_time, 3)

        logger.info(
            "Single-stop optimization complete — %d candidates evaluated",
            len(candidates),
        )
        return candidates

    def optimize_two_stop(
        self,
        driver: str,
        decision_lap: int,
        compounds: Optional[list[str]] = None,
        window: Optional[tuple[int, int]] = None,
        step: int = 2,
    ) -> list[StrategyCandidate]:
        """Find the best two pit-stop strategy.

        Evaluates all feasible (pit1, pit2, compound_sequence) combinations
        with a configurable step size to reduce search space.

        Args:
            driver: Three-letter driver abbreviation.
            decision_lap: Lap at which the decision is made.
            compounds: Compounds to consider for stints 2 and 3.
            window: (earliest, latest) overall pit window.
            step: Lap increment for the search grid (default 2).

        Returns:
            List of ``StrategyCandidate``s sorted by total time.
        """
        if compounds is None:
            compounds = DRY_COMPOUNDS

        baseline_pace = self._get_baseline_pace(driver, decision_lap)
        current_state = self._get_driver_state(driver, decision_lap)
        current_compound = str(current_state["compound"].iloc[0])
        current_age = float(current_state["tyre_age"].iloc[0])

        earliest = window[0] if window else decision_lap
        latest = window[1] if window else (self.total_laps - MIN_STINT_LAPS)
        earliest = max(earliest, decision_lap)
        latest = min(latest, self.total_laps - MIN_STINT_LAPS)

        logger.info(
            "Optimizing two-stop for %s from lap %d (window: %d–%d, step=%d)",
            driver, decision_lap, earliest, latest, step,
        )

        candidates: list[StrategyCandidate] = []
        other_totals = self._project_others_total(driver, decision_lap)

        for pit1 in range(earliest, latest + 1, step):
            for pit2 in range(pit1 + MIN_STINT_LAPS, latest + 1, step):
                if self.total_laps - pit2 < MIN_STINT_LAPS:
                    continue

                for comp2 in compounds:
                    for comp3 in compounds:
                        stint_compounds = [current_compound, comp2, comp3]
                        total_time = self._project_strategy_time(
                            baseline_pace=baseline_pace,
                            current_age=current_age,
                            current_compound=current_compound,
                            decision_lap=decision_lap,
                            pit_laps=[pit1, pit2],
                            stint_compounds=stint_compounds,
                        )
                        stint_1 = pit1 - decision_lap
                        stint_2 = pit2 - pit1
                        stint_3 = self.total_laps - pit2

                        expected_finish = self._estimate_position(total_time, other_totals)
                        risk = self._compute_risk(
                            stint_lengths=[stint_1, stint_2, stint_3],
                            compounds=stint_compounds,
                        )

                        candidates.append(StrategyCandidate(
                            pit_laps=[pit1, pit2],
                            compounds=stint_compounds,
                            total_time=round(total_time, 3),
                            expected_finish=expected_finish,
                            risk=round(risk, 4),
                            stint_lengths=[stint_1, stint_2, stint_3],
                        ))

        candidates.sort(key=lambda c: c.total_time)
        if candidates:
            best_time = candidates[0].total_time
            for c in candidates:
                c.time_delta = round(c.total_time - best_time, 3)

        logger.info(
            "Two-stop optimization complete — %d candidates evaluated",
            len(candidates),
        )
        return candidates

    def find_optimal(
        self,
        driver: str,
        decision_lap: int,
        max_stops: int = 2,
        top_n: int = 5,
        compounds: Optional[list[str]] = None,
    ) -> dict[str, list[StrategyCandidate]]:
        """Find the overall optimal strategy across 1-stop and 2-stop.

        Args:
            driver: Driver abbreviation.
            decision_lap: Current lap.
            max_stops: Maximum number of stops to evaluate (1 or 2).
            top_n: Number of top strategies to return per category.
            compounds: Compounds to evaluate.

        Returns:
            Dictionary with keys ``"1_stop"`` and optionally ``"2_stop"``,
            each mapping to the top N strategies. Also includes
            ``"best_overall"`` with the single best strategy.
        """
        results: dict[str, list[StrategyCandidate]] = {}

        # 1-stop strategies
        one_stop = self.optimize_single_stop(
            driver, decision_lap, compounds=compounds,
        )
        results["1_stop"] = one_stop[:top_n]

        # 2-stop strategies
        if max_stops >= 2:
            two_stop = self.optimize_two_stop(
                driver, decision_lap, compounds=compounds,
            )
            results["2_stop"] = two_stop[:top_n]

        # Find overall best
        all_candidates = one_stop + (two_stop if max_stops >= 2 else [])
        if all_candidates:
            best = min(all_candidates, key=lambda c: c.total_time)
            # Recompute deltas relative to overall best
            for category in results.values():
                for c in category:
                    c.time_delta = round(c.total_time - best.total_time, 3)
            results["best_overall"] = [best]

        return results

    def summary_table(
        self,
        candidates: list[StrategyCandidate],
    ) -> pd.DataFrame:
        """Convert a list of candidates to a summary DataFrame.

        Args:
            candidates: List of strategy candidates.

        Returns:
            DataFrame with one row per candidate, sorted by total time.
        """
        rows = []
        for i, c in enumerate(candidates):
            rows.append({
                "rank": i + 1,
                "stops": len(c.pit_laps),
                "pit_laps": ", ".join(map(str, c.pit_laps)),
                "compounds": " → ".join(c.compounds),
                "stints": " / ".join(map(str, c.stint_lengths)),
                "total_time": c.total_time,
                "delta": f"+{c.time_delta:.1f}s" if c.time_delta > 0 else "BEST",
                "finish": c.expected_finish,
                "risk": c.risk,
            })
        return pd.DataFrame(rows)

    # ──────────────────────────────────────────────────────
    # Internal projection methods
    # ──────────────────────────────────────────────────────

    def _project_strategy_time(
        self,
        baseline_pace: float,
        current_age: float,
        current_compound: str,
        decision_lap: int,
        pit_laps: list[int],
        stint_compounds: list[str],
    ) -> float:
        """Project total remaining race time for a multi-stint strategy.

        Builds the stint schedule from pit_laps and compounds, then
        sums degradation-adjusted lap times across all stints.

        Args:
            baseline_pace: Clean-air baseline lap time.
            current_age: Current tyre age in laps.
            current_compound: Current tyre compound.
            decision_lap: Lap at which the decision is made.
            pit_laps: Ordered list of pit lap numbers.
            stint_compounds: Compound for each stint (len = stops + 1).

        Returns:
            Total projected time for the remaining race (seconds).
        """
        rng = np.random.default_rng(seed=42)
        total = 0.0

        # Build stint boundaries
        boundaries = [decision_lap] + pit_laps + [self.total_laps]
        tyre_age = current_age

        for stint_idx in range(len(boundaries) - 1):
            stint_start = boundaries[stint_idx]
            stint_end = boundaries[stint_idx + 1]
            stint_length = stint_end - stint_start
            compound = stint_compounds[stint_idx]

            # Reset tyre age after pit stops (except first stint)
            if stint_idx > 0:
                tyre_age = 1.0
                total += self.pit_loss  # add pit time loss

            for lap_offset in range(stint_length):
                deg = self.tyre_model.degradation(compound, tyre_age)
                noise = rng.normal(0, self.noise_std) if self.noise_std > 0 else 0.0

                # Fresh tyre bonus (decays over first ~10 laps)
                fresh_bonus = 0.0
                if stint_idx > 0 and tyre_age <= 10:
                    fresh_adv = self.tyre_model.fresh_tyre_advantage(compound)
                    fresh_bonus = max(0.0, fresh_adv * (1.0 - tyre_age / 10.0))

                total += baseline_pace + deg + noise - fresh_bonus
                tyre_age += 1.0

        return total

    def _project_others_total(
        self,
        exclude_driver: str,
        decision_lap: int,
    ) -> list[float]:
        """Project total remaining time for other drivers (stay-out).

        Args:
            exclude_driver: Driver to exclude.
            decision_lap: Current lap number.

        Returns:
            List of projected totals for each other driver.
        """
        remaining = self.total_laps - decision_lap
        totals: list[float] = []

        for drv in self.race_state["driver"].unique():
            if drv == exclude_driver:
                continue
            try:
                state = self._get_driver_state(drv, decision_lap)
                pace = self._get_baseline_pace(drv, decision_lap)
                age = float(state["tyre_age"].iloc[0]) if pd.notna(state["tyre_age"].iloc[0]) else 10.0
                compound = str(state["compound"].iloc[0]) if "compound" in state.columns else "MEDIUM"
            except (ValueError, IndexError):
                continue

            time = self._project_strategy_time(
                baseline_pace=pace,
                current_age=age,
                current_compound=compound,
                decision_lap=decision_lap,
                pit_laps=[],
                stint_compounds=[compound],
            )
            totals.append(time)

        return totals

    def _estimate_position(
        self,
        driver_total: float,
        other_totals: list[float],
    ) -> float:
        """Estimate finishing position given projected times.

        Args:
            driver_total: Driver's projected total time.
            other_totals: Other drivers' projected totals.

        Returns:
            Expected finishing position (1-indexed).
        """
        return 1.0 + sum(1 for t in other_totals if t < driver_total)

    def _compute_risk(
        self,
        stint_lengths: list[int],
        compounds: list[str],
    ) -> float:
        """Compute a risk score for a strategy.

        Risk factors:
        - Very short stints (pit stop variability dominates)
        - Very long stints (cliff risk)
        - Number of pit stops (each adds execution risk)
        - Soft compound in long stints

        Args:
            stint_lengths: Length of each stint in laps.
            compounds: Compound for each stint.

        Returns:
            Risk score ∈ [0, 1].
        """
        risk = 0.0
        n_stops = len(stint_lengths) - 1

        # Pit execution risk: ~0.05 per stop
        risk += n_stops * 0.05

        for length, compound in zip(stint_lengths, compounds):
            cliff = self.tyre_model.cliff_onset_lap(compound)

            # Cliff risk: running beyond or near cliff onset
            if length > cliff:
                risk += 0.15 * ((length - cliff) / cliff)
            elif length > cliff * 0.8:
                risk += 0.05

            # Short stint risk: too short to benefit from fresh tyres
            if length < MIN_STINT_LAPS + 2:
                risk += 0.08

            # Soft on long stint risk
            if compound == "SOFT" and length > 20:
                risk += 0.10

        return min(1.0, max(0.0, risk))

    def _get_driver_state(self, driver: str, lap: int) -> pd.DataFrame:
        """Retrieve driver state at a specific lap."""
        mask = (self.race_state["driver"] == driver) & (self.race_state["lap"] == lap)
        state = self.race_state.loc[mask]
        if state.empty:
            raise ValueError(f"No data for driver={driver}, lap={lap}")
        return state

    def _get_baseline_pace(self, driver: str, lap: int) -> float:
        """Get baseline pace from rolling averages.

        Args:
            driver: Driver abbreviation.
            lap: Current lap number.

        Returns:
            Baseline lap time in seconds.
        """
        state = self._get_driver_state(driver, lap)
        for col in ["rolling_5lap_mean", "rolling_3lap_mean", "lap_time"]:
            if col in state.columns:
                val = state[col].iloc[0]
                if pd.notna(val) and val > 0:
                    return float(val)

        driver_times = self.race_state.loc[
            self.race_state["driver"] == driver, "lap_time"
        ].dropna()
        return float(driver_times.median()) if not driver_times.empty else 90.0
