"""
STRATUM-F1 — GenAI Reasoning Adapter

Translates canonical race state, feature engineering, and simulation
data into a structured, natural-language prompt optimized for Large
Language Models (LLMs) to act as a Race Strategist.
"""

import logging
from typing import Optional

import pandas as pd

from ..simulation.optimizer import StrategyCandidate
from ..simulation.tyre_model import TyreDegradationModel

logger = logging.getLogger(__name__)


class StrategyPromptBuilder:
    """Builds a comprehensive LLM prompt from race data.

    Constructs a 'Race Briefing' document containing:
    - Current race context (lap, track status)
    - Target driver status (position, gaps, tyre age, pace)
    - Rival context (who is ahead, who is behind, undercut risks)
    - Optimizer recommendations (1-stop and 2-stop projections)

    This text can be fed into an LLM along with a system prompt
    asking it to reason about the tactical situation.
    """

    def __init__(
        self,
        race_state: pd.DataFrame,
        tyre_model: TyreDegradationModel,
    ) -> None:
        """Initialize the prompt builder.

        Args:
            race_state: Enriched canonical race state DataFrame.
            tyre_model: Fitted tyre degradation model.
        """
        self.race_state = race_state
        self.tyre_model = tyre_model
        self.total_laps = int(race_state["lap"].max())

    def build_prompt(
        self,
        driver: str,
        lap: int,
        candidates_1_stop: list[StrategyCandidate],
        candidates_2_stop: list[StrategyCandidate],
        best_overall: StrategyCandidate,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate the complete markdown prompt.

        Args:
            driver: Target driver abbreviation.
            lap: Current lap number.
            candidates_1_stop: Top 1-stop strategies from optimizer.
            candidates_2_stop: Top 2-stop strategies from optimizer.
            best_overall: The single best strategy.
            system_prompt: Optional system persona prompt to prepend.

        Returns:
            Formatted markdown string ready for LLM consumption.
        """
        logger.info("Building LLM strategy prompt for %s (Lap %d)", driver, lap)

        parts = []

        if system_prompt:
            parts.append(system_prompt)
            parts.append("---\n")

        parts.append(self._build_header(driver, lap))
        parts.append(self._build_driver_context(driver, lap))
        parts.append(self._build_rival_context(driver, lap))
        parts.append(self._build_tyre_context())
        parts.append(self._build_optimizer_context(
            candidates_1_stop, candidates_2_stop, best_overall
        ))

        parts.append("\n## Request to Strategist\n")
        parts.append(
            "Based on the data above, what is your strategic recommendation? "
            "Consider traffic risks, the tyre degradation curve, and the "
            "actions of nearby rivals. Provide a step-by-step reasoning "
            "followed by your final decision: should we PIT NOW, or STAY OUT?"
        )

        return "\n".join(parts)

    def _get_row(self, driver: str, lap: int) -> pd.Series:
        """Helper to get a specific driver's lap data."""
        mask = (self.race_state["driver"] == driver) & (self.race_state["lap"] == lap)
        df = self.race_state[mask]
        if df.empty:
            raise ValueError(f"No data for {driver} on lap {lap}")
        return df.iloc[0]

    def _build_header(self, driver: str, lap: int) -> str:
        """Build the document header."""
        remaining = self.total_laps - lap
        return (
            f"# RACEDAY STRATEGY BRIEFING\n\n"
            f"**Target Driver:** {driver}\n"
            f"**Current Lap:** {lap} / {self.total_laps} ({remaining} laps remaining)\n"
        )

    def _build_driver_context(self, driver: str, lap: int) -> str:
        """Build context about the target driver's current state."""
        row = self._get_row(driver, lap)
        
        pos = int(row["position"]) if pd.notna(row["position"]) else "Unknown"
        compound = str(row.get("compound", "UNKNOWN"))
        age = float(row.get("tyre_age", 0.0))
        pace = row.get("rolling_3lap_mean", row.get("lap_time", "N/A"))
        if pd.notna(pace):
            pace = f"{pace:.3f}s"
        
        gap = row.get("gap_ahead", "N/A")
        if pd.notna(gap):
            gap = f"{gap:.3f}s"
            
        traffic = row.get("traffic_penalty", 0.0)
        dirty_air = "Yes" if traffic > 0 else "No"
        
        undercut = row.get("undercut_estimate", 0.0)

        return (
            f"## 1. Driver Status\n"
            f"- **Position:** P{pos}\n"
            f"- **Current Tyres:** {compound} (Age: {age:.0f} laps)\n"
            f"- **Recent Pace:** {pace} (3-lap rolling average)\n"
            f"- **Gap to Car Ahead:** {gap}\n"
            f"- **In Dirty Air?** {dirty_air} (Penalty: +{traffic:.2f}s/lap)\n"
            f"- **Estimated Undercut Power:** +{undercut:.2f}s (if we pit now)\n"
        )

    def _build_rival_context(self, driver: str, lap: int) -> str:
        """Build context about drivers immediately ahead and behind."""
        current_lap = self.race_state[self.race_state["lap"] == lap].sort_values("position")
        if current_lap.empty:
            return "## 2. Rival Context\nNo data available for this lap.\n"

        target_row = current_lap[current_lap["driver"] == driver]
        if target_row.empty:
            return "## 2. Rival Context\nTarget driver not found in standings.\n"
            
        pos = int(target_row.iloc[0]["position"])
        
        lines = ["## 2. Rival Context\n"]
        
        # Car Ahead
        if pos > 1:
            ahead = current_lap[current_lap["position"] == pos - 1].iloc[0]
            lines.append(
                f"- **P{pos-1} (Ahead):** {ahead['driver']} "
                f"on {ahead.get('compound', 'N/A')} (Age: {ahead.get('tyre_age', 0):.0f}). "
                f"Pace: {ahead.get('rolling_3lap_mean', ahead['lap_time']):.3f}s"
            )
        else:
            lines.append("- **Ahead:** Clean air (Race Leader).")

        # Car Behind
        if pos < len(current_lap):
            behind = current_lap[current_lap["position"] == pos + 1].iloc[0]
            gap_behind = target_row.iloc[0].get("gap_ahead", 0) # approximation, FastF1 telemetry has actual gap, we use inverse
            # Actually canonical state gap_ahead is relative to the car ahead of the row driver.
            # To get gap BEHIND us, we look at the gap_ahead of the driver P(pos+1)
            actual_gap_behind = behind.get("gap_ahead", "N/A")
            if pd.notna(actual_gap_behind):
                actual_gap_behind = f"{actual_gap_behind:.3f}s"
                
            lines.append(
                f"- **P{pos+1} (Behind):** {behind['driver']} "
                f"on {behind.get('compound', 'N/A')} (Age: {behind.get('tyre_age', 0):.0f}). "
                f"Gap behind us: {actual_gap_behind}. "
                f"Pace: {behind.get('rolling_3lap_mean', behind['lap_time']):.3f}s"
            )
            
        return "\n".join(lines) + "\n"

    def _build_tyre_context(self) -> str:
        """Build context about the tyre degradation model."""
        summary = self.tyre_model.summary()
        
        lines = ["## 3. Tyre Model Data (Fitted from Race)"]
        lines.append("Degradation penalty added to lap time based on tyre age:\n")
        
        for _, row in summary.iterrows():
            comp = row['compound']
            deg10 = row['deg_at_10']
            deg20 = row['deg_at_20']
            cliff = row['cliff_onset']
            lines.append(
                f"- **{comp}**: +{deg10:.2f}s at L10 | +{deg20:.2f}s at L20 "
                f"| Cliff Drop-off at Lap {cliff:.0f}"
            )
            
        return "\n".join(lines) + "\n"

    def _build_optimizer_context(
        self,
        candidates_1_stop: list[StrategyCandidate],
        candidates_2_stop: list[StrategyCandidate],
        best: StrategyCandidate,
    ) -> str:
        """Build context from the simulation optimizer results."""
        lines = ["## 4. Optimizer Projections"]
        
        if not candidates_1_stop and not candidates_2_stop:
            lines.append("No viable pit strategies found (race ends soon or under safety car).")
            return "\n".join(lines) + "\n"

        lines.append("The simulation engine has evaluated all possible future pit windows.\n")
        
        lines.append(f"**RECOMMENDED: {len(best.pit_laps)}-STOP STRATEGY**")
        lines.append(f"- **Pits at:** Laps {', '.join(map(str, best.pit_laps))}")
        lines.append(f"- **Compounds:** {' → '.join(best.compounds)}")
        lines.append(f"- **Expected Finish:** P{best.expected_finish:.0f}")
        lines.append(f"- **Risk Score:** {best.risk:.3f} (0=Safe, 1=High Risk)\n")
        
        lines.append("**Top Alternatives:**")
        
        alt_count = 0
        for c in (candidates_1_stop + candidates_2_stop):
            if c.total_time == best.total_time:
                continue # Skip the best one which we already printed
            if alt_count >= 3:
                break
            
            lines.append(
                f"- {len(c.pit_laps)}-Stop (Pits: {', '.join(map(str, c.pit_laps))}, "
                f"{' → '.join(c.compounds)}) | Delta: +{c.time_delta:.1f}s "
                f"| Project: P{c.expected_finish:.0f} | Risk: {c.risk:.3f}"
            )
            alt_count += 1
            
        return "\n".join(lines) + "\n"
