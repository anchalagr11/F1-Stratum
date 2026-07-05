#!/usr/bin/env python3
"""
STRATUM-F1 — End-to-End Pipeline Script

Demonstrates the full data pipeline:
1. Load a Formula 1 race session via FastF1
2. Build the canonical race state
3. Add strategy features
4. Run the simulator for a target driver and lap
5. Print results

Usage:
    python run_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on the Python path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.load_session import load_race_session
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features
from src.simulation.simulator import RaceSimulator
from src.simulation.tyre_model import TyreDegradationModel
from src.simulation.optimizer import PitWindowOptimizer
from src.visualization.race_charts import RaceVisualizer

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

YEAR: int = 2024
GP: str = "Singapore"
RACE_ID: str = f"{YEAR}_{GP}"

# Driver and lap for simulation demo
TARGET_DRIVER: str = "NOR"  # Lando Norris
TARGET_LAP: int = 20

# Logging setup
LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"


def configure_logging() -> None:
    """Configure root logger for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Execute the full STRATUM-F1 pipeline."""
    configure_logging()
    logger = logging.getLogger("stratum-f1.pipeline")

    # ── Step 1: Ingest ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading race session: %d %s", YEAR, GP)
    logger.info("=" * 60)

    session_data = load_race_session(year=YEAR, gp=GP)

    for key, df in session_data.items():
        logger.info("  %-15s → %d rows, %d columns", key, len(df), len(df.columns))

    # ── Step 2: Build canonical race state ────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Building canonical race state")
    logger.info("=" * 60)

    race_state = build_race_state(session_data, race_id=RACE_ID)
    logger.info("Race state shape: %s", race_state.shape)
    logger.info("Columns: %s", list(race_state.columns))

    # Show sample
    sample = race_state[race_state["driver"] == TARGET_DRIVER].head(5)
    logger.info("Sample (first 5 laps for %s):\n%s", TARGET_DRIVER, sample.to_string(index=False))

    # ── Step 3: Feature engineering ───────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Adding strategy features")
    logger.info("=" * 60)

    enriched = add_strategy_features(race_state)
    logger.info("Enriched shape: %s", enriched.shape)

    feature_sample = enriched[enriched["driver"] == TARGET_DRIVER][
        ["lap", "driver", "traffic_penalty", "undercut_estimate", "pit_loss_estimate"]
    ].head(5)
    logger.info("Feature sample:\n%s", feature_sample.to_string(index=False))

    # ── Step 4: Simulation ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Running simulation for %s on lap %d", TARGET_DRIVER, TARGET_LAP)
    logger.info("=" * 60)

    # Fit tyre degradation model from the race data
    tyre_model = TyreDegradationModel()
    fit_results = tyre_model.fit_from_race_data(enriched)
    logger.info("Tyre model fit results: %s", fit_results)
    logger.info("Tyre model summary:\n%s", tyre_model.summary().to_string(index=False))

    simulator = RaceSimulator(race_state=enriched, tyre_model=tyre_model)

    actions = ["pit_now", "pit_plus_1", "stay_out"]
    results: list[dict] = []

    for action in actions:
        result = simulator.simulate_strategy(
            driver=TARGET_DRIVER,
            lap=TARGET_LAP,
            action=action,
        )
        results.append(result)
        logger.info("  Action: %-12s  Finish: %.1f  Risk: %.4f",
                     result["action"], result["expected_finish"], result["risk"])

    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SIMULATION RESULTS")
    logger.info("=" * 60)

    best = min(results, key=lambda r: r["expected_finish"])
    logger.info(
        "Recommended action for %s on lap %d: %s (expected P%.1f, risk=%.4f)",
        TARGET_DRIVER, TARGET_LAP, best["action"],
        best["expected_finish"], best["risk"],
    )

    # ── Step 5: Pit Window Optimization ───────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Pit window optimization for %s on lap %d", TARGET_DRIVER, TARGET_LAP)
    logger.info("=" * 60)

    optimizer = PitWindowOptimizer(
        race_state=enriched,
        tyre_model=tyre_model,
    )

    optimal = optimizer.find_optimal(
        driver=TARGET_DRIVER,
        decision_lap=TARGET_LAP,
        max_stops=2,
        top_n=3,
    )

    # Print 1-stop results
    if "1_stop" in optimal and optimal["1_stop"]:
        logger.info("Top 1-stop strategies:")
        table_1 = optimizer.summary_table(optimal["1_stop"])
        logger.info("\n%s", table_1.to_string(index=False))

    # Print 2-stop results
    if "2_stop" in optimal and optimal["2_stop"]:
        logger.info("Top 2-stop strategies:")
        table_2 = optimizer.summary_table(optimal["2_stop"])
        logger.info("\n%s", table_2.to_string(index=False))

    # Print overall best
    if "best_overall" in optimal and optimal["best_overall"]:
        best_strat = optimal["best_overall"][0]
        logger.info(
            "OPTIMAL STRATEGY: %d-stop | Pit laps: %s | Compounds: %s | "
            "Total: %.1fs | Finish: P%.0f | Risk: %.3f",
            len(best_strat.pit_laps),
            best_strat.pit_laps,
            " → ".join(best_strat.compounds),
            best_strat.total_time,
            best_strat.expected_finish,
            best_strat.risk,
        )

    # ── Step 6: Visualization ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Generating analysis charts")
    logger.info("=" * 60)

    visualizer = RaceVisualizer(race_state=enriched, race_id=RACE_ID)
    
    # Collect candidates for strategy comparison chart
    all_candidates = []
    if "1_stop" in optimal:
        all_candidates.extend(optimal["1_stop"])
    if "2_stop" in optimal:
        all_candidates.extend(optimal["2_stop"])
        
    chart_paths = visualizer.generate_all(
        reference_driver=TARGET_DRIVER,
        tyre_model=tyre_model,
        optimizer_candidates=all_candidates,
        target_driver=TARGET_DRIVER,
    )
    for p in chart_paths:
        logger.info("  -> %s", p.name)

    # ── Final Summary ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
