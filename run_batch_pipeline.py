#!/usr/bin/env python3
"""
STRATUM-F1 — Multi-Race Pipeline Script

Loads multiple races (or full seasons), builds canonical race states
for each, and optionally runs simulation across all of them.

Usage:
    # Load a few specific races from 2024
    python run_batch_pipeline.py

    # Modify RACES_TO_LOAD below for different configurations
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on the Python path
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.batch_loader import load_multiple_races, build_historical_dataset
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Option A: Load specific races from one season
YEAR: int = 2024
RACES_TO_LOAD: list[str] = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China"]

# Option B: Uncomment below to load full seasons
# MULTI_SEASON_YEARS: list[int] = [2023, 2024]

# Skip races that have already been downloaded
SKIP_EXISTING: bool = True

# Logging
LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"

_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def configure_logging() -> None:
    """Configure root logger for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    """Execute the multi-race pipeline."""
    configure_logging()
    logger = logging.getLogger("stratum-f1.batch_pipeline")

    # ── Step 1: Batch Ingest ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Batch ingestion: %d %s", YEAR, "selected races")
    logger.info("Races: %s", RACES_TO_LOAD)
    logger.info("=" * 60)

    all_sessions, report = load_multiple_races(
        year=YEAR,
        gp_list=RACES_TO_LOAD,
        skip_existing=SKIP_EXISTING,
    )

    logger.info("Loaded %d new sessions (skipped %d, failed %d)",
                report.succeeded, report.skipped, report.failed)

    # ── Step 2: Build canonical states for all races ──────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Building canonical race states")
    logger.info("=" * 60)

    all_race_states: list[pd.DataFrame] = []

    for race_id, session_data in all_sessions.items():
        try:
            race_state = build_race_state(session_data, race_id=race_id)
            all_race_states.append(race_state)
            logger.info("  ✓ %s — %d rows", race_id, len(race_state))
        except Exception as exc:
            logger.error("  ✗ %s — %s", race_id, exc)

    if not all_race_states:
        logger.warning("No race states built. Check ingestion results.")
        return

    # Combine into a single historical DataFrame
    combined = pd.concat(all_race_states, ignore_index=True)
    logger.info("Combined race state: %s", combined.shape)

    # ── Step 3: Feature engineering on combined dataset ────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Adding strategy features to combined dataset")
    logger.info("=" * 60)

    enriched = add_strategy_features(combined)
    logger.info("Enriched shape: %s", enriched.shape)

    # ── Step 4: Save combined dataset ─────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Saving combined historical dataset")
    logger.info("=" * 60)

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / "historical_race_state.parquet"
    enriched.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(enriched), out_path)

    # ── Summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BATCH PIPELINE COMPLETE")
    logger.info("=" * 60)

    races_in_dataset = enriched["race_id"].nunique()
    drivers_in_dataset = enriched["driver"].nunique()
    logger.info("  Races:   %d", races_in_dataset)
    logger.info("  Drivers: %d", drivers_in_dataset)
    logger.info("  Rows:    %d", len(enriched))
    logger.info("  Output:  %s", out_path)


if __name__ == "__main__":
    main()
