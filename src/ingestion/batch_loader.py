"""
STRATUM-F1 — Multi-Race Batch Ingestion

Provides utilities to load multiple race sessions across seasons,
build a combined historical dataset, and track ingestion progress.
Reuses the single-session loader from ``load_session.py``.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .load_session import load_race_session, _PROJECT_ROOT, _sanitize_name

logger = logging.getLogger(__name__)

_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# ──────────────────────────────────────────────────────────────
# 2024 F1 Calendar (fallback schedule when FastF1 is unavailable)
# Extend or replace with other seasons as needed.
# ──────────────────────────────────────────────────────────────

SEASON_SCHEDULES: dict[int, list[str]] = {
    2023: [
        "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan",
        "Miami", "Monaco", "Spain", "Canada", "Austria",
        "Great Britain", "Hungary", "Belgium", "Netherlands",
        "Italy", "Singapore", "Japan", "Qatar", "United States",
        "Mexico", "São Paulo", "Las Vegas", "Abu Dhabi",
    ],
    2024: [
        "Bahrain", "Saudi Arabia", "Australia", "Japan",
        "China", "Miami", "Emilia Romagna", "Monaco", "Canada",
        "Spain", "Austria", "Great Britain", "Hungary", "Belgium",
        "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico", "São Paulo", "Las Vegas",
        "Qatar", "Abu Dhabi",
    ],
}


@dataclass
class IngestionResult:
    """Outcome of a single session ingestion attempt."""

    year: int
    gp: str
    race_id: str
    status: str  # "success" | "skipped" | "failed"
    laps_count: int = 0
    error: Optional[str] = None
    elapsed_sec: float = 0.0


@dataclass
class BatchIngestionReport:
    """Summary report for a batch ingestion run."""

    started_at: str = ""
    completed_at: str = ""
    total_requested: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[IngestionResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """Log a human-readable summary of the batch run."""
        logger.info("=" * 60)
        logger.info("BATCH INGESTION REPORT")
        logger.info("=" * 60)
        logger.info("  Started:   %s", self.started_at)
        logger.info("  Completed: %s", self.completed_at)
        logger.info(
            "  Total: %d | Success: %d | Skipped: %d | Failed: %d",
            self.total_requested, self.succeeded, self.skipped, self.failed,
        )
        for r in self.results:
            icon = "✓" if r.status == "success" else ("⊘" if r.status == "skipped" else "✗")
            detail = f"{r.laps_count} laps" if r.status == "success" else (r.error or "")
            logger.info("  %s %-30s %s (%.1fs)", icon, r.race_id, detail, r.elapsed_sec)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the report as JSON.

        Args:
            path: Output path. Defaults to ``data/processed/ingestion_report.json``.

        Returns:
            Path the report was saved to.
        """
        if path is None:
            _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            path = _PROCESSED_DIR / "ingestion_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info("Report saved to %s", path)
        return path


def get_season_races(year: int) -> list[str]:
    """Return the list of Grand Prix names for a given season.

    Falls back to a hardcoded schedule if the year is known.

    Args:
        year: Championship season (e.g. 2024).

    Returns:
        List of GP names suitable for ``load_race_session``.

    Raises:
        ValueError: If the season schedule is not available.
    """
    if year in SEASON_SCHEDULES:
        return SEASON_SCHEDULES[year]
    raise ValueError(
        f"No schedule available for {year}. "
        f"Known seasons: {sorted(SEASON_SCHEDULES.keys())}. "
        f"Add it to SEASON_SCHEDULES or pass explicit gp_list."
    )


def load_multiple_races(
    year: int,
    gp_list: Optional[list[str]] = None,
    skip_existing: bool = True,
) -> tuple[dict[str, dict[str, pd.DataFrame]], BatchIngestionReport]:
    """Load multiple race sessions for a given season.

    Args:
        year: Championship season year.
        gp_list: Explicit list of GP names to load. If ``None``,
            loads the full season from ``SEASON_SCHEDULES``.
        skip_existing: If ``True``, skip GPs whose raw parquet
            directory already exists.

    Returns:
        Tuple of:
        - Dictionary mapping ``race_id`` → session_data dict.
        - ``BatchIngestionReport`` with per-race outcomes.
    """
    if gp_list is None:
        gp_list = get_season_races(year)

    report = BatchIngestionReport(
        started_at=datetime.now().isoformat(),
        total_requested=len(gp_list),
    )
    all_sessions: dict[str, dict[str, pd.DataFrame]] = {}

    logger.info("Starting batch ingestion: %d %s (%d races)", year, "season", len(gp_list))

    for gp in gp_list:
        race_id = f"{year}_{_sanitize_name(gp)}"
        raw_dir = _PROJECT_ROOT / "data" / "raw" / race_id

        # Skip if data already exists
        if skip_existing and raw_dir.exists() and any(raw_dir.glob("*.parquet")):
            logger.info("Skipping %s (already exists)", race_id)
            report.results.append(IngestionResult(
                year=year, gp=gp, race_id=race_id, status="skipped",
            ))
            report.skipped += 1
            continue

        # Attempt ingestion
        t0 = time.time()
        try:
            session_data = load_race_session(year=year, gp=gp)
            elapsed = time.time() - t0
            laps_count = len(session_data.get("laps", pd.DataFrame()))

            all_sessions[race_id] = session_data
            report.results.append(IngestionResult(
                year=year, gp=gp, race_id=race_id, status="success",
                laps_count=laps_count, elapsed_sec=round(elapsed, 2),
            ))
            report.succeeded += 1
            logger.info("✓ %s — %d laps (%.1fs)", race_id, laps_count, elapsed)

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("✗ %s — %s (%.1fs)", race_id, exc, elapsed)
            report.results.append(IngestionResult(
                year=year, gp=gp, race_id=race_id, status="failed",
                error=str(exc), elapsed_sec=round(elapsed, 2),
            ))
            report.failed += 1

    report.completed_at = datetime.now().isoformat()
    report.print_summary()
    report.save()

    return all_sessions, report


def build_historical_dataset(
    years: list[int],
    gp_list: Optional[list[str]] = None,
    skip_existing: bool = True,
) -> tuple[dict[str, dict[str, pd.DataFrame]], list[BatchIngestionReport]]:
    """Load races across multiple seasons and return a combined dataset.

    Args:
        years: List of season years to load (e.g. ``[2023, 2024]``).
        gp_list: If provided, load only these GPs for each season.
            If ``None``, loads the full calendar for each year.
        skip_existing: Skip races whose raw data already exists.

    Returns:
        Tuple of:
        - Combined dictionary mapping ``race_id`` → session_data.
        - List of ``BatchIngestionReport``s, one per season.
    """
    all_sessions: dict[str, dict[str, pd.DataFrame]] = {}
    all_reports: list[BatchIngestionReport] = []

    for year in years:
        logger.info("━" * 60)
        logger.info("SEASON %d", year)
        logger.info("━" * 60)

        sessions, report = load_multiple_races(
            year=year, gp_list=gp_list, skip_existing=skip_existing,
        )
        all_sessions.update(sessions)
        all_reports.append(report)

    total_races = sum(r.succeeded for r in all_reports)
    logger.info(
        "Historical dataset complete — %d seasons, %d races loaded",
        len(years), total_races,
    )
    return all_sessions, all_reports
