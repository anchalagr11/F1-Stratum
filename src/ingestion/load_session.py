"""
STRATUM-F1 — Data Ingestion Module

Loads Formula 1 race session data using the FastF1 library,
extracts key datasets (laps, weather, track status, results),
and persists them as parquet files for downstream processing.
"""

import logging
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd

logger = logging.getLogger(__name__)

# Project root for data storage (two levels up from this file → src/ingestion/)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DATA_DIR = _PROJECT_ROOT / "data" / "raw"


def _ensure_cache() -> Path:
    """Create and return the FastF1 cache directory."""
    cache_dir = _PROJECT_ROOT / ".fastf1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _sanitize_name(name: str) -> str:
    """Sanitize a string for safe filesystem usage."""
    return name.lower().replace(" ", "_").replace("-", "_")


def load_race_session(year: int, gp: str) -> dict[str, pd.DataFrame]:
    """Load a Formula 1 race session and extract core datasets.

    Uses FastF1 to fetch session data, extracts laps, weather,
    track status, and classification results, then writes each
    dataset to parquet in ``data/raw/<year>_<gp>/``.

    Args:
        year: Championship season year (e.g. 2024).
        gp: Grand Prix identifier accepted by FastF1
            (e.g. ``"Singapore"``, ``"Monza"``).

    Returns:
        Dictionary with keys ``"laps"``, ``"weather"``,
        ``"track_status"``, ``"results"`` mapping to DataFrames.

    Raises:
        ValueError: If the session cannot be loaded or contains no lap data.
        RuntimeError: If data persistence fails.
    """
    logger.info("Loading race session: %d %s", year, gp)

    # Configure FastF1 cache
    cache_dir = _ensure_cache()
    fastf1.Cache.enable_cache(str(cache_dir))

    # Load the race session
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load()
    except Exception as exc:
        logger.error("Failed to load session %d %s: %s", year, gp, exc)
        raise ValueError(
            f"Could not load race session for {year} {gp}. "
            f"Check that the year/GP combination is valid."
        ) from exc

    # Extract datasets
    laps = _extract_laps(session)
    weather = _extract_weather(session)
    track_status = _extract_track_status(session)
    results = _extract_results(session)

    session_data: dict[str, pd.DataFrame] = {
        "laps": laps,
        "weather": weather,
        "track_status": track_status,
        "results": results,
    }

    # Persist raw data
    _save_raw_data(session_data, year, gp)

    logger.info(
        "Session loaded — laps: %d rows, weather: %d rows, "
        "track_status: %d rows, results: %d rows",
        len(laps), len(weather), len(track_status), len(results),
    )
    return session_data


def _extract_laps(session: Any) -> pd.DataFrame:
    """Extract lap-level data from the session.

    Args:
        session: A loaded FastF1 session object.

    Returns:
        DataFrame containing all laps with relevant columns.

    Raises:
        ValueError: If the session contains no lap data.
    """
    laps = session.laps
    if laps is None or laps.empty:
        raise ValueError("Session contains no lap data.")

    # Select useful columns (keep all available; downstream will filter)
    keep_cols = [
        "Driver", "DriverNumber", "LapNumber", "LapTime", "Sector1Time",
        "Sector2Time", "Sector3Time", "Compound", "TyreLife", "FreshTyre",
        "Stint", "Position", "PitInTime", "PitOutTime", "IsPersonalBest",
        "Team", "TrackStatus",
    ]
    available = [c for c in keep_cols if c in laps.columns]
    return laps[available].reset_index(drop=True)


def _extract_weather(session: Any) -> pd.DataFrame:
    """Extract weather telemetry from the session.

    Args:
        session: A loaded FastF1 session object.

    Returns:
        DataFrame of weather readings; empty DataFrame if unavailable.
    """
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            logger.warning("No weather data available for this session.")
            return pd.DataFrame()
        return weather.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Could not extract weather data: %s", exc)
        return pd.DataFrame()


def _extract_track_status(session: Any) -> pd.DataFrame:
    """Extract track status messages (flags, VSC, SC, red flag).

    Args:
        session: A loaded FastF1 session object.

    Returns:
        DataFrame of track status events; empty DataFrame if unavailable.
    """
    try:
        track_status = session.track_status
        if track_status is None or track_status.empty:
            logger.warning("No track status data available.")
            return pd.DataFrame()
        return track_status.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Could not extract track status: %s", exc)
        return pd.DataFrame()


def _extract_results(session: Any) -> pd.DataFrame:
    """Extract final classification / results.

    Args:
        session: A loaded FastF1 session object.

    Returns:
        DataFrame of race results; empty DataFrame if unavailable.
    """
    try:
        results = session.results
        if results is None or results.empty:
            logger.warning("No results data available.")
            return pd.DataFrame()
        return results.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Could not extract results: %s", exc)
        return pd.DataFrame()


def _save_raw_data(
    session_data: dict[str, pd.DataFrame],
    year: int,
    gp: str,
) -> None:
    """Persist raw DataFrames as parquet files.

    Files are written to ``data/raw/<year>_<gp>/<key>.parquet``.

    Args:
        session_data: Dictionary of DataFrames keyed by dataset name.
        year: Championship season year.
        gp: Grand Prix name.

    Raises:
        RuntimeError: If any file fails to write.
    """
    race_dir = _RAW_DATA_DIR / f"{year}_{_sanitize_name(gp)}"
    race_dir.mkdir(parents=True, exist_ok=True)

    for key, df in session_data.items():
        out_path = race_dir / f"{key}.parquet"
        try:
            # Convert timedelta columns to total-seconds floats for parquet
            df_out = _convert_timedeltas(df.copy())
            df_out.to_parquet(out_path, index=False, engine="pyarrow")
            logger.debug("Saved %s → %s (%d rows)", key, out_path, len(df_out))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write {out_path}: {exc}"
            ) from exc

    logger.info("Raw data saved to %s", race_dir)


def _convert_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timedelta columns to float seconds for parquet compatibility.

    Args:
        df: Input DataFrame (modified in place and returned).

    Returns:
        DataFrame with timedelta columns replaced by float seconds.
    """
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    return df
