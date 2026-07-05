"""
STRATUM-F1 — Tyre Degradation Model

Models tyre performance degradation as a function of tyre age and
compound type. Supports:
- Default parametric curves (quadratic + cliff) per compound
- Fitting from historical race data via least-squares regression
- Serialization to/from JSON for reuse across sessions
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_DIR = _PROJECT_ROOT / "data" / "features"

# ──────────────────────────────────────────────────────────────
# Compound-level degradation parameterization
# ──────────────────────────────────────────────────────────────
#
# Model:  deg(age) = a * age + b * age² + cliff_penalty(age)
#
#   where cliff_penalty kicks in at cliff_onset laps:
#     cliff_penalty = c * max(0, age - cliff_onset)²
#
# This captures:
#   - Linear wear (a): steady grip loss per lap
#   - Quadratic wear (b): accelerating degradation
#   - Cliff (c, cliff_onset): sudden performance drop on old tyres
# ──────────────────────────────────────────────────────────────


@dataclass
class CompoundParams:
    """Degradation parameters for a single tyre compound.

    Attributes:
        compound: Compound name (SOFT, MEDIUM, HARD, INTERMEDIATE, WET).
        a: Linear degradation coefficient (s/lap).
        b: Quadratic degradation coefficient (s/lap²).
        c: Cliff penalty coefficient (s/lap² past onset).
        cliff_onset: Lap age at which cliff effect begins.
        fresh_delta: Pace advantage on fresh tyres vs. worn (seconds).
    """

    compound: str
    a: float
    b: float
    c: float
    cliff_onset: float
    fresh_delta: float


# Sensible defaults based on typical F1 tyre behaviour
DEFAULT_COMPOUND_PARAMS: dict[str, CompoundParams] = {
    "SOFT": CompoundParams(
        compound="SOFT",
        a=0.065,
        b=0.0020,
        c=0.010,
        cliff_onset=18.0,
        fresh_delta=1.2,
    ),
    "MEDIUM": CompoundParams(
        compound="MEDIUM",
        a=0.045,
        b=0.0010,
        c=0.008,
        cliff_onset=28.0,
        fresh_delta=0.8,
    ),
    "HARD": CompoundParams(
        compound="HARD",
        a=0.030,
        b=0.0005,
        c=0.006,
        cliff_onset=38.0,
        fresh_delta=0.5,
    ),
    "INTERMEDIATE": CompoundParams(
        compound="INTERMEDIATE",
        a=0.055,
        b=0.0015,
        c=0.012,
        cliff_onset=22.0,
        fresh_delta=0.6,
    ),
    "WET": CompoundParams(
        compound="WET",
        a=0.040,
        b=0.0008,
        c=0.010,
        cliff_onset=30.0,
        fresh_delta=0.4,
    ),
}

# Fallback for unknown compounds
_FALLBACK_PARAMS = CompoundParams(
    compound="UNKNOWN",
    a=0.050,
    b=0.0012,
    c=0.008,
    cliff_onset=25.0,
    fresh_delta=0.7,
)


class TyreDegradationModel:
    """Per-compound tyre degradation model.

    Computes the time penalty (in seconds) for a given tyre age and
    compound using a parameterized curve. Can be initialized with
    defaults or fitted from historical data.

    Attributes:
        params: Dictionary mapping compound name → ``CompoundParams``.
        fitted: Whether the model was fitted from data.
    """

    def __init__(
        self,
        params: Optional[dict[str, CompoundParams]] = None,
    ) -> None:
        """Initialize with compound parameters.

        Args:
            params: Custom parameters per compound. Uses defaults
                if ``None``.
        """
        self.params: dict[str, CompoundParams] = (
            params if params is not None
            else dict(DEFAULT_COMPOUND_PARAMS)
        )
        self.fitted: bool = False

    def degradation(self, compound: str, tyre_age: float) -> float:
        """Compute the cumulative degradation penalty for a given age.

        Args:
            compound: Tyre compound name (e.g. ``"SOFT"``).
            tyre_age: Number of laps on current set of tyres.

        Returns:
            Time penalty in seconds added to baseline lap time.
        """
        p = self._get_params(compound)
        return self._deg_function(tyre_age, p.a, p.b, p.c, p.cliff_onset)

    def degradation_rate(self, compound: str, tyre_age: float) -> float:
        """Compute the marginal degradation rate (delta from previous lap).

        Useful for understanding whether the tyre is still usable.

        Args:
            compound: Tyre compound name.
            tyre_age: Current tyre age in laps.

        Returns:
            Marginal degradation from lap (age-1) to lap (age) in seconds.
        """
        if tyre_age <= 1:
            return self.degradation(compound, tyre_age)
        return (
            self.degradation(compound, tyre_age)
            - self.degradation(compound, tyre_age - 1)
        )

    def fresh_tyre_advantage(self, compound: str) -> float:
        """Return the pace advantage on fresh tyres for a given compound.

        Args:
            compound: Tyre compound name.

        Returns:
            Advantage in seconds per lap on brand-new tyres.
        """
        return self._get_params(compound).fresh_delta

    def cliff_onset_lap(self, compound: str) -> float:
        """Return the lap at which the tyre cliff begins.

        Args:
            compound: Tyre compound name.

        Returns:
            Cliff onset age in laps.
        """
        return self._get_params(compound).cliff_onset

    def predict_stint_times(
        self,
        compound: str,
        baseline_pace: float,
        stint_length: int,
        start_age: float = 1.0,
    ) -> np.ndarray:
        """Project lap times over a full stint.

        Args:
            compound: Tyre compound.
            baseline_pace: Clean-air baseline lap time (seconds).
            stint_length: Number of laps in the stint.
            start_age: Starting tyre age (default 1 = fresh).

        Returns:
            Array of projected lap times for each lap of the stint.
        """
        ages = np.arange(start_age, start_age + stint_length)
        times = np.array([
            baseline_pace + self.degradation(compound, age)
            for age in ages
        ])
        return times

    # ──────────────────────────────────────────────────────
    # Fitting from historical data
    # ──────────────────────────────────────────────────────

    def fit_from_race_data(
        self,
        race_state: pd.DataFrame,
        min_samples: int = 20,
    ) -> dict[str, dict]:
        """Fit degradation curves from a canonical race state DataFrame.

        For each compound present in the data, estimates the parameters
        ``(a, b, c, cliff_onset)`` by fitting the observed relationship
        between tyre age and lap-time residuals.

        Args:
            race_state: Canonical race state with columns ``compound``,
                ``tyre_age``, ``lap_time``, ``driver``.
            min_samples: Minimum data points required per compound
                to attempt fitting.

        Returns:
            Dictionary of fit diagnostics per compound (R², n_samples, etc.).
        """
        logger.info("Fitting tyre degradation model from race data")

        # Clean data: drop NaN lap times and anomalous laps
        df = race_state.dropna(subset=["lap_time", "tyre_age", "compound"]).copy()
        df = df[df["lap_time"] > 0]

        # Compute per-driver baseline (median of their fastest 50% of laps)
        driver_baselines = (
            df.groupby("driver")["lap_time"]
            .apply(lambda s: s.nsmallest(max(1, len(s) // 2)).median())
        )
        df["baseline"] = df["driver"].map(driver_baselines)
        df["residual"] = df["lap_time"] - df["baseline"]

        # Filter out pit laps and safety car laps
        if "pit_this_lap" in df.columns:
            df = df[~df["pit_this_lap"].astype(bool)]
        if "track_status" in df.columns:
            df = df[df["track_status"].astype(str).str.startswith("1")]

        fit_results: dict[str, dict] = {}

        for compound, group in df.groupby("compound"):
            compound_str = str(compound).upper()
            if len(group) < min_samples:
                logger.warning(
                    "Skipping %s — only %d samples (need %d)",
                    compound_str, len(group), min_samples,
                )
                continue

            ages = group["tyre_age"].values.astype(float)
            residuals = group["residual"].values.astype(float)

            try:
                fitted_params, r_squared = self._fit_compound(
                    compound_str, ages, residuals,
                )
                self.params[compound_str] = fitted_params
                fit_results[compound_str] = {
                    "n_samples": len(group),
                    "r_squared": round(r_squared, 4),
                    "a": round(fitted_params.a, 6),
                    "b": round(fitted_params.b, 6),
                    "c": round(fitted_params.c, 6),
                    "cliff_onset": round(fitted_params.cliff_onset, 1),
                }
                logger.info(
                    "Fitted %s: a=%.4f b=%.6f c=%.6f cliff=%.0f (R²=%.3f, n=%d)",
                    compound_str, fitted_params.a, fitted_params.b,
                    fitted_params.c, fitted_params.cliff_onset,
                    r_squared, len(group),
                )
            except Exception as exc:
                logger.warning("Failed to fit %s: %s", compound_str, exc)
                fit_results[compound_str] = {
                    "n_samples": len(group),
                    "error": str(exc),
                }

        self.fitted = True
        return fit_results

    # ──────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model parameters to JSON.

        Args:
            path: Output file path. Defaults to
                ``data/features/tyre_model.json``.

        Returns:
            Path the model was saved to.
        """
        if path is None:
            _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            path = _FEATURES_DIR / "tyre_model.json"

        data = {
            "fitted": self.fitted,
            "compounds": {
                name: asdict(params)
                for name, params in self.params.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Tyre model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "TyreDegradationModel":
        """Load model parameters from JSON.

        Args:
            path: Input file path. Defaults to
                ``data/features/tyre_model.json``.

        Returns:
            Initialized ``TyreDegradationModel`` with loaded parameters.
        """
        if path is None:
            path = _FEATURES_DIR / "tyre_model.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = {
            name: CompoundParams(**cp)
            for name, cp in data["compounds"].items()
        }
        model = cls(params=params)
        model.fitted = data.get("fitted", False)
        logger.info("Tyre model loaded from %s (%d compounds)", path, len(params))
        return model

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _get_params(self, compound: str) -> CompoundParams:
        """Look up parameters for a compound, with fallback."""
        key = compound.upper()
        if key in self.params:
            return self.params[key]
        logger.debug("Unknown compound '%s', using fallback", compound)
        return _FALLBACK_PARAMS

    @staticmethod
    def _deg_function(
        age: float,
        a: float,
        b: float,
        c: float,
        cliff_onset: float,
    ) -> float:
        """Compute degradation penalty for a single age value.

        Model: deg = a × age + b × age² + c × max(0, age − cliff)²
        """
        linear = a * age
        quadratic = b * (age ** 2)
        cliff = c * max(0.0, age - cliff_onset) ** 2
        return linear + quadratic + cliff

    @staticmethod
    def _deg_function_vectorized(
        ages: np.ndarray,
        a: float,
        b: float,
        c: float,
        cliff_onset: float,
    ) -> np.ndarray:
        """Vectorized degradation function for curve fitting."""
        linear = a * ages
        quadratic = b * (ages ** 2)
        cliff = c * np.maximum(0.0, ages - cliff_onset) ** 2
        return linear + quadratic + cliff

    def _fit_compound(
        self,
        compound: str,
        ages: np.ndarray,
        residuals: np.ndarray,
    ) -> tuple[CompoundParams, float]:
        """Fit degradation curve for a single compound.

        Uses scipy.optimize.curve_fit with bounded parameters.

        Args:
            compound: Compound name.
            ages: Tyre age values.
            residuals: Lap-time residuals (observed − baseline).

        Returns:
            Tuple of (fitted CompoundParams, R² score).
        """
        # Get defaults for initial guess
        default = self._get_params(compound)

        # Parameter bounds: [a, b, c, cliff_onset]
        p0 = [default.a, default.b, default.c, default.cliff_onset]
        lower = [0.0, 0.0, 0.0, 5.0]
        upper = [0.5, 0.05, 0.1, 60.0]

        popt, _ = curve_fit(
            self._deg_function_vectorized,
            ages,
            residuals,
            p0=p0,
            bounds=(lower, upper),
            maxfev=5000,
        )

        a_fit, b_fit, c_fit, cliff_fit = popt

        # Compute R²
        predicted = self._deg_function_vectorized(ages, *popt)
        ss_res = np.sum((residuals - predicted) ** 2)
        ss_tot = np.sum((residuals - np.mean(residuals)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        fitted_params = CompoundParams(
            compound=compound,
            a=float(a_fit),
            b=float(b_fit),
            c=float(c_fit),
            cliff_onset=float(cliff_fit),
            fresh_delta=default.fresh_delta,  # keep existing fresh_delta
        )
        return fitted_params, r_squared

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all compound parameters.

        Returns:
            DataFrame with compound params and cliff onset laps.
        """
        rows = []
        for name, p in sorted(self.params.items()):
            rows.append({
                "compound": p.compound,
                "linear_a": p.a,
                "quadratic_b": p.b,
                "cliff_c": p.c,
                "cliff_onset": p.cliff_onset,
                "fresh_delta": p.fresh_delta,
                "deg_at_10": round(self.degradation(name, 10), 3),
                "deg_at_20": round(self.degradation(name, 20), 3),
                "deg_at_30": round(self.degradation(name, 30), 3),
            })
        return pd.DataFrame(rows)
