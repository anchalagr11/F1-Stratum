"""
STRATUM-F1 — Race Visualization Module

Provides publication-quality charts for race analysis:
- Lap time evolution per driver
- Tyre stint lifecycle and compound strategy
- Position changes over the race
- Gap evolution between drivers
- Strategy comparison from the optimizer
- Tyre degradation model curves
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# F1 visual identity
# ──────────────────────────────────────────────────────────────

# Compound colors (F1-standard palette)
COMPOUND_COLORS: dict[str, str] = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#CCCCCC",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0072C6",
    "UNKNOWN": "#888888",
}

# Driver colors (top teams — extend as needed)
DRIVER_COLORS: dict[str, str] = {
    "VER": "#3671C6", "PER": "#3671C6",   # Red Bull
    "NOR": "#FF8000", "PIA": "#FF8000",   # McLaren
    "LEC": "#E8002D", "SAI": "#E8002D",   # Ferrari
    "HAM": "#27F4D2", "RUS": "#27F4D2",   # Mercedes
    "ALO": "#229971", "STR": "#229971",   # Aston Martin
    "GAS": "#2293D1", "OCO": "#2293D1",   # Alpine
    "TSU": "#6692FF", "RIC": "#6692FF",   # RB
    "BOT": "#52E252", "ZHO": "#52E252",   # Kick Sauber
    "MAG": "#B6BABD", "HUL": "#B6BABD",   # Haas
    "ALB": "#64C4FF", "SAR": "#64C4FF",   # Williams
}

_DEFAULT_COLOR = "#AAAAAA"

# Plot style
_STYLE_PARAMS = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "grid.color": "#333355",
    "grid.alpha": 0.5,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#444444",
    "font.size": 10,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "charts"


class RaceVisualizer:
    """Generate race analysis charts from canonical race state data.

    All charts use a dark F1-inspired theme and can be saved to
    ``data/charts/`` or displayed inline.

    Attributes:
        race_state: Canonical race state DataFrame.
        race_id: Race identifier for titles.
        output_dir: Directory for saved chart images.
    """

    def __init__(
        self,
        race_state: pd.DataFrame,
        race_id: str = "race",
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the visualizer.

        Args:
            race_state: Canonical race state DataFrame with columns
                ``driver``, ``lap``, ``lap_time``, ``position``,
                ``compound``, ``tyre_age``, etc.
            race_id: Race identifier used in chart titles and filenames.
            output_dir: Output directory for saved charts.
        """
        self.race_state = race_state.copy()
        self.race_id = race_id
        self.output_dir = output_dir or _OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update(_STYLE_PARAMS)

    def _driver_color(self, driver: str) -> str:
        """Get the color for a driver."""
        return DRIVER_COLORS.get(driver, _DEFAULT_COLOR)

    def _compound_color(self, compound: str) -> str:
        """Get the color for a tyre compound."""
        return COMPOUND_COLORS.get(compound.upper(), COMPOUND_COLORS["UNKNOWN"])

    def _save_fig(self, fig: plt.Figure, name: str) -> Path:
        """Save a figure to the output directory."""
        path = self.output_dir / f"{self.race_id}_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Chart saved → %s", path)
        return path

    # ──────────────────────────────────────────────────────
    # Chart 1: Lap Time Evolution
    # ──────────────────────────────────────────────────────

    def plot_lap_times(
        self,
        drivers: Optional[list[str]] = None,
        highlight_pits: bool = True,
    ) -> Path:
        """Plot lap time evolution for selected drivers.

        Args:
            drivers: Drivers to include. ``None`` = top 5 by final position.
            highlight_pits: Show pit stop markers.

        Returns:
            Path to saved chart.
        """
        df = self.race_state.copy()
        if drivers is None:
            drivers = self._top_drivers(5)
        df = df[df["driver"].isin(drivers)]

        fig, ax = plt.subplots(figsize=(14, 6))

        for drv in drivers:
            drv_data = df[df["driver"] == drv].sort_values("lap")
            color = self._driver_color(drv)
            ax.plot(
                drv_data["lap"], drv_data["lap_time"],
                color=color, linewidth=1.5, alpha=0.9, label=drv,
            )

            if highlight_pits and "pit_this_lap" in drv_data.columns:
                pits = drv_data[drv_data["pit_this_lap"] == True]
                ax.scatter(
                    pits["lap"], pits["lap_time"],
                    color=color, marker="v", s=80, zorder=5, edgecolors="white",
                )

        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        ax.set_title(f"{self.race_id} — Lap Time Evolution", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.3)

        return self._save_fig(fig, "lap_times")

    # ──────────────────────────────────────────────────────
    # Chart 2: Tyre Strategy Timeline
    # ──────────────────────────────────────────────────────

    def plot_tyre_strategy(
        self,
        drivers: Optional[list[str]] = None,
    ) -> Path:
        """Plot a horizontal bar chart showing tyre compound stints.

        Each driver gets a row showing which compound was used on
        each lap, colored by compound type.

        Args:
            drivers: Drivers to include. ``None`` = top 10.

        Returns:
            Path to saved chart.
        """
        df = self.race_state.copy()
        if drivers is None:
            drivers = self._top_drivers(10)
        df = df[df["driver"].isin(drivers)]

        fig, ax = plt.subplots(figsize=(14, max(4, len(drivers) * 0.6)))

        for i, drv in enumerate(reversed(drivers)):
            drv_data = df[df["driver"] == drv].sort_values("lap")
            for _, row in drv_data.iterrows():
                color = self._compound_color(str(row["compound"]))
                ax.barh(
                    i, 1, left=row["lap"] - 1, height=0.6,
                    color=color, edgecolor="none",
                )

        ax.set_yticks(range(len(drivers)))
        ax.set_yticklabels(list(reversed(drivers)))
        ax.set_xlabel("Lap")
        ax.set_title(f"{self.race_id} — Tyre Strategy", fontsize=14, fontweight="bold")

        # Legend
        patches = [
            mpatches.Patch(color=c, label=name)
            for name, c in COMPOUND_COLORS.items()
            if name != "UNKNOWN"
        ]
        ax.legend(handles=patches, loc="upper right", framealpha=0.8)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        return self._save_fig(fig, "tyre_strategy")

    # ──────────────────────────────────────────────────────
    # Chart 3: Position Changes
    # ──────────────────────────────────────────────────────

    def plot_position_changes(
        self,
        drivers: Optional[list[str]] = None,
    ) -> Path:
        """Plot position changes over the race.

        Args:
            drivers: Drivers to include. ``None`` = top 10.

        Returns:
            Path to saved chart.
        """
        df = self.race_state.copy()
        if drivers is None:
            drivers = self._top_drivers(10)
        df = df[df["driver"].isin(drivers)]

        fig, ax = plt.subplots(figsize=(14, 6))

        for drv in drivers:
            drv_data = df[df["driver"] == drv].sort_values("lap")
            color = self._driver_color(drv)
            ax.plot(
                drv_data["lap"], drv_data["position"],
                color=color, linewidth=2, alpha=0.9, label=drv,
                marker="o", markersize=3,
            )

        ax.invert_yaxis()
        ax.set_xlabel("Lap")
        ax.set_ylabel("Position")
        ax.set_title(f"{self.race_id} — Position Changes", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.8, ncol=2)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_yticks(range(1, int(df["position"].max()) + 1))

        return self._save_fig(fig, "position_changes")

    # ──────────────────────────────────────────────────────
    # Chart 4: Gap Evolution
    # ──────────────────────────────────────────────────────

    def plot_gap_evolution(
        self,
        reference_driver: str,
        drivers: Optional[list[str]] = None,
    ) -> Path:
        """Plot cumulative time gap to a reference driver.

        Args:
            reference_driver: Driver to use as the zero baseline.
            drivers: Drivers to compare against. ``None`` = top 5.

        Returns:
            Path to saved chart.
        """
        df = self.race_state.copy()
        if drivers is None:
            drivers = self._top_drivers(5)
        if reference_driver not in drivers:
            drivers = [reference_driver] + drivers[:4]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Compute cumulative times per driver
        cum_times: dict[str, pd.Series] = {}
        for drv in drivers:
            drv_data = df[df["driver"] == drv].sort_values("lap")
            cum_times[drv] = drv_data.set_index("lap")["lap_time"].cumsum()

        ref = cum_times.get(reference_driver)
        if ref is None:
            logger.warning("Reference driver %s not found", reference_driver)
            plt.close(fig)
            return self.output_dir / "gap_evolution_error.png"

        for drv in drivers:
            if drv == reference_driver:
                continue
            color = self._driver_color(drv)
            gap = cum_times[drv] - ref
            # Align on common laps
            common = gap.dropna()
            ax.plot(
                common.index, common.values,
                color=color, linewidth=2, alpha=0.9, label=drv,
            )

        ax.axhline(y=0, color=self._driver_color(reference_driver),
                    linewidth=2, linestyle="--", alpha=0.5,
                    label=f"{reference_driver} (ref)")
        ax.set_xlabel("Lap")
        ax.set_ylabel(f"Gap to {reference_driver} (s)")
        ax.set_title(
            f"{self.race_id} — Gap Evolution (ref: {reference_driver})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="best", framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.3)

        return self._save_fig(fig, "gap_evolution")

    # ──────────────────────────────────────────────────────
    # Chart 5: Tyre Degradation Curves
    # ──────────────────────────────────────────────────────

    def plot_tyre_degradation(
        self,
        tyre_model: Optional[object] = None,
        max_age: int = 40,
    ) -> Path:
        """Plot tyre degradation curves from the model.

        If no model is provided, extracts empirical degradation
        from the race data.

        Args:
            tyre_model: ``TyreDegradationModel`` instance.
            max_age: Maximum tyre age to plot.

        Returns:
            Path to saved chart.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        if tyre_model is not None:
            # Plot model curves
            ages = np.arange(1, max_age + 1, dtype=float)
            for compound in ["SOFT", "MEDIUM", "HARD"]:
                color = self._compound_color(compound)
                deg = [tyre_model.degradation(compound, a) for a in ages]
                rate = [tyre_model.degradation_rate(compound, a) for a in ages]
                ax1.plot(ages, deg, color=color, linewidth=2.5, label=compound)
                ax2.plot(ages, rate, color=color, linewidth=2.5, label=compound)

                # Mark cliff onset
                cliff = tyre_model.cliff_onset_lap(compound)
                if cliff <= max_age:
                    ax1.axvline(x=cliff, color=color, linestyle=":", alpha=0.4)
                    ax2.axvline(x=cliff, color=color, linestyle=":", alpha=0.4)

        else:
            # Empirical: plot observed residuals vs tyre age
            df = self.race_state.dropna(subset=["tyre_age", "lap_time", "compound"])
            driver_medians = df.groupby("driver")["lap_time"].transform("median")
            df = df.copy()
            df["residual"] = df["lap_time"] - driver_medians

            for compound in df["compound"].unique():
                color = self._compound_color(str(compound))
                cdf = df[df["compound"] == compound]
                ax1.scatter(
                    cdf["tyre_age"], cdf["residual"],
                    color=color, alpha=0.3, s=10, label=str(compound),
                )

        ax1.set_xlabel("Tyre Age (laps)")
        ax1.set_ylabel("Cumulative Degradation (s)")
        ax1.set_title("Degradation Curves", fontsize=12, fontweight="bold")
        ax1.legend(framealpha=0.8)
        ax1.grid(True, linestyle="--", alpha=0.3)

        ax2.set_xlabel("Tyre Age (laps)")
        ax2.set_ylabel("Marginal Degradation (s/lap)")
        ax2.set_title("Degradation Rate", fontsize=12, fontweight="bold")
        ax2.legend(framealpha=0.8)
        ax2.grid(True, linestyle="--", alpha=0.3)

        fig.suptitle(
            f"{self.race_id} — Tyre Degradation Model",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()

        return self._save_fig(fig, "tyre_degradation")

    # ──────────────────────────────────────────────────────
    # Chart 6: Strategy Comparison
    # ──────────────────────────────────────────────────────

    def plot_strategy_comparison(
        self,
        candidates: list,
        driver: str,
        max_show: int = 8,
    ) -> Path:
        """Plot a comparison of optimizer strategy candidates.

        Shows a horizontal bar chart of projected total times with
        time deltas and compound labels.

        Args:
            candidates: List of ``StrategyCandidate`` objects.
            driver: Driver name for the title.
            max_show: Maximum number of strategies to display.

        Returns:
            Path to saved chart.
        """
        show = candidates[:max_show]
        n = len(show)

        fig, ax = plt.subplots(figsize=(12, max(4, n * 0.7)))

        labels = []
        times = []
        colors = []
        deltas = []

        for c in reversed(show):
            pit_str = ", ".join(map(str, c.pit_laps))
            comp_str = " → ".join(c.compounds)
            labels.append(f"Pit L{pit_str}\n{comp_str}")
            times.append(c.total_time)
            deltas.append(c.time_delta)
            # Color by first post-pit compound
            colors.append(self._compound_color(c.compounds[-1]))

        # Normalize bar widths relative to best
        best_time = min(times)
        bar_widths = [t - best_time + 0.5 for t in times]

        bars = ax.barh(range(n), bar_widths, color=colors, edgecolor="#444444", height=0.6)

        # Add delta labels
        for i, (bw, delta) in enumerate(zip(bar_widths, reversed(deltas))):
            label = "BEST" if delta == 0 else f"+{delta:.1f}s"
            ax.text(
                bw + 0.3, i, label,
                va="center", ha="left", fontsize=10,
                color="#00FF88" if delta == 0 else "#FF6666",
                fontweight="bold",
            )

        ax.set_yticks(range(n))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Time Delta (s)")
        ax.set_title(
            f"{self.race_id} — Strategy Comparison ({driver})",
            fontsize=14, fontweight="bold",
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        return self._save_fig(fig, "strategy_comparison")

    # ──────────────────────────────────────────────────────
    # Generate all charts
    # ──────────────────────────────────────────────────────

    def generate_all(
        self,
        reference_driver: Optional[str] = None,
        tyre_model: Optional[object] = None,
        optimizer_candidates: Optional[list] = None,
        target_driver: Optional[str] = None,
    ) -> list[Path]:
        """Generate the full suite of race analysis charts.

        Args:
            reference_driver: Driver for gap evolution chart.
            tyre_model: Tyre model for degradation curves.
            optimizer_candidates: Strategy candidates for comparison.
            target_driver: Driver for strategy comparison title.

        Returns:
            List of paths to all generated chart files.
        """
        if reference_driver is None:
            reference_driver = self._top_drivers(1)[0]
        if target_driver is None:
            target_driver = reference_driver

        paths: list[Path] = []

        logger.info("Generating full chart suite for %s", self.race_id)

        paths.append(self.plot_lap_times())
        paths.append(self.plot_tyre_strategy())
        paths.append(self.plot_position_changes())
        paths.append(self.plot_gap_evolution(reference_driver))
        paths.append(self.plot_tyre_degradation(tyre_model))

        if optimizer_candidates:
            paths.append(self.plot_strategy_comparison(
                optimizer_candidates, target_driver,
            ))

        logger.info("Generated %d charts → %s", len(paths), self.output_dir)
        return paths

    # ──────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────

    def _top_drivers(self, n: int) -> list[str]:
        """Get the top N drivers by final position in the race."""
        last_lap = self.race_state["lap"].max()
        final = self.race_state[self.race_state["lap"] == last_lap]
        return (
            final.sort_values("position")["driver"]
            .head(n)
            .tolist()
        )
