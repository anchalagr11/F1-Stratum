"""
STRATUM-F1 — Strategy Validation Backtest (Proof of Concept)

Answers the question that turns a simulator into research: *are the optimizer's
recommendations any good?* For a chosen race it replays the field, asks the
optimizer for the best pit strategy from an early decision lap, and compares
against what the teams actually did and where drivers actually finished.

Metrics reported:
- Pit-lap accuracy: |recommended first pit − actual first pit|, and % within ±k laps.
- Finish accuracy: |projected finish − actual finish| (mean absolute error).

Run:
    python -m validation.backtest --year 2024 --gp Singapore --decision-lap 8
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.load_session import load_race_session
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features
from src.simulation.tyre_model import TyreDegradationModel
from src.simulation.optimizer import PitWindowOptimizer

logger = logging.getLogger("stratum-f1.backtest")


def collapse_stops(pit_laps: list[int]) -> list[int]:
    """Collapse consecutive in/out laps into single stop laps.

    FastF1 flags both the in-lap and out-lap of a stop, so [15, 16] is one
    stop. Returns the sorted list of stop *entry* laps.
    """
    stops = []
    for lap in sorted(set(pit_laps)):
        if not stops or lap > stops[-1] + 1:
            stops.append(lap)
    return stops


def ground_truth(race_state: pd.DataFrame) -> pd.DataFrame:
    """Extract actual first pit lap and final finishing position per driver."""
    last_lap = int(race_state["lap"].max())
    finish = (
        race_state[race_state["lap"] == last_lap]
        .set_index("driver")["position"]
    )
    rows = []
    for driver, grp in race_state.groupby("driver"):
        pit_laps = grp.loc[grp["pit_this_lap"] == True, "lap"].tolist()  # noqa: E712
        stops = collapse_stops(pit_laps)
        rows.append(
            {
                "driver": driver,
                "actual_stops": len(stops),
                "actual_first_pit": stops[0] if stops else None,
                "actual_finish": finish.get(driver, pd.NA),
            }
        )
    return pd.DataFrame(rows).set_index("driver")


def run_backtest(year: int, gp: str, decision_lap: int, tolerance: int = 2) -> dict:
    """Run the validation backtest for one race and return metrics + detail."""
    race_id = f"{year}_{gp.replace(' ', '_').lower()}"
    logger.info("Loading %s %s …", year, gp)
    session = load_race_session(year=year, gp=gp)
    race_state = build_race_state(session, race_id=race_id)
    enriched = add_strategy_features(race_state)

    tyre_model = TyreDegradationModel()
    tyre_model.fit_from_race_data(enriched)
    optimizer = PitWindowOptimizer(race_state=enriched, tyre_model=tyre_model)

    truth = ground_truth(enriched)

    # Positions at the decision lap — used for a naive "persistence" baseline
    # (assume everyone finishes where they currently run).
    grid_at_decision = (
        enriched[enriched["lap"] == decision_lap].set_index("driver")["position"]
    )

    records = []
    for driver in sorted(enriched["driver"].unique()):
        gt = truth.loc[driver]
        # Only evaluate drivers who actually pitted *after* the decision lap
        # (so there's a forward-looking call to compare against).
        if pd.isna(gt["actual_first_pit"]) or gt["actual_first_pit"] <= decision_lap:
            continue
        try:
            result = optimizer.find_optimal(
                driver=driver, decision_lap=decision_lap, max_stops=2, top_n=1
            )
        except Exception as exc:  # noqa: BLE001 - skip drivers the optimizer can't handle
            logger.debug("skip %s: %s", driver, exc)
            continue
        best = result.get("best_overall", [None])[0]
        if best is None or not best.pit_laps:
            continue

        rec_first = best.pit_laps[0]
        pit_err = abs(rec_first - int(gt["actual_first_pit"]))
        fin_err = (
            abs(best.expected_finish - float(gt["actual_finish"]))
            if pd.notna(gt["actual_finish"])
            else None
        )
        # Naive baseline: predict finish = current track position.
        base_pos = grid_at_decision.get(driver)
        base_err = (
            abs(float(base_pos) - float(gt["actual_finish"]))
            if pd.notna(gt["actual_finish"]) and pd.notna(base_pos)
            else None
        )
        records.append(
            {
                "driver": driver,
                "rec_first_pit": rec_first,
                "actual_first_pit": int(gt["actual_first_pit"]),
                "pit_err": pit_err,
                "rec_stops": len(best.pit_laps),
                "actual_stops": int(gt["actual_stops"]),
                "proj_finish": round(best.expected_finish, 1),
                "actual_finish": int(gt["actual_finish"]) if pd.notna(gt["actual_finish"]) else None,
                "finish_err": round(fin_err, 1) if fin_err is not None else None,
                "baseline_err": round(base_err, 1) if base_err is not None else None,
            }
        )

    detail = pd.DataFrame(records)
    n = len(detail)
    metrics = {
        "race": f"{year} {gp}",
        "decision_lap": decision_lap,
        "n_drivers": n,
    }
    if n:
        pit_errs = detail["pit_err"]
        fin_errs = detail["finish_err"].dropna()
        base_errs = detail["baseline_err"].dropna()
        metrics.update(
            {
                "pit_lap_mae": round(pit_errs.mean(), 2),
                "pit_within_tol_pct": round(100 * (pit_errs <= tolerance).mean(), 1),
                "tolerance_laps": tolerance,
                "stop_count_match_pct": round(
                    100 * (detail["rec_stops"] == detail["actual_stops"]).mean(), 1
                ),
                "finish_mae": round(fin_errs.mean(), 2) if len(fin_errs) else None,
                "baseline_finish_mae": round(base_errs.mean(), 2) if len(base_errs) else None,
            }
        )
    return {"metrics": metrics, "detail": detail}


# Races available in the local FastF1 cache — used by --season.
SEASON_2024_CACHED = [
    "Bahrain", "Saudi Arabian", "Australian", "Japanese", "Chinese", "Singapore",
]


def run_season(year: int, gps: list[str], decision_lap: int, tolerance: int = 2) -> None:
    """Run the backtest across several races and aggregate the metrics."""
    per_race = []
    for gp in gps:
        try:
            m = run_backtest(year, gp, decision_lap, tolerance)["metrics"]
            if m.get("n_drivers"):
                per_race.append(m)
        except Exception as exc:  # noqa: BLE001 - skip races that fail to load
            logger.error("skip %s: %s", gp, exc)

    if not per_race:
        print("No races could be evaluated.")
        return

    def wmean(key: str) -> float:
        pairs = [(m[key], m["n_drivers"]) for m in per_race if m.get(key) is not None]
        w = sum(n for _, n in pairs)
        return sum(v * n for v, n in pairs) / w if w else float("nan")

    print("=" * 78)
    print(f"STRATUM-F1 SEASON BACKTEST — {year} ({len(per_race)} races, decision lap {decision_lap})")
    print("=" * 78)
    print(f"{'Race':<26}{'n':>4}{'pitMAE':>9}{'±2%':>7}{'stop%':>7}{'finMAE':>8}{'baseMAE':>9}")
    for m in per_race:
        print(f"{m['race']:<26}{m['n_drivers']:>4}{m['pit_lap_mae']:>9}"
              f"{m['pit_within_tol_pct']:>7}{m['stop_count_match_pct']:>7}"
              f"{str(m['finish_mae']):>8}{str(m['baseline_finish_mae']):>9}")
    print("-" * 78)
    total_n = sum(m["n_drivers"] for m in per_race)
    print(f"{'AGGREGATE (n-weighted)':<26}{total_n:>4}{wmean('pit_lap_mae'):>9.2f}"
          f"{wmean('pit_within_tol_pct'):>7.1f}{wmean('stop_count_match_pct'):>7.1f}"
          f"{wmean('finish_mae'):>8.2f}{wmean('baseline_finish_mae'):>9.2f}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="STRATUM-F1 strategy backtest")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Singapore")
    parser.add_argument("--decision-lap", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument(
        "--season", action="store_true",
        help="Run across all cached races and aggregate metrics.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    if args.season:
        run_season(args.year, SEASON_2024_CACHED, args.decision_lap, args.tolerance)
        return

    out = run_backtest(args.year, args.gp, args.decision_lap, args.tolerance)
    m = out["metrics"]
    detail = out["detail"]

    print("=" * 64)
    print(f"STRATUM-F1 BACKTEST — {m['race']} (decision lap {m['decision_lap']})")
    print("=" * 64)
    if not m["n_drivers"]:
        print("No drivers eligible for evaluation.")
        return
    print(detail.to_string(index=False))
    print("-" * 64)
    print(f"Drivers evaluated       : {m['n_drivers']}")
    print(f"Pit-lap MAE             : {m['pit_lap_mae']} laps")
    print(f"Within ±{m['tolerance_laps']} laps          : {m['pit_within_tol_pct']}%")
    print(f"Stop-count match        : {m['stop_count_match_pct']}%")
    print(f"Finish-position MAE     : {m['finish_mae']} positions")
    print(f"  vs naive baseline MAE : {m['baseline_finish_mae']} positions")
    print("=" * 64)


if __name__ == "__main__":
    main()
