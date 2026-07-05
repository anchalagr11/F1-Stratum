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


def main() -> None:
    parser = argparse.ArgumentParser(description="STRATUM-F1 strategy backtest")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Singapore")
    parser.add_argument("--decision-lap", type=int, default=8)
    parser.add_argument("--tolerance", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(message)s")

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
