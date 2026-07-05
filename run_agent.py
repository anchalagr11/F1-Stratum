"""
STRATUM-F1 — GenAI Agent Runner

Demonstrates the StrategyPromptBuilder. Runs the pipeline for a target
driver and lap, generates the optimizer recommendations, and prints
the generated markdown prompt that would be sent to an LLM.
"""

import logging
from pathlib import Path

from src.ingestion.load_session import load_race_session
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features
from src.simulation.tyre_model import TyreDegradationModel
from src.simulation.optimizer import PitWindowOptimizer
from src.llm.prompt_builder import StrategyPromptBuilder
from src.llm.reasoner import get_reasoner, LLMUnavailableError

# Configuration
YEAR: int = 2024
GP: str = "Singapore"
TARGET_DRIVER: str = "NOR"
TARGET_LAP: int = 20

LOG_FORMAT = "%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
# Silence verbose loggers to focus on output
logging.getLogger("src").setLevel(logging.ERROR)
logging.getLogger("fastf1").setLevel(logging.ERROR)


def main() -> None:
    print(f"============================================================")
    print(f"STRATUM-F1 — AI Strategist Setup ({YEAR} {GP})")
    print(f"============================================================\n")

    print("[1/5] Loading FastF1 telemetry data...")
    session_data = load_race_session(YEAR, GP)

    print("[2/5] Building canonical dataset...")
    race_state = build_race_state(session_data, race_id=f"{YEAR}_{GP}")

    print("[3/5] Computing strategy features...")
    enriched = add_strategy_features(race_state)

    print("[4/5] Fitting tyre model...")
    tyre_model = TyreDegradationModel()
    tyre_model.fit_from_race_data(enriched)

    print(f"[5/5] Optimizing pit windows for {TARGET_DRIVER} on lap {TARGET_LAP}...")
    optimizer = PitWindowOptimizer(race_state=enriched, tyre_model=tyre_model)
    optimal = optimizer.find_optimal(TARGET_DRIVER, TARGET_LAP, max_stops=2, top_n=3)

    print("\n============================================================")
    print("GENERATED LLM PROMPT PAYLOAD")
    print("============================================================\n")

    builder = StrategyPromptBuilder(race_state=enriched, tyre_model=tyre_model)

    prompt = builder.build_prompt(
        driver=TARGET_DRIVER,
        lap=TARGET_LAP,
        candidates_1_stop=optimal.get("1_stop", []),
        candidates_2_stop=optimal.get("2_stop", []),
        best_overall=optimal.get("best_overall", [None])[0],
    )

    print(prompt)

    print("\n============================================================")
    print("LIVE AI STRATEGIST RECOMMENDATION")
    print("============================================================\n")

    try:
        reasoner = get_reasoner()
        result = reasoner.reason(prompt)
        print(result.text)
        print(
            f"\n[Model: {result.model} | "
            f"in: {result.input_tokens} tok | out: {result.output_tokens} tok]"
        )
    except LLMUnavailableError as exc:
        print(f"[Live reasoning unavailable] {exc}")
        print(
            "Set a provider key (GEMINI_API_KEY for the free Gemini backend) "
            "and install its SDK to receive a live recommendation. The briefing "
            "above is the exact payload that would be sent to the model."
        )


if __name__ == "__main__":
    main()
