import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ingestion.load_session import load_race_session
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features
from src.simulation.tyre_model import TyreDegradationModel
from src.simulation.optimizer import PitWindowOptimizer
from src.llm.prompt_builder import StrategyPromptBuilder
from src.llm.reasoner import get_reasoner, LLMUnavailableError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="STRATUM-F1 Real-Time API",
    description="API for Formula 1 strategy simulation and reasoning",
    version="1.0.0"
)

class StrategyRequest(BaseModel):
    year: int
    gp: str
    driver: str
    decision_lap: int
    max_stops: int = 2
    top_n: int = 3
    reason: bool = False  # If True, also call the LLM for a live recommendation.

class StrategyCandidateModel(BaseModel):
    stops: int
    pit_laps: List[int]
    compounds: List[str]
    total_time: float
    time_delta: float
    expected_finish: float
    risk: float

class StrategyResponse(BaseModel):
    best_overall: StrategyCandidateModel
    one_stop_alternatives: List[StrategyCandidateModel]
    two_stop_alternatives: List[StrategyCandidateModel]
    prompt: str
    recommendation: Optional[str] = None  # Live LLM reasoning, if requested.

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/strategy", response_model=StrategyResponse)
def get_strategy(request: StrategyRequest):
    try:
        race_id = f"{request.year}_{request.gp.replace(' ', '_').lower()}"
        
        # 1. Load data
        session_data = load_race_session(year=request.year, gp=request.gp)
        
        # 2. Build canonical state
        race_state = build_race_state(session_data, race_id=race_id)
        
        # 3. Add strategy features
        enriched = add_strategy_features(race_state)
        
        # 4. Fit Tyre Model
        tyre_model = TyreDegradationModel()
        tyre_model.fit_from_race_data(enriched)
        
        # 5. Run Optimizer
        optimizer = PitWindowOptimizer(race_state=enriched, tyre_model=tyre_model)
        optimal = optimizer.find_optimal(
            driver=request.driver,
            decision_lap=request.decision_lap,
            max_stops=request.max_stops,
            top_n=request.top_n
        )
        
        if "best_overall" not in optimal or not optimal["best_overall"]:
            raise HTTPException(status_code=404, detail="No viable strategy found.")
            
        best = optimal["best_overall"][0]
        one_stop = optimal.get("1_stop", [])
        two_stop = optimal.get("2_stop", [])
        
        # 6. Build Prompt
        prompt_builder = StrategyPromptBuilder(race_state=enriched, tyre_model=tyre_model)
        prompt = prompt_builder.build_prompt(
            driver=request.driver,
            lap=request.decision_lap,
            candidates_1_stop=one_stop,
            candidates_2_stop=two_stop,
            best_overall=best
        )
        
        # Format response
        def format_candidate(c):
            return StrategyCandidateModel(
                stops=len(c.pit_laps),
                pit_laps=c.pit_laps,
                compounds=c.compounds,
                total_time=c.total_time,
                time_delta=c.time_delta,
                expected_finish=c.expected_finish,
                risk=c.risk
            )
            
        # 7. Optionally run live LLM reasoning
        recommendation = None
        if request.reason:
            try:
                reasoner = get_reasoner()
                recommendation = reasoner.reason(prompt).text
            except LLMUnavailableError as exc:
                logger.warning("Live reasoning unavailable: %s", exc)
                recommendation = f"[Live reasoning unavailable] {exc}"

        return StrategyResponse(
            best_overall=format_candidate(best),
            one_stop_alternatives=[format_candidate(c) for c in one_stop],
            two_stop_alternatives=[format_candidate(c) for c in two_stop],
            prompt=prompt,
            recommendation=recommendation
        )
        
    except Exception as e:
        logger.error(f"Error processing strategy request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
