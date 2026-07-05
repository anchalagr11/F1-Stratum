# STRATUM-F1: Sprint Plan

**Strategic Telemetry Reasoning and Adaptive Tactical Understanding Model for Formula 1**

> Status verified against source on 2026-07-04.

## 🎯 Project Goal
Build a research-grade system to ingest F1 data, simulate race strategies, and provide an LLM-ready reasoning layer for tactical decision-making.

---

## 🛠️ Phase 1: Data Ingestion & Persistence
**Objective:** Create a robust pipeline to fetch raw data and save it for offline use.

*   [x] **Module:** `src.ingestion`
*   [x] Implement `load_session.py`: Wrapper for `FastF1` to fetch laps, weather, track status, and results.
*   [x] Implement `batch_loader.py`: Support for loading multiple races or full seasons (e.g., 2024 Bahrain, Singapore).
*   [x] Data Format: Persist all raw data as `.parquet` files for high-performance reading.

## 📊 Phase 2: Canonical Race State
**Objective:** Transform scattered raw data into a structured "state of the race."

*   [x] **Module:** `src.canonical`
*   [x] Implement `build_race_state.py`: One row per `(driver, lap)`.
*   [x] Compute positional gaps (Gap Ahead/Behind).
*   [x] Compute rolling pace averages (3-lap and 5-lap means) to smooth out noise.
*   [x] Track tyre life and pit-stop status per lap.

## 🧪 Phase 3: Strategy Feature Engineering
**Objective:** Add derivative metrics that drive strategic insights.

*   [x] **Module:** `src.features`
*   [x] Implement `strategy_features.py`:
    *   **Traffic Penalty:** Identify when a driver is in "dirty air" (< 1.5s gap).
    *   **Undercut Estimate:** Heuristic for potential time gain from pitting early.
    *   **Pit Loss:** Track-specific time lost during pit lane entry/exit.

## 🏎️ Phase 4: Simulation & Optimization Engine
**Objective:** Project future outcomes using mathematical models.

*   [x] **Module:** `src.simulation`
*   [x] **Tyre Model (`tyre_model.py`):** Nonlinear degradation model (Quadratic + Cliff effect). Supports fitting parameters from historical race data.
*   [x] **Race Simulator (`simulator.py`):** Monte Carlo-style projection of future lap times given a chosen action (Pit Now vs. Stay Out).
*   [x] **Pit Window Optimizer (`optimizer.py`):** Brute-force optimization of 1-stop and 2-stop windows across the remaining race distance.

## 🤖 Phase 5: GenAI Reasoning Layer
**Objective:** Bridge the gap between raw data and human-like strategic reasoning.

*   [x] **Module:** `src.llm`
*   [x] Implement `prompt_builder.py`: Translates canonical state and simulation results into a structured Markdown "Race Briefing."
*   [x] Logic: Includes rival context, tyre health warnings, and optimizer recommendations for LLM consumption.
*   [x] **Live LLM call:** `reasoner.py` sends the briefing to Claude and returns a recommendation (see Phase 8).

## 📈 Phase 6: Visualization *(added — not in original plan)*
**Objective:** Render race dynamics and the "Strategy Cliff" as charts.

*   [x] **Module:** `src.visualization`
*   [x] Implement `race_charts.py` (matplotlib): lap times, gap evolution, position changes, tyre degradation, tyre strategy, and strategy comparison.
*   [x] Output: static `.png` charts saved to `data/charts/` (e.g. 2024 Singapore set).
*   [x] Interactive dashboard (Streamlit) — implemented in `dashboard.py` (see Phase 9).

## 🚀 Phase 7: Integration & Validation
**Objective:** End-to-end execution and testing.

*   [x] **Entry Points:** `run_pipeline.py` (single race) and `run_batch_pipeline.py` (historical analysis).
*   [x] **Real-time API:** `src.api.main` (FastAPI) exposes `GET /health` and `POST /strategy`, running the full pipeline → optimizer → prompt. Served via `run_api.py`; smoke-tested by `test_api.py`. *(Was previously listed under Future Work — now complete.)*
*   [x] **Testing:** Unit tests in `tests/` for the optimizer, tyre model, simulator, canonical builders, and features.
*   [x] **Documentation:** `README.md` and project structure.

---

## ✅ Phase 8: Live Reasoning (Complete)
**Objective:** Turn the built prompt into a live LLM recommendation.

*   [x] **Pluggable backends** via `get_reasoner(provider)` (`src.llm.reasoner`), selected by the `STRATUM_LLM_PROVIDER` env var. Default: **`gemini`**.
*   [x] **Gemini backend** (`src.llm.gemini_reasoner.GeminiReasoner`) — free-tier `gemini-2.5-flash`, reads `GEMINI_API_KEY` / `GOOGLE_API_KEY`. Verified live.
*   [x] **Claude backend** (`src.llm.reasoner.StrategyReasoner`) — `claude-opus-4-8` with adaptive thinking, reads `ANTHROPIC_API_KEY` / `ant auth login`.
*   [x] Both share `ReasoningResult` and graceful `LLMUnavailableError` (missing SDK/key) — the pipeline still prints the briefing.
*   [x] Wired into `run_agent.py` (prints a live recommendation) and the `/strategy` API endpoint (opt-in via `reason: true`, returned as `recommendation`).
*   [x] Added `anthropic` and `google-genai` to `requirements.txt`.

## ✅ Phase 9: Interactive Dashboard (Complete)
**Objective:** Interactive Streamlit UI for the "Strategy Cliff" and optimizer output.

*   [x] `dashboard.py` (run with `streamlit run dashboard.py`): sidebar controls for season/GP/driver/decision lap, cached data pipeline, interactive Strategy Cliff chart, optimizer projection tables (1-stop / 2-stop), the race briefing, and an "Ask the AI Strategist" button wired to Phase 8.
*   [x] Added `streamlit` to `requirements.txt`.

## 🔮 Future Work
*   [ ] Add multi-provider support (e.g. Gemini) behind the `StrategyReasoner` interface.
*   [ ] Live/streaming reasoning output in the dashboard (token-by-token).
