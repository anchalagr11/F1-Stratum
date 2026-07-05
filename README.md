# STRATUM-F1 🏎️

**Strategic Telemetry Reasoning and Adaptive Tactical Understanding Model for Formula 1**

A research-grade F1 strategy engine: it ingests real telemetry, fits tyre
degradation curves, brute-forces the optimal pit window, quantifies the outcome
with Monte-Carlo probabilities, and has an LLM race engineer reason over the
result — all in an interactive dashboard.

> **What makes it different:** most F1 projects *visualize* strategy. STRATUM-F1
> *optimizes* it and then **validates** those recommendations against what teams
> actually did — see the numbers below.

---

## ✅ Validated accuracy

Backtested on the 2024 Singapore GP (20 drivers), grading the optimizer against
real pit calls and finishing positions ([full POC](docs/POC_STRATEGY_VALIDATION.md)):

| Metric | Result | Naive baseline |
|---|---|---|
| Finish-position MAE | **0.71 positions** | 1.86 (predict current position) |
| Pit-lap MAE | **5.7 laps** | — |
| Stop-count match | **85%** | — |
| Race winner identified | ✅ NOR → P1 | — |

The engine is **2.6× more accurate** than assuming everyone finishes where they
currently run. Reproduce:

```bash
python -m validation.backtest --year 2024 --gp Singapore --decision-lap 8
```

## Pipeline

```
FastF1 ingest → canonical race state → strategy features →
  tyre model (fitted) → pit-window optimizer → Monte-Carlo outcome → LLM briefing
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Interactive dashboard (recommended) — Strategy Cliff, optimizer, AI engineer
streamlit run dashboard.py

# 2. Full data pipeline (2024 Singapore by default)
python run_pipeline.py

# 3. CLI AI strategist (prints briefing + live recommendation)
python run_agent.py

# 4. REST API
python run_api.py            # POST /strategy  (set reason:true for LLM output)

# 5. Validation backtest
python -m validation.backtest --year 2024 --gp Singapore
```

## AI race engineer (free)

The reasoning layer is provider-pluggable via `STRATUM_LLM_PROVIDER`:

- **Gemini (default, free):** get a key at
  [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then
  `setx GEMINI_API_KEY "..."` (Windows) / `export GEMINI_API_KEY=...`.
  Uses `gemini-2.5-flash`.
- **Claude:** `set STRATUM_LLM_PROVIDER=claude` + `ANTHROPIC_API_KEY`
  (uses `claude-opus-4-8`).

Without a key the app still runs; only the "Ask the AI Strategist" button is
disabled, with a helpful notice.

## Deploy a live demo (free)

The dashboard deploys to **Streamlit Community Cloud** in minutes:

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing
   at `dashboard.py`.
3. In the app's **Secrets**, add `GEMINI_API_KEY="..."` (Streamlit exposes
   secrets as env vars, which the reasoner reads automatically).

## Architecture

```
stratum-f1/
├── src/
│   ├── ingestion/      # FastF1 loading + batch/season loader
│   ├── canonical/      # Per-(driver, lap) race state
│   ├── features/       # Traffic, undercut, pit-loss features
│   ├── simulation/     # Tyre model, race simulator, pit-window optimizer
│   ├── llm/            # Prompt builder + Gemini/Claude reasoners
│   ├── visualization/  # Matplotlib race charts
│   └── api/            # FastAPI service
├── validation/         # Strategy backtest harness
├── dashboard.py        # Streamlit UI
├── run_pipeline.py · run_batch_pipeline.py · run_agent.py · run_api.py
├── docs/               # POC & design docs
└── tests/              # 67 unit/integration tests
```

| Module | Purpose |
|---|---|
| `src.ingestion` | Load F1 sessions via FastF1; persist parquet |
| `src.canonical` | Canonical race state (one row per driver × lap) |
| `src.features` | Strategy features (traffic, undercut, pit loss) |
| `src.simulation` | Fitted tyre model, simulator, pit-window optimizer + Monte-Carlo |
| `src.llm` | Race-briefing builder and live LLM reasoning (Gemini/Claude) |
| `src.api` | FastAPI `/strategy` and `/health` endpoints |
| `validation` | Backtest that grades the optimizer vs. reality |

## Testing

```bash
pytest tests/            # 67 tests (add PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 if a plugin conflicts)
```

## License

Research use only.
