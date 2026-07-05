# STRATUM-F1

**Strategic Telemetry Reasoning and Adaptive Tactical Understanding Model for Formula 1**

A research-grade system for ingesting Formula 1 telemetry data, building canonical race state datasets, simulating counterfactual race strategies, and supporting a future GenAI reasoning layer.

---

## Architecture

```
stratum-f1/
├── data/
│   ├── raw/            # Raw parquet files from FastF1
│   ├── processed/      # Cleaned/canonical datasets
│   └── features/       # Feature-engineered datasets
├── src/
│   ├── ingestion/      # FastF1 data loading
│   ├── canonical/      # Race state builder
│   ├── features/       # Feature engineering
│   └── simulation/     # Strategy simulation engine
├── notebooks/          # Exploratory analysis
├── tests/              # Unit and integration tests
├── run_pipeline.py     # End-to-end pipeline script
└── requirements.txt
```

## Modules

| Module | Purpose |
|---|---|
| `src.ingestion` | Load F1 session data via FastF1 and persist as parquet |
| `src.canonical` | Build a canonical race state (one row per driver × lap) |
| `src.features` | Add strategy-relevant features (traffic, undercut, pit loss) |
| `src.simulation` | Simulate counterfactual pit strategies and estimate outcomes |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (2024 Singapore GP by default)
python run_pipeline.py
```

## Data Pipeline

1. **Ingest** — Load race session from FastF1, save raw parquet files
2. **Canonicalize** — Build per-driver-per-lap race state with rolling averages
3. **Feature Engineer** — Add strategy metrics (traffic penalty, undercut estimate, pit loss)
4. **Simulate** — Project counterfactual outcomes for pit timing decisions

## License

Research use only.
