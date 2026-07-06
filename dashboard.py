"""
STRATUM-F1 — Interactive Strategy Dashboard

A Streamlit UI for exploring the "Strategy Cliff" and the pit-window
optimizer's recommendations for a chosen driver and decision lap.

Run with:
    streamlit run dashboard.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when launched via `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion.load_session import load_race_session
from src.canonical.build_race_state import build_race_state
from src.features.strategy_features import add_strategy_features
from src.simulation.tyre_model import TyreDegradationModel
from src.simulation.optimizer import PitWindowOptimizer, StrategyCandidate
from src.llm.prompt_builder import StrategyPromptBuilder
from src.llm.reasoner import get_reasoner, LLMUnavailableError

logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="STRATUM-F1 Strategy", page_icon="🏎️", layout="wide")


# ──────────────────────────────────────────────────────────────
# Cached data pipeline
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading race data & fitting tyre model…")
def load_pipeline(year: int, gp: str):
    """Load, enrich, and fit the tyre model for a race. Cached per race."""
    race_id = f"{year}_{gp.replace(' ', '_').lower()}"
    session_data = load_race_session(year=year, gp=gp)
    race_state = build_race_state(session_data, race_id=race_id)
    enriched = add_strategy_features(race_state)

    tyre_model = TyreDegradationModel()
    tyre_model.fit_from_race_data(enriched)
    return enriched, tyre_model


def build_cliff_frame(tyre_model: TyreDegradationModel, compounds, max_age: int = 45):
    """Build a tidy DataFrame of degradation penalty vs. tyre age per compound."""
    ages = list(range(0, max_age + 1))
    data = {"Tyre Age (laps)": ages}
    for comp in compounds:
        data[comp] = [tyre_model.degradation(comp, a) for a in ages]
    return pd.DataFrame(data).set_index("Tyre Age (laps)")


def candidates_to_frame(candidates: list[StrategyCandidate]) -> pd.DataFrame:
    """Render a list of strategy candidates as a display DataFrame."""
    rows = []
    for c in candidates:
        rows.append(
            {
                "Stops": len(c.pit_laps),
                "Pit Laps": ", ".join(map(str, c.pit_laps)),
                "Compounds": " → ".join(c.compounds),
                "Δ vs Best (s)": round(c.time_delta, 2),
                "Proj. Finish": f"P{c.expected_finish:.0f}",
                "Risk": round(c.risk, 3),
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────
st.sidebar.title("🏎️ STRATUM-F1")
st.sidebar.caption("Strategic Telemetry Reasoning for Formula 1")

year = st.sidebar.number_input("Season", min_value=2018, max_value=2025, value=2024)
gp = st.sidebar.text_input("Grand Prix", value="Singapore")
driver = st.sidebar.text_input("Driver (3-letter code)", value="NOR").upper()
decision_lap = st.sidebar.number_input("Decision Lap", min_value=1, value=20)
max_stops = st.sidebar.radio("Max stops", options=[1, 2], index=1, horizontal=True)
run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

st.title("Race Strategy — The Strategy Cliff")

if not run:
    st.info(
        "Set the race, driver, and decision lap in the sidebar, then press "
        "**Run Analysis** to load telemetry, fit the tyre model, and optimize "
        "the pit window."
    )
    st.stop()

# ──────────────────────────────────────────────────────────────
# Run the pipeline
# ──────────────────────────────────────────────────────────────
try:
    enriched, tyre_model = load_pipeline(year, gp)
except Exception as exc:  # noqa: BLE001 - surface any load failure to the UI
    st.error(f"Failed to load race data: {exc}")
    st.stop()

if driver not in set(enriched["driver"].unique()):
    st.error(
        f"Driver '{driver}' not found in {year} {gp}. "
        f"Available: {', '.join(sorted(enriched['driver'].unique()))}"
    )
    st.stop()

optimizer = PitWindowOptimizer(race_state=enriched, tyre_model=tyre_model)
optimal = optimizer.find_optimal(
    driver=driver, decision_lap=int(decision_lap), max_stops=int(max_stops), top_n=5
)
best = optimal.get("best_overall", [None])[0]

if best is None:
    st.warning("No viable strategy found for this driver/lap (race may end soon).")
    st.stop()

# ── Headline recommendation ──────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Recommended stops", f"{len(best.pit_laps)}-stop")
col2.metric("Pit lap(s)", ", ".join(map(str, best.pit_laps)))
col3.metric("Projected finish", f"P{best.expected_finish:.0f}")

# ── Uncertainty (Monte Carlo) ────────────────────────────────
dist = optimizer.simulate_finish_distribution(
    driver=driver,
    decision_lap=int(decision_lap),
    pit_laps=best.pit_laps,
    stint_compounds=best.compounds,
)
st.caption(
    f"Monte-Carlo outlook for the recommended strategy — "
    f"finish **P{dist['mean_finish']:.1f}** "
    f"(likely range P{dist['p10']}–P{dist['p90']})"
)
u1, u2, u3 = st.columns(3)
u1.metric("Win probability", f"{dist['p_win']*100:.0f}%")
u2.metric("Podium probability", f"{dist['p_podium']*100:.0f}%")
u3.metric("Points probability", f"{dist['p_points']*100:.0f}%")
finish_hist = (
    pd.Series(dist["positions"]).value_counts(normalize=True).sort_index()
)
finish_hist.index = [f"P{int(p)}" for p in finish_hist.index]
st.bar_chart(finish_hist, height=200, y_label="probability")

# ── Rival undercut check ─────────────────────────────────────
st.subheader("🎯 Undercut check — vs the car ahead")
uc = optimizer.evaluate_undercut(driver, int(decision_lap))
if uc.get("verdict") == "N/A":
    st.caption(uc.get("reason", "No undercut target."))
else:
    gap_txt = f"{uc['gap_s']:.1f}s" if uc.get("gap_s") is not None else "n/a"
    if uc["verdict"] == "UNDERCUT":
        st.success(
            f"**Undercut on {uc['rival']} is ON** — pitting now onto {uc['new_compound']} "
            f"nets ~{uc['net_s']:.1f}s over {uc['response_laps']} laps "
            f"(fresh-tyre gain {uc['undercut_gain_s']:.1f}s vs a {gap_txt} gap)."
        )
    else:
        st.info(
            f"**Hold vs {uc['rival']}** — the undercut gains only "
            f"{uc['undercut_gain_s']:.1f}s, not enough to clear the {gap_txt} gap "
            f"(net {uc['net_s']:.1f}s)."
        )

# ── The Strategy Cliff ───────────────────────────────────────
st.subheader("📉 The Strategy Cliff")
st.caption(
    "Cumulative lap-time penalty (seconds) as tyres age. The steep upturn is "
    "the cliff — where staying out stops paying off."
)
compounds_present = [
    c for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    if c in set(enriched["compound"].dropna().unique())
]
if not compounds_present:
    compounds_present = ["SOFT", "MEDIUM", "HARD"]
cliff_df = build_cliff_frame(tyre_model, compounds_present, max_age=45)
st.line_chart(cliff_df, height=380)

with st.expander("Fitted tyre model parameters"):
    st.dataframe(tyre_model.summary(), use_container_width=True)

# ── Optimizer projections ────────────────────────────────────
st.subheader("🧮 Optimizer Projections")
tab1, tab2 = st.tabs(["1-Stop", "2-Stop"])
with tab1:
    one_stop = optimal.get("1_stop", [])
    if one_stop:
        st.dataframe(candidates_to_frame(one_stop), use_container_width=True)
    else:
        st.write("No viable 1-stop strategies.")
with tab2:
    two_stop = optimal.get("2_stop", [])
    if two_stop:
        st.dataframe(candidates_to_frame(two_stop), use_container_width=True)
    else:
        st.write("No 2-stop strategies evaluated.")

# ── Race briefing + live reasoning ───────────────────────────
st.subheader("🤖 AI Strategist")
builder = StrategyPromptBuilder(race_state=enriched, tyre_model=tyre_model)
prompt = builder.build_prompt(
    driver=driver,
    lap=int(decision_lap),
    candidates_1_stop=optimal.get("1_stop", []),
    candidates_2_stop=optimal.get("2_stop", []),
    best_overall=best,
)

with st.expander("View the race briefing sent to the model"):
    st.markdown(prompt)

if st.button("Ask the AI Strategist", use_container_width=True):
    try:
        reasoner = get_reasoner()
        # Stream the recommendation token-by-token as the model generates it.
        st.write_stream(reasoner.stream_reason(prompt))
        result = reasoner.last_result
        if result and result.input_tokens is not None:
            st.caption(
                f"Model: {result.model} · {result.input_tokens} in / "
                f"{result.output_tokens} out tokens"
            )
    except LLMUnavailableError as exc:
        st.warning(
            f"Live reasoning unavailable: {exc}\n\n"
            "For the free Gemini backend, get a key at "
            "https://aistudio.google.com/apikey and set `GEMINI_API_KEY`."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Reasoning call failed: {exc}")
