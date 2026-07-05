# STRATUM-F1 — Impact Sprint Plan

**Goal:** raise the project from "working demo" to "validated, credible, and
visible." Priorities are ordered by leverage: prove it works → make the model
better → get it in front of people.

> Baseline established by the [Strategy Validation POC](docs/POC_STRATEGY_VALIDATION.md):
> on 2024 Singapore the optimizer has an **11.9-lap pit MAE**, **3.0-position
> finish MAE**, but **85% stop-count accuracy** and correctly picks the winner.
> These are the numbers each sprint drives down.
>
> **Progress (2026-07-05):** Sprints 1–3 delivered. After calibration the same
> race scores **pit MAE 5.7**, **finish MAE 0.71** (vs 1.86 naive baseline — 2.6×
> better), **85% stop-count**, plus Monte-Carlo win/podium/points probabilities
> in the dashboard. 67 tests pass.

---

## Sprint 1 — Prove It Works (Validation Harness) ✅ *POC complete*
**Objective:** a measurable, repeatable score for strategy quality.

*   [x] `validation/backtest.py` — grades optimizer vs. actual pit laps & finish.
*   [x] Metrics: pit-lap MAE, ±2-lap hit rate, stop-count match, finish MAE.
*   [x] First real result captured on 2024 Singapore (see POC doc).
*   [x] Naive persistence baseline added — proves added value (0.71 vs 1.86 MAE).
*   [x] `build_race_state` confirmed to stamp `race_id` (the old parquet was a
    stale artifact; the backtest loads clean single races).
*   [ ] Multi-race sweep (`--season 2024`) averaging metrics across events.

**Acceptance:** ✅ `python -m validation.backtest` outputs a metrics table for
any race, with a baseline comparison. *(Season sweep still open.)*

## Sprint 2 — Make the Model Better (Calibration) ✅
**Objective:** drive the POC's error metrics down using the harness as the scorer.

*   [x] **Fixed the tyre fit:** reject low-R² "magic tyre" fits, keep physical
    defaults, enforce SOFT ≥ MEDIUM ≥ HARD (`tyre_model.py`).
*   [x] **Fixed the finish estimator:** rivals now projected with a nominal stop
    → positions spread; finish MAE **3.0 → 0.71**.
*   [x] **Added track-position penalty** (`stop_penalty`) → pit MAE **11.9 → 5.7**,
    stop-count match back to 85%.
*   [x] Re-ran the backtest after each change; trend recorded in the POC doc.
*   [x] Regression tests added (`tests/test_calibration.py`, 4 tests).

**Acceptance:** ✅ finish MAE 0.71 (< 1.5 target); pit MAE 5.7 (target < 4 —
partially met; residual gap is track-position modelling, a Sprint 5 item).

## Sprint 3 — Quantify Uncertainty (Monte Carlo) ✅
**Objective:** replace point estimates with probabilities — what real pit walls show.

*   [x] `PitWindowOptimizer.simulate_finish_distribution(...)` — Monte-Carlo over
    per-lap noise (total variance ∝ √laps).
*   [x] Reports **win / podium / points probability** and a P10–P90 finish range.
*   [x] Dashboard shows the probabilities + a finish-position distribution chart
    under the headline recommendation.

**Acceptance:** ✅ dashboard shows e.g. "finish P3 (likely P2–P5)" with win/
podium/points % and a distribution bar chart.

## Sprint 4 — Get It In Front of People (Deploy + Story) 🟡 *prepped*
**Objective:** a clickable live demo and a compelling narrative.

*   [x] README rewritten: validation metrics up top, architecture, pipeline,
    run/deploy instructions, provider setup.
*   [x] POC validation table published in the README (credibility numbers).
*   [x] `.streamlit/config.toml` theme + Streamlit Cloud deploy steps documented
    (secrets → `GEMINI_API_KEY`).
*   [ ] **User action:** push to GitHub + click-deploy on share.streamlit.io
    (needs your account) → paste the live URL into the README badge.
*   [ ] Record a demo GIF of the strategist streaming a recommendation.
*   [ ] Write one case-study race narrative.

**Acceptance:** repo is deploy-ready; once you deploy, a public URL + GIF finish
this sprint.

## Sprint 5 — Robustness & Polish (Stretch)
*   [ ] Season-wide validation across ≥ 5 2024 races; report aggregate metrics.
*   [ ] Sidebar model selector (Gemini / Claude) once multi-provider is stable.
*   [ ] Cache warmed race data so first load of common races is instant.
*   [ ] CI: run `tests/` + a fast backtest smoke check on push.

---

## Priority summary

| Rank | Item | Why it matters | Sprint |
|---|---|---|---|
| 1 | Validation harness + multi-race sweep | Turns claims into measured facts | 1 |
| 2 | Model calibration (pit timing, finish) | Fixes the defects the POC exposed | 2 |
| 3 | Deploy + README with demo/diagram/metrics | Visibility — a live link beats a repo | 4 |
| 4 | Uncertainty / probabilities | Credible, pit-wall-grade output | 3 |
| 5 | Season robustness + CI | Shows it generalizes | 5 |

**Recommended order:** Sprint 1 → 2 → 4 (deploy) → 3 → 5. Ship the live demo
(Sprint 4) as soon as the model is calibrated so people see a *correct* engine.
