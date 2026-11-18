# Grounding-check fixtures

Static test fixtures for verifying that an LLM race briefing is **grounded** in
the simulation output (i.e. it doesn't hallucinate pit laps, compounds, or
drivers). Generated from a real cached race (2024 Singapore GP, driver NOR,
decision lap 20).

| File | What it is |
|---|---|
| `sim_singapore_2024.json` | The saved simulation output — ground truth: the field, the recommended strategy (pit lap **26**, **MEDIUM → HARD**), and the tyre model. |
| `briefing_good.md` | The correct briefing generated from that output by `StrategyPromptBuilder`. A grounding check must **pass** this. |
| `briefing_bad.md` | The same briefing with three injected errors. A grounding check must **fail** this and flag all three. |

### Injected errors in `briefing_bad.md`
1. **Wrong pit lap** — recommends pitting on lap **12** instead of **26**.
2. **Wrong compound** — says **MEDIUM → SOFT** instead of **MEDIUM → HARD**.
3. **Phantom driver** — references **VET**, who is not in the race field.

Regenerate with the script referenced in the PR for `feat/grounding-check`
(loads the cached race, re-runs the optimizer, rewrites all three files).
