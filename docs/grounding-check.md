# Grounding check — spec

After the LLM writes a race briefing, verify every factual claim against the
simulation output. Any unsupported claim fails the briefing.

- **Claims to check:** lap numbers, pit-stop count, tyre compounds, driver
  names, time gaps.
- **Source of truth:** the simulation output object, not the raw telemetry.
- **Tolerance:** laps and compounds must match exactly. Gaps within 0.5s.
- **Parsing:** deterministic (regex + the driver list from the sim output). No
  LLM is used to check an LLM.
- **On failure:** return a list of the unsupported claims and block the
  briefing.
- **Out of scope:** do not change the simulation, the prompt, or existing tests.
- **Done when:** the good fixture passes, the bad fixture fails with all three
  errors named (wrong pit lap, wrong compound, unknown driver), and the full
  suite is green.
