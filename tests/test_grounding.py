"""
STRATUM-F1 — Grounding-check tests

Verifies the deterministic grounding checker against the static fixtures:
- the *good* briefing must pass,
- the *bad* briefing must fail and name all three injected errors
  (wrong pit lap, wrong compound, unknown driver).

No LLM is involved; these are pure parsing/assertion tests.
"""

from pathlib import Path

import pytest

from src.llm.grounding import (
    Claim,
    check_briefing,
    extract_claims,
    load_sim_output,
    verify_files,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SIM_PATH = FIXTURES / "sim_singapore_2024.json"
GOOD_PATH = FIXTURES / "briefing_good.md"
BAD_PATH = FIXTURES / "briefing_bad.md"


@pytest.fixture
def sim() -> dict:
    return load_sim_output(SIM_PATH)


@pytest.fixture
def good_briefing() -> str:
    return GOOD_PATH.read_text(encoding="utf-8")


@pytest.fixture
def bad_briefing() -> str:
    return BAD_PATH.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Good briefing: must pass
# --------------------------------------------------------------------------- #
def test_good_briefing_passes(good_briefing: str, sim: dict) -> None:
    result = check_briefing(good_briefing, sim)
    assert result.passed, f"expected pass, got errors: {result.categories}"
    assert result.errors == []
    assert bool(result) is True


def test_verify_files_matches_check_briefing() -> None:
    result = verify_files(GOOD_PATH, SIM_PATH)
    assert result.passed


# --------------------------------------------------------------------------- #
# Bad briefing: must fail and name all three injected errors
# --------------------------------------------------------------------------- #
def test_bad_briefing_fails(bad_briefing: str, sim: dict) -> None:
    result = check_briefing(bad_briefing, sim)
    assert not result.passed
    assert bool(result) is False


def test_bad_briefing_flags_wrong_pit_lap(bad_briefing: str, sim: dict) -> None:
    result = check_briefing(bad_briefing, sim)
    errs = [e for e in result.errors if e.category == "wrong pit lap"]
    assert len(errs) == 1
    # recommends lap 12 instead of the simulation's lap 26
    assert "12" in errs[0].message
    assert "26" in errs[0].message


def test_bad_briefing_flags_wrong_compound(bad_briefing: str, sim: dict) -> None:
    result = check_briefing(bad_briefing, sim)
    errs = [e for e in result.errors if e.category == "wrong compound"]
    assert len(errs) == 1
    assert "SOFT" in errs[0].message
    assert "HARD" in errs[0].message


def test_bad_briefing_flags_unknown_driver(bad_briefing: str, sim: dict) -> None:
    result = check_briefing(bad_briefing, sim)
    errs = [e for e in result.errors if e.category == "unknown driver"]
    assert len(errs) == 1
    assert "VET" in errs[0].message


def test_bad_briefing_names_all_three_errors(bad_briefing: str, sim: dict) -> None:
    """The spec's 'done when': all three errors named, no extras."""
    result = check_briefing(bad_briefing, sim)
    assert sorted(result.categories) == [
        "unknown driver",
        "wrong compound",
        "wrong pit lap",
    ]


# --------------------------------------------------------------------------- #
# Extraction details
# --------------------------------------------------------------------------- #
def test_extract_claims_on_good_briefing(good_briefing: str) -> None:
    claims, parse_errors = extract_claims(good_briefing)
    assert parse_errors == []

    by_kind: dict[str, list[Claim]] = {}
    for c in claims:
        by_kind.setdefault(c.kind, []).append(c)

    # recommended strategy claims
    assert [c.parsed for c in by_kind["pit_laps"]] == [[26]]
    assert [c.parsed for c in by_kind["compounds"]] == [["MEDIUM", "HARD"]]
    assert [c.parsed for c in by_kind["stop_count"]] == [1]

    # three alternatives parsed from the alternatives block
    assert len(by_kind["alternative"]) == 3

    # drivers: target (NOR) from header + VER from rival context, nothing else
    assert {c.parsed for c in by_kind["driver"]} == {"NOR", "VER"}


def test_alternatives_do_not_trip_the_recommended_check(
    good_briefing: str, sim: dict
) -> None:
    """Laps 25/27/20/41 appear in the alternatives block but must not be
    read as the recommended pit lap."""
    result = check_briefing(good_briefing, sim)
    assert result.passed


def test_footer_stay_out_is_not_a_driver(good_briefing: str) -> None:
    """'STAY OUT' in the footer must not be mistaken for a driver code."""
    claims, _ = extract_claims(good_briefing)
    driver_names = {c.parsed for c in claims if c.kind == "driver"}
    assert "OUT" not in driver_names
    assert "PIT" not in driver_names


# --------------------------------------------------------------------------- #
# Unparseable claim is an error, never a silent skip
# --------------------------------------------------------------------------- #
def test_unparseable_pit_lap_is_error_not_skip(good_briefing: str, sim: dict) -> None:
    broken = good_briefing.replace("**Pits at:** Laps 26", "**Pits at:** Laps TWELVE")
    claims, parse_errors = extract_claims(broken)

    # the broken line is recognised but not parsed as a pit-lap claim ...
    assert all(c.kind != "pit_laps" for c in claims)
    # ... and it surfaces as an explicit parse error
    assert any(e.category == "unparseable claim" for e in parse_errors)

    # it must also fail the overall check (not pass by omission)
    result = check_briefing(broken, sim)
    assert not result.passed
    assert any(e.category == "unparseable claim" for e in result.errors)


def test_missing_optimizer_section_is_error(sim: dict) -> None:
    result = check_briefing("# Briefing\n\nNo projections here.\n", sim)
    assert not result.passed
    assert any(e.category == "unparseable claim" for e in result.errors)
