"""
STRATUM-F1 — Grounding Check

Verifies that an LLM-written race briefing is *grounded* in the simulation
output: that it does not hallucinate pit laps, tyre compounds, or drivers.

Design constraints (see docs/grounding-check.md):
- **No LLM is used to check an LLM.** Claim extraction is deterministic:
  regex + the driver list taken from the simulation output.
- **Source of truth** is the simulation output object, not raw telemetry.
- **Laps and compounds must match exactly.** (Time-gap claims are *not*
  checked: the simulation output object carries no gap field, so there is
  nothing to ground them against.)
- **A recognised-but-unparseable claim is an error, never a silent skip.**

The checker parses the briefing section by section rather than scanning the
whole document. This is deliberate: the "Top Alternatives" block legitimately
names other pit laps/compounds, and the footer literally says "STAY OUT", so a
document-wide scan would misfire. Driver references are therefore only read
from the header and the Rival Context section; strategy claims only from the
Optimizer Projections section.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# Tyre compounds are the only all-caps tokens in the driver-reference sections
# that are not driver codes; exclude them from driver extraction.
_KNOWN_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}

# A driver code as it appears in this briefing format: 2-4 uppercase letters.
_DRIVER_TOKEN = re.compile(r"\b[A-Z]{2,4}\b")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    """A single factual assertion extracted from the briefing."""

    kind: str  # "stop_count" | "pit_laps" | "compounds" | "alternative" | "driver"
    raw: str  # the source text the claim was read from
    parsed: Any  # the parsed value (type depends on kind)
    section: str  # briefing section the claim came from


@dataclass
class GroundingError:
    """A claim that is not supported by the simulation output."""

    category: str  # "wrong pit lap" | "wrong compound" | "unknown driver" | ...
    message: str
    claim: Optional[Claim] = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"[{self.category}] {self.message}"


@dataclass
class GroundingResult:
    """Outcome of grounding a briefing against a simulation output."""

    errors: list[GroundingError] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def categories(self) -> list[str]:
        return [e.category for e in self.errors]

    def __bool__(self) -> bool:
        return self.passed


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_sim_output(path: Union[str, Path]) -> dict:
    """Load a simulation output object from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Section splitting
# --------------------------------------------------------------------------- #
def _split_sections(briefing: str) -> dict[str, str]:
    """Split a briefing into {heading: body} by its ``## `` headings.

    The text before the first ``## `` heading is stored under ``"_header"``.
    """
    sections: dict[str, str] = {}
    current = "_header"
    buf: list[str] = []
    for line in briefing.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf)
            current = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf)
    return sections


def _find_section(sections: dict[str, str], *needles: str) -> Optional[str]:
    """Return the body of the first section whose heading contains a needle."""
    for heading, body in sections.items():
        low = heading.lower()
        if any(n.lower() in low for n in needles):
            return body
    return None


# --------------------------------------------------------------------------- #
# Claim extraction
# --------------------------------------------------------------------------- #
def _extract_drivers(text: str, section: str) -> list[Claim]:
    """Extract driver-code references from a block of text."""
    claims: list[Claim] = []
    seen: set[str] = set()
    for token in _DRIVER_TOKEN.findall(text):
        if token in _KNOWN_COMPOUNDS or token in seen:
            continue
        seen.add(token)
        claims.append(Claim(kind="driver", raw=token, parsed=token, section=section))
    return claims


def _parse_int_list(raw: str) -> Optional[list[int]]:
    """Parse ``"26"`` or ``"20, 41"`` into ``[26]`` / ``[20, 41]``.

    Returns ``None`` if any token is not an integer (an unparseable claim).
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return None
    out: list[int] = []
    for p in parts:
        if not re.fullmatch(r"-?\d+", p):
            return None
        out.append(int(p))
    return out


def _parse_compounds(raw: str) -> Optional[list[str]]:
    """Parse ``"MEDIUM → HARD"`` into ``["MEDIUM", "HARD"]``.

    Returns ``None`` if any token is not a known compound.
    """
    parts = [p.strip().upper() for p in raw.split("→") if p.strip()]
    if not parts:
        return None
    if any(p not in _KNOWN_COMPOUNDS for p in parts):
        return None
    return parts


def extract_claims(briefing: str) -> tuple[list[Claim], list[GroundingError]]:
    """Extract grounded claims from a briefing.

    Returns ``(claims, parse_errors)``. A claim line that is recognised (its
    label is present) but whose value cannot be parsed is reported as a parse
    error rather than being dropped.
    """
    sections = _split_sections(briefing)
    claims: list[Claim] = []
    parse_errors: list[GroundingError] = []

    # --- Driver references: header + rival context only --------------------- #
    header = sections.get("_header", "")
    m = re.search(r"Target Driver:\*{0,2}\s*([A-Z]{2,4})", header)
    if m:
        claims.append(
            Claim(kind="driver", raw=m.group(0), parsed=m.group(1), section="header")
        )

    rival = _find_section(sections, "Rival Context")
    if rival is not None:
        claims.extend(_extract_drivers(rival, section="rival_context"))

    # --- Strategy claims: optimizer projections only ------------------------ #
    opt = _find_section(sections, "Optimizer Projections")
    if opt is None:
        parse_errors.append(
            GroundingError(
                category="unparseable claim",
                message="No 'Optimizer Projections' section found in briefing.",
            )
        )
        return claims, parse_errors

    # Recommended stop count.
    m = re.search(r"RECOMMENDED:\s*(\d+)-STOP", opt)
    if m:
        claims.append(
            Claim(kind="stop_count", raw=m.group(0), parsed=int(m.group(1)),
                  section="recommended")
        )

    # Recommended pit laps.
    m = re.search(r"Pits at:\*{0,2}\s*Laps?\s*([^\n]+)", opt)
    if m:
        laps = _parse_int_list(m.group(1))
        if laps is None:
            parse_errors.append(
                GroundingError(
                    category="unparseable claim",
                    message=f"Could not parse recommended pit laps from: "
                            f"{m.group(1).strip()!r}",
                )
            )
        else:
            claims.append(
                Claim(kind="pit_laps", raw=m.group(0).strip(), parsed=laps,
                      section="recommended")
            )
    else:
        parse_errors.append(
            GroundingError(
                category="unparseable claim",
                message="No recommended 'Pits at:' line found in briefing.",
            )
        )

    # Recommended compounds.
    m = re.search(r"Compounds:\*{0,2}\s*([^\n]+)", opt)
    if m:
        comps = _parse_compounds(m.group(1))
        if comps is None:
            parse_errors.append(
                GroundingError(
                    category="unparseable claim",
                    message=f"Could not parse recommended compounds from: "
                            f"{m.group(1).strip()!r}",
                )
            )
        else:
            claims.append(
                Claim(kind="compounds", raw=m.group(0).strip(), parsed=comps,
                      section="recommended")
            )
    else:
        parse_errors.append(
            GroundingError(
                category="unparseable claim",
                message="No recommended 'Compounds:' line found in briefing.",
            )
        )

    # Alternative strategies.
    for am in re.finditer(
        r"\d+-Stop\s*\(Pits:\s*([^,]+(?:,\s*\d+)*),\s*([A-Z →]+)\)", opt
    ):
        laps = _parse_int_list(am.group(1))
        comps = _parse_compounds(am.group(2))
        if laps is None or comps is None:
            parse_errors.append(
                GroundingError(
                    category="unparseable claim",
                    message=f"Could not parse alternative strategy: "
                            f"{am.group(0).strip()!r}",
                )
            )
            continue
        claims.append(
            Claim(kind="alternative", raw=am.group(0).strip(),
                  parsed={"pit_laps": laps, "compounds": comps},
                  section="alternatives")
        )

    return claims, parse_errors


# --------------------------------------------------------------------------- #
# Checking
# --------------------------------------------------------------------------- #
def check_briefing(briefing: str, sim: dict) -> GroundingResult:
    """Ground every checkable claim in ``briefing`` against ``sim``.

    Args:
        briefing: The LLM race-briefing markdown.
        sim: The simulation output object (see fixtures/sim_singapore_2024.json).

    Returns:
        A :class:`GroundingResult`. ``result.passed`` is True only if every
        extracted claim is supported by the simulation output.
    """
    claims, errors = extract_claims(briefing)

    drivers = set(sim.get("drivers_in_race", []))
    best = sim.get("best_strategy", {}) or {}
    best_laps = list(best.get("pit_laps", []))
    best_comps = [str(c).upper() for c in best.get("compounds", [])]
    best_stops = best.get("stops")

    alternatives = [
        (
            list(a.get("pit_laps", [])),
            [str(c).upper() for c in a.get("compounds", [])],
        )
        for a in sim.get("alternatives", []) or []
    ]

    for claim in claims:
        if claim.kind == "driver":
            if claim.parsed not in drivers:
                errors.append(
                    GroundingError(
                        category="unknown driver",
                        message=f"Briefing references driver {claim.parsed!r}, "
                                f"who is not in the race field.",
                        claim=claim,
                    )
                )

        elif claim.kind == "pit_laps":
            if claim.parsed != best_laps:
                errors.append(
                    GroundingError(
                        category="wrong pit lap",
                        message=f"Briefing recommends pitting on lap(s) "
                                f"{claim.parsed}, but the simulation's best "
                                f"strategy pits on {best_laps}.",
                        claim=claim,
                    )
                )

        elif claim.kind == "compounds":
            if claim.parsed != best_comps:
                errors.append(
                    GroundingError(
                        category="wrong compound",
                        message=f"Briefing recommends compounds "
                                f"{' → '.join(claim.parsed)}, but the "
                                f"simulation's best strategy is "
                                f"{' → '.join(best_comps)}.",
                        claim=claim,
                    )
                )

        elif claim.kind == "stop_count":
            if best_stops is not None and claim.parsed != best_stops:
                errors.append(
                    GroundingError(
                        category="wrong pit lap",
                        message=f"Briefing recommends a {claim.parsed}-stop "
                                f"strategy, but the simulation's best strategy "
                                f"has {best_stops} stop(s).",
                        claim=claim,
                    )
                )

        elif claim.kind == "alternative":
            key = (claim.parsed["pit_laps"], claim.parsed["compounds"])
            if key not in alternatives:
                errors.append(
                    GroundingError(
                        category="unsupported alternative",
                        message=f"Briefing lists an alternative "
                                f"(pits {claim.parsed['pit_laps']}, "
                                f"{' → '.join(claim.parsed['compounds'])}) that "
                                f"is not among the simulation's alternatives.",
                        claim=claim,
                    )
                )

    return GroundingResult(errors=errors)


def verify_files(
    briefing_path: Union[str, Path], sim_path: Union[str, Path]
) -> GroundingResult:
    """Convenience wrapper: load a briefing file and a sim-output file, check."""
    briefing = Path(briefing_path).read_text(encoding="utf-8")
    sim = load_sim_output(sim_path)
    return check_briefing(briefing, sim)
