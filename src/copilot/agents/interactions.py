"""Pairwise drug-drug interaction checks over a small curated table (by drug class).

A demonstration table of well-known, class-level interactions relevant to diabetes, CKD and
hypertension care; not a substitute for a licensed interaction database.
"""

from itertools import combinations

from pydantic import BaseModel

from copilot.agents.safety import Severity, drug_classes

# (class or drug, class or drug) -> (severity, message)
INTERACTIONS: dict[frozenset[str], tuple[Severity, str]] = {
    frozenset({"ace_inhibitor", "arb"}): (
        "high",
        "Dual RAAS blockade (ACE inhibitor + ARB) increases hyperkalaemia, hypotension and acute "
        "kidney injury risk without added benefit; avoid the combination.",
    ),
    frozenset({"ace_inhibitor", "potassium_sparing"}): (
        "high",
        "ACE inhibitor with a potassium-sparing diuretic or MRA: risk of hyperkalaemia; "
        "monitor potassium and eGFR closely.",
    ),
    frozenset({"arb", "potassium_sparing"}): (
        "high",
        "ARB with a potassium-sparing diuretic or MRA: risk of hyperkalaemia; monitor potassium "
        "and eGFR closely.",
    ),
    frozenset({"sulfonylurea", "insulin"}): (
        "moderate",
        "Sulfonylurea with insulin increases hypoglycaemia risk; consider dose reduction.",
    ),
    frozenset({"nsaid", "ace_inhibitor"}): (
        "moderate",
        "NSAID with an ACE inhibitor can reduce kidney function and antihypertensive effect.",
    ),
    frozenset({"nsaid", "arb"}): (
        "moderate",
        "NSAID with an ARB can reduce kidney function and antihypertensive effect.",
    ),
    frozenset({"sglt2_inhibitor", "loop_diuretic"}): (
        "moderate",
        "SGLT2 inhibitor with a loop diuretic: additive volume depletion; review diuretic dose.",
    ),
    frozenset({"statin", "macrolide"}): (
        "high",
        "Statin with clarithromycin/erythromycin raises statin levels and myopathy risk; "
        "withhold the statin during the course.",
    ),
}
EXTRA_CLASSES: dict[str, set[str]] = {
    "potassium_sparing": {"spironolactone", "eplerenone", "amiloride", "finerenone"},
    "nsaid": {"ibuprofen", "naproxen", "diclofenac", "celecoxib"},
    "loop_diuretic": {"furosemide", "bumetanide"},
    "macrolide": {"clarithromycin", "erythromycin"},
}


class Interaction(BaseModel):
    drugs: tuple[str, str]
    severity: Severity
    message: str


def classes(drug: str) -> set[str]:
    words = set(drug.lower().split())
    return drug_classes(drug) | {c for c, members in EXTRA_CLASSES.items() if words & members}


def check_interactions(drugs: list[str]) -> list[Interaction]:
    found: list[Interaction] = []
    for a, b in combinations(dict.fromkeys(drugs), 2):  # unique, order-preserving
        for ca in classes(a):
            for cb in classes(b):
                hit = INTERACTIONS.get(frozenset({ca, cb})) if ca != cb else None
                if not hit:
                    continue
                interaction = Interaction(drugs=(a, b), severity=hit[0], message=hit[1])
                if interaction not in found:
                    found.append(interaction)
    return found
