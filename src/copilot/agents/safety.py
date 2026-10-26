"""Deterministic medication safety checks: allergies and kidney-function contraindications.

Run on the patient's current medications (to flag findings) and on every medication change
the reasoning step proposes (to reject unsafe drafts). Thresholds follow the sample care
pathway in the corpus; they demonstrate the mechanism and are not a validated rule set.
"""

import re
from typing import Literal

from pydantic import BaseModel

from copilot.agents.patient import Patient, daily_dose_mg

Severity = Literal["critical", "high", "moderate", "low"]
Change = Literal["start", "stop", "increase", "reduce", "continue"]
SEVERITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
ANTIHYPERTENSIVE_CLASSES = {"ace_inhibitor", "arb", "ccb", "thiazide"}
GLUCOSE_LOWERING_CLASSES = {
    "biguanide",
    "sulfonylurea",
    "sglt2_inhibitor",
    "glp1_agonist",
    "insulin",
}

# Drug class membership by generic name, plus suffix rules for names not listed.
DRUG_CLASSES: dict[str, set[str]] = {
    "ace_inhibitor": {"lisinopril", "ramipril", "enalapril", "perindopril", "captopril"},
    "arb": {"losartan", "candesartan", "valsartan", "irbesartan", "telmisartan", "olmesartan"},
    "sglt2_inhibitor": {"empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin"},
    "sulfonylurea": {"gliclazide", "glimepiride", "glipizide", "glibenclamide"},
    "biguanide": {"metformin"},
    "glp1_agonist": {"semaglutide", "liraglutide", "dulaglutide", "exenatide"},
    "insulin": {"insulin", "glargine", "detemir", "degludec", "aspart", "lispro"},
    "ccb": {"amlodipine", "nifedipine", "felodipine", "lercanidipine"},
    "thiazide": {"indapamide", "chlortalidone", "chlorthalidone", "hydrochlorothiazide"},
    "statin": {"atorvastatin", "rosuvastatin", "simvastatin", "pravastatin"},
    "penicillin": {"penicillin", "amoxicillin", "flucloxacillin", "co-amoxiclav"},
}
SUFFIXES = {
    "pril": "ace_inhibitor",
    "dipine": "ccb",
    "sartan": "arb",
    "gliflozin": "sglt2_inhibitor",
    "statin": "statin",
    "cillin": "penicillin",
}
CLASS_NAMES = {
    "ace inhibitor": "ace_inhibitor",
    "acei": "ace_inhibitor",
    "angiotensin receptor blocker": "arb",
    "arb": "arb",
    "sglt2": "sglt2_inhibitor",
    "sulfonylurea": "sulfonylurea",
    "sulphonylurea": "sulfonylurea",
    "glp-1": "glp1_agonist",
    "statin": "statin",
    "penicillin": "penicillin",
}


def drug_classes(name: str) -> set[str]:
    """Classes a drug (or a class name like "ACE inhibitor") belongs to."""
    lowered = name.lower()
    words = set(re.findall(r"[a-z0-9-]+", lowered))
    classes = {cls for cls, members in DRUG_CLASSES.items() if words & members}
    classes |= {cls for w in words for suffix, cls in SUFFIXES.items() if w.endswith(suffix)}
    classes |= {cls for alias, cls in CLASS_NAMES.items() if re.search(rf"\b{alias}\b", lowered)}
    return classes


class SafetyIssue(BaseModel):
    rule: str
    severity: Severity
    drug: str
    message: str


def check_medication(
    patient: Patient, drug: str, change: Change, dose: str = ""
) -> list[SafetyIssue]:
    """Issues raised by taking `drug` after `change`; `dose` is free text (daily total)."""
    if change == "stop":
        return []
    issues: list[SafetyIssue] = []
    classes = drug_classes(drug) | {drug.lower()}

    for allergy in patient.allergies:
        allergen = drug_classes(allergy.substance) | {allergy.substance.lower()}
        reaction = f" ({allergy.reaction})" if allergy.reaction else ""
        if classes & allergen:
            issues.append(
                SafetyIssue(
                    rule="allergy",
                    severity="critical",
                    drug=drug,
                    message=f"Patient is allergic to {allergy.substance}{reaction}; {drug} is "
                    "in the same class.",
                )
            )
        elif (
            "arb" in classes
            and "ace_inhibitor" in allergen
            and "angioedema" in allergy.reaction.lower()
        ):
            issues.append(
                SafetyIssue(
                    rule="allergy_cross_reactivity",
                    severity="high",
                    drug=drug,
                    message=f"ACE-inhibitor angioedema with {allergy.substance}: an ARB carries a "
                    "risk of recurrent angioedema and needs specialist consideration.",
                )
            )

    egfr_lab = patient.latest_lab("eGFR")
    egfr = egfr_lab.value if egfr_lab else None
    increasing = change in ("start", "increase")

    if "biguanide" in classes and egfr is not None:
        current = patient.medication(drug)
        mg = daily_dose_mg(dose) if dose else None
        if mg is None and change == "continue" and current:
            mg = current.daily_dose_mg
        if egfr < 30:
            issues.append(
                SafetyIssue(
                    rule="metformin_egfr",
                    severity="critical",
                    drug=drug,
                    message=f"Metformin is contraindicated at eGFR {egfr:g} (< 30); stop it.",
                )
            )
        elif egfr < 45 and change == "start":
            issues.append(
                SafetyIssue(
                    rule="metformin_egfr",
                    severity="high",
                    drug=drug,
                    message=f"Do not start metformin at eGFR {egfr:g} (30-44).",
                )
            )
        elif egfr < 45 and (change == "increase" or (mg is not None and mg > 1000)):
            dose_text = f" ({mg:g} mg/day)" if mg is not None else ""
            issues.append(
                SafetyIssue(
                    rule="metformin_egfr",
                    severity="high",
                    drug=drug,
                    message=f"Metformin dose{dose_text} exceeds the 1000 mg/day maximum at "
                    f"eGFR {egfr:g} (30-44); reduce the dose.",
                )
            )

    if "sglt2_inhibitor" in classes and change == "start" and egfr is not None and egfr < 20:
        issues.append(
            SafetyIssue(
                rule="sglt2_egfr",
                severity="high",
                drug=drug,
                message=f"Do not start an SGLT2 inhibitor at eGFR {egfr:g} (< 20).",
            )
        )

    if "sulfonylurea" in classes and change != "reduce" and egfr is not None and egfr < 60:
        issues.append(
            SafetyIssue(
                rule="sulfonylurea_ckd",
                severity="moderate",
                drug=drug,
                message=f"Sulfonylurea at eGFR {egfr:g}: increased hypoglycaemia risk in CKD.",
            )
        )

    potassium = patient.latest_lab("Potassium")
    if classes & {"ace_inhibitor", "arb"} and increasing and potassium and potassium.value > 5.5:
        issues.append(
            SafetyIssue(
                rule="raas_hyperkalaemia",
                severity="high",
                drug=drug,
                message=f"Potassium {potassium.value:g} mmol/L (> 5.5): starting or increasing "
                f"{drug} risks hyperkalaemia.",
            )
        )

    return issues


def current_medication_issues(patient: Patient) -> list[SafetyIssue]:
    issues = [
        issue
        for med in patient.medications
        for issue in check_medication(patient, med.name, "continue", f"{med.dose} {med.frequency}")
    ]
    return sorted(issues, key=lambda i: SEVERITY_ORDER[i.severity])
