"""Deterministic patient analysis: lab/vital thresholds and medication safety -> findings.

Rules decide *what* is abnormal; the guideline corpus (retrieved per finding) decides what
to do about it. Each finding carries the search query used to find that evidence.
"""

from typing import Literal

from pydantic import BaseModel

from copilot.agents.patient import Patient
from copilot.agents.safety import (
    ANTIHYPERTENSIVE_CLASSES,
    GLUCOSE_LOWERING_CLASSES,
    SEVERITY_ORDER,
    Severity,
    current_medication_issues,
    drug_classes,
)

Category = Literal["glycaemic", "blood_pressure", "renal", "electrolytes", "lipids", "medication"]


class Finding(BaseModel):
    id: str
    category: Category
    severity: Severity
    title: str
    detail: str
    query: str  # guideline search query for this finding


def _trend(series, unit: str) -> str:
    if len(series) < 2:
        return ""
    prev = series[-2]
    return f" (was {prev.value:g}{unit} on {prev.date.isoformat()})"


def glycaemic(p: Patient) -> list[Finding]:
    series = p.lab_series("HbA1c")
    if not series:
        return []
    a1c = series[-1].value
    trend = _trend(series, "%")
    rising = len(series) > 1 and a1c - series[-2].value >= 0.5

    if not p.has_problem("diabetes"):
        if a1c >= 6.5:
            return [
                Finding(
                    id="hba1c-diabetes-range",
                    category="glycaemic",
                    severity="high",
                    title="HbA1c in the diabetes range without a diagnosis",
                    detail=f"HbA1c {a1c:g}% (>= 6.5%){trend}.",
                    query="diagnosis of type 2 diabetes HbA1c 6.5% threshold confirmatory test",
                )
            ]
        if a1c >= 5.7:
            return [
                Finding(
                    id="hba1c-prediabetes",
                    category="glycaemic",
                    severity="low",
                    title="HbA1c in the pre-diabetes range",
                    detail=f"HbA1c {a1c:g}% (5.7-6.4%).",
                    query="intermediate hyperglycaemia prediabetes HbA1c lifestyle intervention",
                )
            ]
        return []

    if a1c <= 7.0:
        return []
    meds = ", ".join(
        m.name for m in p.medications if drug_classes(m.name) & GLUCOSE_LOWERING_CLASSES
    )
    ckd = " with chronic kidney disease" if p.has_problem("kidney", "ckd") else ""
    return [
        Finding(
            id="hba1c-above-target",
            category="glycaemic",
            severity="high" if a1c > 8.0 else "moderate",
            title="Glycaemic control above target" + (", worsening" if rising else ""),
            detail=f"HbA1c {a1c:g}%{trend} on {meds or 'no glucose-lowering medication'}.",
            query=f"intensify glucose-lowering therapy type 2 diabetes{ckd} HbA1c above target",
        )
    ]


def blood_pressure(p: Patient) -> list[Finding]:
    readings = p.bp_readings()
    if not readings:
        return []
    last = readings[-1]
    sys_, dia = last.bp_systolic, last.bp_diastolic
    high_risk = p.has_problem("diabetes", "kidney", "ckd", "cardiovascular", "coronary", "stroke")
    target_sys = 130 if high_risk else 140
    reading = f"BP {sys_}/{dia} mmHg on {last.date.isoformat()}"
    antihypertensives = ", ".join(
        m.name for m in p.medications if drug_classes(m.name) & ANTIHYPERTENSIVE_CLASSES
    )
    on_treatment = f" on {antihypertensives}" if antihypertensives else ""

    if sys_ >= 180 or dia >= 110:
        return [
            Finding(
                id="bp-severe",
                category="blood_pressure",
                severity="critical",
                title="Severely raised blood pressure",
                detail=f"{reading} (>= 180/110).",
                query="severe hypertension 180/110 urgent assessment",
            )
        ]
    if sys_ >= 140 or dia >= 90 or sys_ >= target_sys:
        persistent = len(readings) > 1 and readings[-2].bp_systolic >= target_sys
        return [
            Finding(
                id="bp-above-target",
                category="blood_pressure",
                severity="high" if sys_ >= 140 or dia >= 90 else "moderate",
                title="Blood pressure above target" + (", persistently" if persistent else ""),
                detail=f"{reading}{on_treatment}; target systolic < {target_sys} mmHg"
                + (" (diabetes/CKD/CVD)." if high_risk else "."),
                query=f"hypertension not at target{on_treatment} treatment intensification "
                "follow-up interval",
            )
        ]
    return []


def renal(p: Patient) -> list[Finding]:
    findings: list[Finding] = []
    series = p.lab_series("eGFR")
    if series:
        egfr = series[-1].value
        stage = (
            (15, "critical", "kidney failure range (< 15)"),
            (30, "high", "CKD stage 4 (15-29)"),
            (45, "moderate", "CKD stage 3b (30-44)"),
            (60, "low", "CKD stage 3a (45-59)"),
        )
        band = next((s for s in stage if egfr < s[0]), None)
        decline = ""
        severity = band[1] if band else None
        if len(series) > 1:
            prev = series[-2]
            months = max((series[-1].date - prev.date).days / 30.4, 1)
            drop = prev.value - egfr
            if drop >= 5 and months <= 12 or drop >= 0.25 * prev.value:
                decline = f"; fell {drop:g} in {months:.0f} months from {prev.value:g}"
                if severity is None or SEVERITY_ORDER[severity] > SEVERITY_ORDER["high"]:
                    severity = "high"
        if severity:
            findings.append(
                Finding(
                    id="egfr-reduced",
                    category="renal",
                    severity=severity,
                    title="Reduced kidney function" + (", declining" if decline else ""),
                    detail=f"eGFR {egfr:g} ml/min/1.73m2"
                    + (f", {band[2]}" if band else "")
                    + f"{decline}.",
                    query="chronic kidney disease declining eGFR monitoring frequency management "
                    "diabetes",
                )
            )

    uacr = p.latest_lab("UACR")
    if uacr and uacr.value >= 3:
        severe = uacr.value >= 30
        findings.append(
            Finding(
                id="albuminuria",
                category="renal",
                severity="high" if severe else "moderate",
                title=("Severely increased" if severe else "Moderately increased") + " albuminuria",
                detail=f"UACR {uacr.value:g} mg/mmol ({'A3, >= 30' if severe else 'A2, 3-29'}).",
                query="albuminuria UACR diabetes CKD kidney protection treatment",
            )
        )
    return findings


def electrolytes(p: Patient) -> list[Finding]:
    k = p.latest_lab("Potassium")
    if not k or k.value < 5.0:
        return []
    severity: Severity = "critical" if k.value > 6.0 else "high" if k.value > 5.5 else "low"
    return [
        Finding(
            id="potassium-raised",
            category="electrolytes",
            severity=severity,
            title="Raised potassium" if k.value > 5.5 else "Upper-normal potassium",
            detail=f"Potassium {k.value:g} mmol/L; relevant to ACE inhibitor/ARB and "
            "mineralocorticoid antagonist use.",
            query="hyperkalaemia potassium monitoring ACE inhibitor ARB CKD",
        )
    ]


def lipids(p: Patient) -> list[Finding]:
    on_statin = any("statin" in drug_classes(m.name) for m in p.medications)
    high_risk = p.has_problem("diabetes", "kidney", "ckd", "cardiovascular")
    ldl = p.latest_lab("LDL cholesterol")
    if ldl and high_risk and ldl.value > 2.6:
        return [
            Finding(
                id="ldl-raised",
                category="lipids",
                severity="moderate",
                title="LDL cholesterol raised for cardiovascular risk"
                + ("" if on_statin else ", no statin"),
                detail=f"LDL {ldl.value:g} mmol/L; diabetes/CKD confers high cardiovascular risk.",
                query="statin therapy diabetes chronic kidney disease cardiovascular risk LDL",
            )
        ]
    total = p.latest_lab("Total cholesterol")
    if total and total.value >= 5.0 and not on_statin:
        return [
            Finding(
                id="cholesterol-raised",
                category="lipids",
                severity="low",
                title="Raised total cholesterol",
                detail=f"Total cholesterol {total.value:g} mmol/L; assess cardiovascular risk.",
                query="cardiovascular risk assessment cholesterol statin hypertension",
            )
        ]
    return []


def medication(p: Patient) -> list[Finding]:
    return [
        Finding(
            id=f"med-{issue.rule}-{issue.drug.lower()}",
            category="medication",
            severity=issue.severity,
            title=f"Medication safety: {issue.drug}",
            detail=issue.message,
            query=f"{issue.drug} kidney function eGFR dose adjustment contraindication",
        )
        for issue in current_medication_issues(p)
    ]


RULES = (glycaemic, blood_pressure, renal, electrolytes, lipids, medication)


def analyse_patient(p: Patient) -> list[Finding]:
    """All findings, most severe first."""
    findings = [f for rule in RULES for f in rule(p)]
    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])
