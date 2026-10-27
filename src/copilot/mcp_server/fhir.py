"""Mock FHIR R4 boundary: patient records go out as a Bundle and come back as `Patient`.

Only the resources and fields the copilot uses are mapped (Patient, Condition,
MedicationStatement, AllergyIntolerance, Observation for labs and vitals, DocumentReference
for notes). Codes are LOINC where one exists; unmapped labs keep their display text.
"""

import base64

from copilot.agents.patient import Allergy, Lab, Medication, Note, Patient, Vital

LOINC = {
    "HbA1c": "4548-4",
    "eGFR": "33914-3",
    "UACR": "9318-7",
    "Potassium": "2823-3",
    "LDL cholesterol": "13457-7",
    "Total cholesterol": "2093-3",
    "Fasting plasma glucose": "1558-6",
}
VITALS = {  # Patient.vitals field -> (LOINC, display, unit)
    "bp_systolic": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "bp_diastolic": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "heart_rate": ("8867-4", "Heart rate", "/min"),
    "bmi": ("39156-5", "Body mass index", "kg/m2"),
}
VITAL_BY_CODE = {code: field for field, (code, _, _) in VITALS.items()}


def _entry(resource: dict) -> dict:
    return {
        "fullUrl": f"urn:uuid:{resource['resourceType']}-{resource['id']}",
        "resource": resource,
    }


def _observation(
    oid: str, pid: str, category: str, code: str | None, display: str, date, value, unit
):
    coding = [{"system": "http://loinc.org", "code": code, "display": display}] if code else []
    return {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "category": [{"coding": [{"code": category}]}],
        "code": {"coding": coding, "text": display},
        "subject": {"reference": f"Patient/{pid}"},
        "effectiveDateTime": date.isoformat(),
        "valueQuantity": {"value": value, "unit": unit},
    }


def to_bundle(p: Patient) -> dict:
    pid = p.patient_id
    subject = {"reference": f"Patient/{pid}"}
    resources: list[dict] = [
        {
            "resourceType": "Patient",
            "id": pid,
            "gender": p.demographics.get("sex"),
            "extension": [
                {"url": "age", "valueInteger": p.demographics.get("age")},
                {"url": "synthetic", "valueBoolean": p.synthetic},
                {"url": "summary", "valueString": p.summary},
            ],
        }
    ]
    resources += [
        {
            "resourceType": "Condition",
            "id": f"{pid}-cond-{i}",
            "subject": subject,
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "code": {"text": problem},
        }
        for i, problem in enumerate(p.problems)
    ]
    resources += [
        {
            "resourceType": "MedicationStatement",
            "id": f"{pid}-med-{i}",
            "status": "active",
            "subject": subject,
            "medicationCodeableConcept": {"text": m.name},
            "dosage": [
                {
                    "text": f"{m.dose} {m.frequency}".strip(),
                    "doseAndRate": [{"doseQuantityText": m.dose}],
                    "timing": {"code": {"text": m.frequency}},
                }
            ],
        }
        for i, m in enumerate(p.medications)
    ]
    resources += [
        {
            "resourceType": "AllergyIntolerance",
            "id": f"{pid}-allergy-{i}",
            "patient": subject,
            "code": {"text": a.substance},
            "reaction": [{"manifestation": [{"text": a.reaction}]}] if a.reaction else [],
        }
        for i, a in enumerate(p.allergies)
    ]
    resources += [
        _observation(
            f"{pid}-lab-{i}",
            pid,
            "laboratory",
            LOINC.get(lab.test),
            lab.test,
            lab.date,
            lab.value,
            lab.unit,
        )
        for i, lab in enumerate(p.labs)
    ]
    for i, v in enumerate(p.vitals):
        for field, (code, display, unit) in VITALS.items():
            value = getattr(v, field)
            if value is not None:
                resources.append(
                    _observation(
                        f"{pid}-vital-{i}-{field}",
                        pid,
                        "vital-signs",
                        code,
                        display,
                        v.date,
                        value,
                        unit,
                    )
                )
    resources += [
        {
            "resourceType": "DocumentReference",
            "id": f"{pid}-note-{i}",
            "status": "current",
            "subject": subject,
            "date": n.date.isoformat(),
            "author": [{"display": n.author}],
            "content": [
                {
                    "attachment": {
                        "contentType": "text/plain",
                        "data": base64.b64encode(n.text.encode()).decode(),
                    }
                }
            ],
        }
        for i, n in enumerate(p.notes)
    ]
    return {"resourceType": "Bundle", "type": "collection", "entry": [_entry(r) for r in resources]}


def from_bundle(bundle: dict) -> Patient:
    by_type: dict[str, list[dict]] = {}
    for e in bundle.get("entry", []):
        r = e["resource"]
        by_type.setdefault(r["resourceType"], []).append(r)

    patient = by_type["Patient"][0]
    ext = {
        x["url"]: next(v for k, v in x.items() if k.startswith("value"))
        for x in patient.get("extension", [])
    }
    vitals: dict[str, dict] = {}
    labs: list[Lab] = []
    for obs in by_type.get("Observation", []):
        date = obs["effectiveDateTime"][:10]
        value = obs["valueQuantity"]["value"]
        codes = [c.get("code") for c in obs["code"].get("coding", [])]
        field = next((VITAL_BY_CODE[c] for c in codes if c in VITAL_BY_CODE), None)
        if field:
            vitals.setdefault(date, {"date": date})[field] = value
        else:
            labs.append(
                Lab(
                    date=date,
                    test=obs["code"]["text"],
                    value=value,
                    unit=obs["valueQuantity"].get("unit", ""),
                )
            )

    medications = []
    for m in by_type.get("MedicationStatement", []):
        dosage = (m.get("dosage") or [{}])[0]
        medications.append(
            Medication(
                name=m["medicationCodeableConcept"]["text"],
                dose=(dosage.get("doseAndRate") or [{}])[0].get("doseQuantityText", ""),
                frequency=dosage.get("timing", {}).get("code", {}).get("text", ""),
            )
        )

    return Patient(
        patient_id=patient["id"],
        synthetic=ext.get("synthetic", False),
        demographics={"age": ext.get("age"), "sex": patient.get("gender")},
        summary=ext.get("summary", ""),
        problems=[c["code"]["text"] for c in by_type.get("Condition", [])],
        medications=medications,
        allergies=[
            Allergy(
                substance=a["code"]["text"],
                reaction=(a.get("reaction") or [{"manifestation": [{"text": ""}]}])[0][
                    "manifestation"
                ][0]["text"],
            )
            for a in by_type.get("AllergyIntolerance", [])
        ],
        vitals=[Vital.model_validate(v) for v in vitals.values()],
        labs=labs,
        notes=[
            Note(
                date=d["date"],
                author=(d.get("author") or [{"display": ""}])[0]["display"],
                text=base64.b64decode(d["content"][0]["attachment"]["data"]).decode(),
            )
            for d in by_type.get("DocumentReference", [])
        ],
    )
