"""Patient record model (a simplified, FHIR-inspired shape) and loader for synthetic records."""

import datetime as dt
import re

from pydantic import BaseModel, Field

from copilot.config import PROJECT_ROOT

PATIENTS_DIR = PROJECT_ROOT / "data" / "patients"


class Medication(BaseModel):
    name: str
    dose: str = ""
    frequency: str = ""

    @property
    def daily_dose_mg(self) -> float | None:
        return daily_dose_mg(f"{self.dose} {self.frequency}")


class Allergy(BaseModel):
    substance: str
    reaction: str = ""


class Vital(BaseModel):
    date: dt.date
    bp_systolic: int | None = None
    bp_diastolic: int | None = None
    heart_rate: int | None = None
    bmi: float | None = None


class Lab(BaseModel):
    date: dt.date
    test: str
    value: float
    unit: str = ""


class Note(BaseModel):
    date: dt.date
    author: str = ""
    text: str


class Patient(BaseModel):
    patient_id: str
    synthetic: bool = False
    demographics: dict = Field(default_factory=dict)
    summary: str = ""
    problems: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[Lab] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)

    def lab_series(self, test: str) -> list[Lab]:
        """Results for `test` (case-insensitive), oldest first."""
        return sorted(
            (lab for lab in self.labs if lab.test.lower() == test.lower()), key=lambda x: x.date
        )

    def latest_lab(self, test: str) -> Lab | None:
        series = self.lab_series(test)
        return series[-1] if series else None

    def bp_readings(self) -> list[Vital]:
        return sorted(
            (v for v in self.vitals if v.bp_systolic is not None and v.bp_diastolic is not None),
            key=lambda v: v.date,
        )

    def has_problem(self, *keywords: str) -> bool:
        text = " ".join(self.problems).lower()
        return any(k.lower() in text for k in keywords)

    def medication(self, name: str) -> Medication | None:
        return next((m for m in self.medications if m.name.lower() == name.lower()), None)


_DOSE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|g|mcg|microgram)\b", re.IGNORECASE)
_TIMES_PER_DAY = {
    "once": 1,
    "daily": 1,
    "od": 1,
    "twice": 2,
    "bd": 2,
    "bid": 2,
    "three times": 3,
    "tds": 3,
    "tid": 3,
    "four times": 4,
    "qds": 4,
    "qid": 4,
}


def daily_dose_mg(text: str) -> float | None:
    """Total daily dose in mg from text like "1000 mg twice daily"; None if no dose given."""
    match = _DOSE.search(text)
    if not match:
        return None
    amount, unit = float(match.group(1)), match.group(2).lower()
    amount *= {"g": 1000, "mcg": 0.001, "microgram": 0.001}.get(unit, 1)
    lowered = text.lower()
    times = max(
        (n for word, n in _TIMES_PER_DAY.items() if re.search(rf"\b{word}\b", lowered)),
        default=1,
    )
    return amount * times


def load_patient(patient_id: str) -> Patient:
    path = PATIENTS_DIR / f"{patient_id}.json"
    if not path.is_file():
        raise KeyError(patient_id)
    return Patient.model_validate_json(path.read_text())
