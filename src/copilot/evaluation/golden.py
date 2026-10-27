"""Golden-set schemas and loaders (eval/golden/*.yaml)."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from copilot.agents.patient import Patient, load_patient
from copilot.agents.safety import Change
from copilot.config import PROJECT_ROOT
from copilot.ingestion.models import Chunk

GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


class GoldPassage(BaseModel):
    doc_id: str
    contains: str

    def matches(self, chunk: Chunk) -> bool:
        return chunk.doc_id == self.doc_id and _norm(self.contains) in _norm(chunk.text)


class QAItem(BaseModel):
    id: str
    question: str
    expect: Literal["answer", "refuse"]
    refuse_reason: str | None = None
    gold: list[GoldPassage] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)  # case-insensitive regexes
    reference: str = ""


class ProposedChange(BaseModel):
    drug: str
    change: Change
    daily_dose: str = ""
    expect: list[str]  # safety rule names that must fire


class InteractionCase(BaseModel):
    drugs: list[str]
    expect_severity: str | None  # None: no interaction expected


class AgentScenario(BaseModel):
    id: str
    patient_id: str | None = None
    patient: Patient | None = None
    expected_findings: dict[str, str] = Field(default_factory=dict)  # id -> severity
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    interactions: list[InteractionCase] = Field(default_factory=list)

    def load(self) -> Patient:
        return self.patient or load_patient(self.patient_id)


def load_qa(path: Path = GOLDEN_DIR / "qa.yaml") -> list[QAItem]:
    return [QAItem.model_validate(x) for x in yaml.safe_load(path.read_text())]


def load_agent(path: Path = GOLDEN_DIR / "agent.yaml") -> list[AgentScenario]:
    return [AgentScenario.model_validate(x) for x in yaml.safe_load(path.read_text())]
