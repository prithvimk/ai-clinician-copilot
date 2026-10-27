"""PII pseudonymisation around LLM calls.

Identifiers (names, contact details, NHS/SSN/MRN numbers, dates of birth, postcodes) are
swapped for stable placeholders like <PERSON_1> before text leaves the process, and swapped
back in the model's output. Clinical content (drugs, lab names, values, visit dates) is kept,
since the reasoning depends on it.

Detection uses Presidio (spaCy NER + its recognizers) when the configured spaCy model is
installed, unioned with built-in regex recognizers; without the model, regex only.
"""

import json
import logging
import re
from functools import lru_cache
from typing import Protocol, TypeVar

from pydantic import BaseModel

from copilot.agents.interactions import EXTRA_CLASSES
from copilot.agents.safety import CLASS_NAMES, DRUG_CLASSES
from copilot.config import get_settings
from copilot.rag.llm import LLM

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Span = tuple[int, int, str]  # start, end, entity type

# Pattern -> entity; a named group "v" narrows the redacted span to the identifier itself.
# Order matters: on identical spans the earlier (more specific, context-anchored) one wins.
REGEX_RECOGNIZERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "EMAIL"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN"),
    (
        re.compile(r"\bNHS\s*(?:number|no\.?)?[:\s#]*(?P<v>\d{3}\s?\d{3}\s?\d{4})\b", re.I),
        "NHS_NUMBER",
    ),
    (
        re.compile(r"\b(?:MRN|hospital\s+(?:number|no\.?))[:\s#]*(?P<v>[A-Z0-9-]{4,})\b", re.I),
        "MRN",
    ),
    (
        re.compile(
            r"\b(?:DOB|D\.O\.B\.?|date of birth)[:\s]*"
            r"(?P<v>\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4})",
            re.I,
        ),
        "DATE_OF_BIRTH",
    ),
    (
        re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+(?P<v>[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)?)"),
        "PERSON",
    ),
    (re.compile(r"\bName[:\s]+(?P<v>[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)"), "PERSON"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b"), "POSTCODE"),
    (re.compile(r"(?<!\d)(?:\+44\s?|0)(?:\d\s?){9,10}(?!\d)"), "PHONE"),
    (re.compile(r"(?<!\d)\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"), "PHONE"),
]
PRESIDIO_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "UK_NHS",
    "US_SSN",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
]
# Never redact clinical vocabulary NER sometimes mislabels as a name.
CLINICAL_TERMS = (
    {d for members in DRUG_CLASSES.values() for d in members}
    | {d for members in EXTRA_CLASSES.values() for d in members}
    | set(CLASS_NAMES)
    | {"hba1c", "egfr", "uacr", "ldl", "bmi", "gp", "ckd", "nhs"}
)


class Detector(Protocol):
    def find(self, text: str) -> list[Span]: ...


class RegexDetector:
    def find(self, text: str) -> list[Span]:
        spans = []
        for pattern, entity in REGEX_RECOGNIZERS:
            for m in pattern.finditer(text):
                start, end = m.span("v") if "v" in pattern.groupindex else m.span()
                spans.append((start, end, entity))
        return spans


class PresidioDetector:
    def __init__(self, spacy_model: str, threshold: float = 0.5):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()  # raises OSError if the spaCy model isn't installed
        self.analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        self.threshold = threshold
        self.regex = RegexDetector()

    def find(self, text: str) -> list[Span]:
        results = self.analyzer.analyze(
            text, language="en", entities=PRESIDIO_ENTITIES, score_threshold=self.threshold
        )
        return [(r.start, r.end, r.entity_type) for r in results] + self.regex.find(text)


class Pseudonymizer:
    """Per-request mapping, so one value gets the same placeholder across all messages."""

    def __init__(self, detector: Detector):
        self.detector = detector
        self.placeholders: dict[str, str] = {}  # original -> placeholder
        self._counts: dict[str, int] = {}

    def redact(self, text: str) -> str:
        spans = sorted(
            (
                s
                for s in self.detector.find(text)
                if text[s[0] : s[1]].strip().lower() not in CLINICAL_TERMS
            ),
            key=lambda s: (s[0], -s[1]),
        )
        merged: list[Span] = []
        for span in spans:  # drop spans overlapping an earlier (longer-first) one
            if not merged or span[0] >= merged[-1][1]:
                merged.append(span)
        replacements = [
            (start, end, self._placeholder(text[start:end], entity))
            for start, end, entity in merged
        ]
        for start, end, placeholder in reversed(replacements):  # numbered in reading order
            text = text[:start] + placeholder + text[end:]
        return text

    def _placeholder(self, original: str, entity: str) -> str:
        if original not in self.placeholders:
            self._counts[entity] = self._counts.get(entity, 0) + 1
            self.placeholders[original] = f"<{entity}_{self._counts[entity]}>"
        return self.placeholders[original]

    def redact_messages(self, messages: list[dict]) -> list[dict]:
        return [
            {**m, "content": self.redact(m["content"])} if isinstance(m["content"], str) else m
            for m in messages
        ]

    def restore(self, text: str, json_escape: bool = False) -> str:
        for original, placeholder in self.placeholders.items():
            value = json.dumps(original)[1:-1] if json_escape else original
            text = text.replace(placeholder, value)
        return text


class RedactingLLM:
    """LLM wrapper: user/patient text is pseudonymised before the call, restored after."""

    def __init__(self, llm: LLM, detector: Detector):
        self.llm = llm
        self.detector = detector

    def complete(self, system, messages, effort, max_tokens=16000) -> str:
        p = Pseudonymizer(self.detector)
        out = self.llm.complete(system, p.redact_messages(messages), effort, max_tokens)
        _log(p)
        return p.restore(out)

    def parse(self, system, messages, schema: type[T], effort, max_tokens=16000) -> T:
        p = Pseudonymizer(self.detector)
        out = self.llm.parse(system, p.redact_messages(messages), schema, effort, max_tokens)
        _log(p)
        if not p.placeholders:
            return out
        return schema.model_validate_json(p.restore(out.model_dump_json(), json_escape=True))


def _log(p: Pseudonymizer) -> None:
    if p.placeholders:
        log.info("pii redaction: %d value(s) pseudonymised", len(p.placeholders))


@lru_cache
def get_detector() -> Detector:
    model = get_settings().pii_spacy_model
    if model:
        try:
            return PresidioDetector(model)
        except (ImportError, OSError) as e:
            log.warning("Presidio unavailable (%s); using regex PII recognizers only", e)
    return RegexDetector()
