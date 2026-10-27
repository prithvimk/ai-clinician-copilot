"""Input scope checks and output checks for clinician-facing answers."""

import re
from typing import Literal

DISCLAIMER = (
    "Decision support only: generated from guideline excerpts and rule-based checks for "
    "clinician review. Not a substitute for clinical judgement."
)

ScopeFlag = Literal["prompt_injection", "emergency"]

_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\b.{0,30}"
    r"\b(?:previous|prior|above|earlier|all|your|the system)\b"
    r".{0,20}\b(?:instructions?|prompts?|rules)\b"
    r"|\b(?:reveal|print|show|repeat)\b.{0,20}\b(?:system prompt|your instructions)\b"
    r"|\byou are now\b|\bjailbreak\b|\bdeveloper mode\b",
    re.I,
)
_RELATIVE = r"(?:child|baby|son|daughter|wife|husband|partner|mother|father|mum|dad|friend)"
# First-person, happening-now phrasing; a clinician describing "a patient with chest pain" is
# a clinical question, not an emergency.
_EMERGENCY = re.compile(
    rf"\b(?:i am|i'm|i've|i have|i took|i want to|i feel|my {_RELATIVE} (?:is|has|was|took|just))\b"
    r"(?!.{0,60}\bpatients?\b)"
    r".{0,60}?\b(?:chest pain|can'?t breathe|cannot breathe|not breathing|overdos\w*|suicid\w*|"
    r"kill (?:myself|himself|herself)|unconscious|unresponsive|having a stroke|seizure)",
    re.I,
)
SCOPE_REFUSALS: dict[ScopeFlag, str] = {
    "prompt_injection": "I can only answer clinical questions about the indexed guidelines.",
    "emergency": (
        "This sounds like an emergency. Call your local emergency number (e.g. 999 or 911) now. "
        "This tool supports clinicians reviewing guidelines and can't give emergency advice."
    ),
}


def check_scope(text: str) -> ScopeFlag | None:
    """Reasons a request is outside what the copilot should answer, if any."""
    if _INJECTION.search(text):
        return "prompt_injection"
    if _EMERGENCY.search(text):
        return "emergency"
    return None


_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|micrograms?|units?|IU|ml/h)\b", re.I)
_CITED = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*\-(])")


def strip_uncited_dosing(text: str) -> tuple[str, int]:
    """Remove sentences that state a dose without a citation; returns (text, removed count).

    Works line by line so lists and paragraphs keep their layout.
    """
    removed = 0
    lines = []
    for line in text.splitlines():
        kept = []
        for sentence in _SENTENCE_END.split(line):
            if _DOSE.search(sentence) and not _CITED.search(sentence):
                removed += 1
                continue
            kept.append(sentence)
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return "\n".join(lines).strip(), removed
