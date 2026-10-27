"""Metric functions (pure)."""

import re
from statistics import mean

from copilot.agents.interactions import EXTRA_CLASSES
from copilot.agents.safety import DRUG_CLASSES
from copilot.evaluation.golden import GoldPassage
from copilot.ingestion.models import Chunk

# --- retrieval -------------------------------------------------------------------------


def retrieval_scores(retrieved: list[Chunk], gold: list[GoldPassage], k: int) -> dict[str, float]:
    """Recall@k (gold passages found), Precision@k (retrieved chunks matching any gold
    passage), reciprocal rank of the first relevant chunk, and hit (any gold found)."""
    top = retrieved[:k]
    relevant = [any(g.matches(c) for g in gold) for c in top]
    found = [any(g.matches(c) for c in top) for g in gold]
    first = next((i for i, r in enumerate(relevant, 1) if r), None)
    return {
        "recall": sum(found) / len(gold),
        "precision": sum(relevant) / k,
        "mrr": 1 / first if first else 0.0,
        "hit": float(any(found)),
    }


def average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean(r[key] for r in rows) for key in rows[0]} if rows else {}


# --- answers ---------------------------------------------------------------------------


def key_fact_coverage(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 1.0
    return sum(bool(re.search(f, answer, re.I)) for f in key_facts) / len(key_facts)


_NUMBER = re.compile(r"(?<![\w.\[])\d+(?:\.\d+)?(?![\w\]])")
_DRUGS = {d for members in DRUG_CLASSES.values() for d in members} | {
    d for members in EXTRA_CLASSES.values() for d in members
}


def clinical_entities(text: str) -> set[str]:
    """Numbers (citation markers excluded) and drug names mentioned in `text`."""
    without_citations = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    numbers = {
        n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER.findall(without_citations)
    }
    words = set(re.findall(r"[a-z][a-z-]+", text.lower()))
    return numbers | (words & _DRUGS)


def unsupported_entities(answer: str, sources: list[str]) -> set[str]:
    """Deterministic hallucination check: clinical numbers and drug names in the answer that
    appear in none of the sources it was given."""
    source_entities = set().union(*(clinical_entities(s) for s in sources)) if sources else set()
    return clinical_entities(answer) - source_entities


# --- agent -----------------------------------------------------------------------------


def set_scores(predicted: set[str], expected: set[str]) -> dict[str, float]:
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else float(not expected)
    recall = tp / len(expected) if expected else 1.0
    return {"precision": precision, "recall": recall}
