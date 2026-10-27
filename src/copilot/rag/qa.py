"""Cited question answering over the guideline corpus.

Pipeline per question: rewrite follow-ups into a standalone question -> retrieve + rerank ->
refuse if the best evidence is weak -> answer from numbered sources with mandatory [n]
citations -> drop citation IDs that point at no source, and refuse if none remain.
"""

import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel

from copilot.config import get_settings
from copilot.guardrails.checks import DISCLAIMER, SCOPE_REFUSALS, check_scope, strip_uncited_dosing
from copilot.rag.llm import LLM, LLMRefusal, get_llm
from copilot.rag.retriever import Retriever, get_retriever
from copilot.rag.vector_store import SearchHit

RefusalReason = Literal[
    "out_of_scope", "low_evidence", "insufficient_evidence", "uncited", "model_declined"
]

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
REFUSALS: dict[str, str] = {
    "low_evidence": "I couldn't find relevant guidance in the indexed documents for this question.",
    "insufficient_evidence": (
        "The retrieved guideline excerpts don't contain enough information to answer this."
    ),
    "uncited": "I couldn't produce an answer supported by citations to the indexed guidelines.",
    "model_declined": "I can't help with this request.",
}

REWRITE_SYSTEM = """\
You rewrite a clinician's latest message into a standalone search query for a clinical \
guideline database. Resolve pronouns and references ("it", "that drug", "what about in CKD?") \
using the conversation. Keep clinical terms, drug names and numeric values exactly as given. \
If the message is already standalone, return it unchanged. Output only the rewritten question."""

ANSWER_SYSTEM = f"""\
You are a clinical decision-support assistant for clinicians. Answer the question using only \
the numbered guideline excerpts in <sources>; do not add facts from outside them.

Cite every factual statement with the id of the source(s) that support it, in square brackets \
directly after the statement, e.g. "Stop metformin below eGFR 30 [2]." or "... [1][3]". \
Only cite ids that appear in <sources>.

If the sources partly answer the question, answer that part and say plainly what they don't \
cover. If they contain nothing relevant to the question, reply with exactly {INSUFFICIENT} \
and nothing else.

Write for a busy clinician: lead with the direct answer, then key thresholds, monitoring \
intervals or exceptions, as short prose or a brief list. Where sources disagree, say so and \
cite each."""

_CITATION_GROUP = re.compile(r"([ \t]*)\[(\d+(?:\s*,\s*\d+)*)\]")


class Citation(BaseModel):
    id: int
    label: str  # human-readable "Title, p.12 (Section > Sub)"
    chunk_id: str
    doc_id: str
    title: str
    source_file: str
    section: str
    page_start: int | None
    page_end: int | None
    relevance: float
    quote: str  # chunk text, for the UI to show on hover / expand


class Answer(BaseModel):
    question: str
    standalone_question: str
    answer: str
    citations: list[Citation]
    grounded: bool
    refusal_reason: RefusalReason | None = None
    invalid_citation_ids: list[int] = []
    guardrail_flags: list[str] = []  # e.g. "prompt_injection", "uncited_dosing_removed:2"
    disclaimer: str = DISCLAIMER


def format_sources(hits: list[SearchHit]) -> str:
    parts = [
        f'<source id="{i}" citation="{hit.chunk.citation()}">\n{hit.chunk.text}\n</source>'
        for i, hit in enumerate(hits, 1)
    ]
    return "<sources>\n" + "\n".join(parts) + "\n</sources>"


def validate_citations(text: str, n_sources: int) -> tuple[str, list[int], list[int]]:
    """Strip citation ids outside 1..n_sources.

    Returns (cleaned text, valid ids in order of first use, invalid ids).
    """
    valid: list[int] = []
    invalid: list[int] = []

    def fix(match: re.Match) -> str:
        kept = []
        for raw in match.group(2).split(","):
            n = int(raw)
            if 1 <= n <= n_sources:
                kept.append(n)
                if n not in valid:
                    valid.append(n)
            elif n not in invalid:
                invalid.append(n)
        # A fully invalid group is removed along with the space before it.
        return match.group(1) + "".join(f"[{n}]" for n in kept) if kept else ""

    return _CITATION_GROUP.sub(fix, text), valid, invalid


def _citation(n: int, hit: SearchHit) -> Citation:
    c = hit.chunk
    return Citation(
        id=n,
        label=c.citation(),
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        title=c.title,
        source_file=c.source_file,
        section=c.section,
        page_start=c.page_start,
        page_end=c.page_end,
        relevance=round(hit.score, 4),
        quote=c.text,
    )


class CitedQA:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        min_relevance: float,
        answer_effort: str = "medium",
        rewrite_effort: str = "low",
        history_turns: int = 6,
    ):
        self.retriever = retriever
        self.llm = llm
        self.min_relevance = min_relevance
        self.answer_effort = answer_effort
        self.rewrite_effort = rewrite_effort
        self.history_turns = history_turns

    def rewrite(self, history: list[dict], question: str) -> str:
        """Standalone version of `question` given prior turns; no LLM call on a first turn."""
        recent = history[-self.history_turns :] if self.history_turns else []
        if not recent or check_scope(question):  # out-of-scope text is never sent to the LLM
            return question
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        prompt = (
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            f"<latest_message>\n{question}\n</latest_message>"
        )
        try:
            rewritten = self.llm.complete(
                REWRITE_SYSTEM,
                [{"role": "user", "content": prompt}],
                effort=self.rewrite_effort,
                max_tokens=4000,
            )
        except LLMRefusal:
            return question
        return rewritten or question

    def answer(self, question: str, standalone_question: str | None = None) -> Answer:
        query = standalone_question or question

        def refuse(reason: RefusalReason, message: str | None = None, **extra) -> Answer:
            return Answer(
                question=question,
                standalone_question=query,
                answer=message or REFUSALS[reason],
                citations=[],
                grounded=False,
                refusal_reason=reason,
                **extra,
            )

        if scope := check_scope(question):
            return refuse("out_of_scope", SCOPE_REFUSALS[scope], guardrail_flags=[scope])

        hits = self.retriever.retrieve(query)
        if not hits or hits[0].score < self.min_relevance:
            return refuse("low_evidence")

        prompt = f"{format_sources(hits)}\n\n<question>\n{query}\n</question>"
        try:
            raw = self.llm.complete(
                ANSWER_SYSTEM, [{"role": "user", "content": prompt}], effort=self.answer_effort
            )
        except LLMRefusal:
            return refuse("model_declined")

        if raw.strip().strip(".") == INSUFFICIENT:
            return refuse("insufficient_evidence")

        # Doses must be traceable to a source: uncited dosing sentences are removed first.
        raw, removed = strip_uncited_dosing(raw)
        flags = [f"uncited_dosing_removed:{removed}"] if removed else []
        text, valid, invalid = validate_citations(raw, len(hits))
        if not valid:
            return refuse("uncited", invalid_citation_ids=invalid, guardrail_flags=flags)

        return Answer(
            question=question,
            standalone_question=query,
            answer=text,
            citations=[_citation(n, hits[n - 1]) for n in sorted(valid)],
            grounded=True,
            invalid_citation_ids=invalid,
            guardrail_flags=flags,
        )


@lru_cache
def get_qa() -> CitedQA:
    settings = get_settings()
    return CitedQA(
        get_retriever(),
        get_llm(),
        min_relevance=settings.min_relevance,
        answer_effort=settings.answer_effort,
        rewrite_effort=settings.rewrite_effort,
        history_turns=settings.history_turns,
    )
