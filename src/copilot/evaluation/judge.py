"""Claim-level LLM judge for generation quality (RAGAS-style metrics with a Claude judge).

One structured call per answer: the judge splits the answer into atomic claims, and for each
says whether its cited sources support it and whether any provided source does. From that:

    faithfulness        claims supported by any source / claims
    citation_accuracy   cited claims whose citations support them (partial = 0.5) / cited claims
    groundedness        claims that are cited AND supported by their citations / claims
    hallucination_rate  claims supported by no source / claims
    answer_relevance    judge's 1-5 rating of how directly the answer addresses the question,
                        scaled to 0-1
"""

from typing import Literal

from pydantic import BaseModel

from copilot.rag.llm import LLM


class ClaimVerdict(BaseModel):
    claim: str
    cited_ids: list[int]
    supported_by_citations: Literal["yes", "partial", "no", "uncited"]
    supported_by_any_source: bool


class Judgement(BaseModel):
    claims: list[ClaimVerdict]
    relevance: int  # 1-5
    relevance_reason: str


JUDGE_SYSTEM = """\
You evaluate answers from a clinical guideline question-answering system. You are strict and \
literal: a claim is supported only if the source text states it or directly entails it.

Split the answer into atomic factual claims (one fact each; skip pure hedges such as "consult \
a specialist" unless they assert a fact). For each claim:
- cited_ids: the [n] source ids attached to it in the answer (empty if none).
- supported_by_citations: "yes" if those cited sources fully support it, "partial" if they \
support part of it, "no" if they do not, "uncited" if it has no citation.
- supported_by_any_source: whether ANY of the provided sources supports it.

Then rate relevance 1-5: how directly and completely the answer addresses the question \
(5 = fully answers it, 1 = does not address it), with a one-sentence reason."""


def judge_answer(
    llm: LLM, question: str, answer: str, sources: list[tuple[int, str]], effort: str = "medium"
) -> Judgement:
    rendered = "\n".join(f'<source id="{i}">\n{text}\n</source>' for i, text in sources)
    prompt = (
        f"<question>\n{question}\n</question>\n\n<sources>\n{rendered}\n</sources>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )
    return llm.parse(JUDGE_SYSTEM, [{"role": "user", "content": prompt}], Judgement, effort)


def judgement_scores(j: Judgement) -> dict[str, float]:
    claims = j.claims
    n = len(claims) or 1
    cited = [c for c in claims if c.supported_by_citations != "uncited"]
    credit = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    return {
        "faithfulness": sum(c.supported_by_any_source for c in claims) / n if claims else 1.0,
        "citation_accuracy": (
            sum(credit[c.supported_by_citations] for c in cited) / len(cited) if cited else 0.0
        ),
        "groundedness": (
            sum(c.supported_by_citations == "yes" for c in claims) / n if claims else 1.0
        ),
        "hallucination_rate": (
            sum(not c.supported_by_any_source for c in claims) / n if claims else 0.0
        ),
        "answer_relevance": (min(max(j.relevance, 1), 5) - 1) / 4,
    }
