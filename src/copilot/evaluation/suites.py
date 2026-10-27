"""Evaluation suites. Each returns a SuiteResult: aggregate metrics plus per-item rows."""

from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver

from copilot.agents.interactions import check_interactions
from copilot.agents.rules import analyse_patient
from copilot.agents.safety import check_medication
from copilot.agents.service import build_analysis_service
from copilot.agents.workflow import ReviewDecision
from copilot.evaluation.golden import AgentScenario, QAItem
from copilot.evaluation.judge import judge_answer, judgement_scores
from copilot.evaluation.metrics import (
    average,
    key_fact_coverage,
    retrieval_scores,
    set_scores,
    unsupported_entities,
)
from copilot.rag.llm import LLM
from copilot.rag.qa import CitedQA
from copilot.rag.vector_store import SearchHit


@dataclass
class SuiteResult:
    name: str
    metrics: dict[str, float]  # flat "<group>.<metric>" -> value, all higher-is-better unless
    rows: list[dict] = field(default_factory=list)  # noted in LOWER_IS_BETTER
    notes: list[str] = field(default_factory=list)


LOWER_IS_BETTER = {"generation.hallucination_rate", "generation.unsupported_entity_rate"}

Search = Callable[[str, int], list[SearchHit]]


# --- retrieval -------------------------------------------------------------------------


def run_retrieval(
    items: list[QAItem], searches: dict[str, Search], k: int, indexed: set[str]
) -> SuiteResult:
    """`searches`: mode name -> search(query, k); e.g. dense / sparse / hybrid / hybrid+rerank."""
    usable = [i for i in items if i.gold and {g.doc_id for g in i.gold} <= indexed]
    skipped = [i.id for i in items if i.gold and i not in usable]
    metrics: dict[str, float] = {}
    rows = []
    for mode, search in searches.items():
        per_item = []
        for item in usable:
            scores = retrieval_scores([h.chunk for h in search(item.question, k)], item.gold, k)
            per_item.append(scores)
            rows.append({"mode": mode, "id": item.id, **scores})
        for name, value in average(per_item).items():
            metrics[
                f"retrieval.{mode}.{name}@{k}" if name != "mrr" else f"retrieval.{mode}.mrr"
            ] = value
    notes = [f"{len(usable)} questions with gold evidence in the index"]
    if skipped:
        notes.append(f"skipped (gold documents not indexed): {', '.join(skipped)}")
    return SuiteResult("retrieval", metrics, rows, notes)


# --- generation ------------------------------------------------------------------------


def run_generation(items: list[QAItem], qa: CitedQA, judge: LLM | None = None) -> SuiteResult:
    rows = []
    for item in items:
        answer = qa.answer(item.question)
        refused = answer.refusal_reason is not None
        row: dict = {
            "id": item.id,
            "expect": item.expect,
            "refused": refused,
            "refusal_reason": answer.refusal_reason,
            "correct_decision": (item.expect == "refuse") == refused
            and (not item.refuse_reason or item.refuse_reason == answer.refusal_reason),
            "invalid_citations": len(answer.invalid_citation_ids),
            "answer": answer.answer,
        }
        if item.expect == "answer" and not refused:
            # Sources the answer was generated from (retrieval is deterministic).
            hits = qa.retriever.retrieve(answer.standalone_question)
            sources = [h.chunk.text for h in hits]
            row["key_fact_coverage"] = key_fact_coverage(answer.answer, item.key_facts)
            row["unsupported_entities"] = sorted(unsupported_entities(answer.answer, sources))
            if judge:
                j = judge_answer(judge, item.question, answer.answer, list(enumerate(sources, 1)))
                row |= judgement_scores(j)
                row["unsupported_claims"] = [
                    c.claim for c in j.claims if not c.supported_by_any_source
                ]
        rows.append(row)

    answerable = [r for r in rows if r["expect"] == "answer"]
    negatives = [r for r in rows if r["expect"] == "refuse"]
    answered = [r for r in answerable if "key_fact_coverage" in r]
    metrics = {
        "generation.decision_accuracy": sum(r["correct_decision"] for r in rows) / len(rows),
        "generation.answer_rate": len(answered) / len(answerable) if answerable else 1.0,
        "generation.refusal_recall": (
            sum(r["refused"] for r in negatives) / len(negatives) if negatives else 1.0
        ),
        "generation.citation_id_validity": 1.0
        - sum(r["invalid_citations"] > 0 for r in rows) / len(rows),
    }
    if answered:
        metrics["generation.key_fact_coverage"] = sum(
            r["key_fact_coverage"] for r in answered
        ) / len(answered)
        metrics["generation.unsupported_entity_rate"] = sum(
            bool(r["unsupported_entities"]) for r in answered
        ) / len(answered)
        if judge:
            for name in (
                "faithfulness",
                "citation_accuracy",
                "groundedness",
                "hallucination_rate",
                "answer_relevance",
            ):
                metrics[f"generation.{name}"] = sum(r[name] for r in answered) / len(answered)
    notes = [] if judge else ["LLM-judge metrics not run (pass --judge)"]
    return SuiteResult("generation", metrics, rows, notes)


# --- agent -----------------------------------------------------------------------------


def run_agent_rules(scenarios: list[AgentScenario]) -> SuiteResult:
    """Deterministic: findings, safety rules and interaction checks against expectations."""
    rows = []
    finding_scores, severity_hits, change_hits, interaction_hits = [], [], [], []
    for s in scenarios:
        patient = s.load()
        found = {f.id: f.severity for f in analyse_patient(patient)}
        scores = set_scores(set(found), set(s.expected_findings))
        finding_scores.append(scores)
        severity_ok = [found.get(fid) == sev for fid, sev in s.expected_findings.items()]
        severity_hits += severity_ok
        missing = sorted(set(s.expected_findings) - set(found))
        extra = sorted(set(found) - set(s.expected_findings))
        wrong_severity = [
            fid
            for fid, ok in zip(s.expected_findings, severity_ok, strict=True)
            if not ok and fid in found
        ]

        change_failures = []
        for pc in s.proposed_changes:
            fired = {i.rule for i in check_medication(patient, pc.drug, pc.change, pc.daily_dose)}
            ok = fired == set(pc.expect)
            change_hits.append(ok)
            if not ok:
                change_failures.append(
                    f"{pc.change} {pc.drug}: fired {sorted(fired)}, expected {pc.expect}"
                )

        interaction_failures = []
        for case in s.interactions:
            found_x = check_interactions(case.drugs)
            severity = found_x[0].severity if found_x else None
            ok = severity == case.expect_severity
            interaction_hits.append(ok)
            if not ok:
                interaction_failures.append(
                    f"{case.drugs}: got {severity}, expected {case.expect_severity}"
                )

        rows.append(
            {
                "id": s.id,
                **{f"findings_{k}": v for k, v in scores.items()},
                "missing": missing,
                "extra": extra,
                "wrong_severity": wrong_severity,
                "safety_failures": change_failures,
                "interaction_failures": interaction_failures,
            }
        )

    metrics = {
        **{f"agent.findings_{k}": v for k, v in average(finding_scores).items()},
        "agent.severity_accuracy": sum(severity_hits) / len(severity_hits)
        if severity_hits
        else 1.0,
        "agent.safety_rule_accuracy": sum(change_hits) / len(change_hits) if change_hits else 1.0,
        "agent.interaction_accuracy": (
            sum(interaction_hits) / len(interaction_hits) if interaction_hits else 1.0
        ),
    }
    return SuiteResult("agent", metrics, rows)


def run_agent_workflow(scenarios: list[AgentScenario], retriever, llm: LLM) -> SuiteResult:
    """Full workflow with the real LLM: auto-approve and check the final plan."""
    rows = []
    for s in scenarios:
        service = build_analysis_service(retriever, llm, InMemorySaver())
        paused = service.start(s.load(), s.id)
        row = {"id": s.id, "status": paused.status, "error": paused.error}
        if paused.status == "awaiting_review":
            req = paused.review_request
            serious = {f["id"] for f in req["findings"] if f["severity"] in ("critical", "high")}
            draft = req["draft"]
            addressed = {fid for r in draft["recommendations"] for fid in r["finding_ids"]}
            report = service.review(s.id, ReviewDecision(decision="approve")).report
            row |= {
                "validated": not req["issues"],
                "coverage": len(serious & addressed) / len(serious) if serious else 1.0,
                "unsupported": sorted(serious & set(draft["unsupported_findings"])),
                "revisions": report["revisions"],
                "confidence": report["confidence"],
                "issues": [i["message"] for i in req["issues"]],
            }
        rows.append(row)
    done = [r for r in rows if "validated" in r]
    metrics = {
        "agent_llm.completion_rate": len(done) / len(rows),
        "agent_llm.validation_pass_rate": sum(r["validated"] for r in done) / len(rows),
        "agent_llm.serious_finding_coverage": (
            sum(r["coverage"] for r in done) / len(done) if done else 0.0
        ),
        "agent_llm.first_pass_rate": sum(r["revisions"] == 0 for r in done) / len(rows),
    }
    return SuiteResult("agent_llm", metrics, rows)
