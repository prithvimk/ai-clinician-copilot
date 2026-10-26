"""Patient analysis workflow (LangGraph).

    analyze -> retrieve -> reason -> validate --ok--> human_review -> recommend -> END
                  ^          ^          |
                  |          +- issues -+  (unsafe drug change, bad citation, unaddressed finding)
                  +---- finding has no evidence (broaden its query)

analyze and validate are deterministic; only reason calls the LLM. validate loops back at most
`max_revisions` times, then hands the draft to the clinician with its unresolved issues.
human_review pauses the graph (interrupt) until a clinician approves, edits or rejects.
Every step is checkpointed per thread_id, so a failed run resumes from its last good step.
"""

import operator
from typing import Annotated, Literal, TypedDict

import anthropic
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt
from pydantic import BaseModel

from copilot.agents.patient import Patient
from copilot.agents.rules import Finding, analyse_patient
from copilot.agents.safety import Change, check_medication
from copilot.rag.llm import LLM, LLMOutputError
from copilot.rag.retriever import Retriever

DISCLAIMER = (
    "Decision support only: generated from guideline excerpts and rule-based checks for "
    "clinician review. Not a substitute for clinical judgement."
)
PRIORITY_ORDER = {"urgent": 0, "soon": 1, "routine": 2}

# --- LLM output schema (all fields required: constrained decoding fills every one) ----


class MedicationChange(BaseModel):
    drug: str
    change: Change
    daily_dose: str  # total daily dose after the change, e.g. "1000 mg"; "" if not applicable


class Recommendation(BaseModel):
    finding_ids: list[str]
    action: str
    rationale: str
    priority: Literal["urgent", "soon", "routine"]
    medication_changes: list[MedicationChange]
    citations: list[int]


class MonitoringItem(BaseModel):
    test: str
    interval: str
    citations: list[int]


class Draft(BaseModel):
    recommendations: list[Recommendation]
    monitoring: list[MonitoringItem]
    unsupported_findings: list[str]  # finding ids the sources don't cover


# --- workflow records ------------------------------------------------------------------


class Evidence(BaseModel):
    id: int
    chunk_id: str
    label: str
    doc_id: str
    relevance: float
    text: str
    finding_ids: list[str]


class ValidationIssue(BaseModel):
    kind: Literal["citation", "safety", "coverage"]
    message: str
    finding_id: str | None = None
    recommendation: int | None = None  # 0-based index into draft.recommendations


class ReviewDecision(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    recommendations: list[Recommendation] | None = None  # replacement list when editing
    comment: str = ""


class Report(BaseModel):
    patient_id: str
    status: Literal["approved", "rejected"]
    summary: str
    findings: list[Finding]
    actions: list[Recommendation]
    monitoring: list[MonitoringItem]
    references: list[Evidence]
    unresolved_issues: list[ValidationIssue]
    confidence: Literal["high", "medium", "low"]
    confidence_reason: str
    revisions: int
    reviewer_comment: str
    disclaimer: str = DISCLAIMER


class AnalysisState(TypedDict, total=False):
    # Values are plain JSON (model_dump) so any checkpointer can serialise them.
    patient: dict
    findings: list[dict]
    evidence: list[dict]
    draft: dict
    issues: list[dict]
    revisions: int
    review: dict
    report: dict
    trace: Annotated[list[dict], operator.add]  # {"node", "summary"} per step, for the UI


# --- prompts ---------------------------------------------------------------------------

REASONING_SYSTEM = """\
You are a clinical decision-support assistant drafting a management plan for a clinician to \
review. You receive a patient record, findings from rule-based checks, and numbered guideline \
excerpts.

- Base every clinical claim on the excerpts in <sources>. Each recommendation and monitoring \
item cites the source ids that support it; never cite an id that is not listed.
- Link each recommendation to the finding ids it addresses. Address every critical and high \
finding. If the sources do not cover a finding, put its id in unsupported_findings instead of \
recommending from outside knowledge.
- List every medication the plan starts, stops, changes or continues in medication_changes, \
with the total daily dose after the change where a dose applies.
- Respect the patient's allergies and kidney function.
- Priority: urgent (act within days), soon (within weeks), routine (next scheduled review).
- Monitoring items give the test and interval stated in the sources."""


def _format_findings(findings: list[Finding]) -> str:
    return "\n".join(f"- {f.id} [{f.severity}] {f.title}: {f.detail}" for f in findings)


def _format_sources(evidence: list[Evidence]) -> str:
    return "\n".join(
        f'<source id="{e.id}" citation="{e.label}">\n{e.text}\n</source>' for e in evidence
    )


def reasoning_prompt(
    patient: Patient,
    findings: list[Finding],
    evidence: list[Evidence],
    draft: Draft | None,
    issues: list[ValidationIssue],
) -> str:
    parts = [
        f"<patient>\n{patient.model_dump_json(indent=1)}\n</patient>",
        f"<findings>\n{_format_findings(findings)}\n</findings>",
        f"<sources>\n{_format_sources(evidence)}\n</sources>",
    ]
    if draft and issues:
        parts += [
            f"<previous_draft>\n{draft.model_dump_json(indent=1)}\n</previous_draft>",
            "<validation_issues>\n"
            + "\n".join(f"- {i.kind}: {i.message}" for i in issues)
            + "\n</validation_issues>",
            "Revise the previous draft so that every validation issue is resolved.",
        ]
    return "\n\n".join(parts)


# --- validation (pure, also re-run on clinician edits) ---------------------------------


def validate_draft(
    draft: Draft, patient: Patient, findings: list[Finding], evidence: list[Evidence]
) -> list[ValidationIssue]:
    ids = {e.id for e in evidence}
    issues: list[ValidationIssue] = []

    for i, rec in enumerate(draft.recommendations):
        label = f"Recommendation {i + 1} ({rec.action[:60]})"
        bad = [c for c in rec.citations if c not in ids]
        if not rec.citations:
            issues.append(
                ValidationIssue(
                    kind="citation", message=f"{label} has no citation.", recommendation=i
                )
            )
        elif bad:
            issues.append(
                ValidationIssue(
                    kind="citation",
                    message=f"{label} cites unknown source ids {bad}.",
                    recommendation=i,
                )
            )
        for mc in rec.medication_changes:
            for s in check_medication(patient, mc.drug, mc.change, mc.daily_dose):
                if s.severity in ("critical", "high"):
                    issues.append(
                        ValidationIssue(
                            kind="safety",
                            message=f"{label}: {mc.change} {mc.drug}"
                            f"{' ' + mc.daily_dose if mc.daily_dose else ''} - {s.message}",
                            recommendation=i,
                        )
                    )

    for item in draft.monitoring:
        bad = [c for c in item.citations if c not in ids]
        if not item.citations or bad:
            issues.append(
                ValidationIssue(
                    kind="citation",
                    message=f"Monitoring '{item.test}' has missing or unknown citations {bad}.",
                )
            )

    addressed = {fid for rec in draft.recommendations for fid in rec.finding_ids}
    addressed |= set(draft.unsupported_findings)
    for f in findings:
        if f.severity in ("critical", "high") and f.id not in addressed:
            issues.append(
                ValidationIssue(
                    kind="coverage",
                    message=f"Finding {f.id} ({f.title}) is not addressed.",
                    finding_id=f.id,
                )
            )
    return issues


# --- graph -----------------------------------------------------------------------------

LLM_RETRY = RetryPolicy(
    max_attempts=3,
    retry_on=(
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        LLMOutputError,
    ),
)


class AnalysisWorkflow:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        checkpointer: BaseCheckpointSaver,
        min_relevance: float,
        evidence_per_finding: int = 3,
        max_evidence: int = 15,
        max_revisions: int = 2,
        reasoning_effort: str = "high",
    ):
        self.retriever = retriever
        self.llm = llm
        self.min_relevance = min_relevance
        self.evidence_per_finding = evidence_per_finding
        self.max_evidence = max_evidence
        self.max_revisions = max_revisions
        self.reasoning_effort = reasoning_effort

        g = StateGraph(AnalysisState)
        g.add_node("analyze", self.analyze)
        g.add_node("retrieve", self.retrieve, retry_policy=RetryPolicy(max_attempts=3))
        g.add_node("reason", self.reason, retry_policy=LLM_RETRY)
        g.add_node("validate", self.validate)
        g.add_node("human_review", self.human_review)
        g.add_node("recommend", self.recommend)
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "retrieve")
        g.add_edge("retrieve", "reason")
        g.add_edge("reason", "validate")
        g.add_conditional_edges("validate", self.route, ["retrieve", "reason", "human_review"])
        g.add_edge("human_review", "recommend")
        g.add_edge("recommend", END)
        self.graph = g.compile(checkpointer=checkpointer)

    # nodes -----------------------------------------------------------------------------

    def analyze(self, state: AnalysisState) -> AnalysisState:
        findings = analyse_patient(Patient.model_validate(state["patient"]))
        return {
            "findings": [f.model_dump() for f in findings],
            "revisions": 0,
            "trace": [{"node": "analyze", "summary": f"{len(findings)} findings"}],
        }

    def retrieve(self, state: AnalysisState) -> AnalysisState:
        findings = [Finding.model_validate(f) for f in state["findings"]]
        evidence = {e["chunk_id"]: Evidence.model_validate(e) for e in state.get("evidence", [])}
        # On a revision, only findings left without evidence are searched again, with a
        # broader query; existing source ids stay stable so earlier citations remain valid.
        first_pass = not state.get("revisions")
        uncovered = _uncovered(state)
        targets = (
            [(f, f.query) for f in findings]
            if first_pass
            else [(f, f"{f.title}. {f.detail}") for f in findings if f.id in uncovered]
        )

        added: list[Evidence] = []
        for finding, query in targets:
            for hit in self.retriever.retrieve(query, k=self.evidence_per_finding):
                if hit.score < self.min_relevance:
                    continue
                existing = evidence.get(hit.chunk.chunk_id) or next(
                    (e for e in added if e.chunk_id == hit.chunk.chunk_id), None
                )
                if existing:
                    if finding.id not in existing.finding_ids:
                        existing.finding_ids.append(finding.id)
                    existing.relevance = max(existing.relevance, round(hit.score, 4))
                    continue
                added.append(
                    Evidence(
                        id=0,
                        chunk_id=hit.chunk.chunk_id,
                        label=hit.chunk.citation(),
                        doc_id=hit.chunk.doc_id,
                        relevance=round(hit.score, 4),
                        text=hit.chunk.text,
                        finding_ids=[finding.id],
                    )
                )

        if first_pass:  # keep the most relevant sources
            added = sorted(added, key=lambda e: e.relevance, reverse=True)[: self.max_evidence]
        next_id = max((e.id for e in evidence.values()), default=0) + 1
        for offset, e in enumerate(added):
            e.id = next_id + offset
        merged = sorted([*evidence.values(), *added], key=lambda e: e.id)
        return {
            "evidence": [e.model_dump() for e in merged],
            "trace": [
                {
                    "node": "retrieve",
                    "summary": f"{len(targets)} queries, {len(added)} new sources "
                    f"({len(merged)} total)",
                }
            ],
        }

    def reason(self, state: AnalysisState) -> AnalysisState:
        draft = Draft.model_validate(state["draft"]) if state.get("draft") else None
        issues = [ValidationIssue.model_validate(i) for i in state.get("issues", [])]
        prompt = reasoning_prompt(
            Patient.model_validate(state["patient"]),
            [Finding.model_validate(f) for f in state["findings"]],
            [Evidence.model_validate(e) for e in state["evidence"]],
            draft,
            issues,
        )
        new = self.llm.parse(
            REASONING_SYSTEM,
            [{"role": "user", "content": prompt}],
            Draft,
            effort=self.reasoning_effort,
        )
        return {
            "draft": new.model_dump(),
            "trace": [
                {
                    "node": "reason",
                    "summary": f"{len(new.recommendations)} recommendations"
                    + (f" (revision addressing {len(issues)} issues)" if draft and issues else ""),
                }
            ],
        }

    def validate(self, state: AnalysisState) -> AnalysisState:
        issues = validate_draft(
            Draft.model_validate(state["draft"]),
            Patient.model_validate(state["patient"]),
            [Finding.model_validate(f) for f in state["findings"]],
            [Evidence.model_validate(e) for e in state["evidence"]],
        )
        revisions = state.get("revisions", 0) + (1 if issues else 0)
        summary = "passed" if not issues else "; ".join(i.message for i in issues)
        return {
            "issues": [i.model_dump() for i in issues],
            "revisions": revisions,
            "trace": [{"node": "validate", "summary": summary}],
        }

    def route(self, state: AnalysisState) -> str:
        if not state["issues"] or state["revisions"] > self.max_revisions:
            return "human_review"
        return "retrieve" if _uncovered(state) else "reason"

    def human_review(self, state: AnalysisState) -> AnalysisState:
        decision = ReviewDecision.model_validate(
            interrupt(
                {
                    "findings": state["findings"],
                    "draft": state["draft"],
                    "issues": state["issues"],
                    "evidence": [
                        {k: e[k] for k in ("id", "label", "relevance")} for e in state["evidence"]
                    ],
                }
            )
        )
        update: AnalysisState = {
            "review": decision.model_dump(),
            "trace": [{"node": "human_review", "summary": decision.decision}],
        }
        if decision.decision == "edit" and decision.recommendations is not None:
            draft = Draft.model_validate(state["draft"])
            draft.recommendations = decision.recommendations
            # Edits are the clinician's call, but they are re-checked and any issue is kept
            # on the report rather than silently accepted.
            issues = validate_draft(
                draft,
                Patient.model_validate(state["patient"]),
                [Finding.model_validate(f) for f in state["findings"]],
                [Evidence.model_validate(e) for e in state["evidence"]],
            )
            update |= {"draft": draft.model_dump(), "issues": [i.model_dump() for i in issues]}
        return update

    def recommend(self, state: AnalysisState) -> AnalysisState:
        patient = Patient.model_validate(state["patient"])
        findings = [Finding.model_validate(f) for f in state["findings"]]
        draft = Draft.model_validate(state["draft"])
        evidence = {e["id"]: Evidence.model_validate(e) for e in state["evidence"]}
        issues = [ValidationIssue.model_validate(i) for i in state.get("issues", [])]
        review = ReviewDecision.model_validate(state["review"])
        rejected = review.decision == "reject"

        cited = sorted(
            {c for r in draft.recommendations for c in r.citations}
            | {c for m in draft.monitoring for c in m.citations}
        )
        references = [evidence[c] for c in cited if c in evidence]
        confidence, reason = _confidence(findings, draft, issues, references, state["revisions"])
        key = [f.title for f in findings if f.severity in ("critical", "high")]
        summary = patient.summary + (f" Key findings: {'; '.join(key)}." if key else "")

        report = Report(
            patient_id=patient.patient_id,
            status="rejected" if rejected else "approved",
            summary=summary,
            findings=findings,
            actions=[]
            if rejected
            else sorted(draft.recommendations, key=lambda r: PRIORITY_ORDER[r.priority]),
            monitoring=[] if rejected else draft.monitoring,
            references=[] if rejected else references,
            unresolved_issues=issues,
            confidence=confidence,
            confidence_reason=reason,
            revisions=state["revisions"],
            reviewer_comment=review.comment,
        )
        summary = f"{report.status}, {confidence} confidence"
        return {
            "report": report.model_dump(mode="json"),
            "trace": [{"node": "recommend", "summary": summary}],
        }


def _uncovered(state: AnalysisState) -> set[str]:
    """Coverage-issue findings that no retrieved source is linked to."""
    with_evidence = {fid for e in state.get("evidence", []) for fid in e["finding_ids"]}
    return {
        i["finding_id"]
        for i in state.get("issues", [])
        if i["kind"] == "coverage" and i["finding_id"] not in with_evidence
    }


def _confidence(
    findings: list[Finding],
    draft: Draft,
    issues: list[ValidationIssue],
    references: list[Evidence],
    revisions: int,
) -> tuple[Literal["high", "medium", "low"], str]:
    if issues:
        return "low", f"{len(issues)} validation issue(s) unresolved"
    serious = {f.id for f in findings if f.severity in ("critical", "high")}
    unsupported = serious & set(draft.unsupported_findings)
    if unsupported:
        return "low", f"no guideline evidence for {', '.join(sorted(unsupported))}"
    mean = sum(e.relevance for e in references) / len(references) if references else 0.0
    if mean < 0.5:
        return "medium", f"mean evidence relevance {mean:.2f}"
    if revisions:
        return "medium", f"passed validation after {revisions} revision(s)"
    return "high", f"passed validation first time; mean evidence relevance {mean:.2f}"
