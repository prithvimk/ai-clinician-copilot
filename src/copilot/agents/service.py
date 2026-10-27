"""Run, pause, review and resume patient analyses by thread_id."""

from collections.abc import Callable
from functools import lru_cache
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import BaseModel

from copilot.agents.interactions import check_interactions
from copilot.agents.patient import Patient, load_patient
from copilot.agents.workflow import AnalysisWorkflow, InteractionChecker, ReviewDecision
from copilot.config import get_settings
from copilot.rag.chat import get_checkpointer
from copilot.rag.llm import LLM, get_llm
from copilot.rag.retriever import Retriever

Status = Literal["awaiting_review", "completed", "failed", "not_found"]


class AnalysisStatus(BaseModel):
    thread_id: str
    status: Status
    review_request: dict | None = None  # findings, draft, issues, evidence while paused
    report: dict | None = None
    trace: list[dict] = []
    error: str | None = None


class ReviewNotPending(Exception):
    pass


class AnalysisService:
    def __init__(
        self, workflow: AnalysisWorkflow, patient_source: Callable[[str], Patient] = load_patient
    ):
        self.graph = workflow.graph
        self.load_patient = patient_source  # raises KeyError for unknown ids

    def start(self, patient: Patient, thread_id: str) -> AnalysisStatus:
        return self._run({"patient": patient.model_dump(mode="json")}, thread_id)

    def review(self, thread_id: str, decision: ReviewDecision) -> AnalysisStatus:
        if self.status(thread_id).status != "awaiting_review":
            raise ReviewNotPending(thread_id)
        return self._run(Command(resume=decision.model_dump()), thread_id)

    def resume(self, thread_id: str) -> AnalysisStatus:
        """Re-run from the last checkpoint after a failure (completed steps are not redone)."""
        return self._run(None, thread_id)

    def status(self, thread_id: str, error: str | None = None) -> AnalysisStatus:
        snapshot = self.graph.get_state(_config(thread_id))
        values = snapshot.values or {}
        common = {"thread_id": thread_id, "trace": values.get("trace", [])}
        if not values:
            status = "failed" if error else "not_found"
            return AnalysisStatus(status=status, error=error, **common)
        if snapshot.interrupts:
            return AnalysisStatus(
                status="awaiting_review", review_request=snapshot.interrupts[0].value, **common
            )
        if not snapshot.next and "report" in values:
            return AnalysisStatus(status="completed", report=values["report"], **common)
        error = error or "run stopped before completion"
        return AnalysisStatus(status="failed", error=error, **common)

    def _run(self, payload, thread_id: str) -> AnalysisStatus:
        try:
            self.graph.invoke(payload, config=_config(thread_id))
        except Exception as e:  # state up to the failing step is checkpointed; see resume()
            return self.status(thread_id, error=f"{type(e).__name__}: {e}")
        return self.status(thread_id)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def build_analysis_service(
    retriever: Retriever,
    llm: LLM,
    checkpointer: BaseCheckpointSaver,
    patient_source: Callable[[str], Patient] = load_patient,
    interaction_checker: InteractionChecker = check_interactions,
) -> AnalysisService:
    settings = get_settings()
    return AnalysisService(
        AnalysisWorkflow(
            retriever,
            llm,
            checkpointer,
            min_relevance=settings.min_relevance,
            evidence_per_finding=settings.evidence_per_finding,
            max_evidence=settings.max_evidence,
            max_revisions=settings.max_revisions,
            reasoning_effort=settings.reasoning_effort,
            interaction_checker=interaction_checker,
        ),
        patient_source,
    )


@lru_cache
def get_analysis_service() -> AnalysisService:
    """Production wiring: patient records, guideline search and interaction checks all go
    through the MCP tool server (in-process by default, stdio when MCP_SERVER_COMMAND is set)."""
    from copilot.mcp_server.client import McpRetriever, ToolCallError, get_toolbox

    toolbox = get_toolbox()

    def patient_source(patient_id: str) -> Patient:
        try:
            return toolbox.get_patient(patient_id)
        except ToolCallError as e:
            raise KeyError(patient_id) from e

    return build_analysis_service(
        McpRetriever(toolbox),
        get_llm(),
        get_checkpointer(),
        patient_source=patient_source,
        interaction_checker=lambda drugs: toolbox.check_drugs(drugs)[0],
    )
