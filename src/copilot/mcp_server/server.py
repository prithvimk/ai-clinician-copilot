"""MCP server exposing the copilot's clinical tools.

    search_guidelines       hybrid search + rerank over the indexed guideline corpus
    get_patient_record      a patient's record as a FHIR R4 Bundle (mock FHIR store)
    check_drug_interaction  pairwise interactions, plus patient-specific contraindications

Run over stdio for external MCP clients:  uv run python -m copilot.mcp_server
"""

from collections.abc import Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from copilot.agents.interactions import Interaction, check_interactions
from copilot.agents.patient import Patient, load_patient
from copilot.agents.safety import SafetyIssue, check_medication
from copilot.ingestion.models import Chunk
from copilot.mcp_server.fhir import to_bundle


class GuidelineHit(BaseModel):
    relevance: float
    citation: str
    chunk: Chunk


class GuidelineResults(BaseModel):
    hits: list[GuidelineHit]


class InteractionReport(BaseModel):
    interactions: list[Interaction]
    patient_issues: list[SafetyIssue]  # allergy / kidney-function issues for starting each drug


def build_server(
    retriever_factory: Callable[[], object],
    patient_source: Callable[[str], Patient] = load_patient,
) -> MCPServer:
    """`retriever_factory` is called on first search, so the server starts without models."""
    server = MCPServer(
        "clinician-copilot",
        instructions="Clinical decision-support tools over synthetic patient data and an "
        "indexed guideline corpus. Outputs support clinician review only.",
    )
    retriever: list[object] = []

    def get_patient(patient_id: str) -> Patient:
        try:
            return patient_source(patient_id)
        except KeyError:
            raise ToolError(f"Unknown patient {patient_id!r}") from None

    @server.tool()
    def search_guidelines(query: str, k: int = 5) -> GuidelineResults:
        """Search clinical guidelines; returns the k most relevant excerpts with citations."""
        if not query.strip():
            raise ToolError("query must not be empty")
        if not retriever:
            retriever.append(retriever_factory())
        hits = retriever[0].retrieve(query, k=max(1, min(k, 20)))
        return GuidelineResults(
            hits=[
                GuidelineHit(
                    relevance=round(h.score, 4), citation=h.chunk.citation(), chunk=h.chunk
                )
                for h in hits
            ]
        )

    @server.tool()
    def get_patient_record(patient_id: str) -> dict:
        """Fetch a patient's record as a FHIR R4 Bundle (Patient, Condition,
        MedicationStatement, AllergyIntolerance, Observation, DocumentReference)."""
        return to_bundle(get_patient(patient_id))

    @server.tool()
    def check_drug_interaction(
        drugs: list[str], patient_id: str | None = None
    ) -> InteractionReport:
        """Check drug-drug interactions between the listed drugs; with patient_id, also check
        each drug against that patient's allergies and kidney function as if starting it."""
        if not drugs:
            raise ToolError("drugs must not be empty")
        patient = get_patient(patient_id) if patient_id else None
        return InteractionReport(
            interactions=check_interactions(drugs),
            patient_issues=[issue for d in drugs for issue in check_medication(patient, d, "start")]
            if patient
            else [],
        )

    return server


def main() -> None:
    from copilot.rag.retriever import get_retriever

    build_server(get_retriever).run("stdio")
