"""HTTP API. Run with: uv run uvicorn copilot.api.app:app"""

from typing import Annotated
from uuid import uuid4

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from copilot.agents.patient import Patient, load_patient
from copilot.agents.service import (
    AnalysisService,
    AnalysisStatus,
    ReviewNotPending,
    get_analysis_service,
)
from copilot.agents.workflow import ReviewDecision
from copilot.rag.chat import ChatService, get_chat_service
from copilot.rag.qa import Answer

Service = Annotated[ChatService, Depends(get_chat_service)]
Analysis = Annotated[AnalysisService, Depends(get_analysis_service)]

app = FastAPI(
    title="AI Clinician Copilot",
    description="Guideline-grounded clinical Q&A and patient analysis. Decision support only.",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(None, description="Omit to start a new conversation")


class ChatResponse(Answer):
    thread_id: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest, service: Service) -> ChatResponse:
    thread_id = req.thread_id or uuid4().hex
    try:
        answer = service.ask(req.question, thread_id)
    except anthropic.APIError as e:
        raise HTTPException(502, f"LLM provider error: {e.__class__.__name__}") from e
    return ChatResponse(thread_id=thread_id, **answer.model_dump())


@app.get("/chat/{thread_id}/history")
def history(thread_id: str, service: Service) -> list[dict]:
    return service.history(thread_id)


class AnalyzeRequest(BaseModel):
    patient_id: str | None = Field(None, description="ID of a stored (synthetic) record")
    patient: Patient | None = Field(None, description="Inline record; overrides patient_id")


@app.post("/analyze")
def analyze(req: AnalyzeRequest, service: Analysis) -> AnalysisStatus:
    """Run the analysis until it pauses for clinician review (or fails)."""
    if req.patient is not None:
        patient = req.patient
    elif req.patient_id:
        try:
            patient = load_patient(req.patient_id)
        except KeyError:
            raise HTTPException(404, f"Unknown patient {req.patient_id}") from None
    else:
        raise HTTPException(422, "Provide patient_id or patient")
    return service.start(patient, uuid4().hex)


@app.get("/analyze/{thread_id}")
def analysis_status(thread_id: str, service: Analysis) -> AnalysisStatus:
    status = service.status(thread_id)
    if status.status == "not_found":
        raise HTTPException(404, f"Unknown analysis {thread_id}")
    return status


@app.post("/analyze/{thread_id}/review")
def review(thread_id: str, decision: ReviewDecision, service: Analysis) -> AnalysisStatus:
    """Approve, edit or reject the paused draft; resumes the workflow to the final report."""
    try:
        return service.review(thread_id, decision)
    except ReviewNotPending:
        raise HTTPException(409, f"Analysis {thread_id} is not awaiting review") from None


@app.post("/analyze/{thread_id}/resume")
def resume(thread_id: str, service: Analysis) -> AnalysisStatus:
    """Retry a failed analysis from its last completed step."""
    status = service.status(thread_id)
    if status.status != "failed":
        raise HTTPException(409, f"Analysis {thread_id} is {status.status}, not failed")
    return service.resume(thread_id)
