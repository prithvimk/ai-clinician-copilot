"""HTTP API. Run with: uv run uvicorn copilot.api.app:app"""

import logging
import time
from typing import Annotated
from uuid import uuid4

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from copilot.agents.patient import PATIENTS_DIR, Patient
from copilot.agents.service import (
    AnalysisService,
    AnalysisStatus,
    ReviewNotPending,
    get_analysis_service,
)
from copilot.agents.workflow import ReviewDecision
from copilot.config import get_settings
from copilot.feedback import (
    Feedback,
    FeedbackIn,
    FeedbackStore,
    PostgresFeedbackStore,
    get_feedback_store,
)
from copilot.observability.telemetry import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    request_id_var,
    setup_logging,
    setup_tracing,
)
from copilot.rag.chat import ChatService, get_chat_service
from copilot.rag.qa import Answer

Service = Annotated[ChatService, Depends(get_chat_service)]
Analysis = Annotated[AnalysisService, Depends(get_analysis_service)]
Store = Annotated[FeedbackStore | PostgresFeedbackStore, Depends(get_feedback_store)]

log = logging.getLogger("copilot.api")
settings = get_settings()
setup_logging(settings)

app = FastAPI(
    title="AI Clinician Copilot",
    description="Guideline-grounded clinical Q&A and patient analysis. Decision support only.",
)
if setup_tracing(settings):
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready,metrics")


@app.middleware("http")
async def observe(request: Request, call_next):
    """Request id (propagated via X-Request-ID), latency/status metrics and an access log."""
    request_id = request.headers.get("x-request-id") or uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        seconds = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "unmatched")  # template, not raw path
        if route not in ("/metrics", "/health", "/ready"):
            HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
            HTTP_LATENCY.labels(request.method, route).observe(seconds)
            log.info(
                "request",
                extra={
                    "fields": {
                        "method": request.method,
                        "route": route,
                        "status": status,
                        "duration_ms": round(seconds * 1000, 1),
                    }
                },
            )
        request_id_var.reset(token)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(None, description="Omit to start a new conversation")


class ChatResponse(Answer):
    thread_id: str


@app.get("/health")
def health() -> dict:
    """Liveness: the process is serving."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: models loaded and the vector store reachable (first call warms models)."""
    from copilot.rag.retriever import get_retriever

    try:
        store = get_retriever().store
        if not store.client.collection_exists(store.collection):
            raise RuntimeError(f"collection {store.collection!r} missing")
    except Exception as e:
        log.warning("not ready: %s", e)
        raise HTTPException(503, f"not ready: {type(e).__name__}") from None
    return {"status": "ready"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


@app.get("/patients")
def patients() -> list[dict]:
    """Synthetic patient records available for analysis."""
    records = (
        Patient.model_validate_json(f.read_text()) for f in sorted(PATIENTS_DIR.glob("*.json"))
    )
    return [{"patient_id": p.patient_id, "summary": p.summary} for p in records]


@app.post("/analyze")
def analyze(req: AnalyzeRequest, service: Analysis) -> AnalysisStatus:
    """Run the analysis until it pauses for clinician review (or fails)."""
    if req.patient is not None:
        patient = req.patient
    elif req.patient_id:
        try:
            patient = service.load_patient(req.patient_id)
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
def review(
    thread_id: str, decision: ReviewDecision, service: Analysis, store: Store
) -> AnalysisStatus:
    """Approve, edit or reject the paused draft; resumes the workflow to the final report.

    Edits and rejections are also recorded as feedback (original vs. clinician version).
    """
    pending = service.status(thread_id)
    try:
        result = service.review(thread_id, decision)
    except ReviewNotPending:
        raise HTTPException(409, f"Analysis {thread_id} is not awaiting review") from None
    if decision.decision != "approve":
        store.add(
            FeedbackIn(
                kind="analysis",
                thread_id=thread_id,
                rating="down" if decision.decision == "reject" else None,
                comment=decision.comment,
                payload={
                    "decision": decision.decision,
                    "original_draft": (pending.review_request or {}).get("draft"),
                    "edited_recommendations": [r.model_dump() for r in decision.recommendations]
                    if decision.recommendations is not None
                    else None,
                },
            ),
            source="review",
        )
    return result


@app.post("/analyze/{thread_id}/resume")
def resume(thread_id: str, service: Analysis) -> AnalysisStatus:
    """Retry a failed analysis from its last completed step."""
    status = service.status(thread_id)
    if status.status != "failed":
        raise HTTPException(409, f"Analysis {thread_id} is {status.status}, not failed")
    return service.resume(thread_id)


@app.post("/feedback")
def add_feedback(item: FeedbackIn, store: Store) -> Feedback:
    return store.add(item)


@app.get("/feedback")
def list_feedback(store: Store, kind: str | None = None, limit: int = 100) -> list[Feedback]:
    return store.list(kind, limit)
