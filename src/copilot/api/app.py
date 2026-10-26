"""HTTP API. Run with: uv run uvicorn copilot.api.app:app"""

from typing import Annotated
from uuid import uuid4

import anthropic
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from copilot.rag.chat import ChatService, get_chat_service
from copilot.rag.qa import Answer

Service = Annotated[ChatService, Depends(get_chat_service)]

app = FastAPI(
    title="AI Clinician Copilot",
    description="Guideline-grounded clinical Q&A. Decision support only.",
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
