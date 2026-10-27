"""UI -> API end to end: Streamlit AppTest drives the real FastAPI app (in-process), with the
scripted LLM and stub retriever behind it."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from streamlit.testing.v1 import AppTest

from copilot.agents.service import get_analysis_service
from copilot.api.app import app
from copilot.feedback import FeedbackIn, FeedbackStore, get_feedback_store
from copilot.rag.chat import ChatService, get_chat_service
from copilot.rag.qa import CitedQA
from tests.fakes import FakeLLM, StubRetriever
from tests.test_agents import SAFE, service_with

UI = str(Path(__file__).parents[1] / "src" / "copilot" / "ui" / "app.py")


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Route the UI's HTTP client into the FastAPI app with fake services behind it."""
    llm = FakeLLM()
    retriever = StubRetriever("Stop metformin when eGFR falls below 30.")
    chat = ChatService(CitedQA(retriever, llm, min_relevance=0.0), InMemorySaver())
    analysis, _ = service_with(llm)
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    app.dependency_overrides |= {
        get_chat_service: lambda: chat,
        get_analysis_service: lambda: analysis,
        get_feedback_store: lambda: store,
    }
    monkeypatch.setattr(
        "copilot.ui.client.httpx.Client",
        lambda base_url, timeout: TestClient(app, base_url=base_url),
    )
    yield llm, store
    app.dependency_overrides.clear()


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_chat_tab_shows_cited_answer_and_records_feedback(api):
    llm, store = api
    llm.replies = ["Stop metformin when eGFR is below 30 [1]."]
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception

    at.chat_input[0].set_value("When should metformin be stopped?").run()
    assert "Stop metformin when eGFR is below 30 [1]." in markdown(at)
    assert any("[1] Care Pathway" in e.label for e in at.expander)

    at.text_input[0].set_value("Also mention dialysis.")
    button(at, "👍").click().run()
    [record] = store.list()
    assert record.kind == "chat" and record.rating == "up"
    assert record.correction == "Also mention dialysis."
    assert record.payload["question"] == "When should metformin be stopped?"


def test_analysis_tab_review_flow_records_edits(api):
    llm, store = api
    llm.replies = [SAFE]
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert at.selectbox[0].value == "P001"

    button(at, "Run analysis").click().run()
    assert "awaiting your review" in "\n".join(s.value for s in at.subheader)
    assert "Reduce metformin" in markdown(at)

    at.radio[0].set_value("edit")
    edited = SAFE.model_dump()["recommendations"][:2]  # clinician drops the metformin change
    at.text_area[0].set_value(json.dumps(edited))
    button(at, "Submit review").click().run()
    assert not at.exception
    headers = "\n".join(s.value for s in at.subheader)
    assert "Report — ✅ Approved" in headers
    assert "Reduce metformin" not in markdown(at)

    [record] = store.list(kind="analysis")
    assert record.source == "review" and record.payload["decision"] == "edit"
    assert len(record.payload["original_draft"]["recommendations"]) == 3
    assert len(record.payload["edited_recommendations"]) == 2


def test_feedback_api_round_trip(tmp_path):
    store = FeedbackStore(tmp_path / "f.jsonl")
    app.dependency_overrides[get_feedback_store] = lambda: store
    try:
        client = TestClient(app)
        r = client.post("/feedback", json={"kind": "chat", "thread_id": "t", "rating": "down"})
        assert r.status_code == 200 and r.json()["source"] == "user"
        store.add(FeedbackIn(kind="analysis", thread_id="a"), source="review")
        assert [f["kind"] for f in client.get("/feedback").json()] == ["analysis", "chat"]
        assert len(client.get("/feedback", params={"kind": "chat"}).json()) == 1
        assert client.post("/feedback", json={"kind": "x", "thread_id": "t"}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_patients_endpoint():
    ids = [p["patient_id"] for p in TestClient(app).get("/patients").json()]
    assert ids == ["P001", "P002", "P003"]
