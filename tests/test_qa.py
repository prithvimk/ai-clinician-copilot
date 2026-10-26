import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient

from copilot.api.app import app
from copilot.ingestion.indexer import ingest_path
from copilot.rag.chat import ChatService, get_chat_service
from copilot.rag.embeddings import get_embedder
from copilot.rag.llm import LLMRefusal
from copilot.rag.qa import INSUFFICIENT, CitedQA, validate_citations
from copilot.rag.retriever import Retriever, get_reranker
from copilot.rag.vector_store import VectorStore
from tests.fakes import FakeLLM, StubRetriever


@pytest.fixture
def retriever() -> StubRetriever:
    return StubRetriever(
        "Stop metformin when eGFR falls below 30.",
        "Reduce the metformin dose when eGFR is 30-44.",
        "Target blood pressure below 140/90 mmHg.",
    )


def make_qa(retriever, llm, min_relevance=0.0) -> CitedQA:
    return CitedQA(retriever, llm, min_relevance=min_relevance)


# --- retrieval ---------------------------------------------------------------------


def test_rerank_orders_by_relevance(sample_pdf, sample_docx):
    store = VectorStore(QdrantClient(":memory:"), "test", get_embedder())
    ingest_path(sample_pdf.parent, store)
    retriever = Retriever(store, get_reranker(), candidates=10, top_k=3)

    hits = retriever.retrieve("When should metformin be stopped?")
    assert len(hits) == 3
    assert hits[0].chunk.doc_id == "pathway" and "Stop" in hits[0].chunk.text
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)


# --- citation validation -----------------------------------------------------------


def test_validate_citations_drops_unknown_ids():
    text, valid, invalid = validate_citations("Stop metformin [2] [7]. Check HbA1c [1, 9].", 3)
    assert text == "Stop metformin [2]. Check HbA1c [1]."
    assert valid == [2, 1]
    assert invalid == [7, 9]


def test_validate_citations_ignores_non_numeric_brackets():
    text, valid, _ = validate_citations("Target [BP] below 140/90 [1].", 2)
    assert text == "Target [BP] below 140/90 [1]."
    assert valid == [1]


# --- answering ---------------------------------------------------------------------


def test_answer_returns_structured_citations(retriever):
    llm = FakeLLM("Stop metformin when eGFR is below 30 [1]. Unsupported claim [8].")
    answer = make_qa(retriever, llm).answer("When should metformin be stopped?")

    assert answer.grounded and answer.refusal_reason is None
    assert answer.answer == "Stop metformin when eGFR is below 30 [1]. Unsupported claim."
    assert [c.id for c in answer.citations] == [1]
    assert answer.citations[0].chunk_id == "c0"
    assert answer.citations[0].label == "Care Pathway (Diabetes > Metformin)"
    assert answer.invalid_citation_ids == [8]
    prompt = llm.calls[0]["messages"][0]["content"]
    assert '<source id="3"' in prompt and '<source id="4"' not in prompt


def test_low_evidence_refuses_without_calling_llm(retriever):
    llm = FakeLLM()
    answer = make_qa(retriever, llm, min_relevance=0.95).answer("Metformin?")
    assert answer.refusal_reason == "low_evidence" and not answer.grounded
    assert llm.calls == []


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (INSUFFICIENT, "insufficient_evidence"),
        ("Metformin is great.", "uncited"),
        ("Metformin is great [9].", "uncited"),
        (LLMRefusal("declined"), "model_declined"),
    ],
)
def test_ungrounded_replies_are_refused(retriever, reply, reason):
    answer = make_qa(retriever, FakeLLM(reply)).answer("When should metformin be stopped?")
    assert answer.refusal_reason == reason
    assert answer.citations == [] and not answer.grounded


def test_first_turn_is_not_rewritten(retriever):
    llm = FakeLLM()
    assert make_qa(retriever, llm).rewrite([], "What is the BP target?") == "What is the BP target?"
    assert llm.calls == []


# --- conversation memory -----------------------------------------------------------


def test_follow_up_resolves_against_thread_history(retriever):
    llm = FakeLLM(
        "Stop metformin when eGFR is below 30 [1].",  # turn 1 answer
        "When should metformin dose be reduced?",  # turn 2 rewrite
        "Reduce the dose [1].",  # turn 2 answer
        "Reduce the dose [1].",  # other thread: answer only, no rewrite
    )
    service = ChatService(make_qa(retriever, llm), InMemorySaver())

    service.ask("When should metformin be stopped?", "t1")
    follow_up = service.ask("And when should the dose be reduced?", "t1")

    assert follow_up.standalone_question == "When should metformin dose be reduced?"
    assert retriever.queries[-1] == "When should metformin dose be reduced?"
    rewrite_prompt = llm.calls[1]["messages"][0]["content"]
    assert "When should metformin be stopped?" in rewrite_prompt
    assert "eGFR is below 30" in rewrite_prompt
    assert len(service.history("t1")) == 4

    other = service.ask("And when should the dose be reduced?", "t2")
    assert other.standalone_question == "And when should the dose be reduced?"
    assert len(llm.calls) == 4


# --- API ---------------------------------------------------------------------------


def test_chat_endpoint(retriever):
    llm = FakeLLM("Stop below eGFR 30 [1].", "metformin dose reduction", "Reduce at 30-44 [1].")
    service = ChatService(make_qa(retriever, llm), InMemorySaver())
    app.dependency_overrides[get_chat_service] = lambda: service
    try:
        client = TestClient(app)
        first = client.post("/chat", json={"question": "When to stop metformin?"}).json()
        assert first["grounded"] and first["citations"][0]["id"] == 1
        thread_id = first["thread_id"]

        second = client.post(
            "/chat", json={"question": "And reduce the dose?", "thread_id": thread_id}
        ).json()
        assert second["standalone_question"] == "metformin dose reduction"
        assert len(client.get(f"/chat/{thread_id}/history").json()) == 4

        assert client.post("/chat", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides.clear()
