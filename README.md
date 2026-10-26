# AI Clinician Copilot

RAG + agentic clinical decision support: ingests clinical guidelines (PDF/DOCX), analyses patient data through a LangGraph multi-agent workflow, and produces evidence-backed, cited recommendations for clinician review.

> **Decision support only.** Outputs are not a substitute for clinical judgement. All patient data in this repo is synthetic.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the phased build plan.

## Quickstart

```bash
uv sync
cp .env.example .env                      # add API keys
uv run python scripts/download_guidelines.py
uv run python -m copilot.ingestion.cli ingest data/guidelines
uv run python -m copilot.ingestion.cli search "monitoring for hypertension patients"

# Cited Q&A (needs ANTHROPIC_API_KEY)
uv run python -m copilot.ingestion.cli chat          # interactive, follow-ups use history
uv run uvicorn copilot.api.app:app                   # POST /chat {"question", "thread_id"?}
```

## RAG Q&A

`question → rewrite (standalone, from thread history) → hybrid search (dense + BM25, RRF, top 20) → cross-encoder rerank (top 5) → refuse if best relevance < MIN_RELEVANCE → answer with mandatory [n] citations → drop citation IDs that match no source; refuse if none remain`

- Memory: LangGraph checkpointer keyed by `thread_id` — in-process by default, Postgres when `POSTGRES_URL` is set.
- `POST /chat` returns `answer`, `citations[]` (doc, title, page, section, relevance, quoted chunk), `standalone_question`, `grounded`, `refusal_reason` (`low_evidence | insufficient_evidence | uncited | model_declined`) and `thread_id`. `GET /chat/{thread_id}/history` returns the conversation.

## Layout

```
src/copilot/
  config.py        settings (pydantic-settings, .env)
  ingestion/       loaders (PDF/DOCX), section-aware chunker, indexer, CLI
  rag/             embeddings, vector store, rerank retriever, cited QA, chat memory
  agents/          LangGraph workflow                         (Phase 3)
  mcp_server/      MCP tools                                  (Phase 4)
  guardrails/      PII + output validation                    (Phase 4)
  api/             FastAPI (/chat)
data/
  guidelines/      source documents (downloaded, not committed)
  patients/        synthetic patient records (JSON)
eval/              golden set + evaluation runner             (Phase 5)
deploy/            docker-compose, k8s                         (Phase 6)
docs/              architecture + plan
tests/
```
