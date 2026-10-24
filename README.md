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
```

## Layout

```
src/copilot/
  config.py        settings (pydantic-settings, .env)
  ingestion/       loaders (PDF/DOCX), section-aware chunker, indexer, CLI
  rag/             embeddings, vector store, retrieval        (Phase 2: rerank, cited QA, memory)
  agents/          LangGraph workflow                         (Phase 3)
  mcp_server/      MCP tools                                  (Phase 4)
  guardrails/      PII + output validation                    (Phase 4)
  api/             FastAPI                                    (Phase 2+)
data/
  guidelines/      source documents (downloaded, not committed)
  patients/        synthetic patient records (JSON)
eval/              golden set + evaluation runner             (Phase 5)
deploy/            docker-compose, k8s                         (Phase 6)
docs/              architecture + plan
tests/
```
