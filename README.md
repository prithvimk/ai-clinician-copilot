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

## Patient analysis (agentic workflow)

`analyze → retrieve → reason → validate → human_review → recommend` (LangGraph, checkpointed per `thread_id`)

- **analyze** (deterministic): lab/vital thresholds (HbA1c, BP, eGFR + trend, UACR, potassium, lipids) and medication safety checks on current drugs → severity-ranked findings.
- **retrieve**: one guideline query per finding through the Phase 2 reranking retriever; sources deduplicated and numbered.
- **reason** (LLM, structured output): recommendations linked to finding IDs, each with source citations and explicit medication changes; monitoring plan.
- **validate** (deterministic): citations must point at retrieved sources; every proposed drug change is re-checked against allergies (incl. class and ACE-inhibitor→ARB cross-reactivity) and kidney function (metformin, SGLT2, sulfonylurea, RAAS + potassium); every critical/high finding must be addressed. Failures loop back to reasoning (or re-retrieval for findings with no evidence) up to `MAX_REVISIONS`, then go to review flagged.
- **human_review**: pauses for a clinician to approve, edit (edits are re-checked) or reject.
- **recommend**: prioritised actions, monitoring, cited references, unresolved issues, confidence and disclaimer.

API: `POST /analyze {"patient_id": "P001"}` (or an inline `patient`) → `awaiting_review` with the draft; `POST /analyze/{thread_id}/review {"decision": "approve" | "edit" | "reject", ...}` → final report; `POST /analyze/{thread_id}/resume` retries a failed run from its last completed step; `GET /analyze/{thread_id}` returns status.

Safety thresholds follow the synthetic sample care pathway: they demonstrate the mechanism and are not a validated clinical rule set.

## Layout

```
src/copilot/
  config.py        settings (pydantic-settings, .env)
  ingestion/       loaders (PDF/DOCX), section-aware chunker, indexer, CLI
  rag/             embeddings, vector store, rerank retriever, cited QA, chat memory
  agents/          patient model, rules, safety checks, LangGraph analysis workflow
  mcp_server/      MCP tools                                  (Phase 4)
  guardrails/      PII + output validation                    (Phase 4)
  api/             FastAPI (/chat, /analyze)
data/
  guidelines/      source documents (downloaded, not committed)
  patients/        synthetic patient records (JSON)
eval/              golden set + evaluation runner             (Phase 5)
deploy/            docker-compose, k8s                         (Phase 6)
docs/              architecture + plan
tests/
```
