# Implementation Plan

Rubric weights drive the order: Agentic design 25%, RAG 25%, Healthcare reasoning 15%, System design 15%, Evaluation 10%, Code 5%, Demo 5%.

| Phase | Goal | Day |
|---|---|---|
| 1 | Foundation & knowledge ingestion | 1 |
| 2 | RAG Q&A with citations + memory | 1–2 |
| 3 | Agentic clinical reasoning (LangGraph) | 2–3 |
| 4 | MCP tools, guardrails, UI | 3 |
| 5 | Evaluation framework | 4 |
| 6 | Observability & deployment | 4 |
| 7 | Architecture doc & presentation | 5 |

---

## Phase 1 — Foundation & Knowledge Ingestion

**Goal:** clinical documents (PDF/DOCX) become searchable, metadata-rich chunks in a vector store.

- Project scaffold: `uv`, `src/` layout, `pydantic-settings` config, `.env.example`, ruff, pytest.
- Loaders: PDF (PyMuPDF, per-page text + heading detection via font size) and DOCX (python-docx, `Heading N` styles).
- Section-aware chunker: split on headings, then token-bounded windows (~600 tokens, 100 overlap); each chunk carries `doc_id, title, source_file, page_start, page_end, section_path, chunk_id`.
- Embeddings: pluggable provider; default local `BAAI/bge-small-en-v1.5` via fastembed (no API key, on-prem friendly).
- Vector store: Qdrant (embedded local mode for dev; same client works against a server in prod) with dense vectors + BM25 sparse vectors for hybrid search later.
- Idempotent indexing (deterministic chunk IDs, re-ingest replaces a document).
- CLI: `ingest <path>`, `search <query>`, `stats`.
- Data: guideline download script (public WHO/NICE sources, not committed), synthetic patients (JSON).
- Tests: chunker, loaders, ingest→search round trip.

**Done when:** `ingest data/guidelines` indexes the corpus and `search "monitoring for hypertension"` returns relevant chunks with page/section metadata.

## Phase 2 — RAG Q&A

- Hybrid retrieval (dense + BM25, RRF fusion) → cross-encoder rerank (top 20 → 5).
- Query rewriting using chat history (standalone question).
- Answer generation with numbered context, mandatory `[n]` citations, post-hoc citation-ID validation.
- Low-evidence refusal threshold.
- Conversational memory: LangGraph `MemorySaver` (dev) / `PostgresSaver` (prod), keyed by `thread_id`.
- FastAPI `POST /chat` returning answer + structured citations.

**Done when:** the three sample questions return grounded, cited answers and follow-ups resolve against history.

## Phase 3 — Agentic Clinical Reasoning

- Typed `CopilotState`; LangGraph `StateGraph` with nodes:
  1. **Patient Analysis** — Pydantic-structured extraction + deterministic lab rules (HbA1c, BP, eGFR, LDL thresholds).
  2. **Clinical Retrieval** — per-finding query generation, uses Phase 2 retriever.
  3. **Reasoning** — finding → evidence → action, cited per claim.
  4. **Validation** — citation support check, allergy/contraindication checks; conditional edge back to Retrieval/Reasoning (max 2 retries).
  5. **Human review** — `interrupt()` for approve/edit/reject.
  6. **Recommendation** — clinician-facing summary, prioritised actions, monitoring plan, confidence, disclaimer.
- Checkpointing + resume after failure; node-level retries/timeouts.
- `POST /analyze`, `POST /analyze/{thread_id}/review`.

**Done when:** a synthetic uncontrolled-diabetes + CKD patient triggers a validation failure (metformin/eGFR), self-corrects, pauses for review, and resumes to a final cited report.

## Phase 4 — MCP, Guardrails, UI

- MCP server exposing `search_guidelines`, `get_patient_record` (mock FHIR), `check_drug_interaction`; agents call tools through an MCP client.
- Guardrails: Presidio PII redaction before LLM calls; output checks (disclaimer, no uncited dosing, scope refusal).
- Streamlit UI: chat tab, patient-analysis tab with agent trace, review controls, feedback (👍/👎 + edits) persisted for the learning loop.

## Phase 5 — Evaluation Framework

- Golden set (30–50 Q) with gold chunk IDs and reference answers.
- Retrieval: Recall@K, Precision@K, MRR — ablation: dense vs hybrid vs hybrid+rerank.
- Generation: Faithfulness, Answer Relevance (RAGAS), Groundedness, Citation Accuracy (claim-level LLM judge), hallucination detection (claim NLI + clinical entity cross-check).
- Agent eval: patient scenarios with expected findings/flags.
- `eval/run_eval.py` → markdown scorecard; CI gate on regressions.

## Phase 6 — Observability & Deployment

- Langfuse tracing (per-node latency, tokens, cost), structured logs, OpenTelemetry.
- Dockerfiles + docker-compose (api, ui, qdrant, postgres, langfuse).
- Kubernetes manifests / Helm: HPA, StatefulSets, NetworkPolicies, secrets, PDBs.
- DR: Postgres PITR, Qdrant snapshots, re-ingest from versioned sources.

## Phase 7 — Docs & Presentation

- Architecture document (≤5 pages): system, RAG, vector store, agents, observability, security, scalability, tech rationale.
- Production deployment architecture diagram + justification.
- 60-min deck + rehearsed demo scenario.
