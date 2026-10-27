# Architecture

AI Clinician Copilot is a RAG + agentic decision-support system. It answers clinical questions with cited guideline evidence and analyses a patient record into a prioritised, evidence-linked plan that a clinician approves, edits or rejects. It never acts on its own: a human review gate sits before every final report. All patient data is synthetic.

## 1. System overview

```
Clinician ─► Streamlit UI ─► FastAPI ──┬─► Cited Q&A graph (LangGraph)
                                       ├─► Patient-analysis graph (LangGraph, human-in-the-loop)
                                       └─► Feedback store
                                              │
              MCP tools (search_guidelines · get_patient_record · check_drug_interaction)
                                              │
               Qdrant (dense + BM25)   Postgres (checkpoints, feedback)   Anthropic API
```

One image serves every role (API, UI, ingest job). The API is stateless: conversation and analysis threads live in Postgres checkpoints, the index in Qdrant.

## 2. RAG pipeline

**Ingestion.** PDF (PyMuPDF, heading detection from font size) and DOCX (python-docx heading styles) are split on headings, then into ~600-token windows with 100 overlap. Each chunk carries `doc_id, title, page_start/end, section_path, chunk_id`. Chunk ids are deterministic, so re-ingest replaces a document and is idempotent. The corpus manifest (`data/guidelines/sources.yaml`) is versioned; the PDFs are not committed.

**Retrieval.** `question → rewrite to a standalone question from thread history → hybrid search (dense bge-small + BM25, fused with RRF, top 20) → cross-encoder rerank (top 5)`. Dense catches paraphrase; BM25 catches exact drug names, doses and lab terms that embeddings blur; the reranker restores precision before the LLM sees anything.

**Generation.** The model receives numbered sources and must cite `[n]`. Post-hoc checks drop citation ids that match no source and refuse if none remain. If the best rerank score is below `MIN_RELEVANCE` the system refuses **without calling the LLM**. Refusal reasons are explicit (`low_evidence`, `insufficient_evidence`, `uncited`, `model_declined`).

**Why Qdrant.** One client for embedded dev mode and a server in production, native sparse + dense vectors in one collection (no second engine for BM25), payload filtering on `doc_id`/conditions, and snapshots for DR. Because the index is derived from the corpus it can always be rebuilt.

## 3. Agent design

A LangGraph `StateGraph` with a typed state, checkpointed per `thread_id`:

`analyze → retrieve → reason → validate → human_review → recommend`

| Node | Kind | Responsibility |
|---|---|---|
| analyze | deterministic | Lab/vital thresholds (HbA1c, BP, eGFR + trend, UACR, potassium, lipids), medication safety checks → severity-ranked findings |
| retrieve | tool call (MCP) | One guideline query per finding through the reranking retriever; sources de-duplicated and numbered |
| reason | LLM, structured output | Recommendations linked to finding ids, each cited, with explicit medication changes and a monitoring plan |
| validate | deterministic | Citations resolve to retrieved sources; each proposed drug is re-checked against allergies (class and ACE-inhibitor→ARB cross-reactivity), kidney function and interactions; every critical/high finding is addressed |
| human_review | `interrupt()` | Clinician approves, edits (edits are re-validated) or rejects |
| recommend | deterministic | Final report: actions, monitoring, references, unresolved issues, confidence, disclaimer |

**The design choice that matters:** the LLM proposes; deterministic code disposes. Safety-critical facts (is this drug safe at this eGFR? does this allergy cross-react?) are rules, not model judgement. Validation failures loop back to reasoning, or to retrieval when a finding has no evidence, up to `MAX_REVISIONS`, then go to the clinician **flagged** rather than silently passing. A failed run resumes from its last completed step.

**MCP.** Tools are exposed through an MCP server (in-process by default, stdio via `MCP_SERVER_COMMAND`). Patient records are served as a mock FHIR R4 Bundle with LOINC-coded observations, so swapping in a real FHIR server changes one adapter, and external MCP clients can reuse the same tools.

## 4. Healthcare safety and security

- **PII.** Identifiers (names, contacts, NHS/SSN/MRN, DOB, postcodes) are replaced with stable placeholders before every LLM call and restored in the output; clinical values and visit dates are preserved. Presidio + spaCy unioned with regex recognizers.
- **Input scope.** Prompt-injection attempts and first-person emergencies are refused before any retrieval or LLM call.
- **Output.** Sentences stating a dose without a citation are removed and reported as `guardrail_flags`; every answer carries a decision-support disclaimer.
- **Platform.** Restricted Pod Security (non-root, read-only root FS, capabilities dropped, seccomp), default-deny NetworkPolicies, only the UI exposed (TLS), secrets out of band, prompts and patient data never written to spans.

## 5. Observability

OpenTelemetry spans for every graph node, retrieval, QA and LLM call (model, tokens, cache tokens, estimated cost; thread id as the session) exported to Langfuse or any OTLP collector. Prometheus `/metrics` (requests, latency, per-node latency, tokens, cost, refusal outcomes, guardrail flags) with alerts on 5xx rate, p95 `/analyze`, refusal rate (index or retrieval broken) and LLM spend. JSON logs carry request, thread and trace ids. `/health` and `/ready` (models loaded, Qdrant reachable).

## 6. Evaluation

`eval/run_eval.py` writes a markdown scorecard and gates CI on regressions against `eval/baseline.json`.

| Suite | Measures |
|---|---|
| agent | Finding precision/recall, severity, safety-rule and interaction accuracy on 7 scenarios (deterministic, no key) |
| retrieval | Recall@K, Precision@K, MRR; ablation dense vs BM25 vs hybrid vs hybrid + rerank |
| generation | Answer/refuse accuracy, citation validity, key-fact coverage, clinical-entity cross-check; with `--judge`: faithfulness, citation accuracy, groundedness, hallucination rate, relevance |
| agent_llm | Full workflow: completion, validation pass rate, serious-finding coverage |

The 30-question golden set includes 8 items that **must** be refused, so over-answering is measured as well as under-answering.

## 7. Scalability and deployment

See `docs/deployment.md` for the production diagram. Summary: API Deployment with HPA (2–10, prod 3–20) behind cluster-internal service; UI with cookie affinity; Qdrant StatefulSet (vertical first, then distributed/managed); Postgres via CloudNativePG (3 instances, WAL archiving for PITR). Embedding and rerank models are baked into the image so pods need egress only to the Anthropic API. Scaling is bounded by LLM latency and rate limits, not CPU: the API holds no state, so horizontal scale is cheap, and per-finding retrieval is the main fan-out to watch.

## 8. Technology rationale

| Choice | Why | Trade-off |
|---|---|---|
| LangGraph | Typed state, conditional edges, checkpointing and `interrupt()` for human review | Framework lock-in; graph is small enough to port |
| Qdrant | Hybrid dense + sparse, embedded→server, snapshots | Another stateful service; mitigated by rebuildable index |
| Local embeddings + reranker | No PHI-adjacent text leaves the cluster for retrieval; no per-query cost | Smaller models than hosted ones; ablation shows what rerank buys |
| Claude (structured output) | Reliable schema-bound reasoning and citations | Vendor dependency; isolated behind `rag/llm.py` |
| MCP | Standard tool boundary, reusable by other clients | In-process by default, so no network hop in the common path |
| Postgres checkpoints | Stateless API, resumable threads, durable feedback | Checkpoint size grows with thread length |

## 9. Known limits

Safety thresholds follow the synthetic sample care pathway and demonstrate the mechanism; they are not a validated clinical rule set. The golden set covers the sample pathway only; WHO/KDIGO items should be added once those PDFs are indexed. Production use would require clinical governance, a real FHIR integration and regulatory review.
