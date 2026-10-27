# AI Clinician Copilot

RAG + agentic clinical decision support: ingests clinical guidelines (PDF/DOCX), analyses patient data through a LangGraph multi-agent workflow, and produces evidence-backed, cited recommendations for clinician review.

> **Decision support only.** Outputs are not a substitute for clinical judgement. All patient data in this repo is synthetic.

See [docs/implementation-plan.md](docs/implementation-plan.md) for the phased build plan, [docs/architecture.md](docs/architecture.md) for the architecture, [docs/deployment.md](docs/deployment.md) for the production diagram and [docs/demo.md](docs/demo.md) for the demo script.

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
uv run streamlit run src/copilot/ui/app.py           # UI (talks to the API at COPILOT_API_URL)
uv run python -m copilot.mcp_server                  # MCP tool server over stdio, for external clients

# Optional: Presidio NER for PII redaction (regex recognizers are used without it)
uv run python -m spacy download en_core_web_lg
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

## MCP tools, guardrails and UI

**MCP server** (`mcp_server/`): `search_guidelines` (rerank retriever), `get_patient_record` (record as a mock FHIR R4 Bundle: Patient, Condition, MedicationStatement, AllergyIntolerance, Observation with LOINC codes, DocumentReference) and `check_drug_interaction` (class-level interaction table plus patient allergy/kidney checks). The analysis workflow reaches patients, guidelines and interaction checks through an MCP client — in-process by default, or a stdio server via `MCP_SERVER_COMMAND`. Validation flags high-severity interactions introduced by drugs the draft starts.

**Guardrails** (`guardrails/`):
- PII: identifiers (names, contact details, NHS/SSN/MRN numbers, DOB, postcodes) are replaced with stable placeholders before every LLM call and restored in the output; clinical terms, values and visit dates are kept. Presidio + spaCy when the model is installed, unioned with regex recognizers.
- Input scope: prompt-injection attempts and first-person emergencies are refused before any retrieval or LLM call.
- Output: sentences stating a dose without a citation are removed (`guardrail_flags`), and every answer and report carries a decision-support disclaimer.

**UI** (`ui/app.py`, Streamlit): guideline Q&A with expandable citations; patient analysis with findings, agent trace, draft plan, validation issues and approve/edit/reject; 👍/👎 + corrections on answers and reports.

**Feedback** (`feedback.py`): `POST /feedback`, `GET /feedback`; clinician edits and rejections are recorded automatically with the original and edited plan, as JSONL for the learning loop.

## Evaluation

```bash
uv run python eval/run_eval.py                                   # deterministic agent checks
uv run python eval/run_eval.py --suites agent,retrieval,generation,agent_llm --judge
uv run python eval/run_eval.py --suites agent,retrieval --gate eval/baseline.json  # CI gate
uv run python eval/run_eval.py --propose "When should metformin be stopped?"       # annotate gold
```

Writes `eval/reports/scorecard.md` (+ `results.json`). Suites that need models, an index or an API key are skipped with the reason.

| Suite | Measures | Needs |
|---|---|---|
| `agent` | finding precision/recall and severity, safety-rule and interaction accuracy on 7 patient scenarios (`eval/golden/agent.yaml`) | nothing |
| `retrieval` | Recall@K, Precision@K, MRR, hit rate; ablation dense vs sparse (BM25) vs hybrid (RRF) vs hybrid + rerank | index + embedding models |
| `generation` | answer/refuse decision accuracy, refusal recall, key-fact coverage, citation-ID validity, clinical-entity cross-check (numbers and drug names absent from the sources); with `--judge`: faithfulness, citation accuracy, groundedness, hallucination rate, answer relevance (claim-level Claude judge) | API key |
| `agent_llm` | full workflow on each scenario: completion, validation pass rate, coverage of serious findings, first-pass rate | API key |

The golden Q&A set (`eval/golden/qa.yaml`, 30 items incl. 8 that must be refused) matches gold evidence by `doc_id` + phrase, so it survives re-chunking; a test checks every gold phrase exists in the parsed sample pathway. It currently covers the synthetic sample care pathway; WHO/KDIGO items can be added once those PDFs are indexed. The generation metrics follow RAGAS definitions but use a Claude judge rather than the `ragas` package.

`eval/baseline.json` holds the gated metrics; `--gate` fails on any drop beyond `--tolerance` (default 0.02) and reports baseline metrics a run did not measure. Update it with `--write-baseline` after an intentional change. CI (`.github/workflows/ci.yml`) runs lint, tests, indexes the sample pathway and runs the gate.

## Deployment and observability

```bash
docker compose -f deploy/docker-compose.yml up -d --build                 # api, ui, qdrant, postgres, jaeger
docker compose -f deploy/docker-compose.yml --profile tools run --rm ingest
kubectl apply -k deploy/k8s/overlays/prod                                 # Kubernetes
```

- **Tracing:** OpenTelemetry spans for every workflow node, retrieval, QA and LLM call (tokens, estimated cost, thread id as session) → Langfuse (set `LANGFUSE_*`) or any OTLP collector (`OTEL_EXPORTER_OTLP_ENDPOINT`; Jaeger in compose at http://localhost:16686).
- **Metrics:** Prometheus at `/metrics` (requests, latency, node latency, tokens, cost, refusal outcomes, guardrail flags) with alert rules.
- **Logs:** JSON lines with request id, thread id and trace/span ids; `/health` (liveness) and `/ready` (models loaded, Qdrant reachable).
- **Kubernetes** (`deploy/k8s`, Kustomize base + prod overlay): API Deployment with HPA and PDB, UI with cookie affinity, Qdrant StatefulSet, Postgres via CloudNativePG (3 instances, WAL archiving for PITR), ingest Job, snapshot CronJob, default-deny NetworkPolicies, restricted Pod Security, ServiceMonitor and alerts.
- Postgres holds conversation/analysis threads and feedback whenever `POSTGRES_URL` is set, so API replicas are stateless.

See [docs/operations.md](docs/operations.md) for the observability reference, security posture, and disaster recovery (RPO/RTO, PITR restore, snapshots, re-ingest).

## Layout

```
src/copilot/
  config.py        settings (pydantic-settings, .env)
  ingestion/       loaders (PDF/DOCX), section-aware chunker, indexer, CLI
  rag/             embeddings, vector store, rerank retriever, cited QA, chat memory
  agents/          patient model, rules, safety checks, LangGraph analysis workflow
  mcp_server/      MCP tool server, sync client, mock FHIR mapping
  guardrails/      PII pseudonymisation, scope and output checks
  ui/              Streamlit UI + API client
  feedback.py      clinician feedback store
  evaluation/      golden-set loaders, metrics, LLM judge, suites, scorecard
  observability/   tracing (OTel/Langfuse), Prometheus metrics, JSON logs
  api/             FastAPI (/chat, /analyze, /patients, /feedback)
data/
  guidelines/      source documents (downloaded, not committed)
  patients/        synthetic patient records (JSON)
eval/              golden sets, baseline, run_eval.py (scorecard + regression gate)
deploy/            Dockerfile, docker-compose, Kubernetes (Kustomize)
docs/              architecture + plan
tests/
```
