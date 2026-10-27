# Operations: observability, deployment and disaster recovery

## Components

| Component | Runs as | State | Scaling |
|---|---|---|---|
| API (FastAPI + LangGraph + in-process MCP tools) | Deployment `api` | none (threads in Postgres, index in Qdrant) | HPA 2–10 (prod 3–20) on CPU/memory |
| UI (Streamlit) | Deployment `ui` | per-session in memory | 2 replicas, cookie affinity at the ingress |
| Vector store | StatefulSet `qdrant` | guideline index (derived) | vertical; distributed Qdrant / managed for HA |
| Postgres | CloudNativePG `copilot-db` (3 instances) | LangGraph checkpoints (chat + analysis threads), clinician feedback | primary + 2 replicas, automatic failover |
| Ingest | Job `ingest-guidelines` | — | on demand / on corpus change |
| Snapshots | CronJob `qdrant-snapshot` | Qdrant snapshots volume | daily |

One image (`deploy/Dockerfile`) serves every role. Embedding and rerank models are baked in at build time (`BAKE_MODELS=true`), so pods need no outbound access except the Anthropic API.

## Observability

**Traces** — OpenTelemetry spans over OTLP/HTTP. Set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`) to send to Langfuse, or `OTEL_EXPORTER_OTLP_ENDPOINT` for any collector (Jaeger in docker-compose, an OTel Collector in Kubernetes). Nothing is exported when neither is set.

| Span | Attributes |
|---|---|
| `POST /chat`, `POST /analyze`, … | FastAPI auto-instrumentation (route, status, duration) |
| `chat.rewrite`, `chat.answer`, `analysis.{analyze,retrieve,reason,validate,human_review,recommend}` | `langfuse.session.id` = thread id; `copilot.interrupted` when pausing for review; error status + exception on failure |
| `retrieval.search` | candidates, returned, top relevance, doc ids |
| `qa.answer` | outcome (grounded / refusal reason), citation counts, guardrail flags |
| `llm.complete`, `llm.parse` | `gen_ai.request.model`, `gen_ai.response.model`, input/output/cache tokens, `gen_ai.usage.cost` (USD estimate) |

Prompts and patient data are **not** recorded as span attributes; only counts, ids and scores.

**Metrics** — Prometheus at `GET /metrics` (scraped by the `ServiceMonitor`):
`copilot_http_requests_total{method,route,status}`, `copilot_http_request_seconds`, `copilot_workflow_node_seconds{graph,node}`, `copilot_llm_request_seconds{model,operation}`, `copilot_llm_tokens_total{model,kind}`, `copilot_llm_cost_usd_total{model}`, `copilot_answers_total{outcome}`, `copilot_guardrail_flags_total{flag}`. Routes are labelled by template, so ids never create new series.

**Alerts** (`PrometheusRule`): 5xx rate > 5% (page), p95 `/analyze` > 2 min, refusal rate > 50% (index or retrieval broken), estimated LLM spend > $50/h.

**Logs** — one JSON object per line on stdout: `ts, level, logger, msg, request_id, thread_id, trace_id, span_id` plus fields (the access log adds method, route, status, duration_ms). `X-Request-ID` is accepted from the caller or generated, and returned on every response.

## Deploying

```bash
# local / single host
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml --profile tools run --rm ingest

# Kubernetes (operators listed in deploy/k8s/base/kustomization.yaml must be installed)
kubectl create namespace clinician-copilot
kubectl -n clinician-copilot create secret generic copilot-secrets \
  --from-literal=ANTHROPIC_API_KEY=... --from-literal=LANGFUSE_PUBLIC_KEY= --from-literal=LANGFUSE_SECRET_KEY=
kubectl -n clinician-copilot create secret generic backup-s3 \
  --from-literal=ACCESS_KEY_ID=... --from-literal=SECRET_ACCESS_KEY=...
kubectl apply -k deploy/k8s/overlays/prod
kubectl -n clinician-copilot wait --for=condition=complete job/ingest-guidelines --timeout=30m
```

Rollback: `kubectl -n clinician-copilot rollout undo deployment/api` (and `ui`). The API's readiness probe (`/ready`) only passes once models are loaded and Qdrant answers, and `maxUnavailable: 0` keeps capacity during rollouts.

## Security posture

- Namespace enforces the Kubernetes *restricted* Pod Security Standard; every container runs as non-root with a read-only root filesystem, no privilege escalation, all capabilities dropped and the RuntimeDefault seccomp profile. docker-compose mirrors the read-only root.
- `NetworkPolicy` default-deny, then: ingress → UI → API → {Qdrant, Postgres, OTLP collector}; API egress to the internet on 443 only (Anthropic, Langfuse); Qdrant has no egress.
- Only the UI is exposed (TLS via cert-manager); the API is cluster-internal.
- Secrets are created out of band (External Secrets / sealed secrets), never committed.
- Identifiers are pseudonymised before any LLM call (see README, Guardrails).

## Disaster recovery

| Data | Protection | RPO | RTO | Restore |
|---|---|---|---|---|
| Postgres: conversation and analysis threads, feedback | CloudNativePG streaming replicas (HA) + continuous WAL archiving and daily base backups to S3, 30-day retention | ≈ 5 min (WAL archive interval); 0 for single-node failure (replica promotion) | failover: < 1 min; PITR restore: ~30–60 min depending on size | new `Cluster` with `bootstrap.recovery` (below) |
| Qdrant index | daily snapshot (7 kept) + always rebuildable from the corpus | 24 h for snapshots; none for rebuild (deterministic) | snapshot: minutes; re-ingest: ingest time (~minutes for the current corpus) | `scripts/qdrant_snapshot.py --restore <name>` or rerun the ingest Job |
| Corpus | `data/guidelines/sources.yaml` in git (publisher, URL/handle, doc ids); content hashes make re-ingest idempotent | — | — | ingest Job downloads and indexes; mirror the PDFs to versioned object storage, since publisher URLs change |
| Config, manifests | git (GitOps) | — | redeploy | `kubectl apply -k` |

Point-in-time restore of Postgres into a new cluster (then point the app secret at it):

```yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata: {name: copilot-db-restore}
spec:
  instances: 3
  storage: {size: 100Gi}
  bootstrap:
    recovery:
      source: copilot-db
      recoveryTarget: {targetTime: "2026-10-27 08:00:00+00"}
  externalClusters:
    - name: copilot-db
      barmanObjectStore:
        destinationPath: s3://clinician-copilot-backups/postgres
        s3Credentials:
          accessKeyId: {name: backup-s3, key: ACCESS_KEY_ID}
          secretAccessKey: {name: backup-s3, key: SECRET_ACCESS_KEY}
```

Qdrant snapshots live on the `snapshots` volume of the StatefulSet; for off-site copies, sync that volume to object storage (or configure Qdrant's S3 snapshot storage). Because the index is derived, the ingest Job is the recovery path of last resort and needs nothing but the corpus manifest.

**Drills:** restore Postgres to a scratch cluster and re-ingest into a scratch Qdrant quarterly; run `eval/run_eval.py --suites agent,retrieval --gate eval/baseline.json` against the restored stack to confirm behaviour.
