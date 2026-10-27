# Production deployment architecture

```
                         Internet
                            │ TLS (cert-manager)
                      ┌─────▼─────┐
                      │  Ingress  │   only exposed surface
                      └─────┬─────┘
          ┌─────────────────▼──────────────────┐   namespace: clinician-copilot
          │  UI  (Streamlit) ×2, cookie affinity│   restricted PSS, default-deny NetworkPolicy
          └─────────────────┬──────────────────┘
                            │ cluster-internal
          ┌─────────────────▼──────────────────┐
          │  API (FastAPI + LangGraph + MCP)    │   HPA 3–20, PDB, maxUnavailable 0
          └──┬───────────┬──────────────┬──────┘
             │           │              │ egress 443 only
   ┌─────────▼───┐ ┌─────▼───────────┐ ┌▼──────────────────────┐
   │ Qdrant      │ │ Postgres (CNPG) │ │ Anthropic API         │
   │ StatefulSet │ │ 1 primary +     │ │ Langfuse / OTLP       │
   │ + snapshots │ │ 2 replicas, WAL │ └───────────────────────┘
   └──────┬──────┘ └────────┬────────┘
          │ daily snapshot   │ WAL + base backups
          └────────► S3 (versioned) ◄────────┘

   Jobs: ingest-guidelines (on demand) · qdrant-snapshot (daily)
   Telemetry: /metrics → Prometheus (ServiceMonitor + alerts) · spans → OTel collector → Langfuse
```

## Justification

| Decision | Reason |
|---|---|
| Stateless API, state in Postgres/Qdrant | Horizontal scaling and rolling deploys without session loss; any replica resumes any thread |
| Only the UI is exposed | Shrinks the attack surface for a system touching clinical data; the API is reachable only from the UI |
| Default-deny NetworkPolicies | Qdrant has no egress; API egress is limited to 443 for the LLM and tracing |
| Models baked into the image | Pods start without downloading from the internet; deterministic, auditable builds |
| CloudNativePG with PITR | RPO ≈ 5 min, failover under 1 min, recovery to a point before a bad write |
| Rebuildable Qdrant | The index is derived from the versioned corpus manifest, so loss is a recovery-time problem, not a data-loss problem |
| HPA on CPU/memory plus LLM-spend alert | Throughput is bounded by LLM latency; the spend alert catches runaway retries |
| Readiness gated on models + Qdrant | Traffic never reaches a pod that would refuse every question |

Operational detail (alerts, backups, restore commands, drills) is in `operations.md`.
