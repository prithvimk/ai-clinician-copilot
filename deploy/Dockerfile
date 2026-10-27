# syntax=docker/dockerfile:1.7
# One image for the API, UI, MCP server and batch jobs (ingest, eval, snapshots); the
# command selects the role. Build from the repo root:
#   docker build -f deploy/Dockerfile -t clinician-copilot .
# BAKE_MODELS=true downloads the embedding and rerank models at build time so pods start
# without internet access (air-gapped / on-prem clusters).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    MODEL_CACHE_DIR=/app/models
WORKDIR /app

# Dependencies first (cached unless the lockfile changes), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ARG BAKE_MODELS=true
RUN if [ "$BAKE_MODELS" = "true" ]; then \
      /app/.venv/bin/python -c "from copilot.rag.embeddings import get_embedder; \
from copilot.rag.retriever import get_reranker; get_embedder(); get_reranker()"; \
    else mkdir -p /app/models; fi


FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --uid 10001 --user-group --create-home --home-dir /home/app app
WORKDIR /app
COPY --from=build --chown=10001:10001 /app /app
COPY --chown=10001:10001 data/patients ./data/patients
COPY --chown=10001:10001 data/guidelines/sources.yaml ./data/guidelines/sources.yaml
COPY --chown=10001:10001 scripts ./scripts
RUN mkdir -p /app/storage /app/data/guidelines && chown -R 10001:10001 /app/storage /app/data

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models \
    QDRANT_PATH=/app/storage/qdrant \
    FEEDBACK_PATH=/app/storage/feedback.jsonl \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER 10001
EXPOSE 8000 8501
CMD ["uvicorn", "copilot.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
