"""Tracing, metrics and structured logs.

Tracing  OpenTelemetry spans exported over OTLP/HTTP: to Langfuse (LANGFUSE_* settings, which
         build its OTLP endpoint and auth header) or any collector (OTEL_EXPORTER_OTLP_ENDPOINT,
         e.g. Jaeger). Workflow nodes, retrieval, QA and LLM calls each get a span; LLM spans
         carry gen_ai.* token usage and estimated cost; thread_id is the session id.
Metrics  Prometheus counters/histograms served at GET /metrics.
Logs     one JSON object per line with request_id, thread_id and the current trace/span id.

Everything is a no-op until configured, so tests and local runs need no collector.
"""

import base64
import contextvars
import json
import logging
import sys
import time
from collections.abc import Callable

from langgraph.errors import GraphInterrupt
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram

from copilot.config import Settings

tracer = trace.get_tracer("copilot")

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
thread_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "thread_id", default=None
)

# --- metrics ---------------------------------------------------------------------------

HTTP_REQUESTS = Counter(
    "copilot_http_requests_total", "HTTP requests", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "copilot_http_request_seconds",
    "HTTP request latency",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
NODE_LATENCY = Histogram(
    "copilot_workflow_node_seconds",
    "LangGraph node latency",
    ["graph", "node"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
LLM_LATENCY = Histogram(
    "copilot_llm_request_seconds",
    "LLM request latency",
    ["model", "operation"],
    buckets=(0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 300),
)
LLM_TOKENS = Counter("copilot_llm_tokens_total", "LLM tokens", ["model", "kind"])
LLM_COST = Counter("copilot_llm_cost_usd_total", "Estimated LLM cost (USD)", ["model"])
ANSWERS = Counter("copilot_answers_total", "Q&A outcomes", ["outcome"])
GUARDRAIL_FLAGS = Counter("copilot_guardrail_flags_total", "Guardrail interventions", ["flag"])

# USD per million tokens: input, output, cache read, cache write (5-minute TTL).
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
}


def llm_cost(model: str, usage) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    tokens = (
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_read_input_tokens", None) or 0,
        getattr(usage, "cache_creation_input_tokens", None) or 0,
    )
    return sum(t * p for t, p in zip(tokens, price, strict=True)) / 1_000_000


def record_llm_call(span, model: str, operation: str, response, seconds: float) -> None:
    """Token usage and cost on the span and in metrics. `response.usage` covers the attempt
    that produced the message (a server-side fallback's declined attempt is not included)."""
    usage = response.usage
    served_by = getattr(response, "model", None) or model
    kinds = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_read": getattr(usage, "cache_read_input_tokens", None) or 0,
        "cache_write": getattr(usage, "cache_creation_input_tokens", None) or 0,
    }
    for kind, n in kinds.items():
        LLM_TOKENS.labels(served_by, kind).inc(n)
    LLM_LATENCY.labels(served_by, operation).observe(seconds)
    cost = llm_cost(served_by, usage)
    attributes = {
        "gen_ai.system": "anthropic",
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "gen_ai.response.model": served_by,
        "gen_ai.response.finish_reasons": [str(response.stop_reason)],
        "gen_ai.usage.input_tokens": kinds["input"],
        "gen_ai.usage.output_tokens": kinds["output"],
        "gen_ai.usage.cache_read_input_tokens": kinds["cache_read"],
        "gen_ai.usage.cache_creation_input_tokens": kinds["cache_write"],
    }
    if cost is not None:
        LLM_COST.labels(served_by).inc(cost)
        attributes["gen_ai.usage.cost"] = round(cost, 6)
    span.set_attributes(attributes)


def traced_node(graph: str, name: str, fn: Callable) -> Callable:
    """Wrap a LangGraph node: span + latency metric; thread_id becomes the session id.

    A human-review interrupt is control flow, not an error, so it is marked, not recorded.
    """

    # No functools.wraps: LangGraph reads the signature to decide whether to pass `config`.
    def node(state, config):
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        token = thread_id_var.set(thread_id)
        start = time.perf_counter()
        with tracer.start_as_current_span(
            f"{graph}.{name}", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attributes(
                {
                    "copilot.graph": graph,
                    "copilot.node": name,
                    "langfuse.session.id": thread_id or "",
                }
            )
            try:
                return fn(state)
            except GraphInterrupt:
                span.set_attribute("copilot.interrupted", True)
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"{type(e).__name__}: {e}"))
                raise
            finally:
                NODE_LATENCY.labels(graph, name).observe(time.perf_counter() - start)
                thread_id_var.reset(token)

    node.__name__ = f"{graph}_{name}"
    return node


# --- setup -----------------------------------------------------------------------------

_configured = False


def exporter_config(settings: Settings) -> tuple[str, dict[str, str]] | None:
    """(OTLP/HTTP traces endpoint, headers): Langfuse if its keys are set, else a generic
    collector if OTEL_EXPORTER_OTLP_ENDPOINT is set, else None (tracing off)."""
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()
        endpoint = f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
        return endpoint, {"Authorization": f"Basic {token}"}
    if settings.otel_exporter_otlp_endpoint:
        return f"{settings.otel_exporter_otlp_endpoint.rstrip('/')}/v1/traces", {}
    return None


def setup_tracing(settings: Settings) -> bool:
    """Install the exporter if one is configured; returns whether spans are exported."""
    global _configured
    if _configured:
        return True
    config = exporter_config(settings)
    if config is None:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    endpoint, headers = config
    provider = TracerProvider(resource=Resource.create({"service.name": settings.service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    _configured = True
    return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if rid := request_id_var.get():
            entry["request_id"] = rid
        if tid := thread_id_var.get():
            entry["thread_id"] = tid
        ctx = trace.get_current_span().get_span_context()
        if ctx.is_valid:
            entry["trace_id"] = format(ctx.trace_id, "032x")
            entry["span_id"] = format(ctx.span_id, "016x")
        entry |= getattr(record, "fields", {})
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if settings.log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
