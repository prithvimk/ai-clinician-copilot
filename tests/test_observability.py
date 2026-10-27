import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from prometheus_client import REGISTRY

from copilot.agents.patient import load_patient
from copilot.agents.workflow import ReviewDecision
from copilot.api.app import app
from copilot.config import Settings
from copilot.observability.telemetry import (
    JsonFormatter,
    exporter_config,
    llm_cost,
    request_id_var,
    thread_id_var,
)
from tests.fakes import FakeLLM
from tests.test_agents import SAFE, service_with
from tests.test_llm import make_llm, message

_exporter = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def span_capture():
    if not isinstance(trace.get_tracer_provider(), TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_exporter))
        trace.set_tracer_provider(provider)
    yield


@pytest.fixture
def spans():
    _exporter.clear()
    return _exporter


def metric(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# --- tracing ---------------------------------------------------------------------------


def test_workflow_nodes_are_traced_with_session_and_interrupt(spans):
    service, _ = service_with(FakeLLM(SAFE))
    service.start(load_patient("P001"), "trace-thread")
    service.review("trace-thread", ReviewDecision(decision="approve"))

    by_name = {}
    for s in spans.get_finished_spans():
        by_name.setdefault(s.name, []).append(s)
    for node in ("analyze", "retrieve", "reason", "validate", "human_review", "recommend"):
        assert f"analysis.{node}" in by_name, node
    assert by_name["analysis.reason"][0].attributes["langfuse.session.id"] == "trace-thread"
    paused = by_name["analysis.human_review"][0]  # first run pauses for review
    assert paused.attributes["copilot.interrupted"] is True
    assert paused.status.status_code != StatusCode.ERROR
    assert (
        metric("copilot_workflow_node_seconds_count", {"graph": "analysis", "node": "reason"}) >= 1
    )


def test_failed_node_span_records_error(spans):
    service, _ = service_with(FakeLLM(RuntimeError("provider outage")))
    service.start(load_patient("P001"), "fail-thread")
    [reason] = [s for s in spans.get_finished_spans() if s.name == "analysis.reason"]
    assert reason.status.status_code == StatusCode.ERROR
    assert "provider outage" in reason.status.description


def test_llm_span_carries_usage_and_cost(spans):
    llm, _ = make_llm(
        message(
            [{"type": "text", "text": "ok"}],
            usage={"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 5000},
        )
    )
    before = metric("copilot_llm_tokens_total", {"model": "claude-opus-5-5", "kind": "output"})
    llm.complete("sys", [{"role": "user", "content": "q"}], effort="low")

    [span] = [s for s in spans.get_finished_spans() if s.name == "llm.complete"]
    attrs = span.attributes
    assert attrs["gen_ai.request.model"] == "claude-opus-5-5"
    assert attrs["gen_ai.usage.input_tokens"] == 1000 and attrs["gen_ai.usage.output_tokens"] == 200
    assert attrs["gen_ai.usage.cost"] == pytest.approx((1000 * 4 + 200 * 20 + 5000 * 0.2) / 1e6)
    after = metric("copilot_llm_tokens_total", {"model": "claude-opus-5-5", "kind": "output"})
    assert after - before == 200


def test_llm_cost_unknown_model():
    usage = SimpleNamespace(input_tokens=10, output_tokens=10)
    assert llm_cost("some-other-model", usage) is None


# --- exporters -------------------------------------------------------------------------


def test_exporter_selection():
    assert exporter_config(Settings(_env_file=None)) is None
    langfuse = Settings(
        _env_file=None,
        langfuse_host="https://lf.example/",
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        otel_exporter_otlp_endpoint="http://ignored:4318",
    )
    assert exporter_config(langfuse) == (
        "https://lf.example/api/public/otel/v1/traces",
        {"Authorization": "Basic cGs6c2s="},  # base64("pk:sk")
    )
    jaeger = Settings(_env_file=None, otel_exporter_otlp_endpoint="http://jaeger:4318/")
    assert exporter_config(jaeger) == ("http://jaeger:4318/v1/traces", {})


# --- logs ------------------------------------------------------------------------------


def test_json_log_includes_context_and_trace_ids():
    tracer = trace.get_tracer("test")
    r_token, t_token = request_id_var.set("req-1"), thread_id_var.set("thr-1")
    try:
        with tracer.start_as_current_span("x"):
            record = logging.LogRecord(
                "copilot", logging.INFO, __file__, 1, "hello %s", ("world",), None
            )
            record.fields = {"route": "/chat"}
            entry = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(r_token)
        thread_id_var.reset(t_token)
    assert entry["msg"] == "hello world" and entry["route"] == "/chat"
    assert entry["request_id"] == "req-1" and entry["thread_id"] == "thr-1"
    assert len(entry["trace_id"]) == 32 and len(entry["span_id"]) == 16


# --- API -------------------------------------------------------------------------------


def test_request_id_and_http_metrics():
    client = TestClient(app)
    labels = {"method": "GET", "route": "/patients", "status": "200"}
    before = metric("copilot_http_requests_total", labels)

    r = client.get("/patients", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert client.get("/patients").headers["X-Request-ID"]  # generated when absent
    assert metric("copilot_http_requests_total", labels) - before == 2

    body = client.get("/metrics").text
    assert 'copilot_http_requests_total{method="GET",route="/patients",status="200"}' in body
    assert "copilot_llm_tokens_total" in body


def test_unmatched_routes_do_not_explode_label_cardinality():
    client = TestClient(app)
    client.get("/does-not-exist/123")
    client.get("/does-not-exist/456")
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    assert metric("copilot_http_requests_total", labels) >= 2
