"""Thin LLM interface so QA and agents can be tested with a scripted fake."""

import time
from functools import lru_cache
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

from copilot.config import get_settings
from copilot.observability.telemetry import record_llm_call, tracer

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)


class LLMRefusal(Exception):
    """The model (and its server-side fallback) declined the request."""


class LLMOutputError(Exception):
    """The response could not be parsed into the requested schema."""


class LLM(Protocol):
    def complete(
        self, system: str, messages: list[dict], effort: str, max_tokens: int = 16000
    ) -> str: ...

    def parse(
        self,
        system: str,
        messages: list[dict],
        schema: type[T],
        effort: str,
        max_tokens: int = 16000,
    ) -> T: ...


class ClaudeLLM:
    def __init__(self, model: str, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def complete(
        self, system: str, messages: list[dict], effort: str, max_tokens: int = 16000
    ) -> str:
        with tracer.start_as_current_span("llm.complete") as span:
            start = time.perf_counter()
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_config={"effort": effort},
                # Safety-classifier declines are retried server-side on a recommended model.
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            record_llm_call(span, self.model, "complete", response, time.perf_counter() - start)
        _raise_on_refusal(response)
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def parse(
        self,
        system: str,
        messages: list[dict],
        schema: type[T],
        effort: str,
        max_tokens: int = 16000,
    ) -> T:
        """Structured output validated against `schema` (JSON-schema constrained decoding)."""
        with tracer.start_as_current_span("llm.parse") as span:
            start = time.perf_counter()
            response = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_format=schema,
                output_config={"effort": effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
            span.set_attribute("copilot.schema", schema.__name__)
            record_llm_call(span, self.model, "parse", response, time.perf_counter() - start)
        _raise_on_refusal(response)
        if response.parsed_output is None:
            raise LLMOutputError(f"no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output


def _raise_on_refusal(response) -> None:
    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMRefusal(f"model declined (category={category})")


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key or None, timeout=settings.llm_timeout_s
    )
    llm = ClaudeLLM(settings.llm_model, client)
    if not settings.pii_redaction:
        return llm
    from copilot.guardrails.pii import RedactingLLM, get_detector

    return RedactingLLM(llm, get_detector())
