"""Thin LLM interface so QA and agents can be tested with a scripted fake."""

from functools import lru_cache
from typing import Protocol

import anthropic

from copilot.config import get_settings


class LLMRefusal(Exception):
    """The model (and its server-side fallback) declined the request."""


class LLM(Protocol):
    def complete(
        self, system: str, messages: list[dict], effort: str, max_tokens: int = 16000
    ) -> str: ...


class ClaudeLLM:
    def __init__(self, model: str, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def complete(
        self, system: str, messages: list[dict], effort: str, max_tokens: int = 16000
    ) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": effort},
            # Safety-classifier declines are retried server-side on a recommended model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise LLMRefusal(f"model declined (category={category})")
        return "".join(b.text for b in response.content if b.type == "text").strip()


@lru_cache
def get_llm() -> LLM:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    return ClaudeLLM(settings.llm_model, client)
