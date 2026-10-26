import json

import anthropic
import httpx2
import pytest

from copilot.rag.llm import ClaudeLLM, LLMRefusal


def make_llm(reply: dict) -> tuple[ClaudeLLM, list[dict]]:
    """ClaudeLLM wired to a mock transport; returns it and the captured request bodies."""
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=reply)

    client = anthropic.Anthropic(
        api_key="test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return ClaudeLLM("claude-opus-5-5", client), sent


def message(content: list[dict], stop_reason: str = "end_turn", **extra) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        **extra,
    }


def test_request_shape_and_text_extraction():
    llm, sent = make_llm(
        message(
            [
                {"type": "thinking", "thinking": "", "signature": "s"},
                {"type": "text", "text": " Hi [1]. "},
            ]
        )
    )
    assert llm.complete("sys", [{"role": "user", "content": "q"}], effort="low") == "Hi [1]."
    body = sent[0]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "sys"
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert "thinking" not in body and "temperature" not in body


def test_refusal_raises():
    llm, _ = make_llm(
        message(
            [],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": None, "explanation": None},
        )
    )
    with pytest.raises(LLMRefusal):
        llm.complete("sys", [{"role": "user", "content": "q"}], effort="low")


def test_parse_sends_schema_and_validates_output():
    from copilot.agents.workflow import Draft

    payload = {"recommendations": [], "monitoring": [], "unsupported_findings": ["x"]}
    llm, sent = make_llm(message([{"type": "text", "text": json.dumps(payload)}]))
    draft = llm.parse("sys", [{"role": "user", "content": "q"}], Draft, effort="high")

    assert draft == Draft.model_validate(payload)
    config = sent[0]["output_config"]
    assert config["effort"] == "high" and config["format"]["type"] == "json_schema"
    assert "recommendations" in config["format"]["schema"]["properties"]
