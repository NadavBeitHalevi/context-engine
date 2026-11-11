"""Unit tests for services.llm.call_llm, using a stub client (no network)."""

from types import SimpleNamespace
from typing import Any

import pytest

from services.llm import FALLBACK_BETA, LLMRefusalError, call_llm


def _block(type_: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type=type_, **fields)


def _stub_client(response: SimpleNamespace) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return response

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    return client, calls


def _response(content: list[SimpleNamespace], stop_reason: str = "end_turn", stop_details: Any = None):
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=stop_details)


def test_returns_text_and_skips_thinking_blocks():
    client, calls = _stub_client(
        _response([_block("thinking", thinking=""), _block("text", text="  Hello "), _block("text", text="world")])
    )

    assert call_llm(client, "claude-opus-5-5", "sys", "user") == "Hello world"

    sent = calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "user"}]
    assert sent["betas"] == [FALLBACK_BETA]
    assert sent["fallbacks"] == "default"
    assert "output_config" not in sent


def test_passes_output_schema_as_structured_output():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"], "additionalProperties": False}
    client, calls = _stub_client(_response([_block("text", text='{"x": "y"}')]))

    assert call_llm(client, "m", "sys", "user", output_schema=schema) == '{"x": "y"}'
    assert calls[0]["output_config"] == {"format": {"type": "json_schema", "schema": schema}}


def test_raises_on_refusal():
    client, _ = _stub_client(
        _response([], stop_reason="refusal", stop_details=SimpleNamespace(category="cyber"))
    )

    with pytest.raises(LLMRefusalError, match="cyber"):
        call_llm(client, "m", "sys", "user")


def test_raises_on_empty_text():
    client, _ = _stub_client(_response([_block("thinking", thinking="")]))

    with pytest.raises(ValueError, match="empty"):
        call_llm(client, "m", "sys", "user")
