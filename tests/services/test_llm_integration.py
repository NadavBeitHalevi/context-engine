"""Live integration tests for the Claude client, using ANTHROPIC_API_KEY from .env.

These make real (billed, very small) API calls, so they are excluded from the default run.
Run them explicitly with:  uv run pytest -m integration
"""

import json

import pytest

from config import Settings
from services.llm import call_llm, create_llm_client

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    settings = Settings()  # reads .env
    if not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY is not set in .env")
    return settings

@pytest.mark.integration
def test_client_authenticates_and_model_exists(live_settings: Settings):
    client = create_llm_client(live_settings)

    model = client.models.retrieve(live_settings.generation_model)

    assert model.id == live_settings.generation_model


@pytest.mark.integration
def test_call_llm_returns_text(live_settings: Settings):
    client = create_llm_client(live_settings)

    reply = call_llm(
        client,
        live_settings.generation_model,
        system_prompt="You are a terse assistant.",
        user_prompt="Reply with exactly the word: pong",
    )

    assert "pong" in reply.lower()


@pytest.mark.integration
def test_call_llm_structured_output(live_settings: Settings):
    client = create_llm_client(live_settings)
    schema = {
        "type": "object",
        "properties": {"capital": {"type": "string"}},
        "required": ["capital"],
        "additionalProperties": False,
    }

    reply = call_llm(
        client,
        live_settings.generation_model,
        system_prompt="Answer factual questions.",
        user_prompt="What is the capital of France?",
        output_schema=schema,
    )

    assert json.loads(reply)["capital"].lower() == "paris"
