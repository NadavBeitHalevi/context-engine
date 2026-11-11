"""Shared pytest fixtures.

Unit tests must never hit real external services (Anthropic / OpenAI / Pinecone).
Settings are overridden with dummy values so no real keys are needed.
"""

import pytest
from fastapi.testclient import TestClient

from app import app
from config import Settings, get_settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        anthropic_api_key="test-anthropic-key",
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
    )


@pytest.fixture
def client(test_settings: Settings, monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    monkeypatch.setattr("controllers.health_controller.get_settings", lambda: test_settings)
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
