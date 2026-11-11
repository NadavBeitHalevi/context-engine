"""LLM interactions: Claude (Anthropic) for generation, OpenAI for embeddings."""

import logging
from typing import Any

import anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config import Settings

logger = logging.getLogger(__name__)

MAX_TOKENS = 16000
# Server-side refusal fallback: if Claude declines, the API re-runs the request on
# Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_with_retries = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


class LLMRefusalError(ValueError):
    """Raised when Claude (and its fallback) declines the request."""


def create_llm_client(settings: Settings) -> anthropic.Anthropic:
    """Build the Claude client. An empty key falls back to the SDK's env/profile lookup."""
    # The SDK retries 408/409/429/5xx and connection errors with exponential backoff.
    return anthropic.Anthropic(api_key=settings.anthropic_api_key or None, max_retries=5)


def call_llm(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_prompt: str,
    output_schema: dict[str, Any] | None = None,
) -> str:
    """Send a system + user prompt to Claude and return the text reply.

    If output_schema (a JSON Schema) is given, the reply is guaranteed to be JSON matching it.
    """
    logger.debug("Calling LLM model=%s structured=%s", model, output_schema is not None)
    extra: dict[str, Any] = {}
    if output_schema is not None:
        extra["output_config"] = {"format": {"type": "json_schema", "schema": output_schema}}

    response = client.beta.messages.create( # type: ignore
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        betas=[FALLBACK_BETA],
        fallbacks="default",
        **extra,
    )

    if response.stop_reason == "refusal": # type: ignore
        category = response.stop_details.category if response.stop_details else None # type: ignore
        raise LLMRefusalError(f"LLM declined the request (category: {category}).")
    if response.stop_reason == "max_tokens": # type: ignore
        logger.warning("LLM reply hit max_tokens=%d and may be truncated.", MAX_TOKENS)

    # Thinking blocks (Claude Opus 5.5 always thinks) are skipped; only the answer text is returned.
    text = "".join(block.text for block in response.content if block.type == "text").strip() # type: ignore
    if not text:
        raise ValueError("LLM returned an empty response.")
    return text


@_with_retries
def get_embedding(client: OpenAI, model: str, text: str) -> list[float]:
    """Embed a single text (OpenAI; Anthropic has no embeddings endpoint)."""
    response = client.embeddings.create(input=[text.replace("\n", " ")], model=model)
    return response.data[0].embedding
