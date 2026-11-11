"""Live goal -> Librarian -> Writer flow.

Scenario: the user wants a *diplomatic* response about tomorrow's weather in Tel Aviv.
1. Goal analysis (Claude) splits the goal into an `intent_query` (style) and a
   `topic_query` (subject matter).
2. The Librarian searches Pinecone with the intent_query only, and we assert it returns
   the diplomatic blueprint exactly as stored in Pinecone.
3. The Writer applies that blueprint to the facts for the topic_query (a fixed forecast:
   the Knowledge Store isn't loaded yet and the model has no live weather) via Claude.

Needs ANTHROPIC_API_KEY, OPENAI_API_KEY and PINECONE_API_KEY in .env, and the Context
Library loaded (uv run python -m rag_pipline.rag_context_librarian).
Run:  uv run pytest -m integration tests/agents   (live logs show each step)
"""

import json
import logging
import re

import pytest
from openai import OpenAI
from pinecone import Pinecone

from agents.librarian_agent import LibrarianAgent
from agents.writer_agent import WriterAgent
from config import Settings
from rag_pipline.context_library import CONTEXT_BLUEPRINTS
from services.llm import call_llm, create_llm_client
from services.mcp import create_mcp_message
from services.vector_store import query_vector_store

pytestmark = pytest.mark.integration

logger = logging.getLogger("test.librarian")

USER_GOAL = "I would like to get a diplomatic response regarding the weather tomorrow in Tel-Aviv"
EXPECTED_BLUEPRINT_ID = "blueprint_diplomatic_context"

ANALYSIS_SYSTEM_PROMPT = """You are an expert goal analyst. Analyze the user's high-level goal and extract two components:
    1. 'intent_query': A descriptive phrase summarizing the desired style, tone, or format, optimized for searching a context library (e.g., "suspenseful narrative blueprint", "objective technical explanation structure").
    2. 'topic_query': A concise phrase summarizing the factual subject matter required (e.g., "Juno mission objectives and power", "Apollo 11 landing details")."""

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "intent_query": {"type": "string"},
        "topic_query": {"type": "string"},
    },
    "required": ["intent_query", "topic_query"],
    "additionalProperties": False,
}

# Stand-in for the Researcher until the Knowledge Store is loaded.
WEATHER_FACTS = (
    "Forecast for Tel Aviv, tomorrow: hot and humid, high of 31°C and low of 24°C. "
    "40% chance of a short thunderstorm in the afternoon. Winds 15-25 km/h from the west. "
    "Some residents welcome the rain after a dry month; beach-goers and outdoor event "
    "organizers are disappointed."
)


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = Settings()  # reads .env
    if not (s.anthropic_api_key and s.openai_api_key and s.pinecone_api_key):
        pytest.skip("ANTHROPIC_API_KEY, OPENAI_API_KEY and PINECONE_API_KEY must be set in .env")
    logger.info(
        "Setup: index=%s namespace=%s embedding=%s generation=%s",
        s.pinecone_index_name, s.namespace_context, s.embedding_model, s.generation_model,
    )
    return s


@pytest.fixture(scope="module")
def llm_client(settings: Settings):
    return create_llm_client(settings)


@pytest.fixture(scope="module")
def embedding_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@pytest.fixture(scope="module")
def index(settings: Settings):
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.pinecone_index_name):
        pytest.fail("Context Library index missing; run: uv run python -m rag_pipline.rag_context_librarian")
    return pc.Index(settings.pinecone_index_name)


@pytest.fixture(scope="module")
def goal_analysis(llm_client, settings: Settings) -> dict[str, str]:
    logger.info("Step 1: analyze the goal with %s -> %r", settings.generation_model, USER_GOAL)
    reply = call_llm(
        llm_client, settings.generation_model, ANALYSIS_SYSTEM_PROMPT, USER_GOAL, output_schema=ANALYSIS_SCHEMA
    )
    analysis = json.loads(reply)
    logger.info("  intent_query (style): %r", analysis["intent_query"])
    logger.info("  topic_query  (facts): %r", analysis["topic_query"])
    return analysis


@pytest.fixture(scope="module")
def librarian_reply(goal_analysis, index, embedding_client: OpenAI, settings: Settings):
    intent_query = goal_analysis["intent_query"]

    # Show how every candidate scored, so it's visible why the Librarian picks what it picks.
    logger.info("Step 2: score every blueprint against the intent_query")
    candidates = query_vector_store(
        index=index, client=embedding_client, embedding_model=settings.embedding_model,
        query_text=intent_query, namespace=settings.namespace_context, top_k=len(CONTEXT_BLUEPRINTS),
    )
    for rank, match in enumerate(candidates, start=1):
        logger.info("  candidate #%d  %-34s score=%.3f", rank, match["id"], match["score"])

    logger.info("Step 3: Librarian retrieves the best blueprint from Pinecone")
    librarian = LibrarianAgent(embedding_client, index, settings.embedding_model, settings.namespace_context)
    reply = librarian.run(create_mcp_message("Engine", {"intent_query": intent_query}))
    logger.info("  Librarian returned: %s", reply["content"]["blueprint_json"])
    return reply


def test_goal_analysis_splits_style_and_topic(goal_analysis):
    intent, topic = goal_analysis["intent_query"], goal_analysis["topic_query"]

    assert re.search(r"diplomat", intent, re.IGNORECASE), "style should carry the 'diplomatic' intent"
    assert not re.search(r"tel[\s-]?aviv", intent, re.IGNORECASE), "style should not contain the topic"
    assert re.search(r"tel[\s-]?aviv", topic, re.IGNORECASE)
    assert re.search(r"weather|forecast", topic, re.IGNORECASE)
    logger.info("  OK: style and topic are separated")


def test_librarian_fetches_diplomatic_blueprint_from_pinecone(librarian_reply, index, settings: Settings):
    logger.info("Step 4: fetch '%s' directly from Pinecone to compare", EXPECTED_BLUEPRINT_ID)
    stored = index.fetch(ids=[EXPECTED_BLUEPRINT_ID], namespace=settings.namespace_context).vectors
    assert EXPECTED_BLUEPRINT_ID in stored, "diplomatic blueprint is not in Pinecone"

    returned_json = librarian_reply["content"]["blueprint_json"]

    assert librarian_reply["sender"] == "Librarian"
    # What the Librarian returned is exactly what Pinecone holds for the diplomatic blueprint...
    assert returned_json == stored[EXPECTED_BLUEPRINT_ID].metadata["blueprint_json"]
    # ...which is the blueprint defined in the Context Library.
    expected = next(b["blueprint"] for b in CONTEXT_BLUEPRINTS if b["id"] == EXPECTED_BLUEPRINT_ID)
    assert json.loads(returned_json) == expected
    logger.info("  OK: Librarian output == Pinecone metadata == Context Library definition")


def test_writer_generates_diplomatic_weather_response(goal_analysis, librarian_reply, llm_client, settings: Settings):
    facts = f"Topic: {goal_analysis['topic_query']}\n\n{WEATHER_FACTS}"
    logger.info("Step 5: Writer applies the blueprint to the topic facts with %s", settings.generation_model)
    logger.info("  facts:\n%s", facts)
    writer = WriterAgent(llm_client, settings.generation_model)

    reply = writer.run(create_mcp_message("Engine", {"blueprint": librarian_reply["content"], "facts": facts}))
    text = reply["content"]
    logger.info("  Writer response (%d chars):\n%s", len(text), text)

    assert reply["sender"] == "Writer"
    assert isinstance(text, str) and len(text) > 50
    assert re.search(r"tel[\s-]?aviv", text, re.IGNORECASE)
    assert "31" in text  # grounded in the supplied forecast
    logger.info("  OK: response mentions Tel Aviv and the 31°C high")
