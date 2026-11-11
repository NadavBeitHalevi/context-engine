"""Live checks that the Context Library pipeline created what the Librarian needs.

Reads (never writes) the Pinecone index configured in .env. Run the pipeline first:
    uv run python -m rag_pipeline.rag_context_librarian
then:
    uv run pytest -m integration tests/rag_pipeline
"""

import json
import logging

import pytest
from openai import OpenAI
from pinecone import Pinecone

from agents.librarian_agent import LibrarianAgent
from config import Settings
from rag_pipeline.context_library import CONTEXT_BLUEPRINTS
from rag_pipeline.rag_context_librarian import embedding_dimension, namespace_count
from services.mcp import create_mcp_message
from services.vector_store import query_vector_store

pytestmark = pytest.mark.integration

logger = logging.getLogger("test.rag_pipeline")

BLUEPRINTS_BY_ID = {b["id"]: b for b in CONTEXT_BLUEPRINTS}


@pytest.fixture(scope="module")
def settings() -> Settings:
    s = Settings()  # reads .env
    if not (s.pinecone_api_key and s.openai_api_key):
        pytest.skip("PINECONE_API_KEY and OPENAI_API_KEY must be set in .env")
    return s


@pytest.fixture(scope="module")
def pc(settings: Settings) -> Pinecone:
    client = Pinecone(api_key=settings.pinecone_api_key)
    if not client.has_index(settings.pinecone_index_name):
        pytest.fail(
            f"Index '{settings.pinecone_index_name}' does not exist; "
            "run: uv run python -m rag_pipeline.rag_context_librarian"
        )
    return client


@pytest.fixture(scope="module")
def index(pc: Pinecone, settings: Settings):
    return pc.Index(settings.pinecone_index_name)


@pytest.fixture(scope="module")
def embedding_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def test_index_matches_embedding_model(pc: Pinecone, settings: Settings):
    description = pc.describe_index(settings.pinecone_index_name)
    logger.info(
        "Index '%s': ready=%s dimension=%s metric=%s",
        settings.pinecone_index_name, description.status.ready, description.dimension, description.metric,
    )

    assert description.status.ready
    assert description.dimension == embedding_dimension(settings.embedding_model)
    assert description.metric == "cosine"


def test_namespace_holds_exactly_the_defined_blueprints(index, settings: Settings):
    count = namespace_count(index, settings.namespace_context)
    logger.info("Namespace '%s' holds %d vectors; %d blueprints defined", settings.namespace_context, count, len(CONTEXT_BLUEPRINTS))

    assert count == len(CONTEXT_BLUEPRINTS)


def test_each_blueprint_is_stored_with_its_metadata(index, settings: Settings):
    fetched = index.fetch(ids=list(BLUEPRINTS_BY_ID), namespace=settings.namespace_context).vectors

    logger.info("Fetched from Pinecone: %s", sorted(fetched))

    assert set(fetched) == set(BLUEPRINTS_BY_ID)
    for blueprint_id, vector in fetched.items():
        logger.info("  checking metadata of %s", blueprint_id)
        expected = BLUEPRINTS_BY_ID[blueprint_id]
        assert vector.metadata["description"] == expected["description"]
        assert json.loads(vector.metadata["blueprint_json"]) == expected["blueprint"]


@pytest.mark.parametrize(
    "intent, expected_id",
    [
        ("Give me a quick, friendly recap in plain words", "blueprint_casual_summary"),
        ("Explain step by step how this mechanism works, with precise terms", "blueprint_technical_explanation"),
        ("Present this sensitive negotiation tactfully, without taking sides", "blueprint_diplomatic_context"),
        ("Brief the CTO on our AI roadmap and strategic initiatives", "blueprint_CTO_AI_Strategy"),
        ("Present our long-term technology vision to the board", "blueprint_CTO_AI_Strategy"),
        ("Design the system architecture and write ADRs that weigh the trade-offs", "blueprint_AI_Architecture"),
        ("Identify the architecture characteristics and the building blocks of the system", "blueprint_AI_Architecture"),
    ],
)
def test_intent_retrieves_the_right_blueprint(index, embedding_client, settings: Settings, intent, expected_id):
    matches = query_vector_store(
        index=index,
        client=embedding_client,
        embedding_model=settings.embedding_model,
        query_text=intent,
        namespace=settings.namespace_context,
        top_k=1,
    )
    top = matches[0] if matches else None
    logger.info(
        "Intent %r -> %s (score %.3f); expected %s",
        intent, top["id"] if top else None, top["score"] if top else 0.0, expected_id,
    )

    assert matches and matches[0]["id"] == expected_id


def test_librarian_agent_returns_the_stored_blueprint(index, embedding_client, settings: Settings):
    librarian = LibrarianAgent(embedding_client, index, settings.embedding_model, settings.namespace_context)

    logger.info("Librarian request: 'a short, casual summary for a friend'")
    reply = librarian.run(create_mcp_message("Test", {"intent_query": "a short, casual summary for a friend"}))
    logger.info("  Librarian returned: %s", reply["content"]["blueprint_json"])

    assert reply["sender"] == "Librarian"
    assert json.loads(reply["content"]["blueprint_json"]) == BLUEPRINTS_BY_ID["blueprint_casual_summary"]["blueprint"]
