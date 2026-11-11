"""Data preparation: the Context Library (procedural RAG).

The Context Library holds Semantic Blueprints: reusable instructions that tell the
Writer *how* to produce content (tone, structure, goal), as opposed to the Knowledge
Store, which holds *what* to write about.

Retrieval works on intent:
- We embed each blueprint's `description` (the intent it serves), so the Librarian
  agent can match a request like "explain this simply" to the right blueprint.
- The blueprint itself is stored as a JSON string in metadata under `blueprint_json`,
  which is the key `LibrarianAgent` reads back.

Run (writes to the Pinecone index configured in .env):
    uv run python -m rag_pipline.context_library
"""

import json
import logging
from typing import Any, Protocol, TypedDict

from openai import OpenAI

from services.llm import get_embedding

logger = logging.getLogger(__name__)


class ContextBlueprint(TypedDict):
    id: str
    description: str  # embedded: what the blueprint is for
    blueprint: dict[str, Any]  # stored as metadata: how to write


class UpsertIndex(Protocol):
    """The subset of the Pinecone Index interface used for loading data."""

    def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any: ...


CONTEXT_BLUEPRINTS: list[ContextBlueprint] = [
    {
        "id": "blueprint_technical_explanation",
        "description": (
            "Presents information as a clear, objective technical explanation. Breaks down how "
            "something works, step by step, with precise terms. Use for explaining mechanisms, "
            "processes, research findings or scientific topics."
        ),
        "blueprint": {
            "scene_goal": "Make the mechanism or findings easy to follow without losing accuracy.",
            "style_guide": "Neutral, formal tone. Precise terminology, defined on first use. No embellishment.",
            "structure": ["What it is", "How it works", "Why it matters"],
            "instruction": "Organize the source facts into the structure, following the style_guide.",
        },
    },
    {
        "id": "blueprint_casual_summary",
        "description": (
            "Gives a short, friendly, plain-language summary that anyone can read quickly. "
            "Use for quick overviews, simple explanations and casual recaps."
        ),
        "blueprint": {
            "scene_goal": "Get the main point across in a few sentences.",
            "style_guide": "Conversational and light, like telling a friend. No jargon.",
            "instruction": "Summarize the source facts following the style_guide.",
        },
    },
    {
        "id": "blueprint_diplomatic_context",
        "description": (
            "Presents information in a balanced and neutral manner, considering multiple perspectives. "
            "Use for sensitive topics, negotiations, or situations requiring tact and diplomacy."
        ),
        "blueprint": {
            "scene_goal": "Convey the context accurately while maintaining neutrality and respect.",
            "style_guide": "Polite, measured tone. Avoid taking sides or making assumptions.",
            "instruction": "Present the source facts following the style_guide.",
        }
    },
    {
        "id": "blueprint_AI_Architecture",
        "description": (
            "Provides an overview of the AI architecture aligned with the concerns and priorities of a CTO. "
            "Use for technical briefings, architecture reviews, and strategic planning."
        ),
        "blueprint": {
            "scene_goal": "Explain the AI architecture clearly and concisely for decision-making.",
            "style_guide": "Professional and technical tone. Focus on high-level architecture and key components.",
            "instruction": "Organize the source facts into a coherent architecture overview following the style_guide.",
        }
    },
    {
        "id": "blueprint_CTO_AI_Strategy",
        "description": (
            "Provides an overview of the AI strategy from a CTO perspective. "
            "Use for strategic planning, technology roadmaps, and executive briefings."
        ),
        "blueprint": {
            "scene_goal": "Explain the AI strategy clearly and concisely for decision-making.",
            "style_guide": "Professional and strategic tone. Focus on high-level goals and key initiatives.",
            "instruction": "Organize the source facts into a coherent strategy overview following the style_guide.",
        }
    }
]


def build_context_vectors(
    embedding_client: OpenAI,
    embedding_model: str,
    blueprints: list[ContextBlueprint] = CONTEXT_BLUEPRINTS,
) -> list[dict[str, Any]]:
    """Embed each blueprint's description and attach the blueprint as metadata."""
    vectors: list[dict[str, Any]] = []
    for item in blueprints:
        vectors.append(
            {
                "id": item["id"],
                "values": get_embedding(embedding_client, embedding_model, item["description"]),
                "metadata": {
                    "description": item["description"],
                    "blueprint_json": json.dumps(item["blueprint"]),
                },
            }
        )
    return vectors


def upload_context_library(
    index: UpsertIndex,
    embedding_client: OpenAI,
    embedding_model: str,
    namespace: str,
    blueprints: list[ContextBlueprint] = CONTEXT_BLUEPRINTS,
) -> int:
    """Embed the blueprints and upsert them into the Context Library namespace. Returns the count."""
    vectors = build_context_vectors(embedding_client, embedding_model, blueprints)
    if vectors:
        index.upsert(vectors=vectors, namespace=namespace)
    logger.info("Uploaded %d context blueprints to namespace '%s'.", len(vectors), namespace)
    return len(vectors)


def main() -> None:
    from pinecone import Pinecone

    from config import get_settings

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    embedding_client = OpenAI(api_key=settings.openai_api_key)
    upload_context_library(index, embedding_client, settings.embedding_model, settings.namespace_context)


if __name__ == "__main__":
    main()
