"""Data preparation: the Context Library (procedural RAG).

The Context Library holds Semantic Blueprints: reusable instructions that tell the
Writer *how* to produce content (tone, structure, goal), as opposed to the Knowledge
Store, which holds *what* to write about.

Retrieval works on intent:
- We embed each blueprint's `description` (the intent it serves), so the Librarian
  agent can match a request like "explain this simply" to the right blueprint.
- The blueprint itself is stored as a JSON string in metadata under `blueprint_json`,
  which is the key `LibrarianAgent` reads back.

Load it into Pinecone with the pipeline runner (creates/checks the index first):
    uv run python -m rag_pipeline.rag_context_librarian
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
        # The software architect: turns requirements into a design the team can build.
        "id": "blueprint_AI_Architecture",
        "description": (
            "Designs a software system's architecture. Identifies the architecture characteristics the "
            "project needs (scalability, availability, performance, maintainability), compares architecture "
            "styles and their trade-offs, records decisions as Architecture Decision Records (ADRs), and "
            "defines the building blocks: components, their responsibilities and how they interact. Use for "
            "system design, architecture characterization, ADRs, trade-off analysis and component design."
        ),
        "blueprint": {
            "scene_goal": "Produce a reasoned architecture design the team can build from.",
            "style_guide": (
                "Pragmatic, analytical architect's voice. Every decision names the options considered and "
                "their trade-offs; nothing is presented as free. Precise, no buzzwords."
            ),
            "structure": [
                "Architecture characteristics (prioritized, with the requirements that drive them)",
                "Candidate architecture styles and their trade-offs",
                "Decisions, each as an ADR: Title, Status, Context, Decision, Consequences",
                "Building blocks: components, responsibilities and interactions",
                "Risks and open questions",
            ],
            "instruction": (
                "Analyze the source facts and produce the architecture design following the structure; "
                "write every significant decision as an ADR."
            ),
        },
    },
    {
        # The CTO: owns the vision, not the design.
        "id": "blueprint_CTO_AI_Strategy",
        "description": (
            "Sets the technology vision: where the product and its AI capabilities should be in the coming "
            "years, why that matters to the business, and the guiding principles and strategic bets that "
            "align the organization. Use for vision statements, long-term technology strategy, roadmaps, and "
            "executive or board communication."
        ),
        "blueprint": {
            "scene_goal": "Articulate a compelling, coherent technology vision that aligns the organization.",
            "style_guide": (
                "Visionary but grounded, executive level. Business outcomes over implementation detail. "
                "Confident and concise."
            ),
            "structure": [
                "Vision statement",
                "Why now: business and market drivers",
                "Guiding principles",
                "Strategic bets and roadmap horizons",
                "How success is measured",
            ],
            "instruction": (
                "Shape the source facts into a technology vision following the structure; leave design and "
                "implementation details to the architecture team."
            ),
        },
    },
]


def blueprint_metadata(item: ContextBlueprint) -> dict[str, str]:
    """The metadata stored with a blueprint's vector. The single source of truth for its shape,
    used both when uploading and when checking that Pinecone serves the current content."""
    return {
        "description": item["description"],
        "blueprint_json": json.dumps(item["blueprint"]),
    }


def build_context_vectors(
    embedding_client: OpenAI,
    embedding_model: str,
    blueprints: list[ContextBlueprint] = CONTEXT_BLUEPRINTS,
) -> list[dict[str, Any]]:
    """Embed each blueprint's description and attach the blueprint as metadata."""
    return [
        {
            "id": item["id"],
            "values": get_embedding(embedding_client, embedding_model, item["description"]),
            "metadata": blueprint_metadata(item),
        }
        for item in blueprints
    ]


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
