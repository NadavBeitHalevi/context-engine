"""RAG pipeline runner for the Context Library (what the Librarian agent searches).

Steps:
1. Make sure the Pinecone index exists and is usable: create a serverless index sized
   for the embedding model if it's missing; for an existing one, refuse to continue if
   its dimension or metric is wrong, and wait until it's ready.
2. Embed every Semantic Blueprint's description and upsert it into the Context
   Library namespace (see `context_library.py`).
3. Wait until Pinecone serves the *new* content: fetch every blueprint back and compare
   it with what was uploaded (serverless indexes are eventually consistent, and on a
   re-run the old vectors are already there under the same ids).

Upserts use stable blueprint ids, so re-running updates blueprints in place.
`--fresh` first empties the Context Library namespace, which also removes blueprints
you deleted from `CONTEXT_BLUEPRINTS`. It never touches other namespaces.

Run:
    uv run python -m rag_pipeline.rag_context_librarian           # create/update
    uv run python -m rag_pipeline.rag_context_librarian --fresh   # rebuild namespace
"""

import argparse
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import Settings, get_settings
from rag_pipeline.context_library import CONTEXT_BLUEPRINTS, ContextBlueprint, upload_context_library

logger = logging.getLogger(__name__)

# Output size of each OpenAI embedding model; the index dimension must match it.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

INDEX_METRIC = "cosine"
INDEX_CLOUD = "aws"
INDEX_REGION = "us-east-1"
POLL_INTERVAL_S = 2


@dataclass(frozen=True)
class PipelineResult:
    index_name: str
    namespace: str
    created_index: bool
    uploaded: int
    vector_count: int


def embedding_dimension(model: str) -> int:
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        raise ValueError(f"Unknown embedding model '{model}'; add its dimension to EMBEDDING_DIMENSIONS.") from None


def _wait_until(check: Callable[[], bool], what: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not check():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.0f}s waiting for {what}.")
        time.sleep(POLL_INTERVAL_S)


def ensure_index(pc: Pinecone, name: str, dimension: int, timeout_s: float = 120) -> bool:
    """Create the serverless index if missing, check it fits, and wait until it's ready.

    Returns True if the index was created by this call.
    """
    created = False
    if pc.has_index(name):
        existing = pc.describe_index(name)
        if existing.dimension != dimension:
            raise ValueError(f"Index '{name}' has dimension {existing.dimension}, but the embedding model needs {dimension}.")
        if existing.metric != INDEX_METRIC:
            raise ValueError(f"Index '{name}' uses metric '{existing.metric}', but the pipeline needs '{INDEX_METRIC}'.")
        logger.info("Index '%s' already exists (dimension %d, %s).", name, dimension, INDEX_METRIC)
    else:
        logger.info("Creating serverless index '%s' (dimension %d, %s).", name, dimension, INDEX_METRIC)
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=INDEX_METRIC,
            spec=ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
        )
        created = True

    # Also covers an existing index still initializing (e.g. an earlier run timed out).
    _wait_until(lambda: pc.describe_index(name).status.ready, f"index '{name}' to be ready", timeout_s)
    logger.info("Index '%s' is ready.", name)
    return created


def namespace_count(index: Any, namespace: str) -> int:
    summary = index.describe_index_stats().namespaces.get(namespace)
    return summary.vector_count if summary else 0


def clear_namespace(index: Any, namespace: str, timeout_s: float = 60) -> None:
    """Delete every vector in one namespace and wait until it's empty."""
    if namespace_count(index, namespace) == 0:
        return
    logger.info("Clearing namespace '%s'.", namespace)
    index.delete(delete_all=True, namespace=namespace)
    _wait_until(lambda: namespace_count(index, namespace) == 0, f"namespace '{namespace}' to empty", timeout_s)


def _serves_current_blueprints(index: Any, namespace: str, blueprints: list[ContextBlueprint]) -> bool:
    """True once every blueprint can be fetched with exactly the content that was uploaded."""
    fetched = index.fetch(ids=[b["id"] for b in blueprints], namespace=namespace).vectors
    for b in blueprints:
        vector = fetched.get(b["id"])
        metadata = (vector.metadata or {}) if vector else {}
        if metadata.get("description") != b["description"]:
            return False
        if metadata.get("blueprint_json") != json.dumps(b["blueprint"]):
            return False
    return True


def run_pipeline(settings: Settings, fresh: bool = False, blueprints: list[ContextBlueprint] = CONTEXT_BLUEPRINTS) -> PipelineResult:
    """Create the index if needed and load the Context Library into it."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    name = settings.pinecone_index_name
    namespace = settings.namespace_context

    created = ensure_index(pc, name, embedding_dimension(settings.embedding_model))
    index = pc.Index(name)
    if fresh:
        clear_namespace(index, namespace)

    uploaded = upload_context_library(
        index,
        OpenAI(api_key=settings.openai_api_key),
        settings.embedding_model,
        namespace,
        blueprints,
    )
    _wait_until(
        lambda: _serves_current_blueprints(index, namespace, blueprints),
        f"namespace '{namespace}' to serve the uploaded blueprints",
        timeout_s=60,
    )
    logger.info("All %d blueprints are queryable with their current content.", uploaded)

    # Stats lag behind fetches; this count is informational (stale-vector warning only).
    count = namespace_count(index, namespace)
    if count > uploaded:
        logger.warning(
            "Namespace '%s' holds %d vectors but %d blueprints are defined; run with --fresh to drop stale ones.",
            namespace, count, uploaded,
        )
    return PipelineResult(name, namespace, created, uploaded, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Context Library into Pinecone.")
    parser.add_argument("--fresh", action="store_true", help="empty the Context Library namespace first")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    result = run_pipeline(settings, fresh=args.fresh)
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
