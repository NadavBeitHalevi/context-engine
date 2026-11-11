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
4. Remove stale blueprints: any id stored in the namespace that is no longer defined in
   `CONTEXT_BLUEPRINTS` is deleted, so the Librarian can't return a removed blueprint.

The namespace always ends up holding exactly the defined blueprints. `--fresh` also
empties it first, forcing every blueprint to be re-embedded. Other namespaces are
never touched.

Vector *ids* are read with `index.list()`, not `describe_index_stats()`: the stats
counts lag behind writes, so they are never used to decide anything here.

Run:
    uv run python -m rag_pipeline.rag_context_librarian           # create/update
    uv run python -m rag_pipeline.rag_context_librarian --fresh   # re-embed everything
"""

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeProtocolError, RateLimitException, ServiceException

from config import Settings, get_settings
from rag_pipeline.context_library import (
    CONTEXT_BLUEPRINTS,
    ContextBlueprint,
    blueprint_metadata,
    upload_context_library,
)

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
POLL_INTERVAL_S = 2.0

# Errors worth retrying while polling: Pinecone 5xx, 429, and dropped connections.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    ServiceException,
    RateLimitException,
    PineconeProtocolError,
    ConnectionError,
)


@dataclass(frozen=True)
class PipelineResult:
    index_name: str
    namespace: str
    created_index: bool
    uploaded: int
    removed: int  # stale blueprints deleted


def embedding_dimension(model: str) -> int:
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        raise ValueError(f"Unknown embedding model '{model}'; add its dimension to EMBEDDING_DIMENSIONS.") from None


def _wait_until(check: Callable[[], bool], what: str, timeout_s: float) -> None:
    """Poll `check` until it returns True. Transient Pinecone errors are retried until the deadline."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if check():
                return
        except TRANSIENT_ERRORS as exc:
            logger.warning("Transient error while waiting for %s (will retry): %s", what, exc)
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out after {timeout_s:.0f}s waiting for {what}.")
        time.sleep(POLL_INTERVAL_S)


def ensure_index(pc: Any, name: str, dimension: int, timeout_s: float = 120) -> bool:
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


def stored_ids(index: Any, namespace: str) -> set[str]:
    """Every vector id currently in the namespace (empty if the namespace doesn't exist)."""
    return {item.id for page in index.list(namespace=namespace) for item in page.vectors}


def clear_namespace(index: Any, namespace: str, timeout_s: float = 60) -> None:
    """Delete every vector in one namespace and wait until it's empty."""
    if not stored_ids(index, namespace):
        return
    logger.info("Clearing namespace '%s'.", namespace)
    index.delete(delete_all=True, namespace=namespace)
    _wait_until(lambda: not stored_ids(index, namespace), f"namespace '{namespace}' to empty", timeout_s)


def remove_stale(index: Any, namespace: str, keep_ids: set[str], timeout_s: float = 60) -> int:
    """Delete stored vectors whose ids aren't in `keep_ids`. Returns how many were deleted."""
    stale = sorted(stored_ids(index, namespace) - keep_ids)
    if not stale:
        return 0
    logger.info("Removing %d stale blueprint(s) from '%s': %s", len(stale), namespace, stale)
    index.delete(ids=stale, namespace=namespace)
    _wait_until(
        lambda: not (stored_ids(index, namespace) & set(stale)),
        f"stale blueprints to leave '{namespace}'",
        timeout_s,
    )
    return len(stale)


def serves_current_blueprints(index: Any, namespace: str, blueprints: list[ContextBlueprint]) -> bool:
    """True once every blueprint can be fetched with exactly the metadata that was uploaded."""
    if not blueprints:
        return True  # nothing to check (and Pinecone rejects a fetch with no ids)
    fetched = index.fetch(ids=[b["id"] for b in blueprints], namespace=namespace).vectors
    for b in blueprints:
        vector = fetched.get(b["id"])
        if vector is None or (vector.metadata or {}) != blueprint_metadata(b):
            return False
    return True


def run_pipeline(
    settings: Settings,
    fresh: bool = False,
    blueprints: list[ContextBlueprint] = CONTEXT_BLUEPRINTS,
    *,
    pc: Any | None = None,
    embedding_client: Any | None = None,
    timeout_s: float = 60,
) -> PipelineResult:
    """Create the index if needed and make the Context Library namespace hold exactly `blueprints`.

    `pc` and `embedding_client` default to real clients built from settings; tests pass fakes.
    """
    pc = pc or Pinecone(api_key=settings.pinecone_api_key)
    embedding_client = embedding_client or OpenAI(api_key=settings.openai_api_key)
    name = settings.pinecone_index_name
    namespace = settings.namespace_context

    created = ensure_index(pc, name, embedding_dimension(settings.embedding_model))
    index = pc.Index(name)
    if fresh:
        clear_namespace(index, namespace, timeout_s)

    uploaded = upload_context_library(index, embedding_client, settings.embedding_model, namespace, blueprints)
    _wait_until(
        lambda: serves_current_blueprints(index, namespace, blueprints),
        f"namespace '{namespace}' to serve the uploaded blueprints",
        timeout_s,
    )
    logger.info("All %d blueprints are queryable with their current content.", uploaded)

    removed = remove_stale(index, namespace, {b["id"] for b in blueprints}, timeout_s)
    return PipelineResult(name, namespace, created, uploaded, removed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the Context Library into Pinecone.")
    parser.add_argument("--fresh", action="store_true", help="empty the namespace first and re-embed everything")
    args = parser.parse_args()

    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    result = run_pipeline(settings, fresh=args.fresh)
    logger.info("Done: %s", result)


if __name__ == "__main__":
    main()
