"""RAG pipeline runner for the Context Library (what the Librarian agent searches).

Steps:
1. Make sure the Pinecone index exists (create a serverless index sized for the
   embedding model if it doesn't; refuse to continue if an existing index has the
   wrong dimension).
2. Embed every Semantic Blueprint's description and upsert it into the Context
   Library namespace (see `context_library.py`).
3. Wait until Pinecone reports the vectors (serverless indexes are eventually
   consistent), so whatever runs next can query them.

Upserts use stable blueprint ids, so re-running updates blueprints in place.
`--fresh` first empties the Context Library namespace, which also removes blueprints
you deleted from `CONTEXT_BLUEPRINTS`. It never touches other namespaces.

Run:
    uv run python -m rag_pipline.rag_context_librarian           # create/update
    uv run python -m rag_pipline.rag_context_librarian --fresh   # rebuild namespace
"""

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from config import Settings, get_settings
from rag_pipline.context_library import CONTEXT_BLUEPRINTS, upload_context_library

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


def ensure_index(pc: Pinecone, name: str, dimension: int, timeout_s: float = 120) -> bool:
    """Create the serverless index if missing and wait until it's ready. Returns True if created."""
    if pc.has_index(name):
        existing = pc.describe_index(name).dimension
        if existing != dimension:
            raise ValueError(f"Index '{name}' has dimension {existing}, but the embedding model needs {dimension}.")
        logger.info("Index '%s' already exists (dimension %d).", name, dimension)
        return False

    logger.info("Creating serverless index '%s' (dimension %d, %s).", name, dimension, INDEX_METRIC)
    pc.create_index(
        name=name,
        dimension=dimension,
        metric=INDEX_METRIC,
        spec=ServerlessSpec(cloud=INDEX_CLOUD, region=INDEX_REGION),
    )
    deadline = time.monotonic() + timeout_s
    while not pc.describe_index(name).status.ready:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Index '{name}' was not ready after {timeout_s:.0f}s.")
        time.sleep(2)
    logger.info("Index '%s' is ready.", name)
    return True


def namespace_count(index: Any, namespace: str) -> int:
    summary = index.describe_index_stats().namespaces.get(namespace)
    return summary.vector_count if summary else 0


def clear_namespace(index: Any, namespace: str, timeout_s: float = 60) -> None:
    """Delete every vector in one namespace and wait until it's empty."""
    if namespace_count(index, namespace) == 0:
        return
    logger.info("Clearing namespace '%s'.", namespace)
    index.delete(delete_all=True, namespace=namespace)
    _wait_for_count(index, namespace, lambda n: n == 0, "to empty", timeout_s)


def _wait_for_count(index: Any, namespace: str, done, what: str, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        count = namespace_count(index, namespace)
        if done(count):
            return count
        if time.monotonic() > deadline:
            raise TimeoutError(f"Namespace '{namespace}' did not finish {what} within {timeout_s:.0f}s (count={count}).")
        time.sleep(2)


def run_pipeline(settings: Settings, fresh: bool = False) -> PipelineResult:
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
        CONTEXT_BLUEPRINTS,
    )
    count = _wait_for_count(index, namespace, lambda n: n >= uploaded, "indexing", timeout_s=60)
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
