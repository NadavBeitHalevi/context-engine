"""Vector store (Pinecone) retrieval."""

import logging
from typing import Any, Protocol

from openai import OpenAI

from services.llm import get_embedding

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """The subset of the Pinecone Index interface the agents rely on."""

    def query(self, *, vector: list[float], namespace: str, top_k: int, include_metadata: bool) -> Any: ...


def query_vector_store(
    index: VectorIndex,
    client: OpenAI,
    embedding_model: str,
    query_text: str,
    namespace: str,
    top_k: int,
) -> list[Any]:
    """Embed the query and return the top_k matches (with metadata) from a namespace."""
    logger.info("Querying namespace '%s' (top_k=%d)", namespace, top_k)
    vector = get_embedding(client, embedding_model, query_text)
    response = index.query(vector=vector, namespace=namespace, top_k=top_k, include_metadata=True)
    return list(response["matches"])
