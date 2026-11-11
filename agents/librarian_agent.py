"""Librarian agent: retrieves the Semantic Blueprint that matches the desired style."""

import json
import logging

from openai import OpenAI

from services.mcp import MCPMessage, create_mcp_message, get_content_dict
from services.vector_store import VectorIndex, query_vector_store

logger = logging.getLogger(__name__)

DEFAULT_BLUEPRINT = json.dumps({"instruction": "Generate the content neutrally."})


class LibrarianAgent:
    """Input: {"intent_query": str}. Output: {"blueprint_json": str}."""

    name = "Librarian"

    def __init__(self, embedding_client: OpenAI, index: VectorIndex, embedding_model: str, namespace: str) -> None:
        self.embedding_client = embedding_client
        self.index = index
        self.embedding_model = embedding_model
        self.namespace = namespace

    def run(self, message: MCPMessage) -> MCPMessage:
        content = get_content_dict(message, self.name)
        intent_query = content.get("intent_query")
        if not intent_query:
            raise ValueError(f"{self.name} requires 'intent_query'.")

        matches = query_vector_store(
            index=self.index,
            client=self.embedding_client,
            embedding_model=self.embedding_model,
            query_text=intent_query,
            namespace=self.namespace,
            top_k=1,
        )

        if matches:
            match = matches[0]
            logger.info("[%s] Found blueprint '%s' (score %.2f)", self.name, match["id"], match["score"])
            blueprint_json = match["metadata"]["blueprint_json"]
        else:
            logger.warning("[%s] No blueprint found; using default.", self.name)
            blueprint_json = DEFAULT_BLUEPRINT

        return create_mcp_message(self.name, {"blueprint_json": blueprint_json})
