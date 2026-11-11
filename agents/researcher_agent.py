"""Researcher agent: high-fidelity RAG that synthesizes facts with source citations."""

import logging

import anthropic
from openai import OpenAI

from services.llm import call_llm
from services.mcp import MCPMessage, create_mcp_message, get_content_dict
from services.sanitizer import UnsafeContentError, sanitize_input
from services.vector_store import VectorIndex, query_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert research synthesis AI. Provide a clear, factual answer to the user's topic "
    "based *only* on the provided source texts. After the answer, you MUST provide a \"Sources\" "
    "section listing the unique source document names you used."
)

NO_DATA_ANSWER = "No data found on the topic."
UNSAFE_DATA_ANSWER = "Could not generate a reliable answer as retrieved data was suspect."


class ResearcherAgent:
    """Input: {"topic_query": str}. Output: {"answer_with_sources": str}."""

    name = "Researcher"

    def __init__(
        self,
        llm_client: anthropic.Anthropic,
        embedding_client: OpenAI,
        index: VectorIndex,
        generation_model: str,
        embedding_model: str,
        namespace: str,
        top_k: int = 3,
    ) -> None:
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.index = index
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.namespace = namespace
        self.top_k = top_k

    def run(self, message: MCPMessage) -> MCPMessage:
        content = get_content_dict(message, self.name)
        topic = content.get("topic_query")
        if not topic:
            raise ValueError(f"{self.name} requires 'topic_query'.")

        matches = query_vector_store(
            index=self.index,
            client=self.embedding_client,
            embedding_model=self.embedding_model,
            query_text=topic,
            namespace=self.namespace,
            top_k=self.top_k,
        )
        if not matches:
            logger.warning("[%s] No relevant information found.", self.name)
            return self._reply(NO_DATA_ANSWER)

        texts: list[str] = []
        sources: set[str] = set()
        for match in matches:
            metadata = match["metadata"]
            try:
                texts.append(sanitize_input(metadata["text"]))
            except UnsafeContentError:
                logger.warning("[%s] Skipping chunk '%s' that failed sanitization.", self.name, match["id"])
                continue
            if source := metadata.get("source"):
                sources.add(source)

        if not texts:
            logger.error("[%s] All retrieved chunks failed sanitization.", self.name)
            return self._reply(UNSAFE_DATA_ANSWER)

        logger.info("[%s] Synthesizing answer from %d chunks.", self.name, len(texts))
        source_material = "\n\n---\n\n".join(texts)
        user_prompt = (
            f"Topic: {topic}\n\nSources:\n{source_material}\n\n---\n"
            "Synthesize your answer and list the source documents now."
        )
        findings = call_llm(self.llm_client, self.generation_model, SYSTEM_PROMPT, user_prompt)

        source_list = "\n".join(f"- {s}" for s in sorted(sources))
        return self._reply(f"{findings}\n\n**Sources:**\n{source_list}")

    def _reply(self, answer: str) -> MCPMessage:
        return create_mcp_message(self.name, {"answer_with_sources": answer})
