"""Summarizer agent: reduces large text to a concise summary to control token usage."""

import logging

import anthropic

from services.llm import call_llm
from services.mcp import MCPMessage, create_mcp_message, get_content_dict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert summarization AI. Reduce the provided text to its essential points, guided by "
    "the user's specific objective. The summary must be concise, accurate, and directly address the stated goal."
)


class SummarizerAgent:
    """Input: {"text_to_summarize": str, "summary_objective": str}. Output: {"summary": str}."""

    name = "Summarizer"

    def __init__(self, client: anthropic.Anthropic, generation_model: str) -> None:
        self.client = client
        self.generation_model = generation_model

    def run(self, message: MCPMessage) -> MCPMessage:
        content = get_content_dict(message, self.name)
        text = content.get("text_to_summarize")
        objective = content.get("summary_objective")
        if not text or not objective:
            raise ValueError(f"{self.name} requires 'text_to_summarize' and 'summary_objective'.")

        logger.info("[%s] Reducing context.", self.name)
        user_prompt = (
            f"--- OBJECTIVE ---\n{objective}\n\n"
            f"--- TEXT TO SUMMARIZE ---\n{text}\n--- END TEXT ---\n\n"
            "Generate the summary now."
        )
        summary = call_llm(self.client, self.generation_model, SYSTEM_PROMPT, user_prompt)
        return create_mcp_message(self.name, {"summary": summary})
