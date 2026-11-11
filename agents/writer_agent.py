"""Writer agent: applies a Semantic Blueprint to source material to produce the final content."""

import logging
from typing import Any

import anthropic

from services.llm import call_llm
from services.mcp import MCPMessage, create_mcp_message, get_content_dict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert content generation AI. Generate or rewrite content based on the provided SOURCE "
    "MATERIAL, strictly following the rules in the SEMANTIC BLUEPRINT. The SOURCE MATERIAL may contain "
    "both a synthesized answer and a list of sources; ensure the final output is a single, cohesive piece of content."
)

# Keys produced by upstream agents that carry factual material, in priority order.
FACT_KEYS = ("facts", "summary", "answer_with_sources")


def _extract_blueprint(blueprint: Any) -> str | None:
    if isinstance(blueprint, dict):
        return blueprint.get("blueprint_json")
    return blueprint


def _extract_facts(facts: Any) -> str | None:
    if isinstance(facts, dict):
        return next((facts[k] for k in FACT_KEYS if facts.get(k)), None)
    return facts


class WriterAgent:
    """Input: {"blueprint": str|dict, "facts"?: str|dict, "previous_content"?: str}. Output: str."""

    name = "Writer"

    def __init__(self, client: anthropic.Anthropic, generation_model: str) -> None:
        self.client = client
        self.generation_model = generation_model

    def run(self, message: MCPMessage) -> MCPMessage:
        content = get_content_dict(message, self.name)
        blueprint = _extract_blueprint(content.get("blueprint"))
        facts = _extract_facts(content.get("facts"))
        previous_content = content.get("previous_content")

        if not blueprint or not (facts or previous_content):
            raise ValueError(f"{self.name} requires 'blueprint' and either 'facts' or 'previous_content'.")

        if facts:
            source_label, source_material = "SOURCE MATERIAL", facts
        else:
            source_label, source_material = "PREVIOUS CONTENT (For Rewriting)", previous_content

        logger.info("[%s] Applying blueprint to %s.", self.name, source_label.lower())
        user_prompt = (
            f"--- SEMANTIC BLUEPRINT (JSON) ---\n{blueprint}\n\n"
            f"--- {source_label} ---\n{source_material}\n\n"
            "Generate the final content now."
        )
        output = call_llm(self.client, self.generation_model, SYSTEM_PROMPT, user_prompt)
        return create_mcp_message(self.name, output)
