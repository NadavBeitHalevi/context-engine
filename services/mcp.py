"""Model Context Protocol (MCP) message format used between the engine and agents."""

from typing import Any, TypedDict

PROTOCOL_VERSION = "2.0 (Context Engine)"


class MCPMessage(TypedDict):
    protocol_version: str
    sender: str
    content: Any
    metadata: dict[str, Any]


def create_mcp_message(sender: str, content: Any, metadata: dict[str, Any] | None = None) -> MCPMessage:
    """Wrap content in a standardized MCP message."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "sender": sender,
        "content": content,
        "metadata": metadata or {},
    }


def get_content_dict(message: MCPMessage, agent_name: str) -> dict[str, Any]:
    """Return the message content as a dict, or raise if the agent received something else."""
    content = message.get("content")
    if not isinstance(content, dict):
        raise ValueError(f"{agent_name} expects a dict as MCP content, got {type(content).__name__}.")
    return content
