"""MCP Tool Provider — bridges MCP tool definitions with the orchestrator.

Provides a unified interface for tool discovery and execution that follows
MCP semantics while staying compatible with the existing synchronous orchestrator.

Modes:
    - embedded: Direct function calls (production mode, no subprocess overhead)
    - external: Via MCP stdio client (for external MCP servers)

The orchestrator only sees list_tools_openai() and call_tool() — it doesn't
need to know whether tools come from direct Python functions or MCP servers.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.mcp.server import TOOL_SCHEMAS, execute_tool  # embedded mode
from src.agent.tools import (
    search_knowledge_base,
    analyze_symptoms,
    triage_decision,
    search_cnki,
)

# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class MCPToolDef:
    """MCP-compatible tool definition."""
    name: str
    description: str
    parameters: dict  # JSON Schema for the tool's input


@dataclass
class MCPToolProvider:
    """Unified tool provider with MCP compatibility.

    In embedded mode (default), tools are called directly — no subprocess.
    In external mode, tools are discovered from MCP servers via stdio.

    Usage:
        provider = MCPToolProvider()
        tools = provider.list_tools_openai()  # -> OpenAI-compatible format
        result = provider.call_tool("search_knowledge_base", {"query": "..."})
    """

    # External MCP server configs (lazy — only used if mode="external")
    _external_servers: list[dict] = field(default_factory=list)
    _mode: str = "embedded"

    # ── Embedded mode (default) ─────────────────────────────────────────

    def _get_embedded_tools(self) -> dict[str, dict]:
        """Return tool registry in embedded mode."""
        return {
            "search_knowledge_base": {
                "function": search_knowledge_base,
                "description": TOOL_SCHEMAS[0].description,
                "parameters": TOOL_SCHEMAS[0].inputSchema,
            },
            "analyze_symptoms": {
                "function": analyze_symptoms,
                "description": TOOL_SCHEMAS[1].description,
                "parameters": TOOL_SCHEMAS[1].inputSchema,
            },
            "triage_decision": {
                "function": triage_decision,
                "description": TOOL_SCHEMAS[2].description,
                "parameters": TOOL_SCHEMAS[2].inputSchema,
            },
            "search_cnki": {
                "function": search_cnki,
                "description": TOOL_SCHEMAS[3].description,
                "parameters": TOOL_SCHEMAS[3].inputSchema,
            },
        }

    def list_tools_openai(self) -> list[dict]:
        """Return tools in OpenAI-compatible Function Calling format.

        This is the format the orchestrator passes to ollama.chat(tools=...).
        """
        tools = self._get_embedded_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in tools.items()
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool by name. Synchronous — compatible with orchestrator."""
        return execute_tool(name, arguments)

    def get_tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._get_embedded_tools().keys())

    def get_server_count(self) -> int:
        """Return number of tool servers (for UI display)."""
        return 1  # embedded mode: all tools in one logical server

    def get_status(self) -> dict:
        """Return provider status for monitoring panel."""
        tools = self._get_embedded_tools()
        return {
            "mode": self._mode,
            "servers": self.get_server_count(),
            "tools": len(tools),
            "tool_names": list(tools.keys()),
            "protocol": "MCP (Model Context Protocol)",
        }


# ── External MCP client (for connecting to standalone MCP servers) ─────────

class ExternalMCPClient:
    """Async MCP client for connecting to external MCP servers via stdio.

    This demonstrates the full MCP client pattern. In production, the
    embedded MCPToolProvider is used for lower latency.

    Usage:
        client = ExternalMCPClient()
        tools = await client.connect_and_list_tools("python -m src.mcp.server")
        result = await client.call_tool("search_knowledge_base", {"query": "..."})
    """

    def __init__(self):
        self._tools: list[MCPToolDef] = []

    async def connect_and_list_tools(self, command: str) -> list[MCPToolDef]:
        """Launch MCP server subprocess and discover tools."""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        # Parse command into args for subprocess
        parts = command.split()
        server_params = {
            "command": parts[0],
            "args": parts[1:] if len(parts) > 1 else [],
        }

        self._tools = []
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await session.list_tools()

                for t in mcp_tools.tools:
                    self._tools.append(MCPToolDef(
                        name=t.name,
                        description=t.description or "",
                        parameters=t.inputSchema if hasattr(t, 'inputSchema') else {},
                    ))

        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool on an external MCP server."""
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession

        command = "python -m src.mcp.server"
        parts = command.split()
        server_params = {
            "command": parts[0],
            "args": parts[1:],
        }

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                # Extract text from MCP content blocks
                texts = []
                for block in result.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)
                return "\n".join(texts)


# ── Convenience: run external client synchronously ──────────────────────────

def discover_external_tools(command: str = "python -m src.mcp.server") -> list[dict]:
    """Synchronous wrapper: discover tools from an external MCP server.

    Returns tools in OpenAI-compatible format.
    """
    async def _discover():
        client = ExternalMCPClient()
        mcp_tools = await client.connect_and_list_tools(command)
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in mcp_tools
        ]

    return asyncio.run(_discover())
