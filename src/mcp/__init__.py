"""MCP (Model Context Protocol) integration for Pet Vet RAG Agent.

Provides:
- MCPToolProvider: in-process tool discovery and execution via MCP patterns
- MCP Server: standalone stdio server exposing vet tools (for external MCP clients)

Architecture:
    Before (hand-written tools):
        orchestrator -> TOOLS dict -> Python functions

    After (MCP integration):
        orchestrator -> MCPToolProvider.list_tools() -> OpenAI-format schemas
        orchestrator -> MCPToolProvider.call_tool() -> execute via MCP protocol
        (falls back to direct TOOLS dict when MCP not needed)
"""

from src.mcp.tool_provider import MCPToolProvider

__all__ = ["MCPToolProvider"]
