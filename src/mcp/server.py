"""MCP Server for Pet Vet RAG Agent tools.

Exposes the 4 vet tools via MCP stdio protocol, allowing any MCP-compatible
client (Claude Code, Cursor, custom orchestrator) to discover and call them.

Usage as standalone server:
    python -m src.mcp.server

The server registers 4 tools:
    - search_knowledge_base  (宠物兽医文献检索)
    - analyze_symptoms       (症状分析)
    - triage_decision        (就医紧迫性分级)
    - search_cnki            (知网网络检索)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure project root on path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.agent.tools import (
    search_knowledge_base,
    analyze_symptoms,
    triage_decision,
    search_cnki,
)

# ── Tool schemas in MCP format ────────────────────────────────────────────

TOOL_SCHEMAS = [
    Tool(
        name="search_knowledge_base",
        description="搜索宠物兽医知识库（ChromaDB+BM25混合检索），返回相关学术文献摘录。"
                    "覆盖29篇CNKI论文、2277个知识片段。",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用户的自然语言问题，如'猫打喷嚏的原因'",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="analyze_symptoms",
        description="分析宠物症状描述，列出可能的疾病方向。包含相关文献记载。仅供参考，不构成兽医诊断。",
        inputSchema={
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "string",
                    "description": "用户描述的宠物症状，如'猫连续打喷嚏三天，精神不振'",
                },
            },
            "required": ["symptoms"],
        },
    ),
    Tool(
        name="triage_decision",
        description="根据症状严重程度，给出就医紧迫性三级建议：🚨急诊 / ⚠️尽快就医 / ℹ️居家观察。"
                    "内置17种危险信号（呼吸困难、中毒、抽搐等）自动识别。",
        inputSchema={
            "type": "object",
            "properties": {
                "symptoms": {
                    "type": "string",
                    "description": "用户描述的宠物症状",
                },
            },
            "required": ["symptoms"],
        },
    ),
    Tool(
        name="search_cnki",
        description="🌐 当本地知识库无结果时，通过搜索引擎查找知网论文摘要作为补充。"
                    "来源标注为'网络检索'以区别于本地知识库。",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询（宠物猫疾病相关），如'猫杯状病毒 治疗'",
                },
            },
            "required": ["query"],
        },
    ),
]

# ── Tool dispatch ──────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "search_knowledge_base": search_knowledge_base,
    "analyze_symptoms": analyze_symptoms,
    "triage_decision": triage_decision,
    "search_cnki": search_cnki,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with arguments. Synchronous for compatibility."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"错误: 未知工具 '{name}'。可用工具: {', '.join(TOOL_FUNCTIONS.keys())}"
    try:
        return str(func(**arguments))
    except Exception as e:
        return f"工具 '{name}' 执行失败: {e}"


# ── MCP Server ────────────────────────────────────────────────────────────

server = Server("pet-vet-rag-agent")


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return all registered tools in MCP format."""
    return list(TOOL_SCHEMAS)


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return MCP-formatted result."""
    result = execute_tool(name, arguments)
    return [TextContent(type="text", text=result)]


# ── Entry point ────────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
