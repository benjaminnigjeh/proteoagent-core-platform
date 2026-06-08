#!/usr/bin/env python3
"""
ProteoAgent Memory MCP Server (v1.1)

Modes:
  stdio (default)  — Claude Code spawns this as a subprocess
  --http PORT      — SSE server for Docker deployments (default port 8001)
"""

import argparse
import asyncio
import json
import sys

import mcp.types as types
from mcp.server import Server

import memory_store

memory_store.init_db()

app = Server("proteoagent-memory")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="write_memory",
            description="Create or update a persistent memory entry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name":        {"type": "string", "description": "Unique kebab-case slug"},
                    "description": {"type": "string", "description": "One-line index summary"},
                    "type": {
                        "type": "string",
                        "enum": ["user", "feedback", "project", "reference"],
                    },
                    "body": {"type": "string", "description": "Full memory content (markdown)"},
                },
                "required": ["name", "description", "type", "body"],
            },
        ),
        types.Tool(
            name="read_memory",
            description="Retrieve a memory entry by name.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        types.Tool(
            name="list_memories",
            description="List all memory entries grouped by type.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="delete_memory",
            description="Delete a memory entry by name.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        ),
        types.Tool(
            name="search_memories",
            description="Full-text search across name, description, and body.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        types.Tool(
            name="export_memories",
            description="Export all memories as a JSON string (v1.1 format).",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="import_memories",
            description="Import memories from a JSON string produced by export_memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "json_data": {
                        "type": "string",
                        "description": "JSON string from export_memories",
                    }
                },
                "required": ["json_data"],
            },
        ),
    ]


# ── Tool dispatch ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "write_memory":
            result = memory_store.write_memory(
                arguments["name"],
                arguments["description"],
                arguments["type"],
                arguments["body"],
            )
            return [types.TextContent(type="text", text=json.dumps(result))]

        if name == "read_memory":
            entry = memory_store.read_memory(arguments["name"])
            if entry is None:
                return [types.TextContent(
                    type="text",
                    text=f"No memory found with name '{arguments['name']}'",
                )]
            return [types.TextContent(type="text", text=json.dumps(entry, indent=2))]

        if name == "list_memories":
            entries = memory_store.list_memories()
            if not entries:
                return [types.TextContent(type="text", text="No memories stored yet.")]
            by_type: dict[str, list] = {}
            for e in entries:
                by_type.setdefault(e["type"], []).append(e)
            lines = ["# Memory Index"]
            for t, items in sorted(by_type.items()):
                lines.append(f"\n## {t.capitalize()}")
                for item in items:
                    lines.append(f"- **{item['name']}** — {item['description']}")
            return [types.TextContent(type="text", text="\n".join(lines))]

        if name == "delete_memory":
            deleted = memory_store.delete_memory(arguments["name"])
            msg = (
                f"Deleted '{arguments['name']}'"
                if deleted
                else f"No memory found with name '{arguments['name']}'"
            )
            return [types.TextContent(type="text", text=msg)]

        if name == "search_memories":
            results = memory_store.search_memories(arguments["query"])
            if not results:
                return [types.TextContent(type="text", text="No memories matched the query.")]
            return [types.TextContent(type="text", text=json.dumps(results, indent=2))]

        if name == "export_memories":
            return [types.TextContent(type="text", text=memory_store.export_memories())]

        if name == "import_memories":
            result = memory_store.import_memories(arguments["json_data"])
            return [types.TextContent(type="text", text=json.dumps(result))]

        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as exc:
        return [types.TextContent(type="text", text=f"Error: {exc}")]


# ── Entry points ──────────────────────────────────────────────────────────────

async def _run_stdio() -> None:
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def _run_http(port: int) -> None:
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0],
                streams[1],
                app.create_initialization_options(),
            )

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/health", endpoint=lambda r: __import__("starlette.responses", fromlist=["JSONResponse"]).JSONResponse({"status": "ok"})),  # noqa: E501
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ProteoAgent Memory MCP Server")
    parser.add_argument("--http", action="store_true", help="Run as HTTP/SSE server")
    parser.add_argument("--port", type=int, default=8001, help="HTTP port (default: 8001)")
    args = parser.parse_args()

    if args.http:
        _run_http(args.port)
    else:
        asyncio.run(_run_stdio())
