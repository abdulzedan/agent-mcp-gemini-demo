# File: youtube_transcript_mcp_server.py

#!/usr/bin/env python3
import asyncio
import json
import logging
import re
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from mcp import types as mcp_types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YouTubeTranscriptMCPServer")


def get_youtube_transcript_tool_func(video_id_or_url: str) -> dict:
    """
    MCP Tool function to fetch a YouTube transcript.
    Accepts either a raw 11-char video ID or a full URL.
    """
    logger.info(f"Fetching transcript for input: {video_id_or_url}")

    # Determine video ID
    vid = None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id_or_url):
        vid = video_id_or_url
    else:
        # Try parsing URL
        parsed = urlparse(video_id_or_url)
        qs = parse_qs(parsed.query)
        vids = qs.get("v", [])
        if vids:
            vid = vids[0]
        else:
            m = re.search(
                r"(?:youtu\.be\/|\/shorts\/)([A-Za-z0-9_-]{11})", video_id_or_url
            )
            if m:
                vid = m.group(1)

    if not vid:
        return {"status": "error", "message": "Could not parse video ID"}

    try:
        transcript = YouTubeTranscriptApi.get_transcript(vid)
        return {"status": "success", "transcript": transcript}
    except TranscriptsDisabled:
        return {"status": "error", "message": "Transcripts disabled."}
    except NoTranscriptFound:
        return {"status": "error", "message": "No transcript found."}
    except Exception as e:
        logger.exception("Unexpected error fetching transcript")
        return {"status": "error", "message": str(e)}


def google_search_tool(query: str, num_results: int = 5) -> dict:
    # stub/simulation – replace with your actual search integration
    return {"status": "success", "query": query, "results": []}


# ADK Tools (no 'name=' here)
adk_youtube_tool = FunctionTool(func=get_youtube_transcript_tool_func)
adk_search_tool = FunctionTool(func=google_search_tool)
logger.info(f"Initialized MCP tools: {adk_youtube_tool.name}, {adk_search_tool.name}")

# MCP Server
mcp_app = Server("youtube-transcript-mcp-server")


@mcp_app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    logger.info("Listing MCP tools")
    return [
        adk_to_mcp_tool_type(adk_youtube_tool),
        adk_to_mcp_tool_type(adk_search_tool),
    ]


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.info(f"call_tool: {name}, args={arguments}")
    if name == adk_youtube_tool.name:
        resp = get_youtube_transcript_tool_func(arguments.get("video_id", ""))
        return [mcp_types.TextContent(type="text", text=json.dumps(resp))]
    if name == adk_search_tool.name:
        q = arguments.get("query", "")
        n = int(arguments.get("num_results", 5))
        resp = google_search_tool(q, n)
        return [mcp_types.TextContent(type="text", text=json.dumps(resp))]
    return [
        mcp_types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"Unknown tool {name}"}),
        )
    ]


async def run_mcp_server():
    async with mcp.server.stdio.stdio_server() as (r, w):
        await mcp_app.run(
            r,
            w,
            InitializationOptions(
                server_name=mcp_app.name,
                server_version="0.1.0",
                capabilities=mcp_app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    logger.info("Starting YouTube Transcript MCP Server...")
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception:
        logger.exception("Unhandled error")
    finally:
        logger.info("Exiting MCP server")
