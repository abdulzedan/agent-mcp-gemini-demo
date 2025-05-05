# code-agent-gemini/backend/fastapi_build/mcp_servers/youtube_transcript_mcp_server.py
import asyncio
import json
import logging

import mcp.server.stdio

# ADK Tool Imports (for converting our function to MCP schema)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# MCP Server Imports
from mcp import types as mcp_types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

# Configure logging for the MCP server
logging.basicConfig(level=logging.INFO)  # Or load from env
logger = logging.getLogger("YouTubeTranscriptMCPServer")


# --- Define the Core Tool Function ---
def get_youtube_transcript_tool_func(video_url: str) -> dict:
    """Fetches the transcript for a given YouTube video URL.

    Args:
    ----
        video_url (str): The full URL of the YouTube video.
                         Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ

    Returns:
    -------
        dict: A dictionary containing either the transcript segments or an error message.
              On success: {"status": "success", "transcript": [{"text": "...", "start": 0.0, "duration": 0.0}, ...]}
              On failure: {"status": "error", "message": "Error description"}

    """
    logger.info(f"Tool 'get_youtube_transcript_tool_func' called with URL: {video_url}")
    try:
        if "v=" not in video_url:
            video_id_parse_error = "Invalid YouTube URL: Missing 'v=' parameter."
            logger.error(video_id_parse_error)
            return {"status": "error", "message": video_id_parse_error}

        video_id = video_url.split("v=")[1].split("&")[0]
        if not video_id:
            video_id_extract_error = "Could not extract video ID from URL."
            logger.error(video_id_extract_error)
            return {"status": "error", "message": video_id_extract_error}

        logger.info(f"Extracted Video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        logger.info(f"Successfully fetched transcript for video ID: {video_id}")
        return {"status": "success", "transcript": transcript_list}
    except TranscriptsDisabled:
        error_msg = f"Transcripts are disabled for video: {video_url}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except NoTranscriptFound:
        error_msg = f"No transcript found for video: {video_url}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred while fetching transcript for {video_url}: {str(e)}"
        logger.exception(error_msg)  # Log full traceback
        return {"status": "error", "message": error_msg}


# --- Prepare the ADK Tool (for schema conversion) ---
logger.info("Initializing ADK FunctionTool for schema conversion...")
adk_youtube_tool = FunctionTool(get_youtube_transcript_tool_func)
logger.info(f"ADK tool '{adk_youtube_tool.name}' initialized for schema purposes.")

# --- MCP Server Setup ---
logger.info("Creating MCP Server instance...")
mcp_app = Server("youtube-transcript-mcp-server")


@mcp_app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    """MCP handler to list available tools."""
    logger.info("MCP Server: Received list_tools request.")
    mcp_tool_schema = adk_to_mcp_tool_type(adk_youtube_tool)
    logger.info(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
    return [mcp_tool_schema]


@mcp_app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
    """MCP handler to execute a tool call."""
    logger.info(
        f"MCP Server: Received call_tool request for '{name}' with args: {arguments}"
    )

    if name == adk_youtube_tool.name:
        try:
            # Execute the ADK tool's underlying function directly (or tool.run if it were async)
            # Since get_youtube_transcript_tool_func is sync, we call it directly.
            # If it were async, we would await tool.run_async(args=arguments, tool_context=None)
            result_dict = get_youtube_transcript_tool_func(
                **arguments
            )  # Pass args correctly
            logger.info(
                f"MCP Server: Tool '{name}' executed. Result status: {result_dict.get('status')}"
            )

            response_text = json.dumps(result_dict, indent=2)
            return [mcp_types.TextContent(type="text", text=response_text)]

        except Exception as e:
            logger.exception(f"MCP Server: Error executing tool '{name}'")
            error_text = json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to execute tool '{name}': {str(e)}",
                }
            )
            return [mcp_types.TextContent(type="text", text=error_text)]
    else:
        logger.warning(f"MCP Server: Tool '{name}' not found.")
        error_text = json.dumps(
            {"status": "error", "message": f"Tool '{name}' not implemented."}
        )
        return [mcp_types.TextContent(type="text", text=error_text)]


# --- MCP Server Runner ---
async def run_mcp_server():
    """Runs the MCP server over standard input/output."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("MCP Server starting handshake...")
        await mcp_app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=mcp_app.name,
                server_version="0.1.0",
                capabilities=mcp_app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
        logger.info("MCP Server run loop finished.")


if __name__ == "__main__":
    logger.info("Launching YouTube Transcript MCP Server...")
    try:
        asyncio.run(run_mcp_server())
    except KeyboardInterrupt:
        logger.info("\nYouTube Transcript MCP Server stopped by user.")
    except Exception:
        logger.exception("YouTube Transcript MCP Server encountered an unhandled error")
    finally:
        logger.info("YouTube Transcript MCP Server process exiting.")
