import asyncio
import os
import logging
from contextlib import AsyncExitStack
import json
from google.adk.agents import LlmAgent, Agent, BaseAgent
from google.adk.tools import google_search, agent_tool, function_tool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types
import json
from fastapi_build.core.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)

# --- Global services for ADK (can be initialized once) ---
# These are used if you run agents directly with ADK Runner...
# FastAPI will manage sessions differently if integrating ADK more deeply, sooo ake a note of that
adk_session_service = InMemorySessionService()

# Common AsyncExitStack for managing MCP server lifecycle
common_exit_stack = AsyncExitStack()


# --- Agent Definitions --_ #


async def get_youtube_transcript_from_mcp(
    video_url: str, tool_context: ToolContext
) -> str:
    """
    An ADK function_tool that uses MCPToolset to connect to our custom
    YouTube Transcript MCP Server and fetch the transcript.
    This function itself will be wrapped by function_tool and given to an ADK Agent.
    """
    logger.info(f"ADK Tool: Attempting to get transcript for URL: {video_url} via MCP.")

    # Path to the MCP server script - ensure it's correct relative to where ADK runs
    # This assumes the CWD is the `backend` directory when this is run by FastAPI/ADK... and we package the backend using pyproject.toml
    # or an absolute path is used.
    script_path = os.path.abspath(settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH)
    if not os.path.exists(script_path):
        logger.error(f"YouTube MCP Server script not found at: {script_path}")
        return json.dumps(
            {
                "status": "error",
                "message": f"MCP Server script not found: {script_path}",
            }
        )

    logger.info(f"Using YouTube MCP Server script: {script_path}")

    mcp_tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="python3",
            args=[script_path],  # Absolute path to the MCP server script
        ),
        # Share the common_exit_stack to manage the server lifecycle
        # If this tool is called multiple times in one agent run, it might try to start multiple servers
        # if not managed carefully. For simplicity, we create a new connection per call here.
        # More robust: manage a single MCPToolset instance for the YouTube server lifecycle.
        async_exit_stack=tool_context.invocation_context.misc.get(
            "mcp_exit_stack"
        ),  # Get from invocation context
    )

    if not mcp_tools:
        logger.error("Failed to connect to YouTube MCP Server or no tools found.")
        return json.dumps(
            {"status": "error", "message": "Failed to connect to YouTube MCP Server."}
        )

    youtube_transcript_mcp_tool = mcp_tools[0]  # Assuming it's the only tool exposed

    try:
        logger.info(
            f"Calling MCP tool '{youtube_transcript_mcp_tool.name}' with args: {{'video_url': '{video_url}'}}"
        )
        mcp_response_content = await youtube_transcript_mcp_tool.run_async(
            args={
                "video_url": video_url
            },  # Arguments for the MCP tool, i'll standarrize this later
            tool_context=tool_context,
            # we're gong to pas the ADK tool context to the MCP tool
        )
        # The MCP tool returns a list of content parts, we expect one TextContent part with JSON
        if isinstance(mcp_response_content, list) and len(mcp_response_content) > 0:
            if hasattr(mcp_response_content[0], "text"):
                transcript_data_json = mcp_response_content[0].text
                logger.info(f"Received transcript data (JSON string) from MCP server.")
                # The MCP server already formats its output as a JSON string representing the
                return transcript_data_json  # Return the JSON string as is..
            else:
                logger.error(
                    f"Unexpected response part from MCP tool: {mcp_response_content[0]}"
                )
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Unexpected response format from MCP tool",
                    }
                )
        else:  # ADK's .run_async for MCP tools might directly return the processed content (e.g. a dict if conversion happened)
            if isinstance(mcp_response_content, str):  # If it's already a string
                logger.info(
                    f"Received transcript data (string) from MCP server: {mcp_response_content[:200]}..."
                )
                return mcp_response_content
            elif isinstance(mcp_response_content, dict):  # If it's a dict
                logger.info(f"Received transcript data (dict) from MCP server.")
                return json.dumps(mcp_response_content)  # Convert dict to JSON string
            else:
                logger.error(
                    f"Unexpected response type from MCP tool: {type(mcp_response_content)}"
                )
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Unexpected response type from MCP tool: {type(mcp_response_content)}",
                    }
                )

    except Exception as e:
        logger.exception("Error calling YouTube Transcript MCP tool")
        return json.dumps(
            {"status": "error", "message": f"Error calling YouTube MCP tool: {str(e)}"}
        )


# Wrap the function to be used as an ADK tool
youtube_transcript_tool = function_tool(
    func=get_youtube_transcript_from_mcp,
    # The docstring of get_youtube_transcript_from_mcp will be used as description
    ##NOTE : This is a bit of a hack, but it works for now.
)

summarizer_agent = LlmAgent(
    name="VideoSummarizerAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction="""You are a Video Summarizer.
    Given a video transcript (which will be a JSON string containing a list of transcript segments),
    your task is to:
    1. Parse the JSON string to extract the transcript segments. Each segment has 'text', 'start', and 'duration'.
    2. Concatenate the 'text' from all segments to form the full transcript.
    3. Generate a concise, neutral, and informative summary of the full transcript.
    4. The summary should capture the main topics and key points discussed in the video.
    5. Output *only* the summary as a plain text string. Do not include any introductory phrases like "Here is the summary:" or markdown.
    """,
    description="Summarizes a YouTube video transcript.",
    # Input to this agent will be the raw JSON string from youtube_transcript_tool
)

fact_checker_agent = LlmAgent(
    name="FactCheckerAgent",
    model=settings.ADK_GEMINI_MODEL,  # Google Search tool is compatible with Gemini 2 models like "gemini-2.0-flash"...
    # before pushing to publish, NOTE:which models do support grounding search
    instruction="""You are a Fact-Checking Agent.
    You will be given a summary of a YouTube video and the original transcript (JSON string).
    Your tasks are:
    1. Identify key factual claims made in the summary.
    2. Use the 'google_search' tool to verify these claims.
    3. For each claim, state whether it is 'Verified', 'False', or 'Unverified due to lack of information'.
    4. Provide the source URL(s) from your search that support your finding for each claim.
    5. If a claim is false, briefly explain why or provide the correct information.
    6. Present your findings in a clear, structured format. For example:
       Claim: "The sky is green."
       Status: False
       Reason/Correction: The sky is typically blue due to Rayleigh scattering.
       Source: https://www.merriam-webster.com/dictionary/source

       Claim: "Water boils at 100 degrees Celsius at sea level."
       Status: Verified
       Source: https://www.merriam-webster.com/dictionary/source
    7. Ensure your response is grounded using the search results.
    Your final output should be the structured fact-checking report as a plain text string.
    """,
    tools=[google_search],  # ADK's built-in Google Search tool
    description="Fact-checks claims from a video summary using Google Search and provides sources.",
)


# Orchestrator Agent
# This will be the main agent invoked by FastAPI
class YouTubeProcessingOrchestratorAgent(BaseAgent):  # Using BaseAgent for custom flow
    """
    Orchestrates the process of fetching a YouTube transcript, summarizing it,
    and then fact-checking the summary.
    """

    # Pydantic model_config for BaseAgent subclasses
    model_config = {"arbitrary_types_allowed": True}

    # Define sub-agents that will be used
    _youtube_transcript_tool: function_tool
    _summarizer_agent: LlmAgent
    _fact_checker_agent: LlmAgent

    def __init__(
        self,
        name: str,
        youtube_tool: function_tool,
        summarizer: LlmAgent,
        fact_checker: LlmAgent,
        **kwargs,
    ):
        super().__init__(
            name=name, description="Orchestrates YouTube video processing.", **kwargs
        )
        self._youtube_transcript_tool = youtube_tool
        self._summarizer_agent = summarizer
        self._fact_checker_agent = fact_checker
        # Register sub-agents for ADK's awareness if needed for certain features,
        # though direct invocation is primary here.
        self.sub_agents = [summarizer, fact_checker]

    async def _run_async_impl(
        self, ctx: Agent.InvocationContext
    ) -> genai_types.AsyncGeneratorType[Agent.Event, None]:
        """Implements the custom orchestration logic."""
        logger.info(
            f"[{self.name}] Starting YouTube processing workflow for invocation {ctx.invocation_id}."
        )
        user_input_text = ""
        if ctx.user_content and ctx.user_content.parts:
            user_input_text = ctx.user_content.parts[
                0
            ].text  # Expecting YouTube URL here

        if not user_input_text or not (
            "youtube.com/watch?v=" in user_input_text or "youtu.be/" in user_input_text
        ):
            logger.warning(
                f"[{self.name}] Invalid or missing YouTube URL in input: {user_input_text}"
            )
            yield Agent.Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text="Error: Please provide a valid YouTube video URL."
                        )
                    ]
                ),
            )
            return

        video_url = user_input_text
        logger.info(f"[{self.name}] Processing URL: {video_url}")

        # Step 1: Get YouTube Transcript using the MCP-wrapper tool
        # Add the shared exit stack to the context for the tool to use
        ctx.misc["mcp_exit_stack"] = common_exit_stack  # Store it in misc for the tool

        transcript_json_str_event = None
        try:
            logger.info(
                f"[{self.name}] Calling YouTube Transcript Tool for: {video_url}"
            )
            # ADK function_tool.run_async expects keyword arguments
            transcript_json_str_result_dict = (
                await self._youtube_transcript_tool.run_async(
                    args={"video_url": video_url},  # Pass args as a dictionary
                    tool_context=ToolContext.from_invocation_context(
                        ctx,
                        self._youtube_transcript_tool.name,
                        ctx.invocation_id + "-transcript-tool",
                    ),
                )
            )
            # The tool itself returns a JSON string which is the 'result' field of the dict
            transcript_json_str = transcript_json_str_result_dict["result"]

            transcript_json_str_event = Agent.Event(
                author=self.name,  # Or tool name
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            text=f"Transcript fetched (JSON string). Length: {len(transcript_json_str)}"
                        )
                    ]
                ),
                actions=Agent.EventActions(
                    state_delta={"raw_transcript_json": transcript_json_str}
                ),
            )
            yield transcript_json_str_event
            logger.info(
                f"[{self.name}] Transcript fetched successfully. JSON String starts with: {transcript_json_str[:200]}..."
            )
        except Exception as e:
            logger.exception(f"[{self.name}] Error getting transcript for {video_url}")
            yield Agent.Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text=f"Error fetching transcript: {str(e)}")
                    ]
                ),
            )
            return

        # Check if transcript fetching was successful from the JSON string content
        try:
            transcript_data = json.loads(transcript_json_str)
            if transcript_data.get("status") == "error":
                error_message = transcript_data.get(
                    "message", "Unknown error from transcript service."
                )
                logger.error(
                    f"[{self.name}] Transcript service returned error: {error_message}"
                )
                yield Agent.Event(
                    author=self.name,
                    invocation_id=ctx.invocation_id,
                    content=genai_types.Content(
                        parts=[
                            genai_types.Part(
                                text=f"Error from transcript service: {error_message}"
                            )
                        ]
                    ),
                )
                return
        except json.JSONDecodeError:
            logger.error(
                f"[{self.name}] Failed to parse transcript JSON: {transcript_json_str[:500]}"
            )
            yield Agent.Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="Error: Failed to parse transcript data.")
                    ]
                ),
            )
            return

        # Step 2: Summarize the Transcript
        summary_text = "Error: Summarization failed."
        try:
            logger.info(f"[{self.name}] Requesting summary from SummarizerAgent.")
            # Pass the raw JSON transcript string to the summarizer agent
            summarizer_input_content = genai_types.Content(
                parts=[genai_types.Part(text=transcript_json_str)]
            )
            async for event in self._summarizer_agent.run_async(
                ctx, user_content=summarizer_input_content
            ):
                yield event  # Yield intermediate events from summarizer
                if event.is_final_response() and event.content and event.content.parts:
                    summary_text = event.content.parts[0].text
                    ctx.session.state["video_summary"] = (
                        summary_text  # Save to state for fact-checker
                    )
                    logger.info(
                        f"[{self.name}] Summary received: {summary_text[:200]}..."
                    )
        except Exception as e:
            logger.exception(f"[{self.name}] Error during summarization")
            yield Agent.Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text=f"Error during summarization: {str(e)}")
                    ]
                ),
            )
            # Continue to fact-checking with error message if summarization fails partially,
            # or return if it's a critical failure. For now, let's assume we try to fact-check what we have.

        # Step 3: Fact-Check the Summary
        fact_check_report = "Error: Fact-checking failed."
        try:
            logger.info(f"[{self.name}] Requesting fact-check from FactCheckerAgent.")
            # FactCheckerAgent needs both summary and original transcript for context
            # The instruction for FactCheckerAgent expects the transcript as a JSON string.
            fact_checker_input_text = f"Summary:\n{summary_text}\n\nOriginal Transcript (JSON String):\n{transcript_json_str}"
            fact_checker_input_content = genai_types.Content(
                parts=[genai_types.Part(text=fact_checker_input_text)]
            )

            async for event in self._fact_checker_agent.run_async(
                ctx, user_content=fact_checker_input_content
            ):
                yield event  # Yield intermediate events from fact-checker
                if event.is_final_response() and event.content and event.content.parts:
                    fact_check_report = event.content.parts[0].text
                    logger.info(
                        f"[{self.name}] Fact-check report received: {fact_check_report[:200]}..."
                    )
        except Exception as e:
            logger.exception(f"[{self.name}] Error during fact-checking")
            # Update fact_check_report with error to return to user
            fact_check_report = f"Error during fact-checking: {str(e)}"
            # Yield an event for this specific error
            yield Agent.Event(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=fact_check_report)]
                ),
            )

        # Step 4: Combine results and yield final response
        final_combined_response = f"Video Summary:\n{summary_text}\n\nFact-Checking Report:\n{fact_check_report}"

        # The final response should also include grounding information if present from fact-checker
        # We'll assume the last event from fact_checker_agent (if it used Google Search)
        # might have grounding_metadata. We need to forward this.
        # A more robust way would be to capture the final event from fact_checker and use its metadata.
        # For now, we yield a new final event.

        final_event_parts = [genai_types.Part(text=final_combined_response)]
        # If fact_check_report contains renderedContent, it should be sent to the UI.
        # We are sending the text part here. The UI will need to handle `renderedContent` from events.

        yield Agent.Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(parts=final_event_parts),
            turn_complete=True,  # Mark as final usable response for this turn
        )
        logger.info(
            f"[{self.name}] YouTube processing workflow completed for invocation {ctx.invocation_id}."
        )


# --- Create Root Agent Instance ---
# This function will be called by FastAPI to get an initialized agent.
def create_youtube_processing_agent():
    """Creates and returns an instance of the YouTubeProcessingOrchestratorAgent."""
    orchestrator = YouTubeProcessingOrchestratorAgent(
        name="YouTubeFactCheckerOrchestrator",
        youtube_tool=youtube_transcript_tool,
        summarizer=summarizer_agent,
        fact_checker=fact_checker_agent,
    )
    return orchestrator


# Create a global instance for FastAPI to use, or create on demand.
# For simplicity with MCP server lifecycle, creating it here and assuming
# FastAPI app startup/shutdown handles common_exit_stack.aclose()
root_agent_instance = create_youtube_processing_agent()


# Example of how to run this agent setup locally for testing (not part of FastAPI app)
async def local_test_run():
    logger.info("Starting local test run of YouTube Processing Agent...")

    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set or ADC is configured
    # and other GOOGLE_CLOUD_PROJECT etc. are set in your environment or .env

    test_session = adk_session_service.create_session(
        app_name="youtube_test_app", user_id="local_test_user"
    )

    runner = Runner(
        agent=root_agent_instance,
        app_name="youtube_test_app",
        session_service=adk_session_service,
        # artifact_service can be added if agents use artifacts
    )

    # youtube_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # A video with transcript
    youtube_video_url = (
        "https://www.youtube.com/watch?v=EIkeRBP5nDg"  # Example: ADK intro
    )

    logger.info(f"Local Test: Querying with URL: {youtube_video_url}")
    query_content = genai_types.Content(
        role="user", parts=[genai_types.Part(text=youtube_video_url)]
    )

    final_text_response = "No final text response captured."
    async with common_exit_stack:  # Manage MCP server lifecycle
        async for event in runner.run_async(
            session_id=test_session.id,
            user_id=test_session.user_id,
            new_message=query_content,
        ):
            logger.info(
                f"Local Test Event: Author={event.author}, Partial={event.partial}, FinalForTurn={event.turn_complete}"
            )
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.info(f"  Part Text: {part.text[:300]}...")
                    if part.function_call:
                        logger.info(
                            f"  Part Function Call: {part.function_call.name}({part.function_call.args})"
                        )
                    if part.function_response:
                        logger.info(
                            f"  Part Function Response: {part.function_response.name} -> {part.function_response.response}"
                        )
                    if hasattr(part, "grounding_metadata") and part.grounding_metadata:
                        logger.info(
                            f"  Part Grounding Metadata: {part.grounding_metadata}"
                        )

            if (
                event.is_final_response()
                and event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
                final_text_response = event.content.parts[0].text

    logger.info(f"\n--- Local Test Final Combined Output ---")
    logger.info(final_text_response)
    logger.info("Local test run finished.")


if __name__ == "__main__":
    # This allows running `python -m fastapi_build.agents.youtube_processing_agents` for local testing
    # Make sure your .env file is in the `backend` directory or environment variables are set.

    # Correct the working directory if running this script directly for local_test_run
    # This is a bit of a hack for direct execution; in FastAPI, paths are handled from backend root.
    if not str(settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH).startswith(
        "/"
    ):  # if it's a relative path
        # Assume running from code-agent-gemini/backend
        # settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH needs to be relative to CWD or absolute
        # For direct script run, CWD is `backend/fastapi_build/agents`. Script is in `backend/fastapi_build/mcp_servers`
        # So, relative path from here is `../mcp_servers/youtube_transcript_mcp_server.py`
        # current_script_dir = Path(__file__).parent
        # mcp_script_rel_path = Path("../mcp_servers/youtube_transcript_mcp_server.py")
        # settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH = str((current_script_dir / mcp_script_rel_path).resolve())
        # A simpler approach for direct execution: ensure CWD is 'backend'
        print(f"Original MCP script path: {settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH}")
        expected_backend_dir = Path(os.getcwd())
        if not (
            expected_backend_dir / settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH
        ).exists():
            print(
                f"Warning: MCP script might not be found if CWD is not 'backend/'. Current CWD: {os.getcwd()}"
            )
            print(
                "For local testing of this script directly, ensure you run it from the 'backend' directory using `python -m fastapi_build.agents.youtube_processing_agents`"
            )
            # Or adjust YOUTUBE_MCP_SERVER_SCRIPT_PATH in .env to be absolute for direct script testing.

    asyncio.run(local_test_run())
