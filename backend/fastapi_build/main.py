# code-agent-gemini/backend/fastapi_build/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI
from google.adk.runners import Runner
from google.genai import types as genai_types
from pydantic import BaseModel, HttpUrl

from fastapi_build.agents.youtube_processing_agents import (
    adk_session_service,  # Using a global ADK session service for this example
    common_exit_stack,  # Manages MCP server lifecycle
    root_agent_instance,
)

# Import settings and agent-related components from our package
from fastapi_build.core.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)


# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup...")
    # Initialize any resources on startup if needed
    # For example, connect to databases, load models not part of ADK init

    # Ensure common_exit_stack is entered. This is critical if MCP servers
    # are started via StdioServerParameters and managed by this stack.
    await common_exit_stack.__aenter__()
    logger.info("AsyncExitStack for MCP tools entered.")

    yield

    logger.info("FastAPI application shutting down...")
    # Clean up resources on shutdown
    await common_exit_stack.aclose()
    logger.info("AsyncExitStack for MCP tools closed.")


app = FastAPI(
    title="Code Agent Gemini - YouTube Fact Checker",
    description="Processes YouTube videos to summarize content and perform fact-checking using Google ADK and Gemini.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Pydantic Models for API ---
class ProcessVideoRequest(BaseModel):
    video_url: HttpUrl  # FastAPI will validate if it's a URL
    user_id: str = "default_user"  # Optional: to track user sessions
    session_id: str | None = None  # Optional: to continue an existing session


class ProcessVideoResponse(BaseModel):
    session_id: str
    summary: str | None = None
    fact_check_report: str | None = None
    full_agent_output: str  # The combined output from the orchestrator
    error: str | None = None
    # Include a field for grounding information if applicable
    grounding_html_content: str | None = None


# --- ADK Runner Setup ---
# Using the globally instantiated root_agent_instance and adk_session_service from agents module
# This is a simplification. In a more complex app, you might manage Runner instances differently.
adk_runner = Runner(
    agent=root_agent_instance,
    app_name="youtube_fact_checker_fastapi",
    session_service=adk_session_service,
    # artifact_service can be added here if your agents use artifacts
    # artifact_service=InMemoryArtifactService(),
)


# --- API Endpoints ---
@app.post("/process-video/", response_model=ProcessVideoResponse)
async def process_video(request: ProcessVideoRequest = Body(...)):
    """Processes a YouTube video: fetches transcript, summarizes, and fact-checks."""
    logger.info(f"Received request to process video: {request.video_url} for user: {request.user_id}")

    # Manage ADK session
    if request.session_id:
        session = adk_session_service.get_session(
            app_name=adk_runner.app_name,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        if not session:
            logger.warning(f"Session ID {request.session_id} not found for user {request.user_id}. Creating new.")
            session = adk_session_service.create_session(app_name=adk_runner.app_name, user_id=request.user_id)
    else:
        session = adk_session_service.create_session(app_name=adk_runner.app_name, user_id=request.user_id)

    logger.info(f"Using ADK Session ID: {session.id}")

    query_content = genai_types.Content(role="user", parts=[genai_types.Part(text=str(request.video_url))])

    final_combined_output = ""
    error_message = None
    summary_text = None
    fact_check_text = None
    rendered_content_html = None  # For grounding citations

    try:
        # The common_exit_stack is managed by the FastAPI lifespan manager
        # The YouTubeProcessingOrchestratorAgent uses it via ctx.misc
        async for event in adk_runner.run_async(
            session_id=session.id,
            user_id=request.user_id,
            new_message=query_content,
        ):
            logger.debug(
                f"ADK Event: Author={event.author}, FinalForTurn={event.turn_complete}, Partial={event.partial}"
            )
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.debug(f"  Event Text: {part.text[:100]}...")
                    # Capture renderedContent for grounding from Google Search tool
                    if hasattr(part, "tool_code_output") and part.tool_code_output:  # Old ADK versions
                        # Look for renderedContent in tool_code_output if Google Search was used
                        # This part needs verification with latest ADK structure for Google Search grounding
                        pass
                    if event.grounding_metadata:  # Newer ADK way
                        for gm_item in event.grounding_metadata.grounding_attributions:
                            if gm_item.web and gm_item.web.title and gm_item.web.uri:
                                if rendered_content_html is None:
                                    rendered_content_html = ""
                                rendered_content_html += (
                                    f"<p><a href='{gm_item.web.uri}' target='_blank'>" f"{gm_item.web.title}</a></p>"
                                )
                            # The ADK docs state: (
                            #   "The UI code (HTML) is returned in the Gemini response as renderedContent"
                            # )
                            # This might be directly in a Part, or within grounding_metadata.
                            # If event.content.parts[0].rendered_content exists, use that.
                            # This needs to be checked against an actual Gemini response with grounding.
                            # For now, constructing simple HTML from grounding_metadata.
                    if event.author == root_agent_instance.name and event.is_final_response() and part.text:
                        final_combined_output = part.text
                        logger.info(f"Final combined output from orchestrator: {final_combined_output[:200]}...")
                        # Try to parse summary and fact-check from the combined output
                        # This is a simple parsing strategy;
                        # a more robust way would be structured output from the agent.
                        if (
                            "Video Summary:" in final_combined_output
                            and "Fact-Checking Report:" in final_combined_output
                        ):
                            parts = final_combined_output.split("Fact-Checking Report:", 1)
                            summary_text = parts[0].replace("Video Summary:", "").strip()
                            fact_check_text = parts[1].strip()
                        else:
                            summary_text = "Could not parse summary."
                            fact_check_text = "Could not parse fact-check report."

            if event.actions and event.actions.state_delta:
                logger.debug(f"  State Delta: {event.actions.state_delta}")
            if event.error_message:
                logger.error(f"  ADK Event Error: Code={event.error_code}, Message={event.error_message}")
                # error_message = f"Agent error: {event.error_message}" # This might be too verbose

        if not final_combined_output:  # If the agent didn't yield a final text response via orchestrator directly
            last_state = adk_session_service.get_session(adk_runner.app_name, request.user_id, session.id).state
            final_combined_output = (
                f"Summary: {last_state.get('video_summary', 'N/A')}\n"
                f"Fact-Check: {last_state.get('fact_check_report', 'N/A')}"
            )
            summary_text = last_state.get("video_summary")
            fact_check_text = last_state.get("fact_check_report")
            if not summary_text and not fact_check_text and not error_message:
                error_message = "Agent finished but no final output captured in expected format."
                final_combined_output = error_message

    except Exception as e:
        logger.exception(f"Error processing video URL {request.video_url}")
        error_message = f"An unexpected error occurred: {str(e)}"
        # Ensure the exit stack is closed if an error occurs mid-processing within this request.
        # However, the lifespan manager should handle this more globally.
        # await common_exit_stack.aclose() # This might be too aggressive here.

    if error_message and not final_combined_output:
        final_combined_output = f"Error: {error_message}"

    return ProcessVideoResponse(
        session_id=session.id,
        summary=summary_text,
        fact_check_report=fact_check_text,
        full_agent_output=final_combined_output,
        error=error_message,
        grounding_html_content=rendered_content_html,
    )


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Code Agent Gemini - YouTube Fact Checker API"}


# To run this app (from the 'backend' directory):
# uvicorn fastapi_build.main:app --reload --port 8000
# Ensure .env file is present in the 'backend' directory.
