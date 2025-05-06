# ---------------------------------------------------------------------------
# google‑genai compatibility shim (unchanged – keeps Dev‑API not Vertex)
# ---------------------------------------------------------------------------
import os, logging, asyncio, json

os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "false")

# --- Google / GenAI bootstrap ------------------------------------------------
# we *must* import google early so later code can do `google.xxx`
import google  # <<< FIXED NameError

try:
    from google import genai
except ImportError:
    import google.generativeai as genai  # type: ignore

_GENAI_CLIENT = None
try:
    if hasattr(genai, "Client"):
        _GENAI_CLIENT = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except ValueError as ve:
    # Vertex creds absent → fall back to Dev‑API
    if "Project and location" in str(ve):
        if hasattr(genai, "configure"):
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise

# ---------------------------------------------------------------------------
# stdlib / third‑party
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncGenerator, ClassVar, List

# ADK
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event as AdkEvent, EventActions
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import google_search
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioServerParameters,
)
from google.genai import types as genai_types

# local settings
from fastapi_build.core.config import settings

# ---------------------------------------------------------------------------
# Constants
MAX_CLAIMS: int = 10  # hard upper‑bound of claims to process / verify

# ---------------------------------------------------------------------------
# Logging
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)

# Single AsyncExitStack shared by every MCP tool instance we spin up
_common_exit_stack = AsyncExitStack()

# In‑memory sessions for local test harness
_session_service = InMemorySessionService()


# ---------------------------------------------------------------------------
# 1)  MCP wrapper tool – fetch YouTube transcript
# ---------------------------------------------------------------------------
async def _get_yt_transcript(
    video_url: str,
    tool_context: ToolContext,  # ADK passes this automatically
) -> str:
    """Fetches the full transcript for a YouTube video via the local MCP server.

    Always returns a JSON *string* so downstream LLMs can parse it deterministically.
    """
    logger.info("YouTube MCP: requesting transcript for %s", video_url)
    script_path = Path(settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH).expanduser()
    if not script_path.exists():
        return json.dumps(
            {"status": "error", "message": f"MCP script not found: {script_path}"}
        )

    # connect (or spin‑up) the MCP server
    try:
        mcp_tools, _ = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command="python3", args=[str(script_path)]
            ),
            async_exit_stack=_common_exit_stack,
        )
    except Exception as e:
        logger.exception("Could not start MCP server")
        return json.dumps({"status": "error", "message": str(e)})

    yt_tool = mcp_tools[0]
    raw = await yt_tool.run_async(
        args={"video_url": video_url}, tool_context=tool_context
    )

    # Always return canonical JSON string so agents can reliably parse
    if isinstance(raw, str):
        return raw  # already JSON or error string
    if isinstance(raw, list):
        # Typical transcript shape → list[dict(text, start, duration, ...)]
        return json.dumps(raw)
    if isinstance(raw, dict):
        return json.dumps(raw)
    return json.dumps(
        {"status": "error", "message": f"Unexpected MCP return type: {type(raw)}"}
    )


youtube_transcript_tool = FunctionTool(func=_get_yt_transcript)

# ---------------------------------------------------------------------------
# 2)  LLM sub‑agents
# ---------------------------------------------------------------------------

# --- Claim Extractor ---------------------------------------------------------
claim_extractor_agent = LlmAgent(
    name="ClaimExtractorAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are a Claim‑Extraction Agent.\n"
        "Given the raw transcript JSON string of a YouTube video:\n"
        "1. Parse the JSON (list of caption segments), concatenate the text, and split into sentences.\n"
        "2. Identify factual or checkable statements.\n"
        f"3. Return *only* a bullet list (`* `) with **at most {MAX_CLAIMS} items** – one concise claim per line."
    ),
    description="Extracts up to 10 factual claims from a YouTube transcript.",
)

# --- Search‑Planning Agent ---------------------------------------------------
search_planner_agent = LlmAgent(
    name="SearchPlannerAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are a Search‑Planning Agent.\n"
        "Input: a bullet list of factual claims (≤10).\n"
        "For each claim, craft a concise Google query that best verifies it.\n"
        "Return **JSON** array exactly like:\n"
        '[{"claim": "<original>", "query": "<google query string>"}, …]'  # no other keys
    ),
    description="Generates optimal Google queries for each claim (max 10).",
)


# --- Fact Checker ------------------------------------------------------------
class _FactChecker(LlmAgent):
    # Allow the LLM to call the built‑in google_search tool
    tools: ClassVar[List] = [google_search]


fact_checker_agent = _FactChecker(
    name="FactCheckerAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are a Fact‑Checking Agent.\n"
        'Input: JSON list of {"claim": ..., "query": ...}. For **each** item:\n'
        "1. Call **google_search(query, num_results=5)** exactly once.\n"
        "2. Read the returned snippets + URLs.\n"
        "3. Decide a Verdict: **Verified**, **False**, or **Unverified**.\n"
        "4. Output a numbered markdown list where each line is:\n"
        "   <claim> • <Verdict> • <up to 5 source URLs>"
    ),
    description="Uses google_search (Gemini tool call) to verify each claim.",
)


# ---------------------------------------------------------------------------
# 3)  Orchestrator Agent
# ---------------------------------------------------------------------------
class YouTubeProcessingOrchestratorAgent(BaseAgent):
    """Pipeline: transcript ➜ claims (≤10) ➜ search plan ➜ fact check."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        name: str,
        yt_tool: FunctionTool,
        extractor: LlmAgent,
        planner: LlmAgent,
        checker: LlmAgent,
        **kw,
    ):
        super().__init__(
            name=name, description="Orchestrates YouTube fact‑checking.", **kw
        )
        self._yt_tool, self._extractor, self._planner, self._checker = (
            yt_tool,
            extractor,
            planner,
            checker,
        )
        self.sub_agents = [extractor, planner, checker]

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[AdkEvent, None]:
        url = ctx.user_content.parts[0].text if ctx.user_content.parts else ""
        if not ("youtube.com/watch" in url or "youtu.be/" in url):
            yield AdkEvent(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="Error: please send a valid YouTube URL.")
                    ]
                ),
                turn_complete=True,
            )
            return

        # 1) transcript ------------------------------------------------------
        transcript_json = await youtube_transcript_tool.run_async(
            args={"video_url": url},
            tool_context=ToolContext(ctx, function_call_id=f"{ctx.invocation_id}-yt"),
        )
        yield AdkEvent(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=f"Transcript fetched (≈{len(transcript_json)} chars)"
                    )
                ]
            ),
            actions=EventActions(state_delta={"raw_transcript": transcript_json}),
        )

        # 2) claim extraction -----------------------------------------------
        extractor_ctx = ctx.model_copy(
            update={
                "invocation_id": f"{ctx.invocation_id}-extract",
                "agent": self._extractor,
                "user_content": genai_types.Content(
                    role="user", parts=[genai_types.Part(text=transcript_json)]
                ),
            }
        )
        async for ev in self._extractor.run_async(extractor_ctx):
            yield ev
            if ev.is_final_response():
                claims_text_raw = ev.content.parts[0].text
                break

        # enforce hard limit of MAX_CLAIMS ----------------------------------
        claims_lines = [
            ln.strip()
            for ln in claims_text_raw.splitlines()
            if ln.strip().startswith("* ")
        ]
        limited_lines = claims_lines[:MAX_CLAIMS]
        claims_text = "\n".join(limited_lines)
        if len(claims_lines) > MAX_CLAIMS:
            logger.info("Truncated %d → %d claims", len(claims_lines), MAX_CLAIMS)

        # 3) search planning -------------------------------------------------
        planner_ctx = ctx.model_copy(
            update={
                "invocation_id": f"{ctx.invocation_id}-plan",
                "agent": self._planner,
                "user_content": genai_types.Content(
                    role="user", parts=[genai_types.Part(text=claims_text)]
                ),
            }
        )
        async for ev in self._planner.run_async(planner_ctx):
            yield ev
            if ev.is_final_response():
                planned_json = ev.content.parts[0].text
                break

        # 4) fact‑checking ---------------------------------------------------
        checker_ctx = ctx.model_copy(
            update={
                "invocation_id": f"{ctx.invocation_id}-check",
                "agent": self._checker,
                "user_content": genai_types.Content(
                    role="user", parts=[genai_types.Part(text=planned_json)]
                ),
            }
        )
        async for ev in self._checker.run_async(checker_ctx):
            yield ev
            if ev.is_final_response():
                final_report = ev.content.parts[0].text
                break

        # 5) combined answer -------------------------------------------------
        combined = (
            "### Extracted Claims (≤10)\n"
            + f"{claims_text}\n\n"
            + "### Fact‑Checking Report\n"
            + f"{final_report}"
        )
        yield AdkEvent(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(parts=[genai_types.Part(text=combined)]),
            turn_complete=True,
        )


# ---------------------------------------------------------------------------
# Factory & local test harness
# ---------------------------------------------------------------------------
def create_root_agent() -> BaseAgent:
    return YouTubeProcessingOrchestratorAgent(
        name="YouTubeFactCheckerOrchestrator",
        yt_tool=youtube_transcript_tool,
        extractor=claim_extractor_agent,
        planner=search_planner_agent,
        checker=fact_checker_agent,
    )


root_agent_instance = create_root_agent()


async def _local_test():
    """Run `python youtube_processing_agents.py` for a quick smoke‑test."""
    logger.info("Starting local test run…")
    sess = _session_service.create_session(app_name="yt_test", user_id="local")
    runner = Runner(
        agent=root_agent_instance, app_name="yt_test", session_service=_session_service
    )
    yt_url = "https://www.youtube.com/watch?v=dtY--67OSp8"
    async with _common_exit_stack:
        async for ev in runner.run_async(
            user_id=sess.user_id,
            session_id=sess.id,
            new_message=genai_types.Content(
                role="user", parts=[genai_types.Part(text=yt_url)]
            ),
        ):
            logger.info(
                "Event: author=%s partial=%s final=%s",
                ev.author,
                ev.partial,
                ev.turn_complete,
            )
            if ev.content and ev.content.parts:
                logger.info("  %s", ev.content.parts[0].text[:300])


if __name__ == "__main__":
    if not Path(settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH).is_absolute():
        print("Reminder: run from the project root so MCP script path resolves.")
    asyncio.run(_local_test())
