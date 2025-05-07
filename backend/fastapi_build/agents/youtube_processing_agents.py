"""
YouTube fact-checking pipeline built with Google-ADK + LoopAgent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from pydantic import BaseModel, Field, RootModel

# ── Environment ────────────────────────────────────────────────────────────────
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ.pop("GOOGLE_CLOUD_LOCATION", None)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "false")

import google  # noqa: E402
import google.generativeai as genai  # type: ignore

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ── ADK core ───────────────────────────────────────────────────────────────────
from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event as AdkEvent, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools import google_search
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

# ── Project settings ───────────────────────────────────────────────────────────
from fastapi_build.core.config import settings

logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)
_common_exit_stack = AsyncExitStack()


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_video_id(raw: str) -> Optional[str]:
    for pat in [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be\/([A-Za-z0-9_-]{11})",
        r"\/shorts\/([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]:
        if m := re.search(pat, raw):
            return m.group(1)
    return None


# ── MCP transcript-fetch tool ──────────────────────────────────────────────────
async def _get_yt_transcript(video_id: str, tool_context: ToolContext) -> List[dict]:
    script = Path(settings.YOUTUBE_MCP_SERVER_SCRIPT_PATH).expanduser()
    if not script.exists():
        raise RuntimeError(f"MCP script not found: {script}")

    mcp_tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command=sys.executable, args=["-u", str(script)]
        ),
        async_exit_stack=_common_exit_stack,
    )
    raw = await mcp_tools[0].run_async(
        args={"video_id": video_id}, tool_context=tool_context
    )

    payload = (
        raw.content[0].text
        if hasattr(raw, "content")
        else getattr(raw, "text", str(raw))
    )
    resp = json.loads(payload)
    if resp.get("status") != "success":
        raise RuntimeError(f"MCP error: {resp.get('message')}")
    return resp["transcript"]


youtube_transcript_tool = FunctionTool(func=_get_yt_transcript)


# ── 1 · Transcript fetcher ─────────────────────────────────────────────────────
class TranscriptFetcherAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[AdkEvent, None]:
        vid = extract_video_id(ctx.user_content.parts[0].text.strip())
        if not vid:
            yield AdkEvent(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(text="❌ Please provide a valid YouTube URL")
                    ]
                ),
                turn_complete=True,
            )
            return

        try:
            transcript = await youtube_transcript_tool.run_async(
                args={"video_id": vid},
                tool_context=ToolContext(
                    ctx, function_call_id=f"{ctx.invocation_id}-yt"
                ),
            )
        except Exception as exc:
            yield AdkEvent(
                author=self.name,
                invocation_id=ctx.invocation_id,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=f"❌ Transcript error: {exc}")]
                ),
                turn_complete=True,
            )
            return

        ctx.session.state["transcript"] = transcript
        yield AdkEvent(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=f"✅ Transcript fetched ({len(transcript)} segments)"
                    )
                ]
            ),
            actions=EventActions(state_delta={"transcript": transcript}),
            turn_complete=True,
        )


# ── 2 · Claim extraction ───────────────────────────────────────────────────────
MAX_CLAIMS = 5


class ClaimsOutput(BaseModel):
    claims: List[str] = Field(..., max_items=MAX_CLAIMS)


claim_extractor_agent = LlmAgent(
    name="ClaimExtractorAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are an expert claim-extraction agent.\n\n"
        "Here is the video transcript:\n{transcript}\n\n"
        f'Return **only** JSON {{"claims": [...]}} with ≤ {MAX_CLAIMS} factual statements.'
    ),
    output_key="claims",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


# ── 3 · Search planning ────────────────────────────────────────────────────────
class SearchPlanItem(BaseModel):
    claim: str
    query: str


class SearchPlanOutput(RootModel[List[SearchPlanItem]]):
    pass


search_planner_agent = LlmAgent(
    name="SearchPlannerAgent",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are a search-planner.\n\n"
        "Claims to verify:\n{claims}\n\n"
        "For each claim craft a concise Google query.\n"
        "Return **only** a JSON array of {claim, query} objects."
    ),
    output_key="pending_items",  # store list of dicts for the loop
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ── 4 · Fact-check LoopAgent ───────────────────────────────────────────────────


class DequeueAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        state = ctx.session.state

        # --- grab & normalise the work-queue -------------------------------
        raw = state.get("pending_items", [])
        if isinstance(raw, str):
            txt = raw.strip()
            if txt.startswith("```"):
                # drop any Markdown fences
                txt = "\n".join(
                    l for l in txt.splitlines() if not re.match(r"\s*```", l)
                )
            try:
                pending: List[dict] = json.loads(txt)
            except Exception:
                pending = []
            # overwrite with parsed list so we don't do this again
            state["pending_items"] = pending
        else:
            pending = raw

        # --- if queue is empty, exit loop ----------------------------------
        if not pending:
            yield AdkEvent(
                author=self.name,
                invocation_id=ctx.invocation_id,
                actions=EventActions(escalate=True),
                turn_complete=True,
            )
            return

        # --- pop next item --------------------------------------------------
        current = pending.pop(0)
        state["current_item"] = current
        yield AdkEvent(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=genai_types.Content(
                parts=[genai_types.Part(text=f"🔍 Checking: {current['claim']}")]
            ),
            actions=EventActions(
                state_delta={"current_item": current, "pending_items": pending}
            ),
            turn_complete=True,
        )


# 4b – LLM worker (single item, calls google_search)
class Verdict(BaseModel):
    claim: str
    verdict: str
    sources: List[str | int]


class WorkerOutput(RootModel[List[Verdict]]):
    pass


fact_checker_worker = LlmAgent(
    name="FactCheckerWorker",
    model=settings.ADK_GEMINI_MODEL,
    instruction=(
        "You are a professional fact-checker.\n\n"
        "Input:\n{current_item}\n\n"
        "Call google_search(query, num_results=5), then decide **True**, **False**, or **Unverified**.\n"
        "Return **only** JSON array of {claim, verdict, sources}."
    ),
    tools=[google_search],
    output_key="last_verdict",  # DequeueAgent will harvest this
)


# 4c – Collector agent (no LLM, just acknowledgement to keep LoopAgent happy)
class CollectorAgent(BaseAgent):
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[AdkEvent, None]:
        # Just emit a no-op event; DequeueAgent already moved verdict
        yield AdkEvent(
            author=self.name,
            invocation_id=ctx.invocation_id,
            turn_complete=True,
        )


# Compose LoopAgent
fact_check_loop = LoopAgent(
    name="FactCheckLoop",
    sub_agents=[
        DequeueAgent(name="DequeueAgent"),
        fact_checker_worker,
        CollectorAgent(name="CollectorAgent"),
    ],
    max_iterations=20,
)

# ── 5 · Sequential pipeline ───────────────────────────────────────────────────
root_agent = SequentialAgent(
    name="YouTubeFactCheckerPipeline",
    description="End-to-end YouTube fact-checking demo using Google-ADK.",
    sub_agents=[
        TranscriptFetcherAgent(name="TranscriptFetcherAgent"),
        claim_extractor_agent,
        search_planner_agent,
        fact_check_loop,
    ],
)

# Global Instances for FastAPI integration
root_agent_instance = root_agent
adk_session_service = InMemorySessionService()
common_exit_stack = _common_exit_stack


# ── 6 · Local test harness (optional) ──────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover

    async def _local_test() -> None:
        service = InMemorySessionService()
        sess: Session = service.create_session(app_name="yt_test", user_id="local")
        runner = Runner(agent=root_agent, app_name="yt_test", session_service=service)
        url = "https://www.youtube.com/watch?v=80i_FUllRVU"

        async with _common_exit_stack:
            async for ev in runner.run_async(
                user_id=sess.user_id,
                session_id=sess.id,
                new_message=genai_types.Content(
                    role="user", parts=[genai_types.Part(text=url)]
                ),
            ):
                if ev.content and ev.content.parts:
                    print(ev.content.parts[0].text)

    asyncio.run(_local_test())
