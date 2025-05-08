import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_build.agents.youtube_processing_agents import (
    ClaimsOutput,
    CollectorAgent,
    DequeueAgent,
    SearchPlanItem,
    SearchPlanOutput,
    TranscriptFetcherAgent,
    Verdict,
    WorkerOutput,
    _get_yt_transcript,
    claim_extractor_agent,
    common_exit_stack,
    extract_video_id,
    fact_check_loop,
    fact_checker_worker,
    root_agent,
    search_planner_agent,
)
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.events import Event as AdkEvent
from google.adk.sessions import Session
from google.adk.tools.mcp_tool.mcp_toolset import MCPTool, MCPToolset
from google.genai import types as genai_types


@pytest.fixture
def mock_invocation_context():
    ctx = MagicMock()
    ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text="https://www.youtube.com/watch?v=dQw4w9WgXcQ")]
    )
    ctx.invocation_id = "test_inv_id"
    ctx.session = MagicMock(spec=Session)
    ctx.session.state = {}
    ctx.misc = {"async_exit_stack": common_exit_stack}
    return ctx


@pytest.fixture
def mock_tool_context(mock_invocation_context):
    return MagicMock(
        invocation_context=mock_invocation_context, function_call_id="test_func_call_id"
    )


def _assert_event_text_contains(event: AdkEvent, expected_text: str):
    assert event.content and event.content.parts
    assert expected_text in event.content.parts[0].text


@pytest.mark.parametrize(
    "raw_input, expected_id",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=youtu.be", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("http://youtu.be/dQw4w9WgXcQ?t=5", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/abcdefghijk", "abcdefghijk"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("valid__id__", "valid__id__"),
        ("invalid_url", "invalid_url"),
        ("https://www.example.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.example.com/somevideo", None),
        ("not a url at all", None),
        ("dQw4w9WgXc", None),
        ("dQw4w9WgXcQ1", None),
        ("https://www.example.com/watch?v=toolongid123", "toolongid12"),
        ("", None),
    ],
)
def test_extract_video_id(raw_input, expected_id):
    assert extract_video_id(raw_input) == expected_id


@pytest.mark.asyncio
async def test_get_yt_transcript_success(mock_tool_context):
    mock_mcp_tool = AsyncMock(spec=MCPTool)
    mock_mcp_tool.run_async.return_value = genai_types.Part(
        text=json.dumps({"status": "success", "transcript": [{"text": "Hello"}]})
    )

    mock_mcp_toolset = MagicMock(spec=MCPToolset)
    mock_mcp_toolset.__getitem__.return_value = mock_mcp_tool

    with (
        patch(
            "fastapi_build.youtube_processing_agents.MCPToolset.from_server",
            AsyncMock(return_value=(mock_mcp_toolset, AsyncMock())),
        ),
        patch("fastapi_build.youtube_processing_agents.Path.exists", return_value=True),
    ):
        transcript = await _get_yt_transcript("dQw4w9WgXcQ", mock_tool_context)
        assert transcript == [{"text": "Hello"}]
        mock_mcp_tool.run_async.assert_called_once()


@pytest.mark.asyncio
async def test_get_yt_transcript_mcp_error(mock_tool_context):
    mock_mcp_tool = AsyncMock(spec=MCPTool)
    mock_mcp_tool.run_async.return_value = genai_types.Part(
        text=json.dumps({"status": "error", "message": "MCP failed"})
    )
    mock_mcp_toolset = MagicMock(spec=MCPToolset)
    mock_mcp_toolset.__getitem__.return_value = mock_mcp_tool

    with (
        patch(
            "fastapi_build.youtube_processing_agents.MCPToolset.from_server",
            AsyncMock(return_value=(mock_mcp_toolset, AsyncMock())),
        ),
        patch("fastapi_build.youtube_processing_agents.Path.exists", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="MCP error: MCP failed"):
            await _get_yt_transcript("dQw4w9WgXcQ", mock_tool_context)


@pytest.mark.asyncio
async def test_get_yt_transcript_script_not_found(mock_tool_context):
    with patch(
        "fastapi_build.youtube_processing_agents.Path.exists", return_value=False
    ):
        with pytest.raises(RuntimeError, match="MCP script not found"):
            await _get_yt_transcript("dQw4w9WgXcQ", mock_tool_context)


@pytest.mark.asyncio
async def test_transcript_fetcher_agent_valid_url(mock_invocation_context):
    agent = TranscriptFetcherAgent(name="TestTranscriptFetcher")
    expected_transcript = [{"text": "Segment 1"}, {"text": "Segment 2"}]

    with (
        patch(
            "fastapi_build.youtube_processing_agents.extract_video_id",
            return_value="dQw4w9WgXcQ",
        ) as mock_extract,
        patch(
            "fastapi_build.youtube_processing_agents.youtube_transcript_tool.run_async",
            AsyncMock(return_value=expected_transcript),
        ) as _,
    ):
        events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]

        assert len(events) == 1
        event = events[0]
        _assert_event_text_contains(event, "✅ Transcript fetched (2 segments)")
        assert event.turn_complete
        assert event.actions.state_delta["transcript"] == expected_transcript
        assert (
            mock_invocation_context.session.state["transcript"] == expected_transcript
        )
        mock_extract.assert_called_once_with(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        )


@pytest.mark.asyncio
async def test_transcript_fetcher_agent_invalid_url(mock_invocation_context):
    agent = TranscriptFetcherAgent(name="TestTranscriptFetcher")
    mock_invocation_context.user_content = genai_types.Content(
        parts=[genai_types.Part(text="invalid")]
    )

    with patch(
        "fastapi_build.youtube_processing_agents.extract_video_id", return_value=None
    ) as mock_extract:
        events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]

        assert len(events) == 1
        event = events[0]
        _assert_event_text_contains(event, "❌ Please provide a valid YouTube URL")
        assert event.turn_complete
        mock_extract.assert_called_once_with("invalid")


@pytest.mark.asyncio
async def test_transcript_fetcher_agent_tool_error(mock_invocation_context):
    agent = TranscriptFetcherAgent(name="TestTranscriptFetcher")

    with (
        patch(
            "fastapi_build.youtube_processing_agents.extract_video_id",
            return_value="dQw4w9WgXcQ",
        ),
        patch(
            "fastapi_build.youtube_processing_agents.youtube_transcript_tool.run_async",
            AsyncMock(side_effect=RuntimeError("Tool failed")),
        ) as _,
    ):
        events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]

        assert len(events) == 1
        event = events[0]
        _assert_event_text_contains(event, "❌ Transcript error: Tool failed")
        assert event.turn_complete


def test_claim_extractor_agent_config():
    assert isinstance(claim_extractor_agent, LlmAgent)
    assert claim_extractor_agent.name == "ClaimExtractorAgent"
    assert claim_extractor_agent.output_key == "claims"
    assert 'Return **only** JSON {"claims": [...]}' in claim_extractor_agent.instruction


def test_search_planner_agent_config():
    assert isinstance(search_planner_agent, LlmAgent)
    assert search_planner_agent.name == "SearchPlannerAgent"
    assert search_planner_agent.output_key == "pending_items"
    assert (
        "Return **only** a JSON array of {claim, query} objects."
        in search_planner_agent.instruction
    )


@pytest.mark.asyncio
async def test_dequeue_agent_empty_queue(mock_invocation_context):
    agent = DequeueAgent(name="TestDequeueAgent")
    mock_invocation_context.session.state["pending_items"] = []

    events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]
    assert len(events) == 1
    event = events[0]
    assert event.actions.escalate is True
    assert event.turn_complete


@pytest.mark.asyncio
async def test_dequeue_agent_with_items(mock_invocation_context):
    agent = DequeueAgent(name="TestDequeueAgent")
    item1 = {"claim": "Claim 1", "query": "Query 1"}
    item2 = {"claim": "Claim 2", "query": "Query 2"}
    mock_invocation_context.session.state["pending_items"] = [item1, item2]

    events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]
    assert len(events) == 1
    event = events[0]
    _assert_event_text_contains(event, f"🔍 Checking: {item1['claim']}")
    assert event.turn_complete
    assert event.actions.state_delta["current_item"] == item1
    assert event.actions.state_delta["pending_items"] == [item2]
    assert mock_invocation_context.session.state["current_item"] == item1
    assert mock_invocation_context.session.state["pending_items"] == [item2]


@pytest.mark.asyncio
async def test_dequeue_agent_json_string_items(mock_invocation_context):
    agent = DequeueAgent(name="TestDequeueAgent")
    item1 = {"claim": "Claim 1", "query": "Query 1"}
    item_str = f"```json\n{json.dumps([item1])}\n```"
    mock_invocation_context.session.state["pending_items"] = item_str

    events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]
    assert len(events) == 1
    event = events[0]
    _assert_event_text_contains(event, f"🔍 Checking: {item1['claim']}")
    assert mock_invocation_context.session.state["current_item"] == item1
    assert mock_invocation_context.session.state["pending_items"] == []


def test_fact_checker_worker_config():
    assert isinstance(fact_checker_worker, LlmAgent)
    assert fact_checker_worker.name == "FactCheckerWorker"
    assert fact_checker_worker.output_key == "last_verdict"
    assert "Call google_search(query, num_results=5)" in fact_checker_worker.instruction
    assert len(fact_checker_worker.tools) == 1


@pytest.mark.asyncio
async def test_collector_agent(mock_invocation_context):
    agent = CollectorAgent(name="TestCollectorAgent")
    events = [ev async for ev in agent._run_async_impl(mock_invocation_context)]
    assert len(events) == 1
    event = events[0]
    assert event.turn_complete
    assert event.content is None


def test_fact_check_loop_config():
    assert isinstance(fact_check_loop, LoopAgent)
    assert fact_check_loop.name == "FactCheckLoop"
    assert len(fact_check_loop.sub_agents) == 3
    assert isinstance(fact_check_loop.sub_agents[0], DequeueAgent)
    assert isinstance(fact_check_loop.sub_agents[1], LlmAgent)
    assert isinstance(fact_check_loop.sub_agents[2], CollectorAgent)
    assert fact_check_loop.max_iterations == 20


def test_root_agent_config():
    assert isinstance(root_agent, SequentialAgent)
    assert root_agent.name == "YouTubeFactCheckerPipeline"
    assert len(root_agent.sub_agents) == 4
    assert isinstance(root_agent.sub_agents[0], TranscriptFetcherAgent)
    assert isinstance(root_agent.sub_agents[1], LlmAgent)
    assert isinstance(root_agent.sub_agents[2], LlmAgent)
    assert isinstance(root_agent.sub_agents[3], LoopAgent)


def test_pydantic_models_instantiation():
    claims_out = ClaimsOutput(claims=["claim1", "claim2"])
    assert len(claims_out.claims) == 2

    search_plan_item = SearchPlanItem(claim="c1", query="q1")
    search_plan_output = SearchPlanOutput.model_validate(
        [search_plan_item.model_dump()]
    )
    assert len(search_plan_output.root) == 1

    verdict = Verdict(claim="c1", verdict="True", sources=["src1"])
    worker_output = WorkerOutput.model_validate([verdict.model_dump()])
    assert len(worker_output.root) == 1
