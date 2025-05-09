# Code Agent Gemini - Demo

This project is intended to showcase how to leverge the google-adk with a local MCP-server. We do this by attaching the [MCP server](backend/fastapi_build/mcp_servers/youtube_transcript_mcp_server.py) created to the ADK adgent via [MCPToolSet](https://google.github.io/adk-docs/tools/mcp-tools/#mcptoolset-class).

The agent pipeline uses [sequential agents](https://google.github.io/adk-docs/agents/workflow-agents/sequential-agents/) and [loop agents](https://google.github.io/adk-docs/agents/workflow-agents/loop-agents/) to ingest YouTube video (via its URL or ID) through a local MCP-server, extracts its transcripts, identifies factual claims against web search results and validates those claims using a built-in [google_search](https://google.github.io/adk-docs/tools/built-in-tools/).

It features a FastAPI backend that processes YouTube video URLs, extracts transcripts, identifies factual claims, plans search queries, performs fact-checking using simulated Google Search, and presents a report.

> **Note:**: While the FastAPI server can be used to service the agent endpoint, we are mainly going to leverge the build in `adk web` command

## Demo

[![Video_demo](./video/thumbnail.jpg)](https://github.com/user-attachments/assets/3b6aa41c-ac59-4943-9c60-c37696510f07)







## Overview of Agentic Capabilities:

* **YouTube Video Processing**: Accepts a YouTube video URL
* **Transcript Fetching**: Retrieves video transcripts using an [MCP (Multi-Capability Protocol)](https://cloud.google.com/blog/products/ai-machine-learning/mcp-toolbox-for-databases-now-supports-model-context-protocol) server powered by `youtube_transcript_api`
* **Claim Extraction**: Uses a Gemini-powered LLM Agent to identify key factual claims from the transcript
* **Search Planning**: Another LLM Agent devises Google Search queries for each claim
* **Fact-Checking Loop**: An ADK LoopAgent iterates through claims
    * Dequeues a claim.
    * An LLM worker agent (simulates) uses google_search and determines if the claim is True, False, or Unverified
    * Collects 'verdicts'
* **Sequential Orchestration**: All steps are managed by an ADK SequentialAgent
* **FastAPI Backend**: Exposes an API endpoint to trigger the pipeline
* **Dockerized**: Includes Dockerfile and docker-compose for containerized deployment
* **Installable Backend Package**: The `fastapi_build` backend module is structured as an installable Python package



## Directory Structure
```bash
code-agent-gemini/
├─ .devcontainer/
│  └─ devcontainer.json
├─ .github/
│  └─ workflows/
│     └─ python.yaml
├─ backend/
│  ├─ fastapi_build/
│  │  ├─ agents/
│  │  │  ├─ __init__.py
│  │  │  └─ youtube_processing_agents.py
│  │  ├─ core/
│  │  │  ├─ __init__.py
│  │  │  └─ config.py
│  │  ├─ mcp_servers/
│  │  │  ├─ __init__.py
│  │  │  └─ youtube_transcript_mcp_server.py
│  │  ├─ tools/
│  │  │  └─ __init__.py
│  │  ├─ __init__.py
│  │  ├─ agent.py
│  │  └─ main.py
│  ├─ fastapi_build.egg-info/
│  ├─ tests/
│  │  ├─ __init__.py
│  │  └─ test_youtube_processing_agents.py
│  ├─ Dockerfile
│  ├─ pyproject.toml
│  ├─ README.md
│  ├─ requirements-dev.txt
│  └─ requirements.txt
├─ frontend/
├─ video/
│  ├─ demo-video.mp4
│  └─ thunbnail.png
├─ .env.example
├─ .gitignore
├─ .pre-commit-config.yaml
├─ docker-compose.yml
├─ LICENSE.md
├─ pyproject.toml
└─ README.md

```

## Prerequisites

* Python (version 3.11+ recommended, see `backend/Dockerfile` for version used in container)
* Google API Key for Generative AI (Gemini)
    * Obtain from [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console
* `gcloud` CLI (if using Application Default Credentials locally, or for Vertex AI)
* Docker and Docker Compose (for containerized deployment)

## Setup and Installation (Local)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/abdulzedan/agent-mcp-gemini-demo.git
    cd code-agent-gemini
    ```

2.  **Set up Environment Variables:**
    Copy `.env.example` to `.env` in the project root and fill in your `GOOGLE_API_KEY` and other relevant details:
    ```bash
    cp .env.example .env
    # Edit .env with your credentials
    ```
    Example `.env` content:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    GOOGLE_CLOUD_PROJECT="your-gcp-project-id" # Optional, if not using Vertex AI for Gemini this can be a placeholder
    GOOGLE_CLOUD_LOCATION="us-central1"      # Optional, same as above
    LOG_LEVEL="INFO"
    GOOGLE_GENAI_USE_VERTEXAI="false" # Set to true if using Vertex AI Gemini models
    ```

3.  **Backend Setup:**
    Create a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
    Navigate to the backend directory:
    ```bash
    cd backend
    ```
    Install dependencies and the `fastapi_build` package in editable mode:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt # For testing and development tools
    pip install -e . # Installs fastapi_build from pyproject.toml
    ```

4.  **Google Authentication (Local Development):**
    If `GOOGLE_GENAI_USE_VERTEXAI` is `true` or if your agents/tools need broader Google Cloud access, ensure you have Application Default Credentials:
    ```bash
    gcloud auth application-default login
    ```
    For Gemini API directly with an API key (`GOOGLE_GENAI_USE_VERTEXAI="false"`), this step might not be strictly necessary if the key is the only auth needed by `google-generativeai`.

## Running the Application (Local)

1.  **Exposing the ADK Web tool to see monitor agent flows:**
    From the `backend` directory (with the virtual environment activated):
    ```bash
    adk web
    ```
    The adk web UI will load and will be available at `http://localhost:8000`.

> **Note:**: When you open the adk web UI, you will need to select the "fastapi_build" on the agents

2.  **MCP Server (`youtube_transcript_mcp_server.py`):**
    This server is started on-demand by the `TranscriptFetcherAgent` using `StdioServerParameters`. You don't need to run it separately
    For standalone testing of the MCP script:
    ```bash
    # From the backend directory
    python fastapi_build/mcp_servers/youtube_transcript_mcp_server.py
    ```

## Running with Docker

1.  **Ensure your `.env` file is created** in the project root with your `GOOGLE_API_KEY`.

2.  **Build and Run using Docker Compose:**
    From the project root directory (`code-agent-gemini/`):
    ```bash
    docker-compose up --build
    ```
    To run in detached mode:
    ```bash
    docker-compose up --build -d
    ```

3.  **To Stop Docker Compose:**
    ```bash
    docker-compose down
    ```

## API Endpoints

> **Note:**: This is if you are intending to run the FastAPI server. if that is the case,
> please head to the OpenAPI through appending `/docs` to the `http://localhost:8000`

* **`POST /process-video/`**:
    * Processes a YouTube video.
    * **Request Body (JSON):**
        ```json
        {
          "video_url": "[https://www.youtube.com/watch?v=your_video_id](https://www.youtube.com/watch?v=your_video_id)",
          "user_id": "optional_user_identifier",
          "session_id": "optional_session_to_continue"
        }
        ```
    * **Response Body (JSON):**
        ```json
        {
          "session_id": "string",
          "summary": "string | null",
          "fact_check_report": "string | null",
          "full_agent_output": "string",
          "error": "string | null",
          "grounding_html_content": "string | null"
        }
        ```

* **`GET /`**:
    * Welcome message.

## Running Tests

Unit tests are located in `backend/tests/`.

1.  Ensure development dependencies are installed (see Backend Setup).
2.  From the `backend` directory (with virtual environment activated):
    ```bash
    pytest
    ```
    Or from the project root:
    ```bash
    pytest backend/tests
    ```


## Google Agent Development Kit (ADK)

This project heavily utilizes the [Google Agent Development Kit (ADK)](https://developers.google.com/ai/agent-builder/docs) to structure and run the AI agents. Key ADK components used:
* `BaseAgent`, `LlmAgent`, `SequentialAgent`, `LoopAgent`
* `Runner` for executing agents.
* `InMemorySessionService` for session management.
* `FunctionTool` and `MCPToolset` for integrating external capabilities like transcript fetching and Google Search.

## Pre-commit Hooks

This project uses `pre-commit` for code quality. To set it up:
```bash
pip install pre-commit
pre-commit install
