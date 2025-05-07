# code-agent-gemini/backend/fastapi_build/core/config.py
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Determine the path to the .env file relative to this config.py file
# This assumes .env is in the 'backend' directory, and config.py is in 'backend/fastapi_build/core/'
env_path = (
    Path(__file__).resolve().parents[2] / ".env"
)  # Moves up two levels from core/ to backend/
print(f"Attempting to load .env file from: {env_path}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(".env file loaded successfully.")
else:
    print(
        f".env file not found at {env_path}. Relying on environment variables directly."
    )


class Settings(BaseSettings):
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "main-env-demo")
    GOOGLE_CLOUD_LOCATION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    GOOGLE_GENAI_USE_VERTEXAI: bool = (
        os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "TRUE").lower() == "true"
    )

    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8080"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ADK Model Name (as per ADK documentation for tools)
    ADK_GEMINI_MODEL: str = (
        "gemini-2.0-flash"  # Sticking to ADK docs recommendation for Google Search etc.
    )

    # Default path for the MCP server script (relative to backend directory)
    # This will be used by StdioServerParameters
    YOUTUBE_MCP_SERVER_SCRIPT_PATH: str = (
        "fastapi_build/mcp_servers/youtube_transcript_mcp_server.py"
    )


settings = Settings()

# Ensure ADK specific environment variables are set for Vertex AI
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings.GOOGLE_CLOUD_PROJECT)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings.GOOGLE_CLOUD_LOCATION)
os.environ.setdefault(
    "GOOGLE_GENAI_USE_VERTEXAI", str(settings.GOOGLE_GENAI_USE_VERTEXAI).upper()
)

# The following is needed for ADK to discover agents if they are in a different module
# and you are using `adk web` or similar ADK CLI tools.
# For programmatic Runner usage like in FastAPI, direct import is typical.
# os.environ.setdefault('ADK_AGENT_MODULE_PATH', 'fastapi_build.agents.youtube_processing_agents')

# Initialize Google Auth for ADK
# This ensures ADC are loaded if not already explicitly handled by the environment.
# Only do this if not running in an environment where it's already handled (e.g. Cloud Run)
# For local dev, `gcloud auth application-default login` is primary.
try:
    import google.auth

    credentials, project_id = google.auth.default()
    if not project_id and settings.GOOGLE_CLOUD_PROJECT:
        project_id = settings.GOOGLE_CLOUD_PROJECT
    # print(f"Google Auth Default: Credentials type: {type(credentials)}, Project ID: {project_id}")
except Exception as e:
    print(
        f"Google Auth Default error: {e}. Ensure ADC or service account is configured."
    )
