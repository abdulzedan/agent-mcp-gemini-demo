# backend/fastapi_build/__init__.py

# (your existing imports)
from .core.config import settings

# Ensure the `agent` submodule is bound on the package
# so that `fastapi_build.agent` actually exists
from . import agent  # noqa

# (you can still expose root_agent here if you like
#  but ADK will pick it up via fastapi_build.agent.root_agent)
