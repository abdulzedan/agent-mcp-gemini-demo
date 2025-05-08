# backend/fastapi_build/__init__.py

# (your existing imports)
# Ensure the `agent` submodule is bound on the package
# so that `fastapi_build.agent` actually exists
from . import agent  # noqa
from .core.config import settings  # noqa
