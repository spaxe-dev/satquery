"""apps.api thin FastAPI boundary (target architecture).

Currently re-exports backend.app.main for single-port dev.
Domain logic stays in packages/* + backend services; this layer only wires routes.
"""
from backend.app.main import app  # noqa: F401 re-export

__all__ = ["app"]
