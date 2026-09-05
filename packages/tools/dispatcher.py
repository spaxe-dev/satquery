"""Capability dispatcher: orchestrator calls registry, never concrete modules.

Adding/replacing a tool = register() a SpecialistTool. No orchestrator edit.
"""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from packages.tools.base import get_tool_registry


def dispatch(tool_id: str, images: List[np.ndarray], query: str = "", params: dict | None = None) -> Dict[str, Any]:
    reg = get_tool_registry()
    tool = reg.get(tool_id)
    if tool is None:
        return {"answer": f"Unknown tool {tool_id}", "success": False, "error": "unknown_tool", "confidence": None, "evidence": [], "elapsed_ms": 0, "provenance": {"tool_id": tool_id}}
    ok, reason = tool.validate_inputs(images)
    if not ok:
        return {"answer": reason, "success": False, "error": "invalid_inputs", "confidence": None, "evidence": [], "elapsed_ms": 0, "provenance": {"tool_id": tool_id}}
    return tool.execute(images, query, params or {})
