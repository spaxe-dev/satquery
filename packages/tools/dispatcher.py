"""Capability dispatcher: orchestrator calls registry, never concrete modules.

Adding/replacing a tool = register() a SpecialistTool. No orchestrator edit.
Image selection + compatibility live here (not in orchestrator).
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
from packages.tools.base import get_tool_registry
from packages.geospatial.registration import registration_status


def select_images(task: str, tool_id: str, arrays: List[np.ndarray], metas: List[Dict[str, Any]], params: Dict[str, Any]) -> List[np.ndarray]:
    if tool_id == "satquery-rs-vlm":
        if params.get("context") == "change" and len(arrays) >= 2:
            return [arrays[1]]
        return [arrays[0]] if arrays else []
    if tool_id == "change-detector":
        return arrays[:2]
    if tool_id == "optical-analyzer":
        for i, m in enumerate(metas):
            if m.get("modality") == "optical":
                return [arrays[i]]
        return [arrays[0]] if arrays else []
    if tool_id == "sar-analyzer":
        for i, m in enumerate(metas):
            if m.get("modality") == "sar":
                return [arrays[i]]
        if task == "compound_temporal_sar" and len(arrays) >= 3:
            return [arrays[2]]
        if any(m.get("modality") == "unknown" for m in metas) and len(arrays) >= 2:
            return [arrays[1]]
        return [arrays[0]] if arrays else []
    return []


def check_compat(task: str, metas: List[Dict[str, Any]]) -> Tuple[bool, str, List[str], bool]:
    """Returns (ok, reason, warnings, approximate). No silent resize: mismatch refuses or approx."""
    warnings: List[str] = []
    if task in ("temporal_change", "compound_temporal_sar") and len(metas) >= 2:
        m1, m2 = metas[0], metas[1]
        if abs(m1["width"] - m2["width"]) > max(m1["width"], m2["width"]) * 0.15 or abs(m1["height"] - m2["height"]) > max(m1["height"], m2["height"]) * 0.15:
            return False, f"Width/height mismatch: {m1['width']}x{m1['height']} vs {m2['width']}x{m2['height']} — cannot be same area", warnings, False
        reg = registration_status(m1, m2)
        if reg == "unregistered":
            warnings.append("Temporal pair lacks shared georeferencing (no common CRS/bounds); change is pixel-space APPROXIMATE")
            return True, "ok", warnings, True
        if reg == "approximately_aligned":
            warnings.append("Temporal pair approximately aligned (shared dims but no identical georeferencing); APPROXIMATE")
            return True, "ok", warnings, True
        if m1.get("crs") and m2.get("crs") and m1["crs"] != m2["crs"]:
            warnings.append(f"CRS mismatch {m1['crs']} vs {m2['crs']}; APPROXIMATE")
            return True, "ok", warnings, True
    if task == "optical_sar_fusion" and len(metas) >= 2:
        reg = registration_status(metas[0], metas[1])
        if reg == "unregistered":
            warnings.append("Optical/SAR pair unregistered; fusion IoU is pixel-space APPROXIMATE, not geospatial correspondence")
            return True, "ok", warnings, True
        if reg == "approximately_aligned":
            warnings.append("Optical/SAR approximately aligned; APPROXIMATE")
            return True, "ok", warnings, True
    return True, "ok", warnings, False


def dispatch(tool_id: str, images: List[np.ndarray], query: str = "", params: dict | None = None) -> Dict[str, Any]:
    reg = get_tool_registry()
    tool = reg.get(tool_id)
    if tool is None:
        return {"answer": f"Unknown tool {tool_id}", "success": False, "error": "unknown_tool", "confidence": None, "evidence": [], "elapsed_ms": 0, "provenance": {"tool_id": tool_id}}
    ok, reason = tool.validate_inputs(images)
    if not ok:
        return {"answer": reason, "success": False, "error": "invalid_inputs", "confidence": None, "evidence": [], "elapsed_ms": 0, "provenance": {"tool_id": tool_id}}
    # Tiled path for large single-image stats tools (bounded memory, provenance per tile)
    try:
        p = dict(params or {})
        if tool_id in ("optical-analyzer", "sar-analyzer") and images and max(images[0].shape[0], images[0].shape[1]) > 1024:
            meta = p.pop("_meta", {})
            from packages.geospatial.tiling_exec import execute_tiled
            return execute_tiled(tool_id, images[0], meta or {}, query, p, dispatch)
    except Exception:
        pass
    return tool.execute(images, query, params or {})
