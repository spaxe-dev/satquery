"""Compatibility shim: single authoritative planner lives in packages.agent.planner.

This module re-exports it with legacy string-task/dict-plan shapes so existing
callers keep working while logic has ONE source of truth.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from packages.agent import planner as _p
from packages.core.schemas import TaskType

PATTERNS = _p.PATTERNS

_REASONS = {
    "satquery-rs-vlm": "Single-image VLM analysis",
    "change-detector": "Bi-temporal change analysis required",
    "optical-analyzer": "Optical branch of cross-modal pair",
    "sar-analyzer": "SAR branch of cross-modal pair",
    "fusion-engine": "Cross-modal evidence fusion & agreement scoring",
}


def classify_query(query: str) -> List[str]:
    return _p.classify_query(query)


def decide_task(intents: List[str], num_images: int, modalities: List[str], query: str) -> str:
    return _p.decide_task(intents, num_images, modalities, query).value


def plan_tools(task: str, num_images: int, modalities: List[str]) -> List[Dict[str, Any]]:
    t = TaskType(task) if isinstance(task, str) else task
    steps = _p.plan_tools(t, num_images, modalities)
    out = []
    for s in steps:
        out.append({"tool_id": s.tool_id, "reason": _REASONS.get(s.tool_id, s.tool_id), "params": dict(s.params)})
    # Restore legacy per-task reasons/params
    if t == TaskType.TEMPORAL_CHANGE and len(out) == 2:
        out[0] = {"tool_id": "change-detector", "reason": "Bi-temporal change analysis required", "params": {"threshold": "otsu"}}
        out[1] = {"tool_id": "satquery-rs-vlm", "reason": "Change-related VQA supplement", "params": {"task": "vqa", "context": "change"}}
    elif t == TaskType.COMPOUND_TEMPORAL_SAR and len(out) == 3:
        out[2] = {"tool_id": "fusion-engine", "reason": "Temporal-SAR evidence fusion (where applicable)", "params": {"mode": "compound"}}
    return out


def validate_inputs(num_images: int, modalities: List[str], task: str, metas: List[Dict]) -> Tuple[bool, str, List[str]]:
    t = TaskType(task) if isinstance(task, str) else task
    return _p.QueryPlanner().validate(num_images, modalities, t, metas)


def load_registry():
    try:
        import yaml, os
        path = os.path.join(os.path.dirname(__file__), "..", "config", "tool_registry.yaml")
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {"tools": []}
