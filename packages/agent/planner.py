"""
Agent planner — query -> structured AnalysisPlan -> validated -> executor.
LLM/planner emits AnalysisPlan; executor validates policy before running.
This gives flexibility + reliability (deterministic executor).
"""
from __future__ import annotations
from typing import List, Dict, Any
import re
from packages.core.schemas import TaskType, AnalysisPlan, ToolRequest

PATTERNS = {
    "temporal": [r"what changed", r"change between", r"difference between.*dates?", r"before.*after", r"has.*increased", r"has.*decreased", r"temporal", r"t1.*t2", r"inundat", r"flood"],
    "sar": [r"\bsar\b", r"radar", r"risat", r"sentinel-1", r"backscatter", r"speckle"],
    "optical_sar": [r"optical.*sar", r"sar.*optical", r"both.*images?", r"together", r"joint", r"cross.modal", r"fusion", r"corroborat", r"confirm.*sar"],
    "grounding": [r"where is", r"highlight", r"locate", r"show.*region", r"bounding box", r"water body referred", r"grounding"],
    "captioning": [r"describe.*land.cover", r"describe.*image", r"caption", r"what.*visible", r"major objects"],
    "vqa": [r"what is", r"how many", r"is there", r"are there", r"\?", r"identify"],
}

def classify_query(query: str) -> List[str]:
    q = query.lower()
    intents: List[str] = []
    for intent, patterns in PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                intents.append(intent)
                break
    if not intents:
        intents.append("vqa")
    return intents

def decide_task(intents: List[str], num_images: int, modalities: List[str], query: str) -> TaskType:
    q = query.lower()
    has_temporal = any(k in intents for k in ["temporal"])
    has_sar = any(m == "sar" for m in modalities)
    has_optical = any(m == "optical" for m in modalities)
    is_compound = has_temporal and ("sar" in intents or "optical_sar" in intents or (has_sar and num_images >= 3))

    if is_compound and num_images >= 2:
        return TaskType.COMPOUND_TEMPORAL_SAR
    if has_temporal and num_images >= 2:
        return TaskType.TEMPORAL_CHANGE
    if ("optical_sar" in intents or "sar" in intents) and num_images >= 2 and has_sar:
        if has_optical and has_sar:
            return TaskType.OPTICAL_SAR_FUSION
        if num_images == 2:
            return TaskType.OPTICAL_SAR_FUSION
    if "grounding" in intents:
        return TaskType.GROUNDING
    if "captioning" in intents:
        return TaskType.CAPTIONING
    if num_images == 1:
        if "describe" in q or "land-cover" in q or "land cover" in q:
            return TaskType.CAPTIONING
        if "where" in q or "highlight" in q:
            return TaskType.GROUNDING
        return TaskType.SINGLE_VQA
    if num_images == 2:
        if has_optical and has_sar:
            return TaskType.OPTICAL_SAR_FUSION
        if len(set(modalities)) == 1:
            return TaskType.TEMPORAL_CHANGE
    return TaskType.SINGLE_VQA

def plan_tools(task: TaskType, num_images: int, modalities: List[str]) -> List[ToolRequest]:
    plan: List[ToolRequest] = []
    if task == TaskType.SINGLE_VQA:
        plan.append(ToolRequest(tool_id="satquery-rs-vlm", params={"task": "vqa"}))
    elif task == TaskType.CAPTIONING:
        plan.append(ToolRequest(tool_id="satquery-rs-vlm", params={"task": "captioning"}))
    elif task == TaskType.GROUNDING:
        plan.append(ToolRequest(tool_id="satquery-rs-vlm", params={"task": "grounding"}))
    elif task == TaskType.TEMPORAL_CHANGE:
        plan.append(ToolRequest(tool_id="change-detector", params={"threshold": "otsu"}))
        plan.append(ToolRequest(tool_id="satquery-rs-vlm", params={"task": "vqa", "context": "change"}))
    elif task == TaskType.OPTICAL_SAR_FUSION:
        plan.append(ToolRequest(tool_id="optical-analyzer", params={}))
        plan.append(ToolRequest(tool_id="sar-analyzer", params={}))
        plan.append(ToolRequest(tool_id="fusion-engine", params={}))
    elif task == TaskType.COMPOUND_TEMPORAL_SAR:
        plan.append(ToolRequest(tool_id="change-detector", params={}))
        plan.append(ToolRequest(tool_id="sar-analyzer", params={}))
        plan.append(ToolRequest(tool_id="fusion-engine", params={"mode": "compound"}))
    return plan

class QueryPlanner:
    """Produces validated AnalysisPlan from query + image config."""

    ALLOWED_TOOLS = {"satquery-rs-vlm", "change-detector", "optical-analyzer", "sar-analyzer", "fusion-engine"}

    def plan(self, query: str, num_images: int, modalities: List[str], metas: List[Dict[str, Any]]) -> AnalysisPlan:
        intents = classify_query(query)
        task = decide_task(intents, num_images, modalities, query)
        # Validation
        valid, reason, warnings = self.validate(num_images, modalities, task, metas)
        if not valid:
            return AnalysisPlan(task=task, intents=intents, steps=[], warnings=warnings, valid=False, refusal_reason=reason)
        steps = plan_tools(task, num_images, modalities)
        # Policy validation: only allowed tools, correct image counts
        for s in steps:
            if s.tool_id not in self.ALLOWED_TOOLS:
                return AnalysisPlan(task=task, intents=intents, steps=[], warnings=warnings, valid=False, refusal_reason=f"Tool {s.tool_id} not in registry")
        return AnalysisPlan(task=task, intents=intents, steps=steps, warnings=warnings, valid=True)

    def validate(self, num_images: int, modalities: List[str], task: TaskType, metas: List[Dict[str, Any]]):
        warnings: List[str] = []
        if task in (TaskType.TEMPORAL_CHANGE, TaskType.COMPOUND_TEMPORAL_SAR) and num_images < 2:
            return False, f"Temporal change requires at least 2 images but only {num_images} provided", warnings
        if task == TaskType.OPTICAL_SAR_FUSION and num_images < 2:
            return False, f"Optical-SAR fusion requires 2 images but only {num_images} provided", warnings
        if task in (TaskType.TEMPORAL_CHANGE, TaskType.COMPOUND_TEMPORAL_SAR) and len(metas) >= 2:
            m1, m2 = metas[0], metas[1]
            if abs(m1["width"] - m2["width"]) > max(m1["width"], m2["width"]) * 0.15 or abs(m1["height"] - m2["height"]) > max(m1["height"], m2["height"]) * 0.15:
                return False, f"Images cannot be established as a valid bi-temporal pair: dimension mismatch {m1['width']}x{m1['height']} vs {m2['width']}x{m2['height']}", warnings
        if task == TaskType.OPTICAL_SAR_FUSION and len(set(modalities)) == 1 and "unknown" not in modalities:
            warnings.append(f"Both images detected as {modalities[0]}; cross-modal analysis will proceed but corroboration will be limited")
        return True, "ok", warnings
