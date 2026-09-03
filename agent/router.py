"""
SatQuery Agentic Router — deterministic-first, auditable.
Classifies query + image config -> selects tools -> execution plan.
Only observable trace is evaluated; no hidden chain-of-thought.
"""
import re
import yaml
import os
from typing import List, Dict, Any, Tuple

TOOL_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "tool_registry.yaml")

# Keywords for intent classification
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
    intents = []
    for intent, patterns in PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                intents.append(intent)
                break
    if not intents:
        intents.append("vqa")
    return intents

def decide_task(intents: List[str], num_images: int, modalities: List[str], query: str) -> str:
    q = query.lower()
    has_temporal_keywords = any(k in intents for k in ["temporal"])
    has_sar = any(m=="sar" for m in modalities)
    has_optical = any(m=="optical" for m in modalities)
    is_compound = has_temporal_keywords and ("sar" in intents or "optical_sar" in intents or (has_sar and num_images>=3))

    if is_compound and num_images >= 2:
        return "compound_temporal_sar"
    if has_temporal_keywords and num_images >= 2:
        return "temporal_change"
    if ("optical_sar" in intents or "sar" in intents) and num_images >=2 and has_sar:
        # If query explicitly mentions optical+sar joint
        if has_optical and has_sar:
            return "optical_sar_fusion"
        if num_images==2 and any("together" in q or "both" in q or "joint" in q for _ in [1]):
            return "optical_sar_fusion"
        # fallback: if 2 images with mixed modalities, treat as fusion
        if num_images==2:
            return "optical_sar_fusion"
    if "grounding" in intents:
        return "grounding"
    if "captioning" in intents:
        return "captioning"
    # single image default
    if num_images==1:
        # captioning vs vqa disambiguation
        if "describe" in q or "land-cover" in q or "land cover" in q:
            return "captioning"
        if "where" in q or "highlight" in q:
            return "grounding"
        return "single_vqa"
    # 2 images without clear intent -> check modalities
    if num_images==2:
        if has_optical and has_sar:
            return "optical_sar_fusion"
        # assume temporal if two images same modality
        if len(set(modalities))==1:
            return "temporal_change"
    return "single_vqa"

def plan_tools(task: str, num_images: int, modalities: List[str]) -> List[Dict[str,Any]]:
    plan = []
    if task == "single_vqa":
        plan.append({"tool_id": "satquery-rs-vlm", "reason": "Single-image VQA requested", "params": {"task": "vqa"}})
    elif task == "captioning":
        plan.append({"tool_id": "satquery-rs-vlm", "reason": "Captioning / scene description requested", "params": {"task": "captioning"}})
    elif task == "grounding":
        plan.append({"tool_id": "satquery-rs-vlm", "reason": "Text-guided grounding requested", "params": {"task": "grounding"}})
    elif task == "temporal_change":
        plan.append({"tool_id": "change-detector", "reason": "Bi-temporal change analysis required", "params": {"threshold": "otsu"}})
        # also add VQA for change QA
        plan.append({"tool_id": "satquery-rs-vlm", "reason": "Change-related VQA supplement", "params": {"task": "vqa", "context": "change"}})
    elif task == "optical_sar_fusion":
        plan.append({"tool_id": "optical-analyzer", "reason": "Optical branch of cross-modal pair", "params": {}})
        plan.append({"tool_id": "sar-analyzer", "reason": "SAR branch of cross-modal pair", "params": {}})
        plan.append({"tool_id": "fusion-engine", "reason": "Cross-modal evidence fusion & agreement scoring", "params": {}})
    elif task == "compound_temporal_sar":
        plan.append({"tool_id": "change-detector", "reason": "Temporal change component of compound query", "params": {}})
        plan.append({"tool_id": "sar-analyzer", "reason": "SAR corroboration component", "params": {}})
        plan.append({"tool_id": "fusion-engine", "reason": "Temporal-SAR evidence fusion (where applicable)", "params": {"mode": "compound"}})
        # For 3-image hero demo: T1,T2 as temporal, SAR as third
    return plan

def validate_inputs(num_images: int, modalities: List[str], task: str, metas: List[Dict]) -> Tuple[bool, str, List[str]]:
    warnings = []
    # task vs num_images sanity
    if task in ("temporal_change", "compound_temporal_sar") and num_images < 2:
        return False, "Temporal change requires at least 2 images (T1 and T2) but only {} provided".format(num_images), warnings
    if task == "optical_sar_fusion" and num_images < 2:
        return False, "Optical-SAR fusion requires 2 images (optical + SAR) but only {} provided".format(num_images), warnings
    if task == "compound_temporal_sar" and num_images < 2:
        return False, "Compound temporal+SAR query requires at least 2 images", warnings
    # For temporal, check dimensions if metas available
    if task in ("temporal_change", "compound_temporal_sar") and len(metas)>=2:
        m1, m2 = metas[0], metas[1]
        # Use validation helper logic inline
        if abs(m1["width"]-m2["width"]) > max(m1["width"], m2["width"])*0.15 or abs(m1["height"]-m2["height"]) > max(m1["height"], m2["height"])*0.15:
            return False, f"Images cannot be established as a valid bi-temporal pair: dimension mismatch {m1['width']}x{m1['height']} vs {m2['width']}x{m2['height']}", warnings
    # optical-sar: warn but don't reject on modality
    if task=="optical_sar_fusion" and len(set(modalities))==1 and "unknown" not in modalities:
        warnings.append(f"Both images detected as {modalities[0]}; cross-modal analysis will proceed but corroboration will be limited")
    return True, "ok", warnings

def load_registry():
    try:
        with open(TOOL_REGISTRY_PATH, "r") as f:
            return yaml.safe_load(f)
    except:
        return {"tools": []}
