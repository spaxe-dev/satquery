"""
SpecialistTool contract — every tool must implement this.
Registry selects tools by capability, not by filename.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time
import numpy as np

class ToolCapability(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    TEMPORAL_CHANGE = "temporal_change"
    OPTICAL_WATER = "optical_water"
    OPTICAL_BUILT = "optical_built"
    OPTICAL_VEGETATION = "optical_vegetation"
    SAR_WATER = "sar_water"
    SAR_BUILT = "sar_built"
    CROSS_MODAL_FUSION = "cross_modal_fusion"
    EVIDENCE_AGREEMENT = "evidence_agreement"

@dataclass
class ToolSpec:
    id: str
    name: str
    version: str
    capabilities: List[ToolCapability]
    modalities: List[str]
    min_images: int
    max_images: int
    description: str
    module: str

class SpecialistTool(ABC):
    @property
    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Must return {answer, confidence, evidence, elapsed_ms, provenance} — never raise on valid input."""

    def health(self) -> Dict[str, Any]:
        return {"id": self.spec.id, "version": self.spec.version, "status": "ready", "capabilities": [c.value for c in self.spec.capabilities]}

    def validate_inputs(self, images: List[np.ndarray]) -> tuple[bool, str]:
        if len(images) < self.spec.min_images:
            return False, f"{self.spec.id} requires at least {self.spec.min_images} image(s), got {len(images)}"
        if len(images) > self.spec.max_images:
            return False, f"{self.spec.id} accepts at most {self.spec.max_images} image(s), got {len(images)}"
        return True, "ok"

# Concrete wrappers around existing tools (adapter pattern — no rewrite)
class VQATool(SpecialistTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id="satquery-rs-vlm", name="SatQuery-RS-VLM", version="0.3.0",
            capabilities=[ToolCapability.VQA, ToolCapability.CAPTIONING, ToolCapability.GROUNDING],
            modalities=["optical", "sar"], min_images=1, max_images=1,
            description="Vision-language for VQA/captioning/grounding (heuristic + LoRA-swappable)",
            module="tools.vqa_tool"
        )
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from tools.vqa_tool import analyze
        task = (params or {}).get("task", "vqa")
        return analyze(query, images[0], task=task)

class ChangeTool(SpecialistTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id="change-detector", name="ChangeDetector", version="0.3.0",
            capabilities=[ToolCapability.TEMPORAL_CHANGE],
            modalities=["bi_temporal"], min_images=2, max_images=2,
            description="Classical CV change detection (LAB+Otsu+morphology)",
            module="tools.change_tool"
        )
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from tools.change_tool import analyze
        return analyze(images[0], images[1], query)

class OpticalTool(SpecialistTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id="optical-analyzer", name="OpticalAnalyzer", version="0.3.0",
            capabilities=[ToolCapability.OPTICAL_WATER, ToolCapability.OPTICAL_BUILT, ToolCapability.OPTICAL_VEGETATION],
            modalities=["optical"], min_images=1, max_images=1,
            description="NDVI/NDWI proxy + HSV + edge density for optical",
            module="tools.optical_tool"
        )
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from tools.optical_tool import analyze
        return analyze(images[0], query)

class SARTool(SpecialistTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id="sar-analyzer", name="SARAnalyzer", version="0.3.0",
            capabilities=[ToolCapability.SAR_WATER, ToolCapability.SAR_BUILT],
            modalities=["sar"], min_images=1, max_images=1,
            description="Backscatter stats + speckle CV for SAR",
            module="tools.sar_tool"
        )
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from tools.sar_tool import analyze
        return analyze(images[0], query)

class FusionTool(SpecialistTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            id="fusion-engine", name="FusionEngine", version="0.3.0",
            capabilities=[ToolCapability.CROSS_MODAL_FUSION, ToolCapability.EVIDENCE_AGREEMENT],
            modalities=["optical_sar_pair"], min_images=2, max_images=2,
            description="Decision-level IoU/Dice fusion",
            module="tools.fusion_tool"
        )
    def execute(self, images: List[np.ndarray], query: str = "", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Fusion requires optical_result + sar_result dicts, not raw images.
        # Return explicit failure instead of raising so registry callers stay total.
        return {"answer": "Fusion requires optical_result + sar_result; call fusion_tool.fuse() via orchestrator", "success": False, "error": "fusion_requires_prior_results", "confidence": None, "evidence": [], "elapsed_ms": 0, "provenance": {"tool_id": "fusion-engine", "heuristic": True}}

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, SpecialistTool] = {}
        for t in [VQATool(), ChangeTool(), OpticalTool(), SARTool(), FusionTool()]:
            self._tools[t.spec.id] = t

    def get(self, tool_id: str) -> Optional[SpecialistTool]:
        return self._tools.get(tool_id)

    def get_for_capability(self, cap: ToolCapability) -> List[SpecialistTool]:
        return [t for t in self._tools.values() if cap in t.spec.capabilities]

    def list(self) -> List[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def health(self) -> Dict[str, Any]:
        return {tid: t.health() for tid, t in self._tools.items()}

_registry: Optional[ToolRegistry] = None

def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
