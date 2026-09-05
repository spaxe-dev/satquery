"""
Model abstraction — application never imports 'qwen' or 'blip' directly.
Orchestrator asks: 'which model provides capability X?' not 'call Qwen'.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time
import numpy as np

class ModelCapability(str, Enum):
    CAPTIONING = "captioning"
    VQA = "vqa"
    GROUNDING = "grounding"
    MULTIMODAL_REASONING = "multimodal_reasoning"

@dataclass
class ModelInfo:
    model_id: str
    version: str
    capabilities: List[ModelCapability]
    modalities: List[str]
    is_heuristic: bool
    vram_mb: int
    provenance: Dict[str, Any]

class ModelAdapter(ABC):
    """Stable interface every VLM must implement."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def caption(self, image: np.ndarray, query: Optional[str] = None) -> Dict[str, Any]:
        """Returns {answer, caption, confidence, elapsed_ms}"""

    @abstractmethod
    def vqa(self, image: np.ndarray, query: str) -> Dict[str, Any]: ...

    @abstractmethod
    def ground(self, image: np.ndarray, query: str) -> Dict[str, Any]:
        """Returns {boxes: [{x,y,w,h,label,confidence}]}"""

    def health(self) -> Dict[str, Any]:
        return {"model_id": self.info.model_id, "available": self.is_available(), "is_heuristic": self.info.is_heuristic}

# ---------------------------------------------------------------------------
# Heuristic fallback — always available, CPU-only, importable without torch
# ---------------------------------------------------------------------------
class HeuristicFallbackAdapter(ModelAdapter):
    """Wraps tools.vqa_tool heuristics behind the stable ModelAdapter contract."""

    def __init__(self):
        self._info = ModelInfo(
            model_id="satquery-heuristic-v0",
            version="0.3.0",
            capabilities=[ModelCapability.CAPTIONING, ModelCapability.VQA, ModelCapability.GROUNDING],
            modalities=["optical", "sar"],
            is_heuristic=True,
            vram_mb=0,
            provenance={"type": "heuristic", "base": "classical CV + templates"}
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def is_available(self) -> bool:
        return True

    def caption(self, image: np.ndarray, query: Optional[str] = None) -> Dict[str, Any]:
        from tools.vqa_tool import _heuristic_caption
        t0 = time.time()
        cap = _heuristic_caption(image)
        elapsed = int((time.time() - t0) * 1000)
        h, w = image.shape[0], image.shape[1]
        input_q = 0.88 if min(h, w) >= 256 else 0.72
        model_c = 0.62
        return {
            "answer": cap,
            "caption": cap,
            "boxes": [],
            "confidence": {"input_quality": input_q, "model_confidence": model_c, "evidence_agreement": 0.0, "overall": round(0.5*input_q+0.5*model_c,2), "is_heuristic": True},
            "elapsed_ms": elapsed,
            "model_used": self.info.model_id,
        }

    def vqa(self, image: np.ndarray, query: str) -> Dict[str, Any]:
        from tools.vqa_tool import _heuristic_vqa, _heuristic_caption, _heuristic_grounding
        t0 = time.time()
        ans = _heuristic_vqa(image, query)
        cap = _heuristic_caption(image)
        boxes = _heuristic_grounding(image, query) if any(k in query.lower() for k in ["where","highlight","locate","show","box","water","built","forest"]) else []
        elapsed = int((time.time() - t0)*1000)
        h, w = image.shape[0], image.shape[1]
        input_q = 0.88 if min(h, w) >= 256 else 0.72
        model_c = 0.62
        return {
            "answer": ans, "caption": cap, "boxes": boxes,
            "confidence": {"input_quality": input_q, "model_confidence": model_c, "evidence_agreement": 0.0, "overall": round(0.5*input_q+0.5*model_c,2), "is_heuristic": True},
            "elapsed_ms": elapsed, "model_used": self.info.model_id,
        }

    def ground(self, image: np.ndarray, query: str) -> Dict[str, Any]:
        from tools.vqa_tool import _heuristic_grounding
        t0 = time.time()
        boxes = _heuristic_grounding(image, query)
        elapsed = int((time.time() - t0)*1000)
        return {"answer": f"Grounded {len(boxes)} region(s) for query '{query}'", "boxes": boxes, "elapsed_ms": elapsed, "model_used": self.info.model_id}

# ---------------------------------------------------------------------------
# Registry — selects best available model for a capability
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self):
        self._adapters: Dict[str, ModelAdapter] = {}
        self.register(HeuristicFallbackAdapter())

    def register(self, adapter: ModelAdapter):
        self._adapters[adapter.info.model_id] = adapter

    def get_for(self, capability: ModelCapability) -> ModelAdapter:
        # Prefer non-heuristic if available, else fallback
        candidates = [a for a in self._adapters.values() if capability in a.info.capabilities and a.is_available()]
        if not candidates:
            return self._adapters["satquery-heuristic-v0"]
        # Sort: non-heuristic first, then by version
        candidates.sort(key=lambda a: (a.info.is_heuristic, a.info.model_id))
        return candidates[0]

    def list(self) -> List[ModelInfo]:
        return [a.info for a in self._adapters.values()]

# Singleton
_registry: Optional[ModelRegistry] = None

def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        # Try to register real adapters if checkpoints exist (lazy, never fails)
        try:
            from packages.models.blip_adapter import BlipAdapter
            a = BlipAdapter()
            if a.is_available():
                _registry.register(a)
        except Exception:
            pass
        try:
            from packages.models.qwen_adapter import QwenAdapter
            a = QwenAdapter()
            if a.is_available():
                _registry.register(a)
        except Exception:
            pass
    return _registry
