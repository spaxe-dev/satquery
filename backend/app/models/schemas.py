from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class TaskType(str, Enum):
    SINGLE_VQA = "single_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    TEMPORAL_CHANGE = "temporal_change"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    COMPOUND_TEMPORAL_SAR = "compound_temporal_sar"
    INVALID = "invalid"

class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"

class ValidationResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    warnings: List[str] = []
    details: Dict[str, Any] = {}

class ImageMeta(BaseModel):
    width: int
    height: int
    bands: int
    format: str
    modality: Modality
    crs: Optional[str] = None
    transform: Optional[str] = None
    dtype: Optional[str] = None
    has_georef: bool = False

class ConfidenceBreakdown(BaseModel):
    input_quality: float
    model_confidence: float
    evidence_agreement: float
    cross_modal_support: Optional[float] = None
    overall: float
    is_heuristic: bool = True

class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float

class ExecutionStep(BaseModel):
    step: int
    action: str
    detail: str
    status: str  # ok, warn, error, skipped
    tool: Optional[str] = None
    duration_ms: Optional[int] = None

class ToolSelection(BaseModel):
    tool_id: str
    tool_name: str
    reason: str
    params: Dict[str, Any] = {}

class AnalysisResult(BaseModel):
    task: TaskType
    query: str
    answer: str
    caption: Optional[str] = None
    grounding_boxes: List[BoundingBox] = []
    change_description: Optional[str] = None
    change_area_pct: Optional[float] = None
    fused_statement: Optional[str] = None
    optical_evidence: Optional[Dict[str, Any]] = None
    sar_evidence: Optional[Dict[str, Any]] = None
    fusion_evidence: Optional[Dict[str, Any]] = None
    change_map_b64: Optional[str] = None
    optical_overlay_b64: Optional[str] = None
    sar_overlay_b64: Optional[str] = None
    fused_overlay_b64: Optional[str] = None
    confidence: ConfidenceBreakdown
    execution_trace: List[ExecutionStep]
    tools_used: List[ToolSelection]
    validation: ValidationResult
    timestamp: str
    report_id: str
