"""
Core typed contracts for SatQuery AI.
These schemas are the single source of truth.
Agent, tools, models, API, and frontend all depend on these — not on each other's internals.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
import uuid
import datetime

class Modality(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"

class TaskType(str, Enum):
    SINGLE_VQA = "single_vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    TEMPORAL_CHANGE = "temporal_change"
    OPTICAL_SAR_FUSION = "optical_sar_fusion"
    COMPOUND_TEMPORAL_SAR = "compound_temporal_sar"
    INVALID = "invalid"

class RasterMetadata(BaseModel):
    """Geospatial-correct metadata. Replaces ad-hoc dict in validation.py."""
    width: int = Field(..., gt=0, description="Pixels X")
    height: int = Field(..., gt=0, description="Pixels Y")
    bands: int = Field(..., gt=0)
    format: str = Field(..., description=".tif/.png etc")
    modality: Modality = Modality.UNKNOWN
    crs: Optional[str] = Field(None, description="CRS WKT or EPSG string, if georeferenced")
    transform: Optional[str] = Field(None, description="Affine transform string")
    dtype: Optional[str] = None
    has_georef: bool = False
    gsd_m: Optional[float] = Field(None, description="Ground sample distance meters, if known")
    bounds: Optional[List[float]] = Field(None, description="[west,south,east,north] if known")
    nodata: Optional[float] = None
    timestamp: Optional[str] = None

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v):
        if v is not None and len(v) != 4:
            raise ValueError("bounds must be [w,s,e,n]")
        return v

class InputImage(BaseModel):
    """Validated input image with bytes kept outside the schema (passed separately)."""
    image_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    filename: str
    metadata: RasterMetadata
    # array is NOT in schema — carried as numpy outside; preview_b64 is API-only
    preview_b64: Optional[str] = None

class Confidence(BaseModel):
    """Heuristic confidence — always labeled, never calibrated."""
    model_config = {"protected_namespaces": ()}

    input_quality: float = Field(..., ge=0, le=1)
    model_confidence: float = Field(..., ge=0, le=1)
    evidence_agreement: float = Field(..., ge=0, le=1)
    cross_modal_support: Optional[float] = Field(None, ge=0, le=1)
    overall: float = Field(..., ge=0, le=1)
    is_heuristic: bool = True

    def label(self) -> str:
        return "heuristic" if self.is_heuristic else "model"

class BoundingBox(BaseModel):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., gt=0)
    h: int = Field(..., gt=0)
    label: str
    confidence: float = Field(..., ge=0, le=1)

class ExecutionStep(BaseModel):
    step: int
    action: str
    detail: str
    status: Literal["ok", "warn", "error", "skipped"] = "ok"
    tool: Optional[str] = None
    duration_ms: Optional[int] = None
    timestamp: Optional[str] = None

class ToolRequest(BaseModel):
    tool_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    image_ids: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list, description="Step ids this depends on")

class AnalysisPlan(BaseModel):
    """Structured plan — LLM/planner emits this, executor validates it."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task: TaskType
    intents: List[str] = Field(default_factory=list)
    steps: List[ToolRequest] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    valid: bool = True
    refusal_reason: Optional[str] = None

    def tool_ids(self) -> List[str]:
        return [s.tool_id for s in self.steps]

class Evidence(BaseModel):
    """Visual or textual evidence from a tool."""
    evidence_type: Literal["change_map", "water_mask", "built_mask", "overlay", "boxes", "text"] = "text"
    description: Optional[str] = None
    overlay_b64: Optional[str] = None
    boxes: List[BoundingBox] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    tool_id: str
    tool_version: str = "0.3.0"
    success: bool = True
    error: Optional[str] = None
    answer: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: Optional[Confidence] = None
    stats: Dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0
    # Raw masks excluded from API — kept internal but serializable via stats
    provenance: Dict[str, Any] = Field(default_factory=dict)

class AnalysisRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = Field(..., min_length=1, max_length=2000)
    image_ids: List[str] = Field(default_factory=list)
    # Filled by validation
    task_hint: Optional[TaskType] = None

class AnalysisResult(BaseModel):
    request_id: str
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str
    task: TaskType
    answer: str
    caption: Optional[str] = None
    grounding_boxes: List[BoundingBox] = Field(default_factory=list)
    change_description: Optional[str] = None
    change_area_pct: Optional[float] = None
    fused_statement: Optional[str] = None
    confidence: Confidence
    execution_trace: List[ExecutionStep] = Field(default_factory=list)
    tool_results: List[ToolResult] = Field(default_factory=list)
    tools_used: List[Dict[str, Any]] = Field(default_factory=list)
    validation: Dict[str, Any] = Field(default_factory=dict)
    refused: bool = False
    refusal_reason: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    # VLM lifecycle
    vlm_attempted: Optional[bool] = None
    vlm_loaded: Optional[bool] = None
    vlm_generation_valid: Optional[bool] = None
    is_heuristic: bool = True
    fallback_reason: Optional[str] = None
    previews: List[Dict[str, Any]] = Field(default_factory=list)
