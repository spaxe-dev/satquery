"""SatQuery Core — stable contracts for the entire platform."""
from packages.core.schemas import (
    Modality,
    TaskType,
    RasterMetadata,
    InputImage,
    AnalysisRequest,
    ToolRequest,
    ToolResult,
    Evidence,
    Confidence,
    ExecutionStep,
    AnalysisPlan,
    AnalysisResult,
    BoundingBox,
)

__all__ = [
    "Modality", "TaskType", "RasterMetadata", "InputImage",
    "AnalysisRequest", "ToolRequest", "ToolResult", "Evidence",
    "Confidence", "ExecutionStep", "AnalysisPlan", "AnalysisResult", "BoundingBox",
]
