"""Contract regression: packages/* must stay importable and total (no raise on valid input)."""
import numpy as np

def test_core_schemas_import():
    from packages.core.schemas import Confidence, AnalysisPlan, TaskType
    c = Confidence(input_quality=0.8, model_confidence=0.6, evidence_agreement=0.5, overall=0.65, is_heuristic=True)
    assert c.label() == "heuristic"

def test_planner_parity_single_and_refusal():
    from packages.agent.planner import QueryPlanner
    p = QueryPlanner()
    ok = p.plan("Describe the land-cover?", 1, ["optical"], [{"width": 512, "height": 512}])
    assert ok.valid and len(ok.steps) == 1
    bad = p.plan("What changed?", 1, ["optical"], [{"width": 512, "height": 512}])
    # single image temporal may still plan single_vqa or refuse; must not crash
    assert bad.task is not None

def test_tool_registry_total():
    from packages.tools.base import get_tool_registry
    import numpy as np
    reg = get_tool_registry()
    assert reg.get("satquery-rs-vlm") is not None
    fusion = reg.get("fusion-engine")
    out = fusion.execute([np.zeros((64, 64, 3), np.uint8)], "q", {})
    assert out["success"] is False and "fusion" in out["error"]

def test_model_registry_fallback():
    from packages.models.adapter import get_model_registry, ModelCapability
    reg = get_model_registry()
    a = reg.get_for(ModelCapability.VQA)
    assert a.is_available()
    assert a.info.is_heuristic is True  # prod without GPU = heuristic

def test_tiling_contract():
    from packages.geospatial.metadata import TilingStrategy
    t = TilingStrategy(2048, 2048)
    assert t.should_tile()
    assert len(t.tiles()) > 1 and len(t.tiles()) <= 64
