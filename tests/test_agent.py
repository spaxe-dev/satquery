import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.router import classify_query, decide_task, plan_tools, validate_inputs

def test_classify_temporal():
    intents = classify_query("What changed between these two dates?")
    assert "temporal" in intents

def test_classify_optical_sar():
    intents = classify_query("Use optical and SAR together to identify water")
    assert "optical_sar" in intents or "sar" in intents

def test_decide_temporal():
    task = decide_task(["temporal"], 2, ["optical","optical"], "What changed?")
    assert task=="temporal_change"

def test_decide_fusion():
    task = decide_task(["optical_sar"], 2, ["optical","sar"], "Use both together")
    assert task=="optical_sar_fusion"

def test_decide_compound():
    task = decide_task(["temporal","sar"], 3, ["optical","optical","sar"], "What changed and can SAR corroborate?")
    assert task=="compound_temporal_sar"

def test_plan_temporal():
    plan = plan_tools("temporal_change", 2, ["optical","optical"])
    assert any(p["tool_id"]=="change-detector" for p in plan)

def test_plan_fusion():
    plan = plan_tools("optical_sar_fusion", 2, ["optical","sar"])
    assert any(p["tool_id"]=="fusion-engine" for p in plan)

def test_validate_temporal_fail():
    ok, reason, w = validate_inputs(1, ["optical"], "temporal_change", [{"width":512,"height":512}])
    assert ok==False

def test_validate_ok():
    ok, r, w = validate_inputs(1, ["optical"], "single_vqa", [{"width":512,"height":512}])
    assert ok==True
