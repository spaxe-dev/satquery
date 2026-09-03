"""
Tests for explicit VLM lifecycle metadata (vlm_attempted, vlm_loaded, vlm_generation_valid, is_heuristic, fallback_reason)
and that the guard is not weakened.

Both accepted-VLM and rejected-VLM paths are covered without GPU.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from tools import vqa_tool
from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis
from backend.app.main import app

client = TestClient(app)


def _make_img(w=64, h=64, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)


def _png_bytes(arr: np.ndarray):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_vqa_metadata_accepted_path(monkeypatch):
    """Valid BLIP generation accepted -> vlm_attempted True, loaded True, generation_valid True, is_heuristic False, fallback None"""
    img = _make_img(seed=1)
    q = "Describe the land-cover in this Sentinel-2 image."
    valid = "This Sentinel-2 image (BigEarthNet, mini) contains: Pastures, Agro-forestry areas."

    # Reset globals and force VLM success
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: valid)

    res = vqa_tool.analyze(q, img, task="captioning")
    assert res["vlm_attempted"] is True
    assert res["vlm_loaded"] is True
    assert res["vlm_generation_valid"] is True
    assert res["is_heuristic"] is False
    assert res["fallback_reason"] is None
    assert res["confidence"]["is_heuristic"] is False
    assert res["answer"] == valid
    # ensure not exposing stack trace
    assert res["fallback_reason"] is None

    # orchestrator should propagate same and add trace
    b = _png_bytes(img)
    meta = validate_image_file(b, "test.png")
    datas = [{"bytes": b, "filename": "test.png", "array": meta["array"], "meta": meta["meta"]}]
    r = run_analysis(q, datas)
    assert r["vlm_attempted"] is True
    assert r["vlm_loaded"] is True
    assert r["vlm_generation_valid"] is True
    assert r["is_heuristic"] is False
    assert r["fallback_reason"] is None
    assert r["confidence"]["is_heuristic"] is False
    # trace must contain VLM status accepted
    trace_text = " ".join(t["detail"] for t in r["execution_trace"])
    assert "VLM: attempted" in trace_text
    assert "accepted" in trace_text
    assert "generated -> accepted" in trace_text


def test_vqa_metadata_rejected_path_guard(monkeypatch):
    """Garbage BLIP generation rejected by guard -> vlm_generation_valid False, is_heuristic True, fallback_reason generation_rejected, trace shows rejected -> fallback"""
    img = _make_img(seed=2)
    q = "Describe the land-cover and major objects visible in this image."
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _ _ _ _ _"

    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: garbage)

    res = vqa_tool.analyze(q, img, task="captioning")
    assert res["vlm_attempted"] is True
    assert res["vlm_loaded"] is True
    assert res["vlm_generation_valid"] is False
    assert res["is_heuristic"] is True
    assert res["fallback_reason"] is not None
    assert res["fallback_reason"].startswith("generation_rejected:")
    assert "underscore_artifact" in res["fallback_reason"]
    assert res["answer"] != garbage
    assert "_ _" not in res["answer"]
    # no stack trace leaked
    assert "Traceback" not in res["fallback_reason"]
    assert res["confidence"]["is_heuristic"] is True

    # orchestrator trace
    b = _png_bytes(img)
    meta = validate_image_file(b, "test.png")
    datas = [{"bytes": b, "filename": "test.png", "array": meta["array"], "meta": meta["meta"]}]
    r = run_analysis(q, datas)
    assert r["vlm_generation_valid"] is False
    assert r["is_heuristic"] is True
    assert r["fallback_reason"].startswith("generation_rejected:")
    trace_text = " ".join(t["detail"] for t in r["execution_trace"])
    assert "VLM: attempted" in trace_text
    assert "rejected" in trace_text
    assert "heuristic fallback" in trace_text


def test_vqa_metadata_checkpoint_not_found(monkeypatch):
    """No checkpoint on disk -> checkpoint_not_found, vlm_loaded False, generation_valid None"""
    img = _make_img(seed=3)
    q = "Describe the land-cover"
    # Reset globals to simulate fresh start with no checkpoint
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", False, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", None, raising=False)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: None)

    res = vqa_tool.analyze(q, img, task="captioning")
    assert res["vlm_attempted"] is True
    assert res["vlm_loaded"] is False
    assert res["vlm_generation_valid"] is None
    assert res["is_heuristic"] is True
    assert res["fallback_reason"] == "checkpoint_not_found"
    assert "This satellite image shows" in res["answer"]


def test_vqa_metadata_model_load_error(monkeypatch):
    """Checkpoint present but load fails -> model_load_error"""
    img = _make_img(seed=4)
    q = "Describe the land-cover"
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", False, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", None, raising=False)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: False)

    res = vqa_tool.analyze(q, img, task="vqa")
    assert res["vlm_attempted"] is True
    assert res["vlm_loaded"] is False
    assert res["vlm_generation_valid"] is None
    assert res["fallback_reason"] == "model_load_error"
    assert res["is_heuristic"] is True


def test_vqa_metadata_generation_error_safe(monkeypatch):
    """Generation throws exception -> generation_error with type only, no path leak"""
    img = _make_img(seed=5)
    q = "What is visible?"
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    def _boom(a, query):
        raise RuntimeError("/sensitive/path/model.bin failed")

    monkeypatch.setattr(vqa_tool, "_vqa_with_model", _boom)

    res = vqa_tool.analyze(q, img, task="vqa")
    assert res["vlm_generation_valid"] is False
    assert res["fallback_reason"] == "generation_error: RuntimeError"
    # ensure sensitive path not leaked
    assert "/sensitive" not in res["fallback_reason"]
    assert "model.bin" not in res["fallback_reason"]
    assert "Traceback" not in res["fallback_reason"]


def test_vlm_metadata_exposed_via_api(monkeypatch):
    """API /api/analyze exposes vlm_* fields and never leaks stack trace"""
    # Mock valid path via API (using monkeypatch on vqa_tool as used by orchestrator)
    valid = "This Sentinel-2 image (BigEarthNet, mini) contains: Pastures."
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: valid)

    img = _make_img(seed=6)
    b = _png_bytes(img)
    resp = client.post("/api/analyze", data={"query": "Describe the land-cover"}, files={"images": ("test.png", b, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "vlm_attempted" in data
    assert "vlm_loaded" in data
    assert "vlm_generation_valid" in data
    assert "fallback_reason" in data
    assert "is_heuristic" in data
    assert data["vlm_attempted"] is True
    assert data["vlm_generation_valid"] is True
    assert data["is_heuristic"] is False
    assert data["fallback_reason"] is None
    # trace shows accepted
    trace = " ".join(t["detail"] for t in data["execution_trace"])
    assert "VLM:" in trace

    # Now rejected path via API
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _"
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: garbage)
    resp2 = client.post("/api/analyze", data={"query": "Describe the land-cover and major objects visible in this image."}, files={"images": ("test2.png", b, "image/png")})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["vlm_generation_valid"] is False
    assert data2["is_heuristic"] is True
    assert data2["fallback_reason"].startswith("generation_rejected:")
    assert "_ _" not in data2["answer"]
    trace2 = " ".join(t["detail"] for t in data2["execution_trace"])
    assert "rejected" in trace2
    assert "fallback" in trace2


def test_guard_not_weakened_still_rejects_echo(monkeypatch):
    """Ensure guard still flags classic echo even with new metadata plumbing"""
    q = "Use the optical and SAR images together to identify built-up and water-covered regions."
    garbage = "use the optical and sar images together to identify built - up and water - covered regions.s"
    is_invalid, _ = vqa_tool._is_invalid_generation(q, garbage)
    assert is_invalid is True

    img = _make_img(seed=7)
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: garbage)
    res = vqa_tool.analyze(q, img, task="vqa")
    assert res["vlm_generation_valid"] is False
    assert "generation_rejected" in res["fallback_reason"]
