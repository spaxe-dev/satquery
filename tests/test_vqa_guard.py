"""
Regression tests for VLM output-quality guard (issue: smoke LoRA prompt-echo/gibberish).

- valid VLM output is retained (not flagged)
- prompt-echo / underscore / degenerate output falls back to heuristic
- fallback is marked heuristic and does not claim VLM
- orchestrator is_heuristic metadata accurately reflects real vs fallback
- DEMO 4 compound behavior unchanged (no VLM, heuristic)
"""
import os, sys, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image
from pathlib import Path

from tools import vqa_tool
from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis


def make_img(w=120, h=120, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)


# --- Guard unit tests ---

def test_guard_rejects_prompt_echo():
    q = "Use the optical and SAR images together to identify built-up and water-covered regions."
    garbage = "use the optical and sar images together to identify built - up and water - covered regions.s"
    is_invalid, reason = vqa_tool._is_invalid_generation(q, garbage)
    assert is_invalid is True, f"echo should be invalid, got ok reason={reason}"
    assert "echo" in reason or "ratio" in reason


def test_guard_rejects_underscore_garbage():
    q = "Describe the land-cover and major objects visible in this image."
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _ _ _ _ _ _ _ _ _ _ _ _"
    is_invalid, reason = vqa_tool._is_invalid_generation(q, garbage)
    assert is_invalid is True, f"underscore garbage should be invalid, got {reason}"
    assert "underscore" in reason


def test_guard_rejects_short_degenerate():
    q = "Describe the land cover"
    for garbage in ["", "   ", "_", "....", "a"]:
        is_invalid, reason = vqa_tool._is_invalid_generation(q, garbage)
        assert is_invalid is True, f"degenerate '{garbage}' should be invalid"


def test_guard_accepts_valid_landcover():
    q = "Describe the land-cover in this Sentinel-2 image."
    valid = "This Sentinel-2 image (BigEarthNet, mini) contains: Pastures, Agro-forestry areas."
    is_invalid, reason = vqa_tool._is_invalid_generation(q, valid)
    assert is_invalid is False, f"valid caption flagged invalid: {reason}"
    # also a valid RS caption with vegetation phrasing
    valid2 = "This satellite image shows significant vegetation / agricultural or forested cover, notably dark tones suggesting water or shadowed terrain, smooth homogeneous texture suggesting water or uniform fields. Observed resolution ~512x512px; scene appears to contain a mix of natural and anthropogenic features typical of mid-resolution optical imagery."
    is_invalid2, _ = vqa_tool._is_invalid_generation(q, valid2)
    assert is_invalid2 is False


def test_guard_accepts_valid_vqa_yes():
    q = "Is there water in this image?"
    valid = "Yes — dark, low-reflectance regions covering ~15.2% of the scene are consistent with water bodies (rivers/lakes)."
    is_invalid, _ = vqa_tool._is_invalid_generation(q, valid)
    assert is_invalid is False


# --- Integration: analyze with mocked VLM ---

def test_valid_vlm_output_retained(monkeypatch):
    img = make_img(seed=42)
    q = "Describe the land-cover in this Sentinel-2 image."
    valid = "This Sentinel-2 image (BigEarthNet, mini) contains: Pastures, Coniferous forest, Water bodies."
    # Force VLM path
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda arr, query: valid)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    res = vqa_tool.analyze(q, img, task="captioning")
    assert res["answer"] == valid
    assert res["confidence"]["is_heuristic"] is False
    assert "LoRA" in res["model_used"]
    assert "heuristic" not in res["model_used"].lower() or "blip" in res["model_used"].lower()


def test_gibberish_falls_back_to_heuristic(monkeypatch):
    img = make_img(seed=7)
    q = "Use the optical and SAR images together to identify built-up and water-covered regions."
    garbage = "use the optical and sar images together to identify built - up and water - covered regions.s"
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda arr, query: garbage)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    res = vqa_tool.analyze(q, img, task="vqa")
    # Must NOT return garbage
    assert res["answer"] != garbage
    assert res["confidence"]["is_heuristic"] is True
    assert "heuristic" in res["model_used"].lower()
    # heuristic fallback contains RS vision phrasing
    assert "This satellite image shows" in res["answer"] or "Query:" in res["answer"]


def test_underscore_gibberish_falls_back(monkeypatch):
    img = make_img(seed=8)
    q = "Describe the land-cover and major objects visible in this image."
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _ _ _ _ _"
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda arr, query: garbage)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    res = vqa_tool.analyze(q, img, task="captioning")
    assert res["answer"] != garbage
    assert res["confidence"]["is_heuristic"] is True
    assert "_ _" not in res["answer"]


# --- Orchestrator is_heuristic metadata ---

def test_orchestrator_is_heuristic_reflects_valid_vlm(monkeypatch):
    # Single VQA with valid VLM -> is_heuristic False
    q = "Describe the land-cover in this Sentinel-2 image."
    valid = "This Sentinel-2 image (BigEarthNet, mini) contains: Pastures, Broad-leaved forest."
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda arr, query: valid)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    img = make_img(seed=10)
    b, arr = _png_bytes_from_array(img)
    res_meta = validate_image_file(b, "test.png")
    datas = [{"bytes": b, "filename": "test.png", "array": res_meta["array"], "meta": res_meta["meta"]}]
    r = run_analysis(q, datas)
    assert r["task"] == "captioning"
    assert r["confidence"]["is_heuristic"] is False  # fixed bug: was always True
    assert r["answer"] == valid

    # Reset monkeypatch for next case: garbage -> heuristic
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _"
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda arr, query: garbage)
    r2 = run_analysis(q, datas)
    assert r2["confidence"]["is_heuristic"] is True
    assert r2["answer"] != garbage
    assert "This satellite image shows" in r2["answer"]


def test_demo4_unchanged():
    # DEMO 4 hero: T1, T2, SAR — compound_temporal_sar, heuristic pipeline unchanged
    base = Path("data/demo")
    files = [
        base / "temporal_pairs" / "flood_T1.png",
        base / "temporal_pairs" / "flood_T2.png",
        base / "optical_sar_pairs" / "flood_sar.png",
    ]
    for f in files:
        assert f.exists(), f"DEMO4 file missing {f}"
    datas = []
    for f in files:
        b = f.read_bytes()
        res = validate_image_file(b, f.name)
        assert res["valid"], f"{f} invalid {res['reason']}"
        datas.append({"bytes": b, "filename": f.name, "array": res["array"], "meta": res["meta"]})
    q = "What changed between the two dates, and can the SAR imagery corroborate the change?"
    r = run_analysis(q, datas)
    assert r["task"] == "compound_temporal_sar"
    assert r["refused"] is False
    # Must contain temporal description
    assert "Temporal:" in r["answer"]
    # SAR corroboration present
    assert "SAR" in r["answer"]
    # Confidence is heuristic (no VLM involved)
    assert r["confidence"]["is_heuristic"] is True
    # No garbage underscore
    assert "_ _" not in r["answer"]


def _png_bytes_from_array(arr: np.ndarray):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr
