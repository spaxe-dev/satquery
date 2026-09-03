"""
Tests that the single-image captioning fallback is genuinely image-dependent.

Fallback must:
- be deterministic and use only pixel-derived evidence (brightness, color, texture, spatial, water/veg/built indicators)
- never use filenames, BigEarthNet labels, or hardcoded demo answers
- produce distinct captions for materially different images
- still fall back safely when BLIP is garbage (guard preserved)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from tools import vqa_tool


def _synthetic_uniform(r, g, b, w=64, h=64):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    # add slight deterministic texture so Laplacian not zero for some
    return arr


def _synthetic_checkerboard(w=64, h=64, block=8):
    # high texture: alternating black/white blocks
    gray = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 0:
                gray[y:y+block, x:x+block] = 30
            else:
                gray[y:y+block, x:x+block] = 210
    arr = np.stack([gray, gray, gray], axis=2)
    return arr


def _synthetic_green_veg(w=80, h=80):
    # vegetation-like: high G, low R/B, moderate texture noise
    rng = np.random.RandomState(42)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(50 + rng.randint(-10, 10, size=(h, w)), 0, 255)
    arr[:, :, 1] = np.clip(170 + rng.randint(-15, 15, size=(h, w)), 0, 255)
    arr[:, :, 2] = np.clip(60 + rng.randint(-10, 10, size=(h, w)), 0, 255)
    return arr


def test_caption_distinct_for_synthetic_variants():
    green = _synthetic_green_veg()
    dark_water = _synthetic_uniform(15, 25, 75, w=80, h=80)
    bright = _synthetic_uniform(195, 190, 165, w=80, h=80)
    textured = _synthetic_checkerboard(w=80, h=80, block=8)
    smooth = _synthetic_uniform(30, 30, 30, w=80, h=80)

    caps = {
        "green": vqa_tool._heuristic_caption(green),
        "dark": vqa_tool._heuristic_caption(dark_water),
        "bright": vqa_tool._heuristic_caption(bright),
        "textured": vqa_tool._heuristic_caption(textured),
        "smooth": vqa_tool._heuristic_caption(smooth),
    }
    # all must contain quantified evidence
    for k, c in caps.items():
        assert "ExG" in c, f"{k} missing ExG"
        assert "mean" in c.lower(), f"{k} missing brightness mean"
        assert "Laplacian var" in c, f"{k} missing texture"
        assert "edge density" in c, f"{k} missing edge"
        assert "quadrant" in c.lower(), f"{k} missing spatial"

    # pairwise distinct
    keys = list(caps.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            assert caps[keys[i]] != caps[keys[j]], f"captions for {keys[i]} and {keys[j]} should differ (was identical generic)"


def test_caption_distinct_for_real_bigearthnet_images():
    base = Path("datasets/raw/bigearthnet/extracted_for_demo")
    # these are the reported problematic pair + two others for coverage
    files = [
        base / "bigearthnet_mini_007_Land_principally_occupied_by_agriculture.png",
        base / "bigearthnet_mini_015_Discontinuous_urban_fabric_Non_irrigated.png",
        base / "bigearthnet_mini_000_Coniferous_forest_Transitional_woodland_.png",
        base / "bigearthnet_mini_025_Sea_and_ocean.png",
    ]
    caps = {}
    for p in files:
        assert p.exists(), f"missing demo image {p}"
        arr = np.array(Image.open(p).convert("RGB"))
        caps[p.name] = vqa_tool._heuristic_caption(arr)

    # 007 and 015 must differ (the reported failure)
    assert caps[files[0].name] != caps[files[1].name], "007 and 015 produced identical generic caption — fallback not image-dependent"
    # all four should be pairwise distinct
    names = list(caps.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            assert caps[names[i]] != caps[names[j]], f"captions for {names[i]} vs {names[j]} identical"


def test_caption_deterministic():
    arr = _synthetic_green_veg()
    c1 = vqa_tool._heuristic_caption(arr)
    c2 = vqa_tool._heuristic_caption(arr.copy())
    assert c1 == c2, "caption must be deterministic for same pixels"


def test_caption_never_uses_filename_or_label():
    # Same pixels with different filenames/labels must give identical caption
    base = Path("datasets/raw/bigearthnet/extracted_for_demo/bigearthnet_mini_007_Land_principally_occupied_by_agriculture.png")
    arr = np.array(Image.open(base).convert("RGB"))
    cap_via_array = vqa_tool._heuristic_caption(arr)
    # ensure caption does not leak filename or label
    assert "007" not in cap_via_array
    assert "Land_principally" not in cap_via_array
    assert "Land principally" not in cap_via_array
    # also ensure it doesn't hardcode BigEarthNet label phrases as hallucinated ground truth
    # (heuristic must describe visual evidence, not claim labels)
    # The fallback should NOT claim to know labels like "BigEarthNet" as content; it may mention resolution but not label list
    # If it contained label-specific sentence like "contains: Pastures" that would be cheating for 007
    assert "007" not in cap_via_array
    # Test via orchestrator/e2e would also use same caption regardless of filename
    # Emulate two different filenames with same bytes via validation
    from backend.app.services.validation import validate_image_file
    import io
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b = buf.getvalue()
    r1 = validate_image_file(b, "a.png")
    r2 = validate_image_file(b, "bigearthnet_mini_007_Land_principally_occupied_by_agriculture.png")
    cap1 = vqa_tool._heuristic_caption(r1["array"])
    cap2 = vqa_tool._heuristic_caption(r2["array"])
    assert cap1 == cap2, "filename must not affect caption"


def test_caption_contains_quantified_evidence():
    arr = _synthetic_uniform(40, 120, 50)
    cap = vqa_tool._heuristic_caption(arr)
    # must expose numeric evidence, not just vague phrases
    assert "RGB" in cap
    assert "NDVI-proxy" in cap
    assert "veg quad-std" in cap.lower() or "veg quad" in cap.lower()
    assert "bright pixels" in cap.lower()
    assert "very dark" in cap.lower()


def test_invalid_blip_still_falls_back_to_image_dependent_caption(monkeypatch):
    # Combine both requirements: invalid BLIP → guard → image-dependent heuristic (not generic same)
    arr_green = _synthetic_green_veg()
    arr_dark = _synthetic_uniform(15, 25, 75)
    q = "Describe the land-cover in this image."
    garbage = "describe the land - cover and major objects visible in this image. _ _ _ _ _ _ _ _"

    # Force guard path
    monkeypatch.setattr(vqa_tool, "_VLM_AVAILABLE", True, raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL", object(), raising=False)
    monkeypatch.setattr(vqa_tool, "_VLM_MODEL_NAME", "Salesforce/blip-image-captioning-base", raising=False)
    monkeypatch.setattr(vqa_tool, "_try_load_vlm", lambda: True)
    monkeypatch.setattr(vqa_tool, "_vqa_with_model", lambda a, query: garbage)
    monkeypatch.setattr(vqa_tool, "_find_checkpoint", lambda: "models/checkpoints/smoke-bigearthnet-blip")

    res_green = vqa_tool.analyze(q, arr_green, task="captioning")
    res_dark = vqa_tool.analyze(q, arr_dark, task="captioning")

    # both must not contain garbage
    assert "_ _" not in res_green["answer"]
    assert "_ _" not in res_dark["answer"]
    assert res_green["confidence"]["is_heuristic"] is True
    assert res_dark["confidence"]["is_heuristic"] is True

    # and must be distinct (image-dependent), not same generic fallback
    assert res_green["answer"] != res_dark["answer"], "invalid BLIP fallback produced identical generic caption for distinct images"
    assert res_green["caption"] != res_dark["caption"]
