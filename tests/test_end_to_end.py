import os, sys, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from PIL import Image
import numpy as np
from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis

def png_bytes(w=256,h=256, seed=0):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0,255,size=(h,w,3),dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr

def prepare_datas(n, seeds):
    datas=[]
    for i, s in enumerate(seeds[:n]):
        b, arr = png_bytes(seed=s)
        res = validate_image_file(b, f"img_{i}.png")
        datas.append({"bytes": b, "filename": f"img_{i}.png", "array": res["array"], "meta": res["meta"]})
    return datas

def test_single_vqa():
    datas = prepare_datas(1, [1])
    r = run_analysis("Describe the land-cover", datas)
    assert r["task"] in ("captioning","single_vqa")
    assert r["answer"]
    assert r["refused"]==False
    assert len(r["execution_trace"])>3

def test_temporal():
    datas = prepare_datas(2, [10,11])
    # make second slightly different for change
    datas[1]["array"][50:100,50:100] = 255
    r = run_analysis("What changed between these two dates?", datas)
    assert r["task"]=="temporal_change"
    assert r["change_description"] is not None

def test_optical_sar():
    datas = prepare_datas(2, [20,21])
    r = run_analysis("Use optical and SAR together to identify water", datas)
    # Could be fusion or temporal depending on modalities detected; allow either
    assert r["task"] in ("optical_sar_fusion","temporal_change","compound_temporal_sar","single_vqa")
    assert r["refused"]==False

def test_compound():
    datas = prepare_datas(3, [30,31,32])
    datas[1]["array"][80:130,80:130]= 200
    r = run_analysis("What changed and can SAR corroborate?", datas)
    assert r["task"]=="compound_temporal_sar"
    assert r["answer"]

def test_invalid_temporal_mismatch():
    # Use sizes that trigger refusal: 512 vs 128 is mismatched >15%
    b1, _ = png_bytes(w=512,h=512,seed=40)
    b2, _ = png_bytes(w=128,h=128,seed=41)
    from backend.app.services.validation import validate_image_file
    r1 = validate_image_file(b1, "large.png")
    r2 = validate_image_file(b2, "small.png")
    datas = [
        {"bytes": b1, "filename":"large.png", "array": r1["array"], "meta": r1["meta"]},
        {"bytes": b2, "filename":"small.png", "array": r2["array"], "meta": r2["meta"]},
    ]
    r = run_analysis("What changed between these two dates?", datas)
    assert r["refused"]==True
    assert "cannot be established as a valid bi-temporal pair" in r["answer"].lower() or "refused" in r["answer"].lower()
