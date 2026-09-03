import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from tools import vqa_tool, change_tool, optical_tool, sar_tool, fusion_tool

def make_img(w=256,h=256, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0,255,size=(h,w,3),dtype=np.uint8)

def test_vqa():
    img = make_img()
    r = vqa_tool.analyze("Describe the land-cover", img, task="captioning")
    assert "answer" in r
    assert "caption" in r
    assert r["confidence"]["overall"]>0

def test_grounding():
    img = make_img(seed=1)
    r = vqa_tool.analyze("Where is the water?", img, task="grounding")
    assert "boxes" in r

def test_change():
    a = make_img(seed=10)
    b = a.copy()
    # add change patch
    b[80:140,80:140] = 255
    r = change_tool.analyze(a,b, "What changed?")
    assert r["changed_pct"]>=0
    assert r["change_map_b64"] is not None
    assert "description" in r

def test_optical():
    img = make_img(seed=5)
    r = optical_tool.analyze(img, "Identify water")
    assert "water_pct" in r
    assert r["water_overlay_b64"] is not None

def test_sar():
    img = make_img(seed=6)
    # make sar-like single channel
    gray = img.mean(axis=2).astype(np.uint8)
    r = sar_tool.analyze(gray, "Identify water")
    assert "water_pct" in r
    assert "speckle_cv" in r

def test_fusion():
    img = make_img(seed=7)
    opt = optical_tool.analyze(img, "water")
    sar = sar_tool.analyze(img, "water")
    h,w = img.shape[0], img.shape[1]
    fused = fusion_tool.fuse(opt, sar, h, w)
    assert "mean_iou" in fused
    assert "corroboration" in fused
