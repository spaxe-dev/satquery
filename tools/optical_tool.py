"""Optical analyzer — NDVI/NDWI heuristics + land-cover evidence maps."""
import numpy as np
import cv2
import io, base64
from PIL import Image
from typing import Dict, Any

def _to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr,0,255).astype(np.uint8)

def analyze(image_array: np.ndarray, query: str="") -> Dict[str, Any]:
    import time
    t0 = time.time()
    h,w = image_array.shape[0], image_array.shape[1]
    if image_array.ndim == 2:
        # panchromatic fallback
        gray = image_array
        # synthesize RGB as gray
        rgb = np.stack([gray,gray,gray], axis=2)
    else:
        rgb = image_array[:,:,:3] if image_array.shape[2]>=3 else np.repeat(image_array[:,:,:1],3,axis=2)
        gray = cv2.cvtColor(_to_uint8(rgb), cv2.COLOR_RGB2GRAY)

    rgb_u8 = _to_uint8(rgb)
    r,g,b = rgb_u8[:,:,0].astype(float), rgb_u8[:,:,1].astype(float), rgb_u8[:,:,2].astype(float)
    # pseudo NDVI using Excess Green vs Red, and NDWI using Green vs NIR-proxy (use R as NIR proxy if no NIR)
    # For true Sentinel-2 we would use B04/B08 but demo uses RGB
    eps = 1e-6
    ndvi_proxy = (g - r) / (g + r + eps)  # green as NIR proxy
    ndwi_proxy = (g - b) / (g + b + eps)  # not ideal but heuristic varies
    # Better: use brightness + HSV
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    # Water: require dark + blue dominance + low NDVI
    # Tightened to reduce false positives on dark vegetation
    blue_dominant = (b > r + 12) & (b > g - 8)
    dark = gray < 88
    low_veg = ndvi_proxy < 0.08
    water_score = (dark.astype(float)*0.4 + blue_dominant.astype(float)*0.4 + low_veg.astype(float)*0.2)
    # Require at least 0.65 (needs dark+blue or dark+blue+low_veg)
    water_mask = (water_score >= 0.65).astype(np.uint8)*255
    # Fallback: also accept classic HSV water when very dark and blue hue
    hsv_water = ((hsv[:,:,0] >= 85) & (hsv[:,:,0] <= 135) & (hsv[:,:,1] > 40) & (hsv[:,:,2] < 110) & dark).astype(np.uint8)*255
    water_mask = np.maximum(water_mask, hsv_water)
    # Enforce blue dominance morphologically: remove regions without blue
    if np.mean(water_mask>0) > 0:
        # keep only where blue_dominant holds
        water_mask = (water_mask>0) & blue_dominant
        water_mask = water_mask.astype(np.uint8)*255
    # Vegetation: high NDVI proxy + green hue
    veg_mask = ((ndvi_proxy > 0.12) & (hsv[:,:,0] >= 30) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] > 35)).astype(np.uint8)*255
    # Built-up: high edge density + medium brightness + low NDVI
    edges = cv2.Canny(gray, 50,150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    edge_dense = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # built candidate where edge dense and not water/veg and medium brightness
    built_candidate = ( (edge_dense>0) & (water_mask==0) & (veg_mask==0) & (gray>70) & (gray<200) ).astype(np.uint8)*255
    # clean masks
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k2)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k2)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, k2)
    built_candidate = cv2.morphologyEx(built_candidate, cv2.MORPH_OPEN, k2)

    water_pct = float(np.mean(water_mask>0)*100)
    veg_pct = float(np.mean(veg_mask>0)*100)
    built_pct = float(np.mean(built_candidate>0)*100)

    # encode overlays as base64 for frontend
    def encode_overlay(mask, color):
        overlay = np.zeros((h,w,3), dtype=np.uint8)
        overlay[mask>0] = color
        pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    water_overlay = encode_overlay(water_mask, [30,120,255])
    built_overlay = encode_overlay(built_candidate, [255,180,30])

    # textual evidence
    evidence = []
    evidence.append(f"Water-like regions: {water_pct:.1f}% of scene (low brightness + NDWI proxy + low saturation)")
    evidence.append(f"Vegetative cover: {veg_pct:.1f}% (NDVI-proxy >0.12 + green hue)")
    evidence.append(f"Built-up texture: {built_pct:.1f}% (edge density + non-water/non-veg mid-brightness)")

    q = query.lower()
    if "water" in q:
        ans = f"Optical evidence: water-covered regions estimated at {water_pct:.1f}% of the scene. " + ("Prominent water bodies are visible." if water_pct>3 else "No major water bodies dominate the scene.")
    elif "built" in q or "urban" in q:
        ans = f"Optical evidence: built-up texture covers ~{built_pct:.1f}% of the scene. " + ("Notable settlement/urban fabric present." if built_pct>4 else "Built-up area is limited.")
    else:
        ans = f"Optical land-cover summary: water ~{water_pct:.1f}%, vegetation ~{veg_pct:.1f}%, built-up texture ~{built_pct:.1f}%."

    # confidence
    input_q = 0.90 if min(h,w)>=256 else 0.75
    # model confidence based on mask coherence (if masks are neither 0 nor huge)
    coherence = 1.0 if 0.5 < water_pct+veg_pct+built_pct < 85 else 0.6
    model_conf = 0.62 + 0.2*coherence
    overall = 0.5*input_q + 0.5*model_conf

    elapsed_ms = int((time.time()-t0)*1000)
    return {
        "water_mask": water_mask,
        "veg_mask": veg_mask,
        "built_mask": built_candidate,
        "water_pct": round(water_pct,2),
        "veg_pct": round(veg_pct,2),
        "built_pct": round(built_pct,2),
        "water_overlay_b64": water_overlay,
        "built_overlay_b64": built_overlay,
        "evidence": evidence,
        "answer": ans,
        "confidence": {
            "input_quality": round(float(input_q),2),
            "model_confidence": round(float(model_conf),2),
            "evidence_agreement": round(float(coherence),2),
            "overall": round(float(overall),2),
            "is_heuristic": True
        },
        "model_used": "OpticalAnalyzer (NDVI-proxy/NDWI-proxy + HSV + edge density)",
        "elapsed_ms": elapsed_ms,
    }
