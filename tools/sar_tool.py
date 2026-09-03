"""SAR analyzer — backscatter statistics, speckle-aware water/built-up detection."""
import numpy as np
import cv2
import io, base64
from PIL import Image
from typing import Dict, Any

def analyze(image_array: np.ndarray, query: str="") -> Dict[str, Any]:
    import time
    t0 = time.time()
    # SAR is typically single-channel; handle RGB composites too
    if image_array.ndim == 3:
        if image_array.shape[2] >=3:
            # Try to recover SAR-like intensity: assume single band replicated or RGB composite
            # Use luminance as backscatter proxy
            gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float)
            # also keep single band if one channel dominates
        else:
            gray = image_array[:,:,0].astype(float)
    else:
        gray = image_array.astype(float)
    h,w = gray.shape
    # normalize to 0-255 if needed
    if gray.max() <= 1.5:
        gray = gray*255
    gray_u8 = np.clip(gray,0,255).astype(np.uint8)

    # Speckle metric: coefficient of variation in local windows
    # Use local std / mean
    mean = cv2.blur(gray, (7,7))
    sqmean = cv2.blur(gray*gray, (7,7))
    std = np.sqrt(np.maximum(sqmean - mean*mean, 0))
    cv_map = std / (mean + 1e-6)
    speckle_cv = float(np.mean(cv_map))

    # Backscatter stats
    # SAR water: smooth, low backscatter (dark), low variance
    # SAR built-up: bright, high variance, double-bounce
    low_thresh = float(np.percentile(gray_u8, 30))
    high_thresh = float(np.percentile(gray_u8, 75))

    # Water mask: low intensity + low local variance (smooth)
    # Apply median despeckle first to reduce speckle fragmentation
    gray_desp = cv2.medianBlur(gray_u8, 5)
    smooth_mask = (std < np.percentile(std, 45)).astype(np.uint8)
    # Use despeckled for thresholding but keep original for variance
    water_mask = ((gray_desp < low_thresh) & (smooth_mask==1)).astype(np.uint8)*255
    # clean — use smaller kernel to preserve fragmented water
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, k_small)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, k)
    # Fill small holes
    water_mask = cv2.medianBlur(water_mask, 5)

    # Built-up: high intensity + high variance (rough/double bounce)
    # Also bright point scatterers
    rough_mask = (std > np.percentile(std, 55)).astype(np.uint8)
    built_mask = ((gray_u8 > high_thresh) & (rough_mask==1)).astype(np.uint8)*255
    built_mask = cv2.morphologyEx(built_mask, cv2.MORPH_OPEN, k_small)
    # Remove tiny speckles
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(built_mask, connectivity=8)
    cleaned = np.zeros_like(built_mask)
    min_area = built_mask.size * 0.0008
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels==i]=255
    built_mask = cleaned

    # Also despeckling via Lee-like (median) for preview quality
    despeckled = cv2.medianBlur(gray_u8, 5)

    water_pct = float(np.mean(water_mask>0)*100)
    built_pct = float(np.mean(built_mask>0)*100)
    mean_backscatter = float(np.mean(gray_u8))
    std_backscatter = float(np.std(gray_u8))

    def encode(mask, color):
        overlay = np.zeros((h,w,3), dtype=np.uint8)
        overlay[mask>0]=color
        pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    water_overlay = encode(water_mask, [30,120,255])
    built_overlay = encode(built_mask, [255,200,40])

    # Evidence text
    q = query.lower()
    evidence = [
        f"Mean backscatter (0-255): {mean_backscatter:.1f} ± {std_backscatter:.1f}",
        f"Speckle CV (7x7): {speckle_cv:.3f} — {'high speckle texture' if speckle_cv>0.25 else 'moderate speckle'}",
        f"SAR water-like (low + smooth): {water_pct:.1f}%",
        f"SAR built-like (bright + rough): {built_pct:.1f}%",
    ]
    # SAR advantage statement
    sar_advantage = " SAR penetrates cloud and operates day/night; water appears dark/smooth due to specular reflection, built-up appears bright due to double-bounce."

    if "water" in q:
        ans = f"SAR evidence: low-backscatter smooth regions cover ~{water_pct:.1f}% — consistent with water (specular, low return).{sar_advantage}"
    elif "built" in q:
        ans = f"SAR evidence: bright rough regions cover ~{built_pct:.1f}% — consistent with built-up / double-bounce scattering.{sar_advantage}"
    else:
        ans = f"SAR backscatter summary: mean {mean_backscatter:.0f}, water-like {water_pct:.1f}%, built-like {built_pct:.1f}%.{sar_advantage}"

    input_q = 0.86 if min(h,w)>=256 else 0.72
    # SAR confidence is lower if speckle is extreme or image is RGB-like (not true SAR)
    is_rgb_like = image_array.ndim==3 and image_array.shape[2]>=3
    sar_penalty = 0.12 if is_rgb_like else 0.0  # composite is less reliable
    model_conf = max(0.5, 0.68 - sar_penalty + (0.1 if 0.15<speckle_cv<0.4 else -0.05))
    overall = 0.5*input_q + 0.5*model_conf

    elapsed_ms = int((time.time()-t0)*1000)
    return {
        "water_mask": water_mask,
        "built_mask": built_mask,
        "water_pct": round(water_pct,2),
        "built_pct": round(built_pct,2),
        "mean_backscatter": round(mean_backscatter,1),
        "std_backscatter": round(std_backscatter,1),
        "speckle_cv": round(speckle_cv,3),
        "water_overlay_b64": water_overlay,
        "built_overlay_b64": built_overlay,
        "evidence": evidence,
        "answer": ans,
        "despeckled_b64": encode(despeckled, [0,0,0]), # placeholder
        "confidence": {
            "input_quality": round(float(input_q),2),
            "model_confidence": round(float(model_conf),2),
            "evidence_agreement": 0.70,
            "overall": round(float(overall),2),
            "is_heuristic": True
        },
        "model_used": "SARAnalyzer (backscatter stats, Lee-median despeckle, CV speckle metric)",
        "elapsed_ms": elapsed_ms,
    }
