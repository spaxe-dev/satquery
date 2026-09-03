"""
Change detection tool — real CV pipeline (not hallucinated VLM).
Produces:
 - change map (visual)
 - change mask (binary)
 - change description
 - area pct
Works on any pair, CPU-only, no GPU required.
"""
import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict, Any

def _ensure_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            return arr[:,:,0]
    return arr

def _resize_match(a: np.ndarray, b: np.ndarray):
    # resize b to a's dims
    h, w = a.shape[0], a.shape[1]
    if b.shape[0] != h or b.shape[1] != w:
        if b.ndim == 3:
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return b

def _encode_mask_overlay(mask: np.ndarray, base_shape) -> str:
    # colorize mask: red overlay
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[mask > 0] = [255, 40, 40]
    # add alpha blending preview is done frontend; here just mask
    # encode as PNG base64
    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _encode_change_map(diff: np.ndarray) -> str:
    # diff is float 0-255; colorize with jet
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(colored)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def analyze(image_t1: np.ndarray, image_t2: np.ndarray, query: str = "") -> Dict[str, Any]:
    import time
    t0 = time.time()
    # match sizes
    if image_t1.shape[0] != image_t2.shape[0] or image_t1.shape[1] != image_t2.shape[1]:
        image_t2 = _resize_match(image_t1, image_t2)

    g1 = _ensure_gray(image_t1)
    g2 = _ensure_gray(image_t2)

    # also color diff if both RGB
    if image_t1.ndim == 3 and image_t2.ndim == 3 and image_t1.shape[2] >=3 and image_t2.shape[2] >=3:
        # Use perceptual diff in LAB
        lab1 = cv2.cvtColor(image_t1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(image_t2, cv2.COLOR_RGB2LAB)
        diff_lab = np.linalg.norm(lab1.astype(float) - lab2.astype(float), axis=2)
        # combine with gray diff
        diff_gray = cv2.absdiff(g1, g2).astype(float)
        diff = 0.6*diff_lab + 0.4*diff_gray
    else:
        diff = cv2.absdiff(g1, g2).astype(float)

    # Denoise diff
    diff_blur = cv2.GaussianBlur(diff.astype(np.uint8), (5,5), 1.0).astype(float)
    # Otsu threshold on diff
    diff_u8 = np.clip(diff_blur, 0, 255).astype(np.uint8)
    # Use adaptive + Otsu hybrid
    try:
        _, mask_otsu = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = _
    except:
        _, mask_otsu = cv2.threshold(diff_u8, 30, 255, cv2.THRESH_BINARY)
        otsu_thresh = 30
    # If Otsu is too low (<12) or too high (>80), use percentile
    mean_diff = float(np.mean(diff_u8))
    if otsu_thresh < 12 or otsu_thresh > 90:
        p85 = float(np.percentile(diff_u8, 85))
        _, mask_otsu = cv2.threshold(diff_u8, max(18, p85), 255, cv2.THRESH_BINARY)
        otsu_thresh = max(18, p85)

    # Morphology cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # remove tiny components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    min_area = mask.size * 0.001  # 0.1% of image
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    mask = cleaned

    # metrics
    changed_pixels = int(np.count_nonzero(mask))
    total = mask.size
    changed_pct = changed_pixels / total * 100 if total else 0

    # change map encoding
    try:
        change_map_b64 = _encode_change_map(diff)
        overlay_b64 = _encode_mask_overlay(mask, mask.shape)
    except Exception as e:
        change_map_b64 = None
        overlay_b64 = None

    # describe change
    # direction heuristic: compare brightness in changed region
    if changed_pixels > 0:
        # mean brightness inside changed region
        mean1 = float(np.mean(g1[mask>0])) if np.any(mask>0) else 0
        mean2 = float(np.mean(g2[mask>0])) if np.any(mask>0) else 0
        brightness_delta = mean2 - mean1
        # NDVI-like for optical if RGB
        if image_t1.ndim==3 and image_t2.ndim==3:
            # excess green change
            def exg(arr):
                r,g,b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
                return 2*g - r - b
            eg1 = exg(image_t1)[mask>0]
            eg2 = exg(image_t2)[mask>0]
            veg_delta = float(np.mean(eg2) - np.mean(eg1)) if eg2.size else 0
        else:
            veg_delta = 0
    else:
        brightness_delta = 0
        veg_delta = 0

    # Build description
    if changed_pct < 0.3:
        magnitude = "negligible change"
        detail = "No substantial bi-temporal change detected; differences are within noise/illumination variation."
    elif changed_pct < 2.0:
        magnitude = "minor change"
        detail = f"Small localized changes covering {changed_pct:.2f}% of the scene."
    elif changed_pct < 8.0:
        magnitude = "moderate change"
        detail = f"Moderate changes covering {changed_pct:.2f}% of the scene."
    else:
        magnitude = "extensive change"
        detail = f"Extensive changes covering {changed_pct:.2f}% of the scene."

    # direction / type inference
    type_hints = []
    if abs(veg_delta) > 8:
        if veg_delta < 0:
            type_hints.append("vegetation loss / clearing")
        else:
            type_hints.append("vegetation gain / regrowth")
    if brightness_delta > 15:
        type_hints.append("brightening (possible construction, bare soil, or drying)")
    elif brightness_delta < -15:
        type_hints.append("darkening (possible inundation, shadow, or vegetation)")
    if not type_hints and changed_pct >= 0.3:
        type_hints.append("structural or land-cover modification")

    # built-up increase/decrease heuristic answer
    built_question = any(k in query.lower() for k in ["built", "urban", "settlement", "increase", "decrease"])
    built_answer = ""
    if built_question and changed_pct > 0.3:
        # edge density in changed region
        edges1 = cv2.Canny(g1, 50,150)
        edges2 = cv2.Canny(g2, 50,150)
        e1 = float(np.mean(edges1[mask>0] > 0)) if changed_pixels else 0
        e2 = float(np.mean(edges2[mask>0] > 0)) if changed_pixels else 0
        if e2 > e1 * 1.2:
            built_answer = " Built-up texture appears to have increased in the changed regions."
        elif e2 < e1 * 0.8:
            built_answer = " Built-up texture appears to have decreased in the changed regions."
        else:
            built_answer = " Built-up change is inconclusive from texture alone; visual inspection of the change map is recommended."

    description = f"{magnitude.capitalize()} detected. {detail} "
    if type_hints:
        description += f" Interpretation: {', '.join(type_hints)}."
    if brightness_delta != 0 and changed_pct >=0.3:
        description += f" Mean brightness shift in changed areas: {brightness_delta:+.1f} (T2−T1)."
    if built_answer:
        description += built_answer
    if changed_pct >=0.3:
        description += " Refer to the change map overlay (red = changed) for spatial location."

    # confidence heuristic
    # input quality by size, model confidence by otsu separability, evidence by area
    input_q = 0.92 if min(g1.shape) >= 256 else 0.78
    # separability: how bimodal is diff histogram? Use Otsu variance
    hist = cv2.calcHist([diff_u8], [0], None, [256], [0,256]).flatten()
    hist = hist / hist.sum()
    # entropy
    entropy = -np.sum(hist[hist>0] * np.log(hist[hist>0]+1e-9))
    # low entropy -> more separable
    sep_score = max(0, min(1, (4.5 - entropy)/2.0))
    model_conf = 0.55 + 0.35*sep_score
    # evidence agreement: if changed area is neither tiny nor huge, confidence higher
    if changed_pct < 0.3 or changed_pct > 45:
        evidence = 0.58
    elif 1 <= changed_pct <= 15:
        evidence = 0.86
    else:
        evidence = 0.73
    overall = 0.30*input_q + 0.40*model_conf + 0.30*evidence

    elapsed_ms = int((time.time()-t0)*1000)
    return {
        "change_map_b64": change_map_b64,
        "overlay_b64": overlay_b64,
        "mask": mask,  # internal, not serialized directly
        "changed_pct": round(float(changed_pct), 2),
        "changed_pixels": changed_pixels,
        "otsu_thresh": float(otsu_thresh),
        "mean_diff": round(mean_diff,1),
        "description": description,
        "brightness_delta": round(float(brightness_delta),1),
        "veg_delta": round(float(veg_delta),1),
        "confidence": {
            "input_quality": round(float(input_q),2),
            "model_confidence": round(float(model_conf),2),
            "evidence_agreement": round(float(evidence),2),
            "overall": round(float(overall),2),
            "is_heuristic": True
        },
        "model_used": "ChangeDetector (LAB+gray diff, Otsu, morphology, CC filtering)",
        "elapsed_ms": elapsed_ms,
    }
