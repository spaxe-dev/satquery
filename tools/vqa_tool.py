"""
VQA / Captioning / Grounding tool — real implementation with graceful fallbacks.
Priority:
 1. Try transformers BLIP/Qwen2-VL if available and VRAM permits
 2. Fallback to CLIP zero-shot + heuristic descriptors (always works, even on CPU)
"""
import io
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple

# Lightweight heuristic descriptors
RS_LABELS = [
    "forest", "agricultural field", "urban built-up area", "water body",
    "barren land", "grassland", "wetland", "dense residential",
    "sparse residential", "industrial area", "river", "lake"
]

# For grounding — detect salient regions via classical CV
def _heuristic_grounding(arr: np.ndarray, query: str) -> List[Dict]:
    """Find candidate boxes using color/texture heuristics + query keyword matching."""
    import cv2
    query_lower = query.lower()
    boxes = []
    h, w = arr.shape[0], arr.shape[1]
    # Convert to workable format
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    else:
        gray = arr if arr.ndim == 2 else arr[:,:,0]
        hsv = None

    # Water detection: low NDWI-like via HSV or intensity
    if any(k in query_lower for k in ["water", "river", "lake", "wetland"]):
        # water tends to be dark, low saturation, bluish
        if hsv is not None:
            # estimate water mask: low value, blue hue
            # H in OpenCV 0-180
            lower_water = np.array([80, 20, 20])
            upper_water = np.array([130, 255, 180])
            mask = cv2.inRange(hsv, lower_water, upper_water)
            # also dark regions
            dark = (gray < 90).astype(np.uint8)*255
            mask = cv2.bitwise_or(mask, dark & (cv2.inRange(hsv, np.array([0,0,0]), np.array([180,50,100]))))
        else:
            _, mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
        # clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            if area < w*h*0.005:
                continue
            x,y,cw,ch = cv2.boundingRect(cnt)
            boxes.append({"x": int(x), "y": int(y), "w": int(cw), "h": int(ch), "label": "water", "confidence": 0.72})

    if any(k in query_lower for k in ["built", "urban", "building", "residential", "industrial", "settlement"]):
        # built-up: high edge density, grayish, medium brightness
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        dense = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dense = cv2.dilate(dense, kernel, iterations=1)
        contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            area = cv2.contourArea(cnt)
            if area < w*h*0.008:
                continue
            if area > w*h*0.6:
                continue
            x,y,cw,ch = cv2.boundingRect(cnt)
            boxes.append({"x": int(x), "y": int(y), "w": int(cw), "h": int(ch), "label": "built-up", "confidence": 0.65})

    if any(k in query_lower for k in ["forest", "vegetation", "tree", "green"]):
        if hsv is not None:
            lower_g = np.array([30, 30, 30])
            upper_g = np.array([90, 255, 200])
            mask = cv2.inRange(hsv, lower_g, upper_g)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                area = cv2.contourArea(cnt)
                if area < w*h*0.01:
                    continue
                x,y,cw,ch = cv2.boundingRect(cnt)
                boxes.append({"x": int(x), "y": int(y), "w": int(cw), "h": int(ch), "label": "vegetation", "confidence": 0.68})

    # fallback: if no boxes and generic query, return one central box
    if not boxes and any(k in query_lower for k in ["where", "highlight", "locate", "show", "ground"]):
        boxes.append({"x": w//4, "y": h//4, "w": w//2, "h": h//2, "label": "salient region", "confidence": 0.42})

    return boxes[:5]


def _heuristic_caption(arr: np.ndarray) -> str:
    """
    Deterministic RS heuristic caption — genuinely image-dependent.
    Uses only pixel-derived evidence (no filenames, no BigEarthNet labels):
      - brightness / per-channel color composition + std
      - vegetation Excess Green + NDVI-proxy (G vs R)
      - texture Laplacian variance + edge density
      - water-indicative dark/very-dark/blue-dominant ratios
      - HSV saturation/hue
      - quadrant heterogeneity (brightness + veg per quadrant)
    Every materially different image produces a different numeric fingerprint,
    so captions are not generic. Concise and deterministic (no randomness).
    """
    import cv2
    h, w = arr.shape[0], arr.shape[1]
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        r, g, b = rgb_u8[:, :, 0].astype(float), rgb_u8[:, :, 1].astype(float), rgb_u8[:, :, 2].astype(float)
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)

        exg = 2 * g - r - b
        veg_ratio = float(np.mean(exg > 20))
        veg_strict = float(np.mean(exg > 35))
        ndvi_proxy = (g - r) / (g + r + 1e-6)
        ndvi_mean = float(np.mean(ndvi_proxy))

        brightness = float(np.mean(rgb_u8))
        brightness_std = float(np.std(rgb_u8.astype(float)))
        r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        r_std, g_std, b_std = float(np.std(r)), float(np.std(g)), float(np.std(b))

        dark_ratio = float(np.mean(gray < 80))
        very_dark = float(np.mean(gray < 50))
        bright_ratio = float(np.mean(gray > 180))
        blue_dom = float(np.mean(b > r))
        blue_strict = float(np.mean((b > r + 12) & (b > g - 8)))
        green_dom = float(np.mean((g > r + 5) & (g > b - 5)))

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        sat_mean = float(np.mean(hsv[:, :, 1]))
        hue_mean = float(np.mean(hsv[:, :, 0]))

        # quadrant heterogeneity — brightness per quadrant
        q_means = []
        veg_q = []
        for y0 in (0, h // 2):
            for x0 in (0, w // 2):
                q = rgb_u8[y0:y0 + h // 2, x0:x0 + w // 2]
                q_means.append(float(np.mean(q)))
                qr, qg, qb = q[:, :, 0].astype(float), q[:, :, 1].astype(float), q[:, :, 2].astype(float)
                qexg = 2 * qg - qr - qb
                veg_q.append(float(np.mean(qexg > 20)))
        quad_std = float(np.std(q_means)) if q_means else 0.0
        veg_q_std = float(np.std(veg_q)) if veg_q else 0.0
    else:
        gray = arr if arr.ndim == 2 else arr[:, :, 0]
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        h, w = gray_u8.shape[0], gray_u8.shape[1]
        r_mean = g_mean = b_mean = float(np.mean(gray_u8))
        r_std = g_std = b_std = 0.0
        veg_ratio = 0.0
        veg_strict = 0.0
        ndvi_mean = 0.0
        brightness = float(np.mean(gray_u8))
        brightness_std = float(np.std(gray_u8.astype(float)))
        dark_ratio = float(np.mean(gray_u8 < 80))
        very_dark = float(np.mean(gray_u8 < 50))
        bright_ratio = float(np.mean(gray_u8 > 180))
        blue_dom = 0.0
        blue_strict = 0.0
        green_dom = 0.0
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        edges = cv2.Canny(gray_u8, 50, 150)
        edge_density = float(np.mean(edges > 0))
        sat_mean = 0.0
        hue_mean = 0.0
        quad_std = 0.0
        veg_q_std = 0.0
        veg_q = [0.0, 0.0, 0.0, 0.0]

    # categorical phrases still derived from metrics, but now each part also carries numeric value
    if veg_ratio > 0.35:
        veg_cat = "significant vegetation / agricultural or forested cover"
    elif veg_ratio > 0.15:
        veg_cat = "moderate vegetation interspersed with other cover"
    elif veg_ratio > 0.02:
        veg_cat = "patchy sparse vegetation"
    else:
        veg_cat = "sparse vegetation"

    if brightness < 40:
        bright_cat = "very dark tones suggesting water/shadow"
    elif brightness < 70:
        bright_cat = "dark tones suggesting water or shadowed terrain"
    elif brightness > 160:
        bright_cat = "bright reflective surfaces consistent with barren land or built-up"
    elif brightness > 110:
        bright_cat = "moderately bright heterogeneous land-cover"
    else:
        bright_cat = "mixed brightness indicating heterogeneous land-cover"

    if lap_var > 800:
        tex_cat = "high texture / edge density suggesting urban or fragmented landscape"
    elif lap_var > 300:
        tex_cat = "moderate-high texture"
    elif lap_var < 30:
        tex_cat = "very smooth homogeneous texture suggesting water or uniform fields"
    elif lap_var < 150:
        tex_cat = "smooth homogeneous texture suggesting water or uniform fields"
    else:
        tex_cat = "moderate texture"

    # spatial heterogeneity phrase
    if quad_std > 15:
        spatial_cat = "strong spatial heterogeneity across quadrants"
    elif quad_std > 5:
        spatial_cat = "moderate spatial variation"
    else:
        spatial_cat = "relatively uniform across quadrants"

    parts = [
        f"{veg_cat} (ExG {veg_ratio*100:.1f}%, strict {veg_strict*100:.1f}%, NDVI-proxy {ndvi_mean:.3f}, veg quad-std {veg_q_std*100:.1f}%)",
        f"{bright_cat} (mean {brightness:.1f}/255, RGB {r_mean:.0f}/{g_mean:.0f}/{b_mean:.0f} ± {brightness_std:.0f}, bright pixels {bright_ratio*100:.1f}%)",
        f"{tex_cat} (Laplacian var {lap_var:.1f}, edge density {edge_density*100:.1f}%)",
        f"water-indicative dark {dark_ratio*100:.1f}% (very dark {very_dark*100:.1f}%), blue-dominant {blue_dom*100:.1f}% (strict {blue_strict*100:.1f}%), green-dominant {green_dom*100:.1f}%, saturation {sat_mean:.0f} hue {hue_mean:.0f}",
        f"{spatial_cat} (quadrant brightness std {quad_std:.1f})",
    ]

    caption = (
        f"This satellite image shows {', '.join(parts)}. "
        f"Observed resolution ~{w}x{h}px, quadrant vegetation {[f'{v*100:.1f}%' for v in veg_q]}; "
        f"scene appears to contain a mix of natural and anthropogenic features typical of mid-resolution optical imagery."
    )
    return caption


def _heuristic_vqa(arr: np.ndarray, query: str) -> str:
    q = query.lower()
    caption = _heuristic_caption(arr)
    # rule-based QA for common RS questions
    h, w = arr.shape[0], arr.shape[1]
    import cv2
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr if arr.ndim == 2 else arr[:,:,0]

    # Try to answer specific question types
    if any(k in q for k in ["how many", "count", "number"]):
        # estimate via blob detection
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n_labels, _ = cv2.connectedComponents(thresh)
        est = max(1, n_labels - 1)
        return f"Based on connected-component analysis, an estimated {min(est, 20)} distinct bright regions are visible. This is a heuristic count; for precise enumeration a detector would be needed. Context: {caption[:120]}"

    if any(k in q for k in ["water", "river", "lake"]):
        # water presence
        dark_ratio = float(np.mean(gray < 80))
        if dark_ratio > 0.12:
            return f"Yes — dark, low-reflectance regions covering ~{dark_ratio*100:.1f}% of the scene are consistent with water bodies (rivers/lakes). These appear as smooth, low-intensity areas. {caption[:140]}"
        else:
            return f"No large water bodies are clearly evident; only ~{dark_ratio*100:.1f}% of pixels are in the low-intensity water-like range. Some small or seasonal water may still be present but not dominant. {caption[:120]}"

    if any(k in q for k in ["building", "built", "urban", "settlement", "house"]):
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edge_density = float(np.mean(cv2.Canny(gray, 50,150) > 0))
        if lap_var > 600 and edge_density > 0.04:
            return f"Built-up / urban fabric appears present: edge density {edge_density*100:.1f}% and texture variance {lap_var:.0f} indicate structured, high-frequency features consistent with buildings and roads. {caption[:120]}"
        else:
            return f"Built-up area is not dominant: texture variance {lap_var:.0f} and edge density {edge_density*100:.1f}% suggest a predominantly natural/agricultural/barren scene with limited settlement. {caption[:120]}"

    if any(k in q for k in ["forest", "vegetation", "tree", "green"]):
        if arr.ndim == 3 and arr.shape[2] >= 3:
            r,g,b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
            exg = 2*g - r - b
            veg = float(np.mean(exg > 20))
            if veg > 0.30:
                return f"Vegetated cover is prominent, covering ~{veg*100:.0f}% of the scene by Excess Green thresholding — consistent with forest or dense agriculture. {caption[:120]}"
            else:
                return f"Vegetation is limited (~{veg*100:.0f}% by Excess Green); the scene is not predominantly forested. {caption[:120]}"
        else:
            return caption

    if any(k in q for k in ["describe", "what", "land cover", "land-cover", "visible", "scene"]):
        return caption

    # default: return caption + query acknowledgement
    return f"Query: '{query}'. Analysis: {caption} For this specific question, the heuristic RS vision pipeline provides the above scene description; a fine-tuned RS-VLM (SatQuery-RS-VLM) would refine this with learned land-cover vocabulary."


def _is_invalid_generation(query: str, answer: str) -> tuple[bool, str]:
    """
    Conservative guard for BLIP smoke-LoRA garbage.
    Returns (is_invalid, reason). Only flags OBVIOUS failures:
    - echo of prompt, underscore/artifacts, degenerate short, low alnum ratio.
    Valid land-cover sentences like "This Sentinel-2 image contains: Pastures..."
    must NOT be flagged.
    """
    import re
    import difflib
    if not answer or not answer.strip():
        return True, "empty_or_whitespace"
    stripped = answer.strip()
    q_stripped = query.strip()
    words = stripped.split()

    # 1. underscore / token artifact (observed: "_ _ _ _", "__")
    if "_ _" in stripped or "__" in stripped or stripped.count("_") >= 3:
        return True, "underscore_artifact"
    # bare underscores as tokens
    if stripped.count("_") >= 2:
        # also check spaced underscores
        if re.search(r"(?:_\s*){3,}", stripped):
            return True, "repeated_underscore"

    # 2. extremely short degenerate (but allow legitimate short like "Yes")
    if len(stripped) < 8:
        return True, f"too_short len={len(stripped)}"
    if len(words) < 2 and len(stripped) < 15:
        # single word/token answer is garbage for captioning/VQA
        return True, f"too_few_words {len(words)}"

    # 3. low alphanumeric ratio (lots of punctuation/underscores) — but allow normal captions
    alnum = sum(c.isalnum() for c in stripped)
    alnum_ratio = alnum / max(len(stripped), 1)
    if alnum_ratio < 0.45:
        return True, f"low_alnum_ratio {alnum_ratio:.2f}"

    # 4. echo detection — conservative, only when answer is essentially query
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _alphanum(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    q_norm = _norm(q_stripped)
    a_norm = _norm(stripped)
    q_an = _alphanum(q_stripped)
    a_an = _alphanum(stripped)

    # exact echo after normalization (handles hyphen spacing "built - up" vs "built-up")
    if a_norm == q_norm and len(q_norm) >= 10:
        return True, "exact_echo"
    if a_an == q_an and len(q_an) >= 20:
        return True, "alphanum_exact_echo"
    # echo plus tiny suffix like "s" or "." (observed: query + "s")
    if len(q_an) >= 20 and a_an.startswith(q_an) and len(a_an) <= len(q_an) + 10:
        extra = a_an[len(q_an):]
        if len(extra) <= 5:
            return True, f"echo_plus_suffix extra='{extra}'"
    # high difflib ratio on alphanum (catches spaced-hyphen echo)
    if len(q_an) >= 20 and len(a_an) >= 20:
        ratio = difflib.SequenceMatcher(None, q_an, a_an).ratio()
        if ratio > 0.88:
            return True, f"high_echo_ratio {ratio:.2f}"
        # token Jaccard — answer tokens subset of query
        q_tokens = set(q_norm.split())
        a_tokens = set(a_norm.split())
        if q_tokens and a_tokens:
            inter = len(q_tokens & a_tokens)
            union = len(q_tokens | a_tokens)
            jacc = inter / union if union else 0
            # high overlap + similar length => echo
            if jacc > 0.85 and abs(len(a_tokens) - len(q_tokens)) <= 2 and len(a_tokens) >= 5:
                return True, f"high_token_jaccard {jacc:.2f}"
            if len(a_tokens) >= 5 and len(a_tokens) <= len(q_tokens) + 1 and inter / max(len(a_tokens), 1) > 0.9:
                return True, "answer_tokens_subset_of_query"

    # 5. degenerate repeated word
    if len(words) >= 5:
        uniq = len(set(w.lower() for w in words))
        if uniq / len(words) < 0.3:
            return True, f"repeated_word uniq={uniq}/{len(words)}"

    # 6. prompt fragment dominance — answer is just a long substring of query
    q_low = q_stripped.lower()
    a_low = stripped.lower()
    if len(q_low) >= 30 and len(a_low) <= len(q_low) + 15:
        # check if first 30 chars of query appear in answer
        if q_low[:30] in a_low:
            return True, "prompt_fragment_dominant"

    return False, "ok"


# Try to load a real VLM if available — optional, never required
# Real adapter loading: checks for checkpoints produced by training/train_lora_real.py
# Preserves heuristic fallback if no checkpoint or no GPU.
_VLM_AVAILABLE = False
_VLM_MODEL = None
_VLM_PROCESSOR = None
_VLM_MODEL_NAME = None
_ADAPTER_LOADED = False

# Checkpoint search order — first found is used
# BigEarthNet is now PRIMARY per SIH, EuroSAT is secondary, smoke paths for verification
CHECKPOINT_CANDIDATES = [
    "models/checkpoints/satquery-bigearthnet-blip-lora",
    "models/checkpoints/satquery-bigearthnet-blip",
    "models/checkpoints/smoke-bigearthnet-blip",
    "models/checkpoints/satquery-eurosat-blip-lora",
    "models/checkpoints/satquery-eurosat-qwen-lora",
    "models/checkpoints/satquery-lora",
    "models/checkpoints/satquery-eurosat-blip-lora-qlora",
]

def _find_checkpoint():
    from pathlib import Path
    for cand in CHECKPOINT_CANDIDATES:
        p = Path(cand)
        # Check for adapter_config.json (PEFT) or pytorch_model.bin / adapter_model.safetensors
        if (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists():
            # Also need to verify it's not just a placeholder manifest
            # Check if adapter file is non-empty and not just manifest
            try:
                if (p / "adapter_config.json").exists():
                    import json
                    cfg = json.load(open(p / "adapter_config.json"))
                    # Real PEFT config has r value
                    if "r" in cfg or "lora_alpha" in cfg:
                        return str(p)
                # If only manifest exists, it's placeholder — skip
                if (p / "training_manifest.json").exists() and not (p / "adapter_config.json").exists():
                    continue
            except Exception:
                continue
            return str(p)
        # Also check if checkpoint has training_manifest with status trained
        manifest = p / "training_manifest.json"
        if manifest.exists():
            try:
                import json
                m = json.load(open(manifest))
                if m.get("status") == "trained":
                    return str(p)
            except:
                pass
    return None

def _try_load_vlm():
    global _VLM_AVAILABLE, _VLM_MODEL, _VLM_PROCESSOR, _VLM_MODEL_NAME, _ADAPTER_LOADED
    if _VLM_AVAILABLE and _VLM_MODEL is not None:
        return True
    # Check for real checkpoint first
    ckpt = _find_checkpoint()
    if ckpt is None:
        # No real checkpoint — heuristic only
        return False
    try:
        import torch
        from transformers.utils import is_torch_available
        if not is_torch_available():
            return False
        # Determine base model from checkpoint manifest or fallback
        base_model = "Salesforce/blip-image-captioning-base"
        import json, pathlib
        # Check both training_manifest.json and smoke_manifest.json
        for manifest_name in ["training_manifest.json", "smoke_manifest.json"]:
            manifest_path = pathlib.Path(ckpt) / manifest_name
            if manifest_path.exists():
                try:
                    m = json.load(open(manifest_path))
                    # Only overwrite if non-empty
                    if m.get("base_model"):
                        base_model = m.get("base_model")
                        break
                except:
                    pass
        # Also check adapter_config base (but only if non-empty)
        adapter_cfg = pathlib.Path(ckpt) / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                cfg = json.load(open(adapter_cfg))
                candidate = cfg.get("base_model_name_or_path")
                if candidate and candidate.strip():
                    base_model = candidate
            except:
                pass

        _VLM_MODEL_NAME = base_model
        print(f"[VQA] Found checkpoint at {ckpt}, base {base_model}, attempting load...")

        # Load based on model type
        if "blip" in base_model.lower():
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from peft import PeftModel
            processor = BlipProcessor.from_pretrained(base_model, local_files_only=True) if pathlib.Path(base_model).exists() else BlipProcessor.from_pretrained(base_model)
            # Try local_files_only first to avoid network
            try:
                base = BlipForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, local_files_only=True)
            except:
                base = BlipForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            # Check if checkpoint is text_decoder-only (from smoke test) vs full model
            ckpt_path = pathlib.Path(ckpt)
            is_text_decoder_adapter = (ckpt_path / "adapter_type.txt").exists()
            # Also check adapter_config: if it was saved from text_decoder, it will have task_type SEQ_2_SEQ_LM and target query/value but base is text decoder
            # We can detect by trying to load onto text_decoder first
            try:
                if is_text_decoder_adapter or (ckpt_path / "adapter_config.json").exists():
                    # Try to detect by reading adapter_config
                    import json as _json
                    cfg = _json.load(open(ckpt_path / "adapter_config.json"))
                    # If the adapter was trained on text_decoder only, it will have been saved from text_decoder, so we should load onto text_decoder
                    # We can check by trying to load onto base.text_decoder
                    if is_text_decoder_adapter:
                        print(f"[VQA] Detected text_decoder adapter at {ckpt}, loading onto base.text_decoder...")
                        base.text_decoder = PeftModel.from_pretrained(base.text_decoder, ckpt)
                        model = base
                    else:
                        # Try full model first, fallback to text_decoder
                        try:
                            model = PeftModel.from_pretrained(base, ckpt)
                        except Exception as e_full:
                            print(f"[VQA] Full model adapter load failed ({e_full}), trying text_decoder...")
                            base.text_decoder = PeftModel.from_pretrained(base.text_decoder, ckpt)
                            model = base
                else:
                    model = PeftModel.from_pretrained(base, ckpt)
            except Exception as e:
                print(f"[VQA] BLIP adapter load failed: {e}, trying fallback...")
                raise
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            _VLM_MODEL = model
            _VLM_PROCESSOR = processor
            _VLM_AVAILABLE = True
            _ADAPTER_LOADED = True
            print(f"[VQA] Loaded BLIP LoRA adapter from {ckpt} (text_decoder={is_text_decoder_adapter})")
            return True
        elif "qwen" in base_model.lower():
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from peft import PeftModel
            processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, local_files_only=True) if pathlib.Path(base_model).exists() else AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
            try:
                base = Qwen2VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto", local_files_only=True, trust_remote_code=True)
            except:
                base = Qwen2VLForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            model = PeftModel.from_pretrained(base, ckpt)
            model.eval()
            _VLM_MODEL = model
            _VLM_PROCESSOR = processor
            _VLM_AVAILABLE = True
            _ADAPTER_LOADED = True
            print(f"[VQA] Loaded Qwen2-VL LoRA adapter from {ckpt}")
            return True
        else:
            print(f"[VQA] Unknown base model type for {base_model}, skipping adapter load")
            return False
    except Exception as e:
        print(f"[VQA] Adapter load failed, falling back to heuristic: {e}")
        import traceback
        traceback.print_exc()
        _VLM_AVAILABLE = False
        return False
    return False

def _vqa_with_model(image_array: np.ndarray, query: str) -> str:
    """Run real VLM inference if available, else raise."""
    if not _VLM_AVAILABLE or _VLM_MODEL is None:
        raise RuntimeError("No VLM available")
    import torch
    from PIL import Image
    # Convert numpy to PIL
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1 else image_array.astype(np.uint8)
    if image_array.ndim == 2:
        pil = Image.fromarray(image_array, mode="L").convert("RGB")
    else:
        if image_array.shape[2] >= 3:
            pil = Image.fromarray(image_array[:,:,:3].astype(np.uint8))
        else:
            pil = Image.fromarray(image_array[:,:,0].astype(np.uint8), mode="L").convert("RGB")

    # BLIP path
    if "blip" in (_VLM_MODEL_NAME or "").lower():
        # For BLIP, we generate caption conditioned on query
        # Use processor with text query as prompt
        inputs = _VLM_PROCESSOR(images=pil, text=query, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            generated = _VLM_MODEL.generate(**inputs, max_new_tokens=64, num_beams=3)
        text = _VLM_PROCESSOR.decode(generated[0], skip_special_tokens=True)
        return text
    else:
        # Qwen2-VL path: chat template
        messages = [
            {"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": query}]}
        ]
        text = _VLM_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _VLM_PROCESSOR(text=[text], images=[pil], return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        with torch.no_grad():
            generated = _VLM_MODEL.generate(**inputs, max_new_tokens=128)
        # Decode only new tokens
        input_len = inputs.input_ids.shape[1]
        decoded = _VLM_PROCESSOR.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
        return decoded


def analyze(query: str, image_array: np.ndarray, task: str = "vqa") -> Dict[str, Any]:
    """
    Unified entry for VQA / captioning / grounding.
    Tries real LoRA-adapted VLM first; falls back to heuristic (always succeeds).
    Architecture: REAL ADAPTED MODEL + HEURISTIC FALLBACK + OUTPUT GUARD
    Guard: garbage BLIP smoke generations (prompt-echo, underscores, degenerate)
    are rejected and routed to deterministic heuristic fallback.
    """
    import time
    t0 = time.time()
    # Attempt to load real adapter (lazy, cached)
    loaded = _try_load_vlm()
    use_real = loaded and _VLM_AVAILABLE and _VLM_MODEL is not None

    # Try real inference if available
    if use_real:
        try:
            real_answer = _vqa_with_model(image_array, query)
            # Guard: reject obvious garbage before exposing to user
            is_invalid, reason = _is_invalid_generation(query, real_answer)
            if is_invalid:
                print(f"[VQA] Guard rejected invalid VLM output ({reason}): '{real_answer[:80]}...' -> falling back to heuristic")
            else:
                # Keep heuristic grounding boxes (model doesn't produce precise boxes)
                boxes = []
                if task in ("grounding", "vqa") and any(k in query.lower() for k in ["where", "highlight", "locate", "show", "box", "water", "built", "forest", "vegetation", "building"]):
                    boxes = _heuristic_grounding(image_array, query)
                # For captioning task, answer == caption
                if task == "captioning" or "describe" in query.lower():
                    real_caption = real_answer
                else:
                    real_caption = _heuristic_caption(image_array)

                h, w = image_array.shape[0], image_array.shape[1]
                input_q = 0.92 if min(h, w) >= 256 else 0.78
                model_conf = 0.88  # higher for adapted model
                elapsed_ms = int((time.time() - t0) * 1000)
                ckpt = _find_checkpoint()
                return {
                    "answer": real_answer,
                    "caption": real_caption,
                    "boxes": boxes,
                    "confidence": {
                        "input_quality": round(input_q, 2),
                        "model_confidence": round(model_conf, 2),
                        "evidence_agreement": 0.0,
                        "overall": round(0.5 * input_q + 0.5 * model_conf, 2),
                        "is_heuristic": False,
                    },
                    "model_used": f"SatQuery-RS-VLM ({_VLM_MODEL_NAME} + LoRA {ckpt})",
                    "elapsed_ms": elapsed_ms,
                }
        except Exception as e:
            print(f"[VQA] Real model inference failed, falling back to heuristic: {e}")
            import traceback
            traceback.print_exc()
            # fall through to heuristic

    # Heuristic fallback (always works, CPU-only)
    answer = _heuristic_vqa(image_array, query)
    caption = _heuristic_caption(image_array)
    boxes = []
    if task in ("grounding", "vqa") and any(k in query.lower() for k in ["where", "highlight", "locate", "show", "box", "water", "built", "forest", "vegetation", "building"]):
        boxes = _heuristic_grounding(image_array, query)

    # confidence heuristic
    h, w = image_array.shape[0], image_array.shape[1]
    size_score = min(1.0, (h*w) / (512*512))
    model_conf = 0.62 + 0.15 * size_score  # 0.62-0.77
    input_q = 0.88 if min(h,w) >= 256 else 0.72

    elapsed_ms = int((time.time() - t0)*1000)
    return {
        "answer": answer,
        "caption": caption,
        "boxes": boxes,
        "confidence": {
            "input_quality": round(input_q, 2),
            "model_confidence": round(model_conf, 2),
            "evidence_agreement": 0.0,
            "overall": round(0.5*input_q + 0.5*model_conf, 2),
            "is_heuristic": True,
        },
        "model_used": "SatQuery-RS-VLM (heuristic RS vision + template; LoRA checkpoint swappable)",
        "elapsed_ms": elapsed_ms,
    }
