"""
Input validation — metadata-first, flag-don't-reject where possible.
Supports GeoTIFF/TIFF via rasterio if available, else PIL fallback.
"""
import os
import io
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

MAX_PIXELS = 16_000_000  # warn above, refuse above 64M
HARD_MAX_PIXELS = 64_000_000
MAX_BANDS = 12

# Heuristic: SAR images are typically single-band, low dynamic range, speckle texture
def _detect_modality_from_array(arr: np.ndarray, filename: str = "") -> str:
    fname = filename.lower()
    # filename hints
    if "sar" in fname or "s1" in fname or "risat" in fname:
        return "sar"
    if "optical" in fname or "s2" in fname or "sentinel2" in fname or "cartosat" in fname:
        return "optical"
    # band count heuristic
    if arr.ndim == 2:
        return "sar"
    if arr.ndim == 3:
        if arr.shape[2] == 1 or (arr.ndim == 3 and arr.shape[0] == 1):
            return "sar"
        # check if single channel repeated
        if arr.shape[2] >= 3:
            # optical is likely RGB; check color variance
            # SAR false-color composites can look grayish — but optical has more color diversity
            r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
            color_var = np.std([np.mean(r), np.mean(g), np.mean(b)])
            # very low color variance + low saturation suggests SAR composite or panchromatic
            # but we err toward optical for 3-band
            return "optical"
    return "unknown"


def validate_image_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = os.path.splitext(filename)[1].lower()
    result = {
        "valid": True,
        "reason": None,
        "warnings": [],
        "meta": {},
        "array": None,
    }
    if ext not in SUPPORTED_EXTS:
        result["valid"] = False
        result["reason"] = f"Unsupported format '{ext}'. Supported: {SUPPORTED_EXTS}"
        return result

    # Try rasterio first for GeoTIFF
    meta = {}
    arr = None
    if HAS_RASTERIO and ext in (".tif", ".tiff"):
        try:
            with rasterio.MemoryFile(file_bytes) as mem:
                with mem.open() as ds:
                    meta["width"] = ds.width
                    meta["height"] = ds.height
                    meta["bands"] = ds.count
                    meta["crs"] = str(ds.crs) if ds.crs else None
                    meta["transform"] = str(ds.transform) if ds.transform else None
                    # Numeric preservation (no longer read-then-discard only)
                    try:
                        meta["transform_list"] = list(ds.transform)[:6] if ds.transform else None
                    except Exception:
                        meta["transform_list"] = None
                    try:
                        b = ds.bounds
                        meta["bounds"] = [float(b.left), float(b.bottom), float(b.right), float(b.top)] if b else None
                    except Exception:
                        meta["bounds"] = None
                    meta["nodata"] = float(ds.nodata) if ds.nodata is not None else None
                    meta["gsd_m"] = None  # populated when catalog provides it; never invented
                    meta["dtype"] = str(ds.dtypes[0]) if ds.dtypes else None
                    meta["has_georef"] = ds.crs is not None
                    # read as array (first 3 bands max for preview)
                    count = min(ds.count, 3)
                    arr = ds.read(indexes=list(range(1, count+1)))
                    # rasterio gives (bands, H, W) -> convert to HWC if needed
                    if arr.ndim == 3:
                        arr = np.transpose(arr, (1,2,0))
                        if arr.shape[2] == 1:
                            arr = arr[:,:,0]
                    # normalize for preview if needed
                    if arr.dtype != np.uint8:
                        # scale to 0-255 for processing
                        arr = _normalize_to_uint8(arr)
                    meta["format"] = ext
        except Exception as e:
            result["warnings"].append(f"Rasterio read warning: {e}; falling back to PIL")

    if arr is None:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.load()
            # handle palette etc
            if img.mode == "P":
                img = img.convert("RGB")
            elif img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            arr = np.array(img)
            meta["width"] = img.width
            meta["height"] = img.height
            meta["bands"] = 1 if arr.ndim == 2 else arr.shape[2] if arr.ndim == 3 else 1
            meta["format"] = ext
            meta["crs"] = None
            meta["transform"] = None
            meta["transform_list"] = None
            meta["bounds"] = None
            meta["nodata"] = None
            meta["gsd_m"] = None
            meta["has_georef"] = False
            meta["dtype"] = str(arr.dtype)
        except Exception as e:
            result["valid"] = False
            result["reason"] = f"Cannot decode image: {e}"
            return result

    # dimension checks
    h, w = meta["height"], meta["width"]
    if meta.get("bands", 1) and meta["bands"] > MAX_BANDS:
        result["valid"] = False
        result["reason"] = f"Too many bands: {meta['bands']} (max {MAX_BANDS})"
        return result
    npix = h * w
    if npix > HARD_MAX_PIXELS:
        result["valid"] = False
        result["reason"] = f"Image too large: {w}x{h} ({npix} px, max {HARD_MAX_PIXELS})"
        return result
    if npix > MAX_PIXELS:
        result["warnings"].append(f"Large image ({w}x{h}); tiled/windowed processing required")
    if h < 64 or w < 64:
        result["warnings"].append(f"Image very small ({w}x{h}); results may be unreliable")
    if h > 4096 or w > 4096:
        result["warnings"].append(f"Image large ({w}x{h}); will be resized for inference")
    if h < 10 or w < 10:
        result["valid"] = False
        result["reason"] = f"Image dimensions too small: {w}x{h}"
        return result

    modality = _detect_modality_from_array(arr, filename)
    meta["modality"] = modality
    result["meta"] = meta
    result["array"] = arr
    return result


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:,:,:3]
    # handle float data (reflectance etc)
    if arr.dtype in (np.float32, np.float64):
        # clip and scale assuming 0-1 or 0-10000 reflectance
        if arr.max() <= 1.5:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        elif arr.max() <= 10000:
            arr = (arr / 10000 * 255).clip(0,255).astype(np.uint8)
        else:
            # generic percentile stretch
            p2, p98 = np.percentile(arr, (2,98))
            arr = ((arr - p2) / max(p98 - p2, 1) * 255).clip(0,255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr / 65535 * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def validate_temporal_pair(meta1: Dict, meta2: Dict) -> Tuple[bool, str, List[str]]:
    warnings = []
    # dimension compatibility
    if abs(meta1["width"] - meta2["width"]) > max(meta1["width"], meta2["width"]) * 0.15:
        return False, f"Width mismatch: {meta1['width']} vs {meta2['width']} — cannot be same area", warnings
    if abs(meta1["height"] - meta2["height"]) > max(meta1["height"], meta2["height"]) * 0.15:
        return False, f"Height mismatch: {meta1['height']} vs {meta2['height']} — cannot be same area", warnings
    # CRS check if available
    if meta1.get("crs") and meta2.get("crs") and meta1["crs"] != meta2["crs"]:
        warnings.append(f"CRS mismatch: {meta1['crs']} vs {meta2['crs']}")
    # modality should match for temporal (both optical or both SAR ideally)
    # but we allow cross but warn
    if meta1.get("modality") != meta2.get("modality") and meta1.get("modality") != "unknown" and meta2.get("modality") != "unknown":
        warnings.append(f"Modality mismatch in temporal pair: {meta1['modality']} vs {meta2['modality']} — analysis will proceed but may be less reliable")
    return True, "ok", warnings


def validate_optical_sar_pair(meta_opt: Dict, meta_sar: Dict) -> Tuple[bool, str, List[str]]:
    warnings = []
    # allow either order, but check at least one is sar-like
    mods = {meta_opt.get("modality"), meta_sar.get("modality")}
    if "sar" not in mods and "unknown" not in mods:
        warnings.append("Neither image detected as SAR; cross-modal analysis may be limited — proceeding with heuristic modality assignment")
    # spatial overlap proxy: dimension similarity
    if abs(meta_opt["width"] - meta_sar["width"]) > max(meta_opt["width"], meta_sar["width"]) * 0.25:
        warnings.append(f"Size differs notably: {meta_opt['width']}x{meta_opt['height']} vs {meta_sar['width']}x{meta_sar['height']} — assuming same area but results are approximate")
    return True, "ok", warnings
