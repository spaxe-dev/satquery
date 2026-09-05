"""Registration status + coordinate transforms (SOURCE<->PROCESSING<->PREVIEW, aspect-preserved)."""
from __future__ import annotations
from typing import Dict, Any, Tuple


def registration_status(m1: Dict[str, Any], m2: Dict[str, Any]) -> str:
    """unregistered | approximately_aligned | registered (structured, never silent)."""
    c1, c2 = m1.get("crs"), m2.get("crs")
    b1, b2 = m1.get("bounds"), m2.get("bounds")
    t1, t2 = m1.get("transform_list"), m2.get("transform_list")
    if c1 and c2 and b1 and b2:
        if c1 == c2 and b1 == b2 and (t1 == t2 or (t1 is None or t2 is None)):
            return "registered"
        if c1 == c2:
            return "approximately_aligned"
    # pixel fallback: same dims => approx, else unregistered
    if m1.get("width") == m2.get("width") and m1.get("height") == m2.get("height"):
        return "approximately_aligned" if (m1.get("has_georef") or m2.get("has_georef")) else "unregistered"
    return "unregistered"


def preview_scale(src_w: int, src_h: int, preview_w: int, preview_h: int) -> Tuple[float, float, float, float]:
    """Aspect-preserved fit: returns (scale, pad_x, pad_y, fitted) for letterbox mapping. No stretch."""
    s = min(preview_w / src_w, preview_h / src_h)
    fitted_w, fitted_h = src_w * s, src_h * s
    return s, (preview_w - fitted_w) / 2.0, (preview_h - fitted_h) / 2.0, s


def box_to_preview(box: Dict[str, float], src_w: int, src_h: int, preview_w: int, preview_h: int) -> Dict[str, float]:
    s, px, py, _ = preview_scale(src_w, src_h, preview_w, preview_h)
    return {"x": box["x"] * s + px, "y": box["y"] * s + py, "w": box["w"] * s, "h": box["h"] * s}
