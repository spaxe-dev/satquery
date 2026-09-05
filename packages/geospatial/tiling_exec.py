"""Tile → execute → aggregate for single-image stats tools (bounded memory, provenance per tile)."""
from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
from packages.geospatial.metadata import TilingStrategy


def execute_tiled(tool_id: str, array: np.ndarray, meta: Dict[str, Any], query: str, params: Dict[str, Any], dispatch_fn) -> Dict[str, Any]:
    h, w = array.shape[0], array.shape[1]
    strat = TilingStrategy(full_h=h, full_w=w)
    tiles = strat.tiles()
    results = []
    for i, (y0, x0, th, tw) in enumerate(tiles):
        chip = array[y0:y0 + th, x0:x0 + tw]
        r = dispatch_fn(tool_id, [chip], query, params)
        area = th * tw
        results.append({"tile_id": i, "window": [x0, y0, tw, th], "area": area, "result": r})
    # Aggregate pct stats area-weighted where present
    total = sum(r["area"] for r in results) or 1
    agg: Dict[str, Any] = {}
    for key in ("water_pct", "veg_pct", "built_pct"):
        vals = [(r["result"].get(key), r["area"]) for r in results if isinstance(r["result"].get(key), (int, float))]
        if vals:
            agg[key] = round(sum(v * a for v, a in vals) / total, 2)
    first = results[0]["result"] if results else {}
    out = dict(first)
    out.update(agg)
    out["provenance"] = {"tiled": True, "tiles": len(results), "tile_size": strat.tile_size, "overlap": strat.overlap,
                         "source": {"w": w, "h": h, "crs": meta.get("crs"), "transform": meta.get("transform_list"), "bounds": meta.get("bounds")},
                         "tile_windows": [r["window"] for r in results]}
    out["answer"] = (first.get("answer", "") + f" [tiled {len(results)} windows, area-weighted]").strip()
    return out
