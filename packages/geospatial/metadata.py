"""
Geospatial correctness — RasterMetadata extraction, CRS handling, tiling strategy.
Frontend never handles raster logic; this is the single geospatial boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

@dataclass
class TileConfig:
    tile_size: int = 512
    overlap: int = 32
    max_tiles: int = 64  # safety for huge images

@dataclass
class TilingStrategy:
    """Plans tiling for large images without loading full raster into memory."""
    full_h: int
    full_w: int
    tile_size: int = 512
    overlap: int = 32

    def tiles(self) -> List[Tuple[int, int, int, int]]:
        """Returns list of (y0, x0, h, w) — inclusive tiles covering the image."""
        tiles = []
        stride = self.tile_size - self.overlap
        for y in range(0, self.full_h, stride):
            for x in range(0, self.full_w, stride):
                h = min(self.tile_size, self.full_h - y)
                w = min(self.tile_size, self.full_w - x)
                if h < 32 or w < 32:
                    continue
                tiles.append((y, x, h, w))
                if len(tiles) >= 64:
                    return tiles
        return tiles

    def should_tile(self) -> bool:
        return self.full_h > 1024 or self.full_w > 1024

class RasterMetadataExtractor:
    """Centralizes rasterio → metadata logic (currently in validation.py)."""

    @staticmethod
    def validate_dimensions(h: int, w: int) -> tuple[bool, Optional[str], List[str]]:
        warnings: List[str] = []
        if h < 10 or w < 10:
            return False, f"Image too small: {w}x{h}", warnings
        if h < 64 or w < 64:
            warnings.append(f"Small image ({w}x{h}); results may be unreliable")
        if h > 4096 or w > 4096:
            warnings.append(f"Large image ({w}x{h}); will be tiled/resized for inference")
        if h > 10000 or w > 10000:
            warnings.append(f"Very large image ({w}x{h}); tiling required, processing will be chunked")
        return True, None, warnings

    @staticmethod
    def temporal_compatible(meta1: Dict[str, Any], meta2: Dict[str, Any], tol: float = 0.15) -> tuple[bool, str, List[str]]:
        warnings: List[str] = []
        if abs(meta1["width"] - meta2["width"]) > max(meta1["width"], meta2["width"]) * tol:
            return False, f"Width mismatch {meta1['width']} vs {meta2['width']}", warnings
        if abs(meta1["height"] - meta2["height"]) > max(meta1["height"], meta2["height"]) * tol:
            return False, f"Height mismatch {meta1['height']} vs {meta2['height']}", warnings
        if meta1.get("crs") and meta2.get("crs") and meta1["crs"] != meta2["crs"]:
            warnings.append(f"CRS mismatch {meta1['crs']} vs {meta2['crs']}")
        return True, "ok", warnings
