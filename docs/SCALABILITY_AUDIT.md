# Scalability Audit

Progression: LOCAL PROTOTYPE (now) -> SINGLE NODE (caps+tiling) -> ASYNC WORKER (queue) -> OBJECT STORAGE+DB -> GPU POOL -> HORIZONTAL.

- 10 images: OK (sequential, <2GB transient if <2048px).
- 100 images: reports/ bloat (b64 JSON), no pagination -> cap previews 512px, paginate `/reports`, prune after N days.
- 10k images: impossible now (no DB/index/STAC, cwd-relative fs). Need object storage + job table.
- 100 concurrent: fails now (sync numpy blocks event loop, singleton VLM no lock). Need `run_in_executor` + queue (arq/celery) + timeout/cancel.
- Large GeoTIFF: fails now (full `ds.read()`, float LAB copies, no overviews/windowed-read). `TilingStrategy` exists but unwired. Wire: `should_tile>1024` -> tiled infer + stitch; enforce `max_tiles 64`; use `rasterio windows + overviews`.
- Multi-band/sub-meter: bands>3 dropped, GSD ignored -> preserve bands, scale by GSD ratio, refuse mismatched GSD without resample note.
- GPU: heuristic has `size_score` penalty; VLM needs 24GB for train, ~6GB infer; gate by env, LRU cache, cold-start preload.
- Transitions: add bytes cap + tiling (single-node); add queue + storage (async); add DB + STAC (10k); add GPU pool autoscale (horizontal).
