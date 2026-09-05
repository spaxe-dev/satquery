# Security & Reliability

## Findings (routes.py, validation.py, main.py, generate_pdf.py, vqa_tool.py)

- Unbounded `await upl.read()` (4 files, no bytes cap) -> OOM/decompression bomb. FIX: `MAX_BYTES=25MB`, `Image.MAX_IMAGE_PIXELS`, pre-check `w*h*bands`.
- Ext-only check, no magic/MIME. FIX: magic via rasterio/PIL already tries; add explicit reject + `python-magic` note.
- Traversal: `get_report(report_id)` no regex; `generate_pdf _tmp_{id}_{filename}` uses upload filename. FIX: `^[a-f0-9]{8}$`, sanitize filename via basename + uuid.
- CWD-relative `REPORTS_DIR`, b64 persisted unbounded. FIX: absolute from settings, strip b64 >512px, retention job.
- CORS `*`+credentials, no auth/rate-limit. FIX: env allowlist, note demo-open, add rate-limit TODO.
- `trust_remote_code=True` + auto-load any adapter_config. FIX: gate by `SATQUERY_ENABLE_VLM`, allowlist checkpoint dirs, `local_files_only` first.
- `traceback.print_exc` + 500 `{e}` leaks internals. FIX: structured error code, no stack to client.
- Reliability: no timeout/retry/cancel/queue; singleton VLM no lock. FIX: per-tool timeout, `run_in_executor`, job id + cancel (roadmap).

Limits: 4 images, 25MB each, 4096px warn/tiling, 10px min refuse, 15% temporal refuse, 25% opt-SAR warn.
