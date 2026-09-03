from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import io, os, json, base64, datetime, uuid
from PIL import Image
import numpy as np

from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis
from pathlib import Path

router = APIRouter()

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

@router.get("/health")
def health():
    return {"status": "ok", "service": "SatQuery AI", "version": "0.3.0"}

@router.get("/registry")
def registry():
    import yaml, os
    p = os.path.join(os.path.dirname(__file__), "../../../config/tool_registry.yaml")
    # try multiple paths
    candidates = [
        "config/tool_registry.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "tool_registry.yaml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with open(c, "r") as f:
                return yaml.safe_load(f)
    return {"tools": []}

@router.post("/analyze")
async def analyze(
    query: str = Form(...),
    images: List[UploadFile] = File(...),
):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(images) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 images allowed")

    image_datas = []
    for upl in images:
        data = await upl.read()
        if len(data) == 0:
            raise HTTPException(status_code=400, detail=f"Empty file: {upl.filename}")
        # validate
        res = validate_image_file(data, upl.filename or "image.png")
        if not res["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid image {upl.filename}: {res['reason']}")
        # attach original bytes for report
        image_datas.append({
            "bytes": data,
            "filename": upl.filename,
            "array": res["array"],
            "meta": res["meta"],
        })

    try:
        result = run_analysis(query, image_datas)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Convert numpy arrays / non-serializable
    # Prepare JSON-safe response
    # Encode preview thumbnails
    previews = []
    for d in image_datas:
        arr = d["array"]
        # create thumbnail base64
        try:
            # ensure uint8
            if arr.dtype != np.uint8:
                arr_thumb = np.clip(arr,0,255).astype(np.uint8)
            else:
                arr_thumb = arr
            if arr_thumb.ndim==2:
                pil = Image.fromarray(arr_thumb, mode="L").convert("RGB")
            else:
                if arr_thumb.shape[2]>=3:
                    pil = Image.fromarray(arr_thumb[:,:,:3])
                else:
                    pil = Image.fromarray(arr_thumb[:,:,0], mode="L").convert("RGB")
            pil.thumbnail((512,512))
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            previews.append({"filename": d["filename"], "meta": d["meta"], "preview_b64": b64})
        except Exception as e:
            previews.append({"filename": d["filename"], "meta": d["meta"], "preview_b64": None})

    # Clean result for JSON — remove raw masks (numpy)
    def clean(obj):
        if isinstance(obj, np.ndarray):
            return None
        if isinstance(obj, dict):
            return {k: clean(v) for k,v in obj.items() if k not in ("mask", "water_mask","veg_mask","built_mask") or isinstance(v,str) }
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        return obj

    # Build serializable response — includes explicit VLM lifecycle metadata for UI
    resp = {
        "task": result["task"],
        "query": result["query"],
        "answer": result["answer"],
        "caption": result.get("caption"),
        "grounding_boxes": result.get("grounding_boxes", []),
        "change_description": result.get("change_description"),
        "change_area_pct": result.get("change_area_pct"),
        "fused_statement": result.get("fused_statement"),
        "change_map_b64": result.get("change_map_b64"),
        "optical_overlay_b64": result.get("optical_overlay_b64"),
        "sar_overlay_b64": result.get("sar_overlay_b64"),
        "fused_overlay_b64": result.get("fused_overlay_b64"),
        "fused_union_b64": result.get("fused_union_b64"),
        "confidence": result["confidence"],
        "execution_trace": result["execution_trace"],
        "tools_used": result["tools_used"],
        "validation": result["validation"],
        "timestamp": result["timestamp"],
        "report_id": result["report_id"],
        "refused": result.get("refused", False),
        # Explicit VLM lifecycle metadata (safe, no stack traces)
        "vlm_attempted": result.get("vlm_attempted"),
        "vlm_loaded": result.get("vlm_loaded"),
        "vlm_generation_valid": result.get("vlm_generation_valid"),
        "is_heuristic": result.get("is_heuristic", result.get("confidence", {}).get("is_heuristic")),
        "fallback_reason": result.get("fallback_reason"),
        "previews": previews,
        "optical_evidence": {
            "water_pct": result.get("optical_evidence",{}).get("water_pct") if result.get("optical_evidence") else None,
            "veg_pct": result.get("optical_evidence",{}).get("veg_pct") if result.get("optical_evidence") else None,
            "built_pct": result.get("optical_evidence",{}).get("built_pct") if result.get("optical_evidence") else None,
            "evidence": result.get("optical_evidence",{}).get("evidence") if result.get("optical_evidence") else None,
        } if result.get("optical_evidence") else None,
        "sar_evidence": {
            "water_pct": result.get("sar_evidence",{}).get("water_pct") if result.get("sar_evidence") else None,
            "built_pct": result.get("sar_evidence",{}).get("built_pct") if result.get("sar_evidence") else None,
            "mean_backscatter": result.get("sar_evidence",{}).get("mean_backscatter") if result.get("sar_evidence") else None,
            "speckle_cv": result.get("sar_evidence",{}).get("speckle_cv") if result.get("sar_evidence") else None,
            "evidence": result.get("sar_evidence",{}).get("evidence") if result.get("sar_evidence") else None,
        } if result.get("sar_evidence") else None,
        "fusion_evidence": {
            "mean_iou": result.get("fusion_evidence",{}).get("mean_iou") if result.get("fusion_evidence") else None,
            "water_iou": result.get("fusion_evidence",{}).get("water_iou") if result.get("fusion_evidence") else None,
            "built_iou": result.get("fusion_evidence",{}).get("built_iou") if result.get("fusion_evidence") else None,
            "agreement_label": result.get("fusion_evidence",{}).get("agreement_label") if result.get("fusion_evidence") else None,
            "corroboration": result.get("fusion_evidence",{}).get("corroboration") if result.get("fusion_evidence") else None,
        } if result.get("fusion_evidence") else None,
    }

    # Save report JSON
    try:
        report_path = REPORTS_DIR / f"report_{result['report_id']}.json"
        # Build full report with config
        full_report = {
            **resp,
            "query": query,
            "image_metas": [d["meta"] for d in image_datas],
        }
        # remove large b64 from saved file? Keep but maybe large — keep for now
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)
    except Exception as e:
        print(f"Failed to save report: {e}")

    return JSONResponse(content=resp)

@router.get("/reports/{report_id}")
def get_report(report_id: str):
    p = REPORTS_DIR / f"report_{report_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    with open(p, "r") as f:
        data = json.load(f)
    return data

@router.get("/reports")
def list_reports():
    files = sorted(REPORTS_DIR.glob("report_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [{"report_id": f.stem.replace("report_",""), "path": str(f), "mtime": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()} for f in files[:20]]
