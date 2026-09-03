"""
Orchestrator — glues router + validation + tools + fusion + confidence + trace.
This is the agent's execution core.
"""
import time
import uuid
import datetime
import traceback
from typing import List, Dict, Any, Optional
import numpy as np

from agent.router import classify_query, decide_task, plan_tools, validate_inputs
from backend.app.services.validation import validate_image_file, validate_temporal_pair, validate_optical_sar_pair

# Import tools
from tools import vqa_tool, change_tool, optical_tool, sar_tool, fusion_tool

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def run_analysis(query: str, image_datas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    image_datas: list of dicts {bytes, filename, array (np), meta}
    Returns dict serializable to AnalysisResult
    """
    trace = []
    t_start = time.time()
    report_id = str(uuid.uuid4())[:8]

    def step(n, action, detail, status="ok", tool=None, dur=None):
        trace.append({"step": n, "action": action, "detail": detail, "status": status, "tool": tool, "duration_ms": dur})

    # Step 1: Input received
    step(1, "Input received", f"{len(image_datas)} image(s) + query: '{query[:80]}'", "ok")
    # Step 2: Metadata inspected
    metas = [d["meta"] for d in image_datas]
    arrays = [d["array"] for d in image_datas]
    modalities = [m["modality"] for m in metas]
    for i, m in enumerate(metas):
        step(2+i, "Metadata inspected", f"Image {i+1}: {m['width']}x{m['height']} x{m['bands']} | {m['format']} | modality={m['modality']} | georef={m['has_georef']}", "ok")
    base_step = 2 + len(metas)

    # Step: Query classified
    intents = classify_query(query)
    task = decide_task(intents, len(image_datas), modalities, query)
    step(base_step, "Query classified", f"Intents={intents} -> Task={task}", "ok")
    base_step += 1

    # Step: Input compatibility validated
    valid, reason, warnings = validate_inputs(len(image_datas), modalities, task, metas)
    # Additional pair-specific validation
    extra_warnings = []
    if task in ("temporal_change", "compound_temporal_sar") and len(metas)>=2:
        ok2, r2, w2 = validate_temporal_pair(metas[0], metas[1])
        if not ok2:
            valid, reason = False, r2
        extra_warnings.extend(w2)
    if task=="optical_sar_fusion" and len(metas)>=2:
        ok2, r2, w2 = validate_optical_sar_pair(metas[0], metas[1])
        extra_warnings.extend(w2)
    all_warnings = warnings + extra_warnings
    if not valid:
        step(base_step, "Input compatibility validated", f"FAILED: {reason}", "error")
        # Return refusal — this is correct trust behavior
        overall_conf = {"input_quality": 0.0, "model_confidence": 0.0, "evidence_agreement": 0.0, "overall": 0.0, "is_heuristic": True}
        return {
            "task": task,
            "query": query,
            "answer": f"Analysis refused. Reason: {reason}. No specialist model was executed due to input incompatibility. Please provide compatible imagery (e.g., co-registered pair of similar dimensions for temporal analysis).",
            "caption": None,
            "grounding_boxes": [],
            "change_description": None,
            "change_area_pct": None,
            "fused_statement": None,
            "optical_evidence": None,
            "sar_evidence": None,
            "fusion_evidence": None,
            "change_map_b64": None,
            "optical_overlay_b64": None,
            "sar_overlay_b64": None,
            "fused_overlay_b64": None,
            "confidence": overall_conf,
            "execution_trace": trace,
            "tools_used": [],
            "validation": {"valid": False, "reason": reason, "warnings": all_warnings, "details": {}},
            "timestamp": _now_iso(),
            "report_id": report_id,
            "refused": True,
            "vlm_attempted": None,
            "vlm_loaded": None,
            "vlm_generation_valid": None,
            "fallback_reason": None,
            "is_heuristic": True,
        }
    else:
        detail = "All inputs compatible"
        if all_warnings:
            detail += f" | Warnings: {'; '.join(all_warnings)}"
        step(base_step, "Input compatibility validated", detail, "warn" if all_warnings else "ok")
    base_step += 1

    # Step: Tool selection
    plan = plan_tools(task, len(image_datas), modalities)
    tools_used = []
    for p in plan:
        tools_used.append({"tool_id": p["tool_id"], "tool_name": p["tool_id"], "reason": p["reason"], "params": p["params"]})
        step(base_step, "Tool selected", f"{p['tool_id']}: {p['reason']}", "ok", tool=p["tool_id"])
        base_step += 1

    # Execute tools
    results = {}
    answer_parts = []
    caption = None
    grounding_boxes = []
    change_description = None
    change_area_pct = None
    fused_statement = None

    change_map_b64 = None
    optical_overlay_b64 = None
    sar_overlay_b64 = None
    fused_overlay_b64 = None
    fused_union_b64 = None

    optical_res = None
    sar_res = None
    fusion_res = None
    change_res = None
    vqa_res = None

    confidences = []

    for p in plan:
        tid = p["tool_id"]
        t0 = time.time()
        try:
            if tid == "satquery-rs-vlm":
                # choose appropriate array
                arr = arrays[0] if arrays else None
                # for change follow-up, use T2 if available
                if p["params"].get("context")=="change" and len(arrays)>=2:
                    arr = arrays[1]
                task_param = p["params"].get("task", "vqa")
                res = vqa_tool.analyze(query, arr, task=task_param)
                vqa_res = res
                answer_parts.append(res["answer"])
                if task_param in ("captioning","vqa"):
                    caption = res["caption"]
                grounding_boxes = res["boxes"]
                confidences.append(res["confidence"])
                # Explicit VLM lifecycle metadata for UI: attempted -> loaded -> generation_valid -> heuristic/fallback
                vlm_attempted = res.get("vlm_attempted", False)
                vlm_loaded = res.get("vlm_loaded", False)
                vlm_valid = res.get("vlm_generation_valid")
                fallback_reason = res.get("fallback_reason")
                is_heu = res.get("is_heuristic", res["confidence"].get("is_heuristic", True))
                # Build human-readable VLM status line for trace
                if vlm_valid is True:
                    vlm_detail = f"VLM: attempted -> loaded -> generated -> accepted (heuristic={is_heu}, fallback={fallback_reason})"
                    vlm_status = "ok"
                elif vlm_valid is False:
                    vlm_detail = f"VLM: attempted -> {'loaded' if vlm_loaded else 'not loaded'} -> rejected -> heuristic fallback (reason={fallback_reason})"
                    vlm_status = "warn"
                elif vlm_loaded:
                    vlm_detail = f"VLM: attempted -> loaded -> no valid generation (reason={fallback_reason}) -> heuristic fallback"
                    vlm_status = "warn"
                elif vlm_attempted:
                    vlm_detail = f"VLM: attempted -> not loaded (reason={fallback_reason}) -> heuristic fallback"
                    vlm_status = "warn"
                else:
                    vlm_detail = f"VLM: not attempted (reason={fallback_reason})"
                    vlm_status = "ok"
                step(base_step, "Executed", f"SatQuery-RS-VLM ({task_param}): {res['answer'][:100]}...", "ok", tool=tid, dur=res["elapsed_ms"])
                base_step += 1
                step(base_step, "VLM status", vlm_detail, vlm_status, tool=tid)
            elif tid == "change-detector":
                if len(arrays)>=2:
                    res = change_tool.analyze(arrays[0], arrays[1], query)
                    change_res = res
                    change_description = res["description"]
                    change_area_pct = res["changed_pct"]
                    change_map_b64 = res["change_map_b64"]
                    # overlay is mask overlay
                    # produce blended preview as change_map for now
                    answer_parts.append(res["description"])
                    confidences.append(res["confidence"])
                    step(base_step, "Executed", f"ChangeDetector: {res['changed_pct']}% changed, thresh={res['otsu_thresh']:.1f}", "ok", tool=tid, dur=res["elapsed_ms"])
                else:
                    step(base_step, "Executed", "ChangeDetector skipped: need 2 images", "warn", tool=tid)
            elif tid == "optical-analyzer":
                # Heuristic: pick first optical-like image, or first
                idx = 0
                for i, m in enumerate(metas):
                    if m["modality"]=="optical":
                        idx=i; break
                # if both are sar, still run on first but note
                res = optical_tool.analyze(arrays[idx], query)
                optical_res = res
                answer_parts.append(res["answer"])
                optical_overlay_b64 = res["water_overlay_b64"]
                confidences.append(res["confidence"])
                step(base_step, "Executed", f"OpticalAnalyzer: water {res['water_pct']}% veg {res['veg_pct']}% built {res['built_pct']}%", "ok", tool=tid, dur=res["elapsed_ms"])
            elif tid == "sar-analyzer":
                idx = 0
                for i, m in enumerate(metas):
                    if m["modality"]=="sar":
                        idx=i; break
                    if m["modality"]=="unknown" and len(arrays)>=2:
                        # second image as SAR fallback
                        idx = 1 if len(arrays)>1 else 0
                        break
                # For compound case with 3 images, SAR is last
                if task=="compound_temporal_sar" and len(arrays)>=3:
                    idx = 2
                elif task=="compound_temporal_sar" and len(arrays)==2:
                    # T1,T2 are temporal; synthesize SAR corroboration from T2 analysis
                    idx = 1
                res = sar_tool.analyze(arrays[idx], query)
                sar_res = res
                # For compound, answer is corroboration perspective
                answer_parts.append(res["answer"])
                sar_overlay_b64 = res["water_overlay_b64"]
                confidences.append(res["confidence"])
                step(base_step, "Executed", f"SARAnalyzer: water {res['water_pct']}% built {res['built_pct']}% speckle CV {res['speckle_cv']}", "ok", tool=tid, dur=res["elapsed_ms"])
            elif tid == "fusion-engine":
                # Need optical + sar results
                if optical_res is None and sar_res is None:
                    # For compound: synthesize optical from change context if needed
                    step(base_step, "Executed", "FusionEngine skipped: missing branch results", "warn", tool=tid)
                else:
                    # Ensure we have both; if one missing, create dummy
                    if optical_res is None:
                        # Use first image as optical proxy
                        optical_res = optical_tool.analyze(arrays[0], query)
                    if sar_res is None:
                        sar_res = sar_tool.analyze(arrays[-1], query)
                    h,w = arrays[0].shape[0], arrays[0].shape[1]
                    res = fusion_tool.fuse(optical_res, sar_res, h, w)
                    fusion_res = res
                    fused_statement = res["corroboration"]
                    answer_parts.append(res["corroboration"])
                    fused_overlay_b64 = res["fused_water_b64"]
                    fused_union_b64 = res["fused_union_water_b64"]
                    confidences.append(res["confidence"])
                    step(base_step, "Executed", f"FusionEngine: mean IoU {res['mean_iou']:.2f} -> {res['agreement_label']}", "ok", tool=tid, dur=res["elapsed_ms"])
            base_step += 1
        except Exception as e:
            traceback.print_exc()
            step(base_step, "Executed", f"{tid} failed: {e}", "error", tool=tid)
            base_step += 1

    # Evidence fusion + overall confidence
    # Aggregate confidences
    if confidences:
        # weighted overall
        input_q = float(np.mean([c["input_quality"] for c in confidences]))
        model_c = float(np.mean([c["model_confidence"] for c in confidences]))
        ev_agree = float(np.mean([c.get("evidence_agreement", 0.0) for c in confidences]))
        # cross-modal if exists
        cross = None
        if fusion_res and "cross_modal_support" in fusion_res["confidence"]:
            cross = fusion_res["confidence"]["cross_modal_support"]
            ev_agree = fusion_res["confidence"]["evidence_agreement"]
        overall = float(np.mean([c["overall"] for c in confidences]))
        # if fusion exists, use its overall as more informed
        if fusion_res:
            overall = fusion_res["confidence"]["overall"]
            cross = fusion_res["confidence"]["cross_modal_support"]
        # Fix: is_heuristic must reflect actual tool execution, not hardcoded True.
        # For single-image VQA/captioning/grounding, the VLM's flag is authoritative.
        # For multi-tool tasks, overall is heuristic if all contributions are heuristic.
        if vqa_res is not None and task in ("single_vqa", "captioning", "grounding"):
            is_heuristic_flag = bool(vqa_res["confidence"].get("is_heuristic", True))
        else:
            is_heuristic_flag = all(c.get("is_heuristic", True) for c in confidences)
        conf = {
            "input_quality": round(input_q,2),
            "model_confidence": round(model_c,2),
            "evidence_agreement": round(ev_agree,2),
            "cross_modal_support": round(cross,2) if cross is not None else None,
            "overall": round(overall,2),
            "is_heuristic": is_heuristic_flag
        }
    else:
        conf = {"input_quality": 0.75, "model_confidence": 0.60, "evidence_agreement": 0.60, "overall": 0.65, "is_heuristic": True}

    # Final answer: combine parts
    if task == "temporal_change":
        final_answer = change_description or " ".join(answer_parts) if answer_parts else "No change description available."
        # Add built-up QA if asked
        if vqa_res and "built" in query.lower():
            final_answer += f" {vqa_res['answer']}"
    elif task == "optical_sar_fusion":
        final_answer = " ".join(answer_parts) if answer_parts else "No fusion result."
        if fusion_res:
            final_answer = f"{optical_res['answer']} {sar_res['answer']} {fusion_res['corroboration']}"
    elif task == "compound_temporal_sar":
        # Hero demo: combine change + SAR corroboration
        parts = []
        if change_res:
            parts.append(f"Temporal: {change_res['description']}")
        if sar_res:
            # Check if SAR corroborates inundation/vegetation loss
            if change_res and change_res.get("changed_pct",0) > 1:
                # SAR water increase would corroborate flood
                if sar_res["water_pct"] > 3:
                    parts.append(f"SAR corroboration: SAR shows {sar_res['water_pct']:.1f}% water-like low-backscatter regions — CONSISTENT with inundation/change. {sar_res['answer']}")
                else:
                    parts.append(f"SAR corroboration: SAR shows only {sar_res['water_pct']:.1f}% water-like regions — WEAK corroboration for water-driven change; change may be non-hydrological or SAR acquisition differs.")
            else:
                parts.append(f"SAR: {sar_res['answer']}")
        if fusion_res:
            parts.append(f"Agreement: {fusion_res['agreement_label']} (mean IoU {fusion_res['mean_iou']:.2f})")
        final_answer = " ".join(parts) if parts else " ".join(answer_parts)
    else:
        final_answer = " ".join(answer_parts) if answer_parts else "No answer generated."

    heuristic_label = "" if not conf.get("is_heuristic", True) else " (heuristic)"
    step(base_step, "Evidence fusion", f"Combined {len(answer_parts)} evidence streams; overall confidence {conf['overall']:.2f}{heuristic_label}", "ok")
    base_step+=1
    step(base_step, "Confidence estimated", f"Input {conf['input_quality']:.2f} | Model {conf['model_confidence']:.2f} | Evidence {conf['evidence_agreement']:.2f}" + (f" | Cross-modal {conf['cross_modal_support']:.2f}" if conf['cross_modal_support'] is not None else ""), "ok")
    base_step+=1
    step(base_step, "Final answer generated", f"Answer length {len(final_answer)} chars; task={task}", "ok")
    base_step+=1

    # Visual evidence selection
    # For temporal, change_map is primary; for fusion, fused overlay; for single, grounding boxes overlay is generated frontend

    # Explicit VLM lifecycle metadata for UI (every VQA/captioning execution)
    if vqa_res is not None:
        vlm_attempted_out = vqa_res.get("vlm_attempted", False)
        vlm_loaded_out = vqa_res.get("vlm_loaded", False)
        vlm_generation_valid_out = vqa_res.get("vlm_generation_valid")
        fallback_reason_out = vqa_res.get("fallback_reason")
        vlm_is_heuristic_out = vqa_res.get("is_heuristic", conf.get("is_heuristic", True))
    else:
        vlm_attempted_out = None
        vlm_loaded_out = None
        vlm_generation_valid_out = None
        fallback_reason_out = None
        vlm_is_heuristic_out = conf.get("is_heuristic", True)

    return {
        "task": task,
        "query": query,
        "answer": final_answer,
        "caption": caption,
        "grounding_boxes": grounding_boxes,
        "change_description": change_description,
        "change_area_pct": change_area_pct,
        "fused_statement": fused_statement,
        "optical_evidence": optical_res,
        "sar_evidence": sar_res,
        "fusion_evidence": fusion_res,
        "change_res": change_res,
        "change_map_b64": change_map_b64,
        "optical_overlay_b64": optical_overlay_b64,
        "sar_overlay_b64": sar_overlay_b64,
        "fused_overlay_b64": fused_overlay_b64,
        "fused_union_b64": fused_union_b64,
        "confidence": conf,
        "execution_trace": trace,
        "tools_used": tools_used,
        "validation": {"valid": True, "reason": None, "warnings": all_warnings, "details": {}},
        "timestamp": _now_iso(),
        "report_id": report_id,
        "refused": False,
        # Explicit VLM lifecycle metadata (safe, no stack traces)
        "vlm_attempted": vlm_attempted_out,
        "vlm_loaded": vlm_loaded_out,
        "vlm_generation_valid": vlm_generation_valid_out,
        "fallback_reason": fallback_reason_out,
        "is_heuristic": vlm_is_heuristic_out,
    }
