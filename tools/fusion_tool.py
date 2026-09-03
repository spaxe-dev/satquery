"""Fusion engine — decision-level fusion + agreement scoring."""
import numpy as np
import cv2
import io, base64
from PIL import Image
from typing import Dict, Any

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a,b).sum()
    union = np.logical_or(a,b).sum()
    return float(inter / union) if union else 0.0

def _dice(a,b):
    aa = a>0; bb=b>0
    inter = np.logical_and(aa,bb).sum()
    return float(2*inter / (aa.sum()+bb.sum()+1e-9))

def fuse(optical_result: Dict[str, Any], sar_result: Dict[str, Any], h: int, w: int) -> Dict[str, Any]:
    import time
    t0 = time.time()
    # Extract masks — ensure same size
    opt_water = optical_result.get("water_mask")
    opt_built = optical_result.get("built_mask")
    sar_water = sar_result.get("water_mask")
    sar_built = sar_result.get("built_mask")

    # If missing, create empty
    def _prep(mask):
        if mask is None:
            return np.zeros((h,w), dtype=np.uint8)
        if mask.shape[0]!=h or mask.shape[1]!=w:
            return cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        return mask

    opt_water = _prep(opt_water)
    opt_built = _prep(opt_built)
    sar_water = _prep(sar_water)
    sar_built = _prep(sar_built)

    # Agreement scores
    water_iou = _iou(opt_water, sar_water)
    built_iou = _iou(opt_built, sar_built)
    water_dice = _dice(opt_water, sar_water)
    built_dice = _dice(opt_built, sar_built)

    # Fused masks: intersection = high confidence, union = inclusive
    fused_water_high = np.logical_and(opt_water>0, sar_water>0).astype(np.uint8)*255
    fused_water_union = np.logical_or(opt_water>0, sar_water>0).astype(np.uint8)*255
    fused_built_high = np.logical_and(opt_built>0, sar_built>0).astype(np.uint8)*255
    fused_built_union = np.logical_or(opt_built>0, sar_built>0).astype(np.uint8)*255

    water_high_pct = float(np.mean(fused_water_high>0)*100)
    water_union_pct = float(np.mean(fused_water_union>0)*100)
    built_high_pct = float(np.mean(fused_built_high>0)*100)
    built_union_pct = float(np.mean(fused_built_union>0)*100)

    # Agreement level
    avg_iou = (water_iou + built_iou)/2
    if avg_iou > 0.5:
        agreement_label = "strong agreement"
    elif avg_iou > 0.25:
        agreement_label = "moderate agreement"
    elif avg_iou > 0.08:
        agreement_label = "weak agreement"
    else:
        agreement_label = "low agreement / possible disagreement"

    # Corroboration statement
    statements = []
    # Water
    if water_iou > 0.35:
        statements.append(f"Water: optical ({optical_result.get('water_pct',0):.1f}%) and SAR ({sar_result.get('water_pct',0):.1f}%) agree strongly (IoU {water_iou:.2f}); fused high-confidence water {water_high_pct:.1f}% — SAR corroborates optical water detection (SAR water is dark/smooth, robust to cloud).")
    elif water_iou > 0.12:
        statements.append(f"Water: partial agreement (IoU {water_iou:.2f}); optical {optical_result.get('water_pct',0):.1f}% vs SAR {sar_result.get('water_pct',0):.1f}%; fused union {water_union_pct:.1f}% suggests complementary evidence — disagreement may indicate cloud-affected optical or narrow water not resolved in one modality.")
    else:
        if optical_result.get('water_pct',0) > 2 and sar_result.get('water_pct',0) > 2:
            statements.append(f"Water: modalities disagree (IoU {water_iou:.2f}) despite both detecting water-like regions — possible co-registration offset, differing resolutions, or one modality affected by speckle/cloud. Treat fused result as low confidence.")
        elif optical_result.get('water_pct',0) > 2 or sar_result.get('water_pct',0) > 2:
            statements.append(f"Water: only one modality shows clear water (optical {optical_result.get('water_pct',0):.1f}% vs SAR {sar_result.get('water_pct',0):.1f}%) — evidence is not corroborated; SAR can still be useful under cloud, optical under clear sky.")
        else:
            statements.append(f"Water: neither modality shows substantial water (optical {optical_result.get('water_pct',0):.1f}% / SAR {sar_result.get('water_pct',0):.1f}%) — consistent absence.")
    # Built
    if built_iou > 0.30:
        statements.append(f"Built-up: agreement (IoU {built_iou:.2f}); optical {optical_result.get('built_pct',0):.1f}% vs SAR {sar_result.get('built_pct',0):.1f}%; fused high-confidence built {built_high_pct:.1f}% — SAR double-bounce corroborates optical texture evidence.")
    elif built_iou > 0.10:
        statements.append(f"Built-up: partial agreement (IoU {built_iou:.2f}); union {built_union_pct:.1f}% — modalities are complementary; SAR is sensitive to vertical structure, optical to spectral/texture.")
    else:
        statements.append(f"Built-up: low agreement (IoU {built_iou:.2f}); optical {optical_result.get('built_pct',0):.1f}% vs SAR {sar_result.get('built_pct',0):.1f}% — difference reflects sensor physics (SAR sees structure, optical sees reflectance).")

    corroboration = " ".join(statements) + f" Overall cross-modal agreement: {agreement_label} (mean IoU {avg_iou:.2f}, water Dice {water_dice:.2f}, built Dice {built_dice:.2f})."

    # Encoded fused overlays
    def encode(mask, color):
        overlay = np.zeros((h,w,3), dtype=np.uint8)
        overlay[mask>0]=color
        pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    fused_water_b64 = encode(fused_water_high, [30,120,255])
    fused_built_b64 = encode(fused_built_high, [255,180,30])
    # Union as separate
    fused_union_water_b64 = encode(fused_water_union, [80,160,255])

    # Confidence
    input_q = float(np.mean([optical_result["confidence"]["input_quality"], sar_result["confidence"]["input_quality"]]))
    model_conf = float(np.mean([optical_result["confidence"]["model_confidence"], sar_result["confidence"]["model_confidence"]]))
    evidence_agreement = float(0.35 + 0.55*avg_iou)  # map IoU to agreement
    cross_support = float(0.3 + 0.6*avg_iou)
    overall = 0.25*input_q + 0.30*model_conf + 0.25*evidence_agreement + 0.20*cross_support

    elapsed_ms = int((time.time()-t0)*1000)
    return {
        "water_iou": round(float(water_iou),3),
        "built_iou": round(float(built_iou),3),
        "water_dice": round(float(water_dice),3),
        "built_dice": round(float(built_dice),3),
        "mean_iou": round(float(avg_iou),3),
        "agreement_label": agreement_label,
        "fused_water_high_pct": round(water_high_pct,2),
        "fused_water_union_pct": round(water_union_pct,2),
        "fused_built_high_pct": round(built_high_pct,2),
        "fused_built_union_pct": round(built_union_pct,2),
        "corroboration": corroboration,
        "fused_water_b64": fused_water_b64,
        "fused_built_b64": fused_built_b64,
        "fused_union_water_b64": fused_union_water_b64,
        "confidence": {
            "input_quality": round(float(input_q),2),
            "model_confidence": round(float(model_conf),2),
            "evidence_agreement": round(float(evidence_agreement),2),
            "cross_modal_support": round(float(cross_support),2),
            "overall": round(float(overall),2),
            "is_heuristic": True
        },
        "model_used": "FusionEngine (IoU/Dice agreement, intersection=high-conf, union=inclusive)",
        "elapsed_ms": elapsed_ms,
    }
