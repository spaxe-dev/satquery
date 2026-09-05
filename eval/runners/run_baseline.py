"""Eval runner: executes cases via run_analysis, records provenance. No ground-truth invention."""
import os, sys, json, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis

def load(p):
    with open(p, 'rb') as f:
        d = f.read()
    r = validate_image_file(d, os.path.basename(p))
    return {'bytes': d, 'filename': os.path.basename(p), 'array': r['array'], 'meta': r['meta']}

CASES = [
    {"case_id": "single_vqa", "task": "captioning", "files": ["data/demo/single_image/scene_1.png"], "query": "Describe the land-cover and major objects visible in this image.", "eval_type": "REGRESSION", "expect_refusal": False, "expected_task": "captioning", "min_trace": 5},
    {"case_id": "temporal_change", "task": "temporal_change", "files": ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png"], "query": "What changed between these two dates, and where did the change occur?", "eval_type": "REGRESSION", "expect_refusal": False, "expected_task": "temporal_change", "min_trace": 5},
    {"case_id": "optical_sar", "task": "optical_sar_fusion", "files": ["data/demo/optical_sar_pairs/pair1_optical.png", "data/demo/optical_sar_pairs/pair1_sar.png"], "query": "Use the optical and SAR images together to identify built-up and water-covered regions.", "eval_type": "REGRESSION", "expect_refusal": False, "expected_task": "optical_sar_fusion", "min_trace": 5},
    {"case_id": "compound_hero", "task": "compound_temporal_sar", "files": ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png", "data/demo/optical_sar_pairs/flood_sar.png"], "query": "What changed between the two dates, and can the SAR imagery corroborate the change?", "eval_type": "REGRESSION", "expect_refusal": False, "expected_task": "compound_temporal_sar", "min_trace": 5},
    {"case_id": "invalid_refusal", "task": "temporal_change", "files": ["data/demo/invalid_inputs/small_128.png", "data/demo/invalid_inputs/large_512.png"], "query": "What changed between these two dates?", "eval_type": "REGRESSION", "expect_refusal": True, "expected_task": "temporal_change", "min_trace": 2},
    {"case_id": "grounding_probe", "task": "grounding", "files": ["data/demo/single_image/scene_1.png"], "query": "Highlight the water body referred to in the query.", "eval_type": "QUALITATIVE", "expect_refusal": False, "expected_task": "grounding", "min_trace": 3, "note": "boxes are heuristic uncalibrated; review visually only"},
]

if __name__ == "__main__":
    impl = os.environ.get("SATQUERY_EVAL_IMPL", "heuristic-baseline-v0")
    out = {"impl": impl, "ts": datetime.datetime.utcnow().isoformat() + "Z", "results": []}
    for c in CASES:
        t0 = time.time()
        try:
            datas = [load(p) for p in c["files"]]
            res = run_analysis(c["query"], datas)
            status = "REFUSED" if res.get("refused") else "OK"
            passed = (res.get("refused") == c["expect_refusal"]) and (len(res.get("execution_trace", [])) >= c["min_trace"])
            if not c["expect_refusal"]:
                passed = passed and bool(res.get("answer")) and res.get("task") == c["expected_task"]
            out["results"].append({"case_id": c["case_id"], "implementation": impl, "model_tool": ",".join([t.get("tool_id", "") for t in res.get("tools_used", [])]), "heuristic": res.get("is_heuristic", True), "time_s": round(time.time() - t0, 2), "status": status, "passed": passed, "task": res.get("task"), "confidence": res.get("confidence"), "warnings": res.get("validation", {}).get("warnings", []), "eval_type": c["eval_type"]})
            print(f"{c['case_id']}: {status} passed={passed} task={res.get('task')}")
        except Exception as e:
            out["results"].append({"case_id": c["case_id"], "implementation": impl, "status": "ERROR", "passed": False, "error": str(e), "eval_type": c["eval_type"]})
            print(f"{c['case_id']}: ERROR {e}")
    os.makedirs("eval/reports", exist_ok=True)
    fn = f"eval/reports/baseline_{impl}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"wrote {fn}")
    n_pass = sum(1 for r in out["results"] if r.get("passed"))
    print(f"{n_pass}/{len(out['results'])} passed")
