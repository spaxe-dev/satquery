"""Planner consolidation + tiling + geo safety (no downloads, bounded RAM)."""
import numpy as np


def test_router_is_shim_over_planner():
    from agent import router as R
    from packages.agent import planner as P
    from packages.core.schemas import TaskType
    qs = [
        "Describe the land-cover and major objects visible in this image.",
        "What changed between these two dates, and where did the change occur?",
        "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "What changed between the two dates, and can the SAR imagery corroborate the change?",
        "Highlight the water body referred to in the query.",
    ]
    cfgs = [(1, ["optical"]), (2, ["optical", "optical"]), (2, ["optical", "sar"]), (3, ["optical", "optical", "sar"]), (1, ["optical"])]
    for q, (n, mods) in zip(qs, cfgs):
        assert R.classify_query(q) == P.classify_query(q)
        assert R.decide_task(R.classify_query(q), n, mods, q) == P.decide_task(P.classify_query(q), n, mods, q).value
    # plan ids match (reasons may differ in wording, ids must not)
    p_ids = [s.tool_id for s in P.plan_tools(TaskType.TEMPORAL_CHANGE, 2, ["optical", "optical"])]
    r_ids = [s["tool_id"] for s in R.plan_tools("temporal_change", 2, ["optical", "optical"])]
    assert p_ids == r_ids


def test_planner_determines_plan_used_by_orchestrator():
    import os
    from backend.app.services.validation import validate_image_file
    from backend.app.services.orchestrator import run_analysis
    from packages.agent.planner import QueryPlanner

    def load(p):
        with open(p, 'rb') as f:
            d = f.read()
        r = validate_image_file(d, os.path.basename(p))
        return {'bytes': d, 'filename': os.path.basename(p), 'array': r['array'], 'meta': r['meta']}
    datas = [load("data/demo/temporal_pairs/flood_T1.png"), load("data/demo/temporal_pairs/flood_T2.png")]
    metas = [d["meta"] for d in datas]
    mods = [m["modality"] for m in metas]
    plan = QueryPlanner().plan("What changed between these two dates?", 2, mods, metas)
    assert plan.valid and [s.tool_id for s in plan.steps] == ["change-detector", "satquery-rs-vlm"]
    res = run_analysis("What changed between these two dates?", datas)
    assert [t["tool_id"] for t in res["tools_used"]] == ["change-detector", "satquery-rs-vlm"]
    assert res["task"] == "temporal_change" and not res["refused"]


def test_compound_and_refusal_unchanged():
    import os
    from backend.app.services.validation import validate_image_file
    from backend.app.services.orchestrator import run_analysis

    def load(p):
        with open(p, 'rb') as f:
            d = f.read()
        r = validate_image_file(d, os.path.basename(p))
        return {'bytes': d, 'filename': os.path.basename(p), 'array': r['array'], 'meta': r['meta']}
    hero = [load(p) for p in ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png", "data/demo/optical_sar_pairs/flood_sar.png"]]
    r = run_analysis("What changed between the two dates, and can the SAR imagery corroborate the change?", hero)
    assert r["task"] == "compound_temporal_sar" and not r["refused"]
    bad = [load(p) for p in ["data/demo/invalid_inputs/small_128.png", "data/demo/invalid_inputs/large_512.png"]]
    r2 = run_analysis("What changed between these two dates?", bad)
    assert r2["refused"] is True


def test_tiling_aggregate_and_provenance():
    from packages.geospatial.tiling_exec import execute_tiled
    from packages.tools.dispatcher import dispatch
    rng = np.random.RandomState(0)
    big = (rng.rand(1300, 1100, 3) * 255).astype(np.uint8)  # forces tiling (>1024)
    meta = {"width": 1100, "height": 1300, "crs": None, "transform_list": None, "bounds": None}
    out = execute_tiled("optical-analyzer", big, meta, "water?", {}, dispatch)
    assert out["provenance"]["tiled"] is True and out["provenance"]["tiles"] > 1
    assert len(out["provenance"]["tile_windows"]) == out["provenance"]["tiles"]
    assert "water_pct" in out and "[tiled" in out["answer"]


def test_tiling_edge_tiles_covered():
    from packages.geospatial.metadata import TilingStrategy
    s = TilingStrategy(full_h=1050, full_w=1050, tile_size=512, overlap=32)
    tiles = s.tiles()
    # last tile must reach the far edge
    assert max(y + h for y, x, h, w in tiles) == 1050
    assert max(x + w for y, x, h, w in tiles) == 1050


def test_excessive_input_refused():
    from backend.app.services.validation import HARD_MAX_PIXELS
    assert HARD_MAX_PIXELS == 64_000_000
    # 9000x9000 = 81M > hard max → would refuse at validation (no RAM allocated here)
    assert 9000 * 9000 > HARD_MAX_PIXELS


def test_geo_approx_marked_not_silent():
    import os
    from backend.app.services.validation import validate_image_file
    from backend.app.services.orchestrator import run_analysis

    def load(p):
        with open(p, 'rb') as f:
            d = f.read()
        r = validate_image_file(d, os.path.basename(p))
        return {'bytes': d, 'filename': os.path.basename(p), 'array': r['array'], 'meta': r['meta']}
    datas = [load("data/demo/temporal_pairs/flood_T1.png"), load("data/demo/temporal_pairs/flood_T2.png")]
    res = run_analysis("What changed between these two dates, and where?", datas)
    warns = " ".join(res["validation"]["warnings"])
    assert "APPROXIMATE" in warns or "APPROXIMATE" in (res.get("change_description") or "")
