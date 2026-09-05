"""Proves orchestrator uses registry: replacing a tool changes output without editing orchestrator."""
import numpy as np
from packages.tools.base import get_tool_registry, SpecialistTool, ToolSpec, ToolCapability


class FakeOptical(SpecialistTool):
    @property
    def spec(self):
        return ToolSpec(id="optical-analyzer", name="FakeOptical", version="9.9.9",
                        capabilities=[ToolCapability.OPTICAL_WATER], modalities=["optical"],
                        min_images=1, max_images=1, description="fake", module="fake")

    def execute(self, images, query="", params=None):
        return {"answer": "FAKE_OPTICAL_MARKER", "water_pct": 0.0, "veg_pct": 0.0, "built_pct": 0.0,
                "water_overlay_b64": None, "confidence": {"input_quality": 1.0, "model_confidence": 1.0, "evidence_agreement": 1.0, "overall": 1.0, "is_heuristic": True},
                "elapsed_ms": 1, "provenance": {"fake": True}}


def test_orchestrator_uses_registry_replacement():
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from backend.app.services.validation import validate_image_file
    from backend.app.services.orchestrator import run_analysis
    reg = get_tool_registry()
    orig = reg.get("optical-analyzer")
    reg._tools["optical-analyzer"] = FakeOptical()
    try:
        def load(p):
            with open(p, 'rb') as f:
                d = f.read()
            r = validate_image_file(d, os.path.basename(p))
            return {'bytes': d, 'filename': os.path.basename(p), 'array': r['array'], 'meta': r['meta']}
        datas = [load("data/demo/optical_sar_pairs/pair1_optical.png"), load("data/demo/optical_sar_pairs/pair1_sar.png")]
        res = run_analysis("Use the optical and SAR images together to identify built-up and water-covered regions.", datas)
        assert "FAKE_OPTICAL_MARKER" in res["answer"], "orchestrator did not use registry replacement"
    finally:
        reg._tools["optical-analyzer"] = orig
