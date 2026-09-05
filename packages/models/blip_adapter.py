"""BLIP pretrained adapter (CPU-runnable). Gated by SATQUERY_ENABLE_VLM=1. No download at import; no fake."""
from __future__ import annotations
import os, time
import numpy as np
from packages.models.adapter import ModelAdapter, ModelInfo, ModelCapability

MODEL_ID = "blip-base-pretrained"
CAPTION_CKPT = os.environ.get("SATQUERY_BLIP_CAPTION_CKPT", "Salesforce/blip-image-captioning-base")
VQA_CKPT = os.environ.get("SATQUERY_BLIP_VQA_CKPT", "Salesforce/blip-vqa-base")


class BlipAdapter(ModelAdapter):
    def __init__(self):
        self._cap = None
        self._vqa = None
        self._blocker: str | None = None
        self._info = ModelInfo(model_id=MODEL_ID, version="1.0.0",
                               capabilities=[ModelCapability.CAPTIONING, ModelCapability.VQA],
                               modalities=["optical"], is_heuristic=False, vram_mb=1024,
                               provenance={"type": "pretrained", "caption_ckpt": CAPTION_CKPT, "vqa_ckpt": VQA_CKPT, "license": "BSD-3-Clause"})

    @property
    def info(self):
        return self._info

    def is_available(self) -> bool:
        if os.environ.get("SATQUERY_ENABLE_VLM", "0") != "1":
            self._blocker = "SATQUERY_ENABLE_VLM!=1"
            return False
        try:
            import transformers  # noqa
            import torch  # noqa
            return True
        except Exception as e:
            self._blocker = f"missing deps: {e}"
            return False

    def _ensure(self, kind: str):
        from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
        from PIL import Image
        if kind == "caption" and self._cap is None:
            p = BlipProcessor.from_pretrained(CAPTION_CKPT)
            m = BlipForConditionalGeneration.from_pretrained(CAPTION_CKPT)
            m.eval()
            self._cap = (p, m)
        if kind == "vqa" and self._vqa is None:
            p = BlipProcessor.from_pretrained(VQA_CKPT)
            m = BlipForQuestionAnswering.from_pretrained(VQA_CKPT)
            m.eval()
            self._vqa = (p, m)

    @staticmethod
    def _to_pil(image: np.ndarray):
        from PIL import Image
        arr = np.clip(image, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGB")
        return Image.fromarray(arr[:, :, :3])

    def caption(self, image, query=None):
        t0 = time.time()
        # Capability guard: RGB optical only; SAR/temporal refused upstream
        self._ensure("caption")
        p, m = self._cap
        img = self._to_pil(image).resize((384, 384))
        out = p.decode(m.generate(**p(img, return_tensors="pt"), max_new_tokens=64)[0], skip_special_tokens=True)
        return {"answer": out, "caption": out, "boxes": [], "confidence": None, "elapsed_ms": int((time.time() - t0) * 1000), "model_used": MODEL_ID, "provenance": {"heuristic": False}}

    def vqa(self, image, query: str):
        t0 = time.time()
        self._ensure("vqa")
        p, m = self._vqa
        img = self._to_pil(image).resize((384, 384))
        out = p.decode(m.generate(**p(img, query, return_tensors="pt"))[0], skip_special_tokens=True)
        cap = None
        try:
            cap = self.caption(image)["caption"]
        except Exception:
            pass
        return {"answer": out, "caption": cap, "boxes": [], "confidence": None, "elapsed_ms": int((time.time() - t0) * 1000), "model_used": MODEL_ID, "provenance": {"heuristic": False}}

    def ground(self, image, query: str):
        return {"answer": "Grounding not supported by BLIP-base (heuristic boxes remain authoritative)", "boxes": [], "elapsed_ms": 0, "model_used": MODEL_ID, "provenance": {"unsupported": "grounding"}}
