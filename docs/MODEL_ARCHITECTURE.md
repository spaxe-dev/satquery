# Model Architecture

`ModelDescriptor {id,version,capabilities,modalities,is_heuristic,vram,provenance}` + `ModelAdapter {info,is_available,caption,vqa,ground,health}` + `ModelRegistry {register,get_for, list}`. `HeuristicFallbackAdapter` always available. Real adapters (`blip_adapter`, `qwen_adapter`) optional, gated by `SATQUERY_ENABLE_VLM=1` + checkpoint with `adapter_config.json` LoRA keys; smoke dir quarantined from prod candidates. Orchestrator asks capability, never imports model names. Future: VQA/caption/grounding/change/multimodal. No training in prod path.
