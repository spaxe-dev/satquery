# Tool Architecture

`ToolSpec {id,name,version,capabilities,modalities,min/max_images,description,module}` + `SpecialistTool {spec,execute,health,validate_inputs}` + `ToolRegistry`. Tools: `satquery-rs-vlm, change-detector, optical-analyzer, sar-analyzer, fusion-engine`. `execute(images,query,params)->{answer,confidence,evidence,elapsed_ms,provenance}` never raises on valid input; `FusionTool` must implement `execute` via `fuse()` not raise (fix P1). Domain logic in tools, orchestration in agent, API/UI call registries only. Each tool has unit test on synthetic + demo fixtures.
