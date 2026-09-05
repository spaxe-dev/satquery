export type Confidence = {
  input_quality: number;
  model_confidence: number;
  evidence_agreement: number;
  cross_modal_support?: number | null;
  overall: number;
  is_heuristic: boolean;
};

export type ExecutionStep = {
  step: number;
  action: string;
  detail: string;
  status: "ok" | "warn" | "error" | "skipped";
  tool?: string;
  duration_ms?: number;
};

export type AnalysisResponse = {
  task: string;
  query: string;
  answer: string;
  caption?: string;
  grounding_boxes: Array<{ x: number; y: number; w: number; h: number; label: string; confidence: number }>;
  change_description?: string;
  change_area_pct?: number;
  fused_statement?: string;
  change_map_b64?: string;
  optical_overlay_b64?: string;
  sar_overlay_b64?: string;
  fused_overlay_b64?: string;
  fused_union_b64?: string;
  confidence: Confidence;
  execution_trace: ExecutionStep[];
  tools_used: Array<{ tool_id: string; tool_name: string; reason: string; params: any }>;
  validation: { valid: boolean; reason?: string; warnings: string[] };
  timestamp: string;
  report_id: string;
  refused: boolean;
  vlm_attempted?: boolean;
  vlm_loaded?: boolean;
  vlm_generation_valid?: boolean | null;
  is_heuristic?: boolean;
  fallback_reason?: string;
  previews: Array<{ filename: string; meta: any; preview_b64: string }>;
  optical_evidence?: any;
  sar_evidence?: any;
  fusion_evidence?: any;
};

export async function analyze(query: string, files: File[]): Promise<AnalysisResponse> {
  const fd = new FormData();
  fd.append("query", query);
  for (const f of files) fd.append("images", f, f.name);
  const r = await fetch("/api/analyze", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Analysis failed");
  return j as AnalysisResponse;
}

export async function getRegistry() {
  const r = await fetch("/api/registry");
  return r.json();
}

export async function getHealth() {
  const r = await fetch("/api/health");
  return r.json();
}
