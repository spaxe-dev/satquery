import { z } from "zod";

export const ConfidenceSchema = z.object({
  input_quality: z.number().min(0).max(1),
  model_confidence: z.number().min(0).max(1),
  evidence_agreement: z.number().min(0).max(1),
  overall: z.number().min(0).max(1),
  is_heuristic: z.boolean(),
});
export type ConfidenceT = z.infer<typeof ConfidenceSchema>;

export const ExecutionStepSchema = z.object({
  step: z.number(),
  action: z.string(),
  detail: z.string(),
  status: z.enum(["ok", "warn", "error", "skipped"]),
  tool: z.string().optional(),
  duration_ms: z.number().optional(),
});

export const AnalysisResponseSchema = z.object({
  task: z.string(),
  answer: z.string(),
  report_id: z.string(),
  refused: z.boolean(),
  confidence: ConfidenceSchema,
  execution_trace: z.array(ExecutionStepSchema),
  is_heuristic: z.boolean().optional(),
  fallback_reason: z.string().optional(),
}).passthrough();

export const DEMOS = [
  { id: "single", label: "Single-image VQA", files: ["data/demo/single_image/scene_1.png"], query: "Describe the land-cover and major objects visible in this image." },
  { id: "temporal", label: "Bi-temporal change", files: ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png"], query: "What changed between these two dates, and where?" },
  { id: "optical-sar", label: "Optical + SAR", files: ["data/demo/optical_sar_pairs/pair1_optical.png", "data/demo/optical_sar_pairs/pair1_sar.png"], query: "Use the optical and SAR images together to identify built-up and water-covered regions." },
  { id: "hero", label: "Compound + SAR corroboration", files: ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png", "data/demo/optical_sar_pairs/flood_sar.png"], query: "What changed between the dates, and can SAR corroborate it?" },
  { id: "invalid", label: "Invalid refusal", files: ["data/demo/invalid_inputs/small_128.png", "data/demo/invalid_inputs/large_512.png"], query: "What changed between these two dates?" },
] as const;
