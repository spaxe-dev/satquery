export function ConfidencePanel({ confidence, isHeuristic }: { confidence: { overall: number; input_quality: number; model_confidence: number; evidence_agreement: number }; isHeuristic?: boolean }) {
  return (
    <div className="rounded-xl border p-3">
      <div className="text-sm font-semibold">Confidence {confidence.overall.toFixed(2)} {isHeuristic === false ? "(model)" : "(heuristic · uncalibrated)"}</div>
      <div className="text-xs text-zinc-500">Input {confidence.input_quality.toFixed(2)} · Model {confidence.model_confidence.toFixed(2)} · Evidence {confidence.evidence_agreement.toFixed(2)}</div>
    </div>
  );
}
