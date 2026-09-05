export function HeuristicBadge({ isHeuristic }: { isHeuristic?: boolean }) {
  if (isHeuristic === false) return <span className="text-xs px-2 py-0.5 rounded bg-green-900 text-green-200">model</span>;
  return <span title="Uncalibrated heuristic quality score, not a probability" className="text-xs px-2 py-0.5 rounded bg-amber-900 text-amber-200">heuristic · uncalibrated</span>;
}

export function TraceView({ trace }: { trace: Array<{ step: number; action: string; detail: string; status: string; tool?: string }> }) {
  return (
    <ol className="space-y-1 text-sm">
      {trace.map((s) => (
        <li key={s.step} className="flex gap-2">
          <span className="text-slate-500">{s.step}.</span>
          <span className="font-medium">{s.action}</span>
          <span className="text-slate-400 truncate">{s.detail}</span>
          {s.tool && <span className="text-xs text-slate-500">[{s.tool}]</span>}
        </li>
      ))}
    </ol>
  );
}
