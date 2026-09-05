export function EvidencePanel({ result }: { result: any }) {
  const items: string[] = [];
  if (result.change_map_b64) items.push("change_map");
  if (result.optical_overlay_b64) items.push("optical_overlay");
  if (result.sar_overlay_b64) items.push("sar_overlay");
  if (result.fused_overlay_b64) items.push("fused_overlay");
  if (!items.length) return <div className="text-sm text-zinc-500">No visual evidence (refused or heuristic text-only).</div>;
  return <div className="text-sm">Evidence: {items.join(", ")} (pixel-space unless georeferenced; see metadata)</div>;
}
