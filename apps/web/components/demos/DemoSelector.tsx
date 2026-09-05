export function DemoSelector({ onRun }: { onRun: (id: number) => void }) {
  const demos = [
    { id: 1, label: "Single VQA" },
    { id: 2, label: "Temporal change" },
    { id: 3, label: "Optical + SAR" },
    { id: 4, label: "Hero: change + SAR" },
    { id: 5, label: "Invalid refusal" },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {demos.map((d) => (
        <button key={d.id} onClick={() => onRun(d.id)} className="px-3 py-1.5 rounded-lg border text-sm font-medium">{d.id}. {d.label}</button>
      ))}
    </div>
  );
}
