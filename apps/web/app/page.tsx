"use client";
import { useEffect, useState, useRef, useCallback } from "react";
import {
  Satellite,
  Layers,
  Cpu,
  Activity,
  Database,
  FileText,
  Settings,
  ShieldCheck,
  BarChart3,
  Search,
  Upload,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Eye,
  Zap,
  Globe,
  Image as ImageIcon,
  MapPinned,
  Braces,
  History,
  Download,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Play,
  X,
} from "lucide-react";
import { Card, CardHeader, CardTitle, Badge, Button } from "@/components/ui";
import { analyze, getHealth, getRegistry, type AnalysisResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------
const NAV = [
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "analysis", label: "Analysis", icon: Search, badge: "PRIMARY" },
  { id: "datasets", label: "Datasets", icon: Database },
  { id: "models", label: "Models", icon: Cpu },
  { id: "tools", label: "Tools", icon: Layers },
  { id: "runs", label: "Runs", icon: History },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "health", label: "System Health", icon: Activity },
  { id: "settings", label: "Settings", icon: Settings },
];

// ---------------------------------------------------------------------------
// Demo definitions — matches backend data/demo
// ---------------------------------------------------------------------------
const DEMOS = [
  { id: 1, label: "DEMO 1", sub: "Single", query: "Describe the land-cover and major objects visible in this image.", files: ["data/demo/single_image/scene_1.png"] },
  { id: 2, label: "DEMO 2", sub: "Temporal", query: "What changed between these two dates, and where did the change occur?", files: ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png"] },
  { id: 3, label: "DEMO 3", sub: "Opt+SAR", query: "Use the optical and SAR images together to identify built-up and water-covered regions.", files: ["data/demo/optical_sar_pairs/pair1_optical.png", "data/demo/optical_sar_pairs/pair1_sar.png"] },
  { id: 4, label: "DEMO 4", sub: "Hero ★", query: "What changed between the two dates, and can the SAR imagery corroborate the change?", files: ["data/demo/temporal_pairs/flood_T1.png", "data/demo/temporal_pairs/flood_T2.png", "data/demo/optical_sar_pairs/flood_sar.png"] },
  { id: 5, label: "DEMO 5", sub: "Invalid", query: "What changed between these two dates?", files: ["data/demo/invalid_inputs/small_128.png", "data/demo/invalid_inputs/large_512.png"] },
];

export default function Page() {
  const [nav, setNav] = useState("overview");
  const [health, setHealth] = useState<any>(null);
  const [registry, setRegistry] = useState<any>(null);
  const [recent, setRecent] = useState<AnalysisResponse[]>([]);
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [traceOpen, setTraceOpen] = useState(true);
  const [evidenceTab, setEvidenceTab] = useState<"all" | "change" | "optical" | "sar" | "fusion" | "ground">("all");
  const [compare, setCompare] = useState(50);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Health polling
  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
    getRegistry().then(setRegistry).catch(() => {});
    const id = setInterval(() => getHealth().then(setHealth).catch(() => {}), 8000);
    return () => clearInterval(id);
  }, []);

  // File previews local
  useEffect(() => {
    const urls = files.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
  }, [files]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const list = Array.from(e.dataTransfer.files).slice(0, 4 - files.length);
    if (list.length) setFiles((prev) => [...prev, ...list].slice(0, 4));
  }, [files.length]);

  const handleFiles = (list: FileList | null) => {
    if (!list) return;
    const arr = Array.from(list).slice(0, 4 - files.length);
    setFiles((prev) => [...prev, ...arr].slice(0, 4));
  };

  const runAnalysis = async () => {
    if (!query.trim() || files.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyze(query, files);
      setResult(res);
      setRecent((prev) => [res, ...prev].slice(0, 8));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const runDemo = async (id: number) => {
    const demo = DEMOS.find((d) => d.id === id)!;
    setQuery(demo.query);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const fetched: File[] = [];
      for (const path of demo.files) {
        const r = await fetch("/" + path);
        if (!r.ok) throw new Error(`Failed to fetch ${path}: ${r.status}`);
        const ct = r.headers.get("content-type") || "";
        if (ct.includes("text/html")) throw new Error(`Got HTML for ${path} — API not reachable`);
        const blob = await r.blob();
        const name = path.split("/").pop()!;
        fetched.push(new File([blob], name, { type: blob.type || "image/png" }));
      }
      setFiles(fetched);
      // wait a tick for state, then analyze with fetched files directly
      const res = await analyze(demo.query, fetched);
      setResult(res);
      setRecent((prev) => [res, ...prev].slice(0, 8));
      setNav("analysis");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setFiles([]);
    setResult(null);
    setError(null);
  };

  return (
    <div className="flex min-h-screen">
      {/* SIDEBAR */}
      <aside className="hidden lg:flex w-[260px] shrink-0 bg-zinc-900 text-zinc-100 flex-col sticky top-0 h-screen overflow-auto scrollbar-thin">
        <div className="px-5 pt-6 pb-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-white text-zinc-900 flex items-center justify-center font-extrabold mono text-sm">SQ</div>
            <div>
              <div className="font-extrabold tracking-tight text-sm leading-none">SATQUERY AI</div>
              <div className="text-[11px] tracking-[0.14em] text-zinc-400 font-semibold mt-0.5">SIH26167 • ISRO</div>
            </div>
          </div>
          <div className="mt-3 inline-flex px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[11px] font-bold tracking-wide">
            ● AGENTIC ORCHESTRATION
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = nav === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setNav(item.id)}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-semibold transition",
                  active ? "bg-white text-zinc-900" : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="flex-1 text-left">{item.label}</span>
                {item.badge && <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-400 text-zinc-900 font-extrabold">{item.badge}</span>}
              </button>
            );
          })}
        </nav>

        <div className="p-3 border-t border-zinc-800 space-y-3">
          <div className="rounded-xl bg-zinc-800 p-3">
            <div className="text-xs font-bold flex items-center gap-2">
              <span className={cn("w-2 h-2 rounded-full", health?.status === "ok" ? "bg-emerald-400" : "bg-red-400")} />
              Backend {health?.status === "ok" ? `online v${health.version}` : health ? health.status : "checking…"}
            </div>
            <div className="text-[11px] text-zinc-400 mono mt-1">localhost:8000 • CORS open for dev</div>
          </div>
          <div className="text-[11px] mono text-zinc-500 text-center">SatQuery v1.0 • Deterministic-first • Offline-capable</div>
        </div>
      </aside>

      {/* MOBILE TOP BAR */}
      <div className="lg:hidden fixed top-0 inset-x-0 z-40 bg-zinc-900 text-white flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-white text-zinc-900 flex items-center justify-center font-extrabold mono text-xs">SQ</div>
          <span className="font-extrabold text-sm">SATQUERY AI</span>
        </div>
        <select value={nav} onChange={(e) => setNav(e.target.value)} className="bg-zinc-800 border border-zinc-700 rounded-full px-3 py-1 text-sm">
          {NAV.map((n) => (
            <option key={n.id} value={n.id}>{n.label}</option>
          ))}
        </select>
      </div>

      {/* MAIN */}
      <main className="flex-1 min-w-0 lg:ml-0 mt-14 lg:mt-0">
        {nav === "overview" && (
          <Overview
            health={health}
            registry={registry}
            recent={recent}
            onRunDemo={runDemo}
            setNav={setNav}
          />
        )}
        {nav === "analysis" && (
          <AnalysisWorkspace
            query={query}
            setQuery={setQuery}
            files={files}
            previews={previews}
            result={result}
            loading={loading}
            error={error}
            traceOpen={traceOpen}
            setTraceOpen={setTraceOpen}
            evidenceTab={evidenceTab}
            setEvidenceTab={setEvidenceTab}
            compare={compare}
            setCompare={setCompare}
            onDrop={onDrop}
            handleFiles={handleFiles}
            fileInputRef={fileInputRef}
            runAnalysis={runAnalysis}
            runDemo={runDemo}
            clearAll={clearAll}
            setFiles={setFiles}
          />
        )}
        {nav === "datasets" && <DatasetsView />}
        {nav === "models" && <ModelsView registry={registry} />}
        {nav === "tools" && <ToolsView registry={registry} />}
        {nav === "runs" && <RunsView recent={recent} setNav={setNav} />}
        {nav === "reports" && <ReportsView recent={recent} />}
        {nav === "health" && <HealthView health={health} registry={registry} />}
        {nav === "settings" && <SettingsView />}
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------
function Overview({ health, registry, recent, onRunDemo, setNav }: any) {
  const toolCount = registry?.tools?.length ?? 5;
  const stats = [
    { label: "Analyses", value: recent.length, sub: "this session", icon: BarChart3, color: "bg-zinc-900 text-white" },
    { label: "Successful", value: recent.filter((r: any) => !r.refused).length, sub: "non-refused", icon: CheckCircle2, color: "bg-emerald-600 text-white" },
    { label: "Avg Latency", value: recent.length ? `${Math.round(recent.reduce((a: number, r: any) => a + (r.execution_trace?.length || 0), 0) / recent.length * 45)}ms` : "—", sub: "est. CPU 512²", icon: Clock, color: "bg-white border text-zinc-900" },
    { label: "Tools", value: toolCount, sub: "registry ready", icon: Layers, color: "bg-white border text-zinc-900" },
  ];
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Remote Sensing Intelligence</h1>
          <p className="text-zinc-500 mt-1">Ask satellite imagery a question. The agent selects specialist models, fuses evidence, and returns auditable results.</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setNav("analysis")} className="gap-2">
            <Sparkles className="w-4 h-4" /> Open Analysis Workspace
          </Button>
          <a href="/api/registry" target="_blank" className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl border border-zinc-300 bg-white text-sm font-semibold">
            Registry <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <Card key={s.label} className="p-4">
            <div className="flex items-center justify-between">
              <div className={cn("w-9 h-9 rounded-xl flex items-center justify-center", s.color)}>
                <s.icon className="w-4.5 h-4.5" />
              </div>
              <span className="text-[11px] mono px-2 py-1 rounded-full bg-zinc-100 border">{s.sub}</span>
            </div>
            <div className="mt-3 text-2xl font-extrabold">{s.value}</div>
            <div className="text-xs font-semibold text-zinc-500">{s.label}</div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-12 gap-5">
        <Card className="col-span-12 lg:col-span-8 p-0 overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-4 h-4" /> Quick Start — 5 Verified Demos
            </CardTitle>
            <span className="text-xs mono px-2 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700">click to run end-to-end</span>
          </CardHeader>
          <div className="p-4 grid grid-cols-2 lg:grid-cols-5 gap-3">
            {DEMOS.map((d) => (
              <button
                key={d.id}
                onClick={() => onRunDemo(d.id)}
                className={cn(
                  "rounded-2xl border-2 p-4 text-left hover:shadow-md transition",
                  d.id === 4 ? "border-amber-300 bg-amber-50" : d.id === 5 ? "border-red-200 bg-red-50" : d.id === 1 ? "border-zinc-900 bg-zinc-900 text-white" : "bg-white"
                )}
              >
                <div className="text-xs font-extrabold mono">{d.label}</div>
                <div className="text-sm font-bold">{d.sub}</div>
                <div className="text-[11px] mt-1 leading-snug opacity-70 line-clamp-3">{d.query}</div>
                <div className="mt-2 text-[11px] mono opacity-60">{d.files.length} image(s)</div>
              </button>
            ))}
          </div>
          <div className="px-4 pb-4 text-xs text-zinc-500">
            Demos fetch from <span className="mono bg-zinc-100 px-1.5 py-0.5 rounded">/data/demo</span> via FastAPI static mount (not SPA fallback) — validated by <span className="mono">test_demo_static.py</span>.
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-4 p-0 overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4" /> System Health
            </CardTitle>
            <Badge className={cn(health?.status === "ok" ? "bg-emerald-600 text-white" : "bg-red-600 text-white")}>
              {health?.status === "ok" ? "● healthy" : "● checking"}
            </Badge>
          </CardHeader>
          <div className="p-4 space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-500">Backend</span>
              <span className="font-semibold mono">{health?.status ?? "—"} {health?.version ? `v${health.version}` : ""}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Registry</span>
              <span className="font-semibold mono">{registry ? `${registry.tools?.length} tools` : "loading"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Agent</span>
              <span className="font-semibold">deterministic-first</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-500">Models</span>
              <span className="font-semibold mono">heuristic + LoRA-swappable</span>
            </div>
            <div className="pt-3 border-t text-xs text-zinc-500">
              Confidence is always labeled <span className="font-bold text-amber-700">heuristic</span> when not from a fine-tuned checkpoint. See <span className="mono">docs/MODEL_CARD.md</span>.
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-12 gap-5">
        <Card className="col-span-12 lg:col-span-7 p-0 overflow-hidden">
          <CardHeader>
            <CardTitle>Recent Analyses</CardTitle>
            <span className="text-xs mono px-2 py-1 rounded-full bg-zinc-100 border">{recent.length} in session</span>
          </CardHeader>
          <div className="divide-y">
            {recent.length === 0 ? (
              <div className="p-10 text-center text-sm text-zinc-500">No analyses yet — run a demo or open the Analysis workspace.</div>
            ) : (
              recent.map((r: AnalysisResponse) => (
                <div key={r.report_id} className="px-4 py-3 flex items-center gap-3">
                  <div className={cn("w-2.5 h-2.5 rounded-full", r.refused ? "bg-red-500" : "bg-emerald-500")} />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold truncate">{r.query}</div>
                    <div className="text-xs mono text-zinc-500">
                      {r.task} • {r.tools_used.length} tools • conf {r.confidence.overall.toFixed(2)} {r.confidence.is_heuristic ? "(heuristic)" : ""} • {new Date(r.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                  <Badge className="bg-zinc-900 text-white mono text-[11px]">{r.report_id}</Badge>
                </div>
              ))
            )}
          </div>
        </Card>

        <Card className="col-span-12 lg:col-span-5 p-0 overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Globe className="w-4 h-4" /> Data Sources
            </CardTitle>
          </CardHeader>
          <div className="p-4 grid grid-cols-2 gap-3 text-xs">
            {[
              { name: "Uploaded GeoTIFF", modality: "Optical/SAR", status: "ready" },
              { name: "Sentinel-2 (via STAC)", modality: "Optical 10m", status: "interface ready" },
              { name: "Sentinel-1", modality: "SAR", status: "interface ready" },
              { name: "Cartosat-2S", modality: "Optical sub-m", status: "eval set" },
              { name: "RISAT", modality: "SAR", status: "eval set" },
              { name: "BigEarthNet / reBEN", modality: "S1+S2 120×120", status: "adapter scaffold" },
            ].map((s) => (
              <div key={s.name} className="rounded-xl border bg-zinc-50 p-3">
                <div className="font-bold">{s.name}</div>
                <div className="mono text-zinc-500">{s.modality}</div>
                <div className={cn("mt-1 inline-flex px-2 py-0.5 rounded-full text-[11px] font-bold", s.status === "ready" ? "bg-emerald-600 text-white" : "bg-zinc-900 text-white")}>{s.status}</div>
              </div>
            ))}
          </div>
          <div className="px-4 pb-4 text-xs text-zinc-500">
            Future integrations are via <span className="mono">packages/geospatial</span> and <span className="mono">DataSource → Connector → Catalog → Asset</span> — see <span className="mono">docs/DATA_SOURCES.md</span>.
          </div>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Analysis Workspace — centerpiece
// ---------------------------------------------------------------------------
function AnalysisWorkspace(props: any) {
  const {
    query, setQuery, files, previews, result, loading, error, traceOpen, setTraceOpen, evidenceTab, setEvidenceTab, compare, setCompare, onDrop, handleFiles, fileInputRef, runAnalysis, runDemo, clearAll, setFiles,
  } = props;

  return (
    <div className="max-w-[1480px] mx-auto px-4 lg:px-6 py-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h1 className="text-2xl font-extrabold tracking-tight flex items-center gap-2">
            <Satellite className="w-6 h-6" /> Analysis Workspace
          </h1>
          <p className="text-sm text-zinc-500">Drop imagery, ask a question, inspect the agent&apos;s plan and evidence.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={clearAll}>Clear</Button>
          <a href="/api/docs" target="_blank" className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl border border-zinc-300 bg-white text-sm font-semibold">
            API Docs <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-5">
        {/* LEFT: Images + Query */}
        <div className="col-span-12 lg:col-span-7 space-y-5">
          {/* Image workspace */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4" /> IMAGE WORKSPACE
              </CardTitle>
              <div className="flex items-center gap-2">
                <Badge className="bg-zinc-900 text-white mono">{files.length} image(s) • max 4</Badge>
                <Button variant="outline" className="text-xs py-1 px-2" onClick={clearAll}>
                  Clear
                </Button>
              </div>
            </CardHeader>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className="m-4 rounded-2xl border-2 border-dashed border-zinc-300 bg-zinc-50 hover:bg-zinc-100 transition p-7 text-center cursor-pointer"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-900 text-white flex items-center justify-center mx-auto">
                <Upload className="w-5 h-5" />
              </div>
              <div className="text-sm font-bold mt-2">Drop GeoTIFF / TIFF / PNG / JPEG here</div>
              <div className="text-xs text-zinc-500 mt-1">or click to browse — supports single, bi-temporal, and optical+SAR pairs</div>
              <div className="mt-3 flex flex-wrap justify-center gap-2 text-[11px] mono">
                <span className="px-2 py-1 bg-white border rounded-full">GeoTIFF ✓</span>
                <span className="px-2 py-1 bg-white border rounded-full">PNG (benchmarks)</span>
                <span className="px-2 py-1 bg-white border rounded-full">JPEG (benchmarks)</span>
              </div>
              <input ref={fileInputRef} type="file" multiple accept=".tif,.tiff,.png,.jpg,.jpeg" className="hidden" onChange={(e) => handleFiles(e.target.files)} />
            </div>

            {files.length > 0 && (
              <div className="px-4 pb-4 grid grid-cols-2 md:grid-cols-3 gap-3">
                {files.map((f: File, i: number) => (
                  <div key={i} className="relative rounded-xl overflow-hidden border border-zinc-200 bg-zinc-50 group">
                    <img src={previews[i]} alt={f.name} className="w-full h-36 object-cover" />
                    <div className="p-2">
                      <div className="text-xs font-semibold truncate">{f.name}</div>
                      <div className="text-[11px] mono text-zinc-500">{(f.size / 1024).toFixed(0)} KB • {f.type || "unknown"}</div>
                    </div>
                    <button onClick={(e) => { e.stopPropagation(); setFiles((prev: File[]) => prev.filter((_, idx) => idx !== i)); }} className="absolute top-1.5 right-1.5 w-7 h-7 rounded-full bg-black/70 text-white flex items-center justify-center">
                      <X className="w-4 h-4" />
                    </button>
                    <div className="absolute bottom-10 left-1.5 text-[10px] mono px-1.5 py-0.5 rounded-full bg-zinc-900 text-white">#{i + 1}</div>
                  </div>
                ))}
              </div>
            )}

            {result?.validation?.warnings?.length ? (
              <div className="mx-4 mb-4 text-xs mono bg-amber-50 border border-amber-200 text-amber-800 px-3 py-2 rounded-xl">
                ⚠ {result.validation.warnings.join(" • ")}
              </div>
            ) : null}
          </Card>

          {/* Query */}
          <Card className="p-5">
            <div className="font-bold text-sm">QUERY</div>
            <div className="text-xs text-zinc-500 mt-1">Natural language — the agent classifies intent and builds a validated execution plan.</div>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              rows={3}
              placeholder="e.g. What changed between these two dates, and where did the change occur?  •  Use optical and SAR together to identify water..."
              className="mt-3 w-full rounded-xl border border-zinc-300 bg-zinc-50 focus:bg-white focus:ring-2 focus:ring-zinc-900 focus:border-zinc-900 outline-none px-3.5 py-3 text-sm"
            />
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              {[
                "Describe the land-cover and major objects visible in this image.",
                "What changed between these two dates, and where did the change occur?",
                "Use the optical and SAR images together to identify built-up and water-covered regions.",
                "What changed between the two dates, and can the SAR imagery corroborate the change?",
                "Highlight the water body referred to in the query.",
              ].map((q) => (
                <button key={q} onClick={() => setQuery(q)} className="px-3 py-1.5 rounded-full border border-zinc-300 bg-white hover:bg-zinc-50 font-semibold text-xs">
                  {q.slice(0, 28)}…
                </button>
              ))}
            </div>
            <div className="mt-4 flex items-center gap-3">
              <Button onClick={runAnalysis} disabled={loading || !query.trim() || files.length === 0} className="flex-1 py-3 gap-2">
                {loading ? <Clock className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} {loading ? "ANALYZING…" : "ANALYZE"}
              </Button>
              <span className="text-xs text-zinc-500 mono">{files.length} image(s) • {query.length} chars</span>
            </div>

            <div className="mt-4 grid grid-cols-5 gap-2 text-[11px]">
              {DEMOS.map((d) => (
                <button key={d.id} onClick={() => runDemo(d.id)} className={cn("py-2 rounded-xl border font-bold", d.id === 4 ? "border-amber-300 bg-amber-50" : d.id === 5 ? "border-red-300 bg-red-50" : d.id === 1 ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white")}>
                  {d.label}
                  <br />
                  <span className="font-normal">{d.sub}</span>
                </button>
              ))}
            </div>
            {error && <div className="mt-3 text-sm bg-red-50 border border-red-200 text-red-800 px-3 py-2 rounded-xl">Error: {error}</div>}
          </Card>

          {/* Visual evidence */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Eye className="w-4 h-4" /> VISUAL EVIDENCE
              </CardTitle>
              <div className="flex gap-1 text-xs">
                {(["all", "change", "optical", "sar", "fusion", "ground"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setEvidenceTab(t)}
                    className={cn("px-2.5 py-1 rounded-full border font-semibold capitalize", evidenceTab === t ? "bg-zinc-900 text-white border-zinc-900" : "bg-white hover:bg-zinc-50")}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </CardHeader>

            <div className="p-4">
              {!result ? (
                <div className="text-sm text-zinc-500 text-center py-10">Run an analysis to see change maps, optical/SAR overlays, and grounding boxes.</div>
              ) : (
                <EvidenceView result={result} tab={evidenceTab} compare={compare} setCompare={setCompare} />
              )}
            </div>
          </Card>
        </div>

        {/* RIGHT: Plan + Result + Trace */}
        <div className="col-span-12 lg:col-span-5 space-y-5">
          {/* Analysis plan */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Braces className="w-4 h-4" /> ANALYSIS PLAN
              </CardTitle>
              <span className="text-xs mono px-2 py-1 rounded-full bg-zinc-900 text-white">{result ? result.task : "awaiting query"}</span>
            </CardHeader>
            <div className="p-4">
              {!result ? (
                <div className="text-sm text-zinc-500">
                  Query → <span className="mono bg-zinc-100 px-1 rounded">classify</span> → <span className="mono bg-zinc-100 px-1 rounded">validate</span> → <span className="mono bg-zinc-100 px-1 rounded">plan tools</span> → <span className="mono bg-zinc-100 px-1 rounded">execute</span> → <span className="mono bg-zinc-100 px-1 rounded">fuse</span> → answer.
                  <div className="mt-3 text-xs mono bg-zinc-900 text-zinc-100 p-3 rounded-xl">Validated plan will appear here with tool reasons and params — auditable per SIH trace requirement.</div>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {result.tools_used.map((t: any, i: number) => (
                      <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-xl bg-zinc-900 text-white text-xs mono">
                        <span className="w-6 h-6 rounded-full bg-white text-zinc-900 flex items-center justify-center font-extrabold">{i + 1}</span>
                        {t.tool_id}
                        {i < result.tools_used.length - 1 && <ChevronRight className="w-3 h-3 text-zinc-400" />}
                      </div>
                    ))}
                    {result.tools_used.length === 0 && <span className="text-sm text-red-600">Refused — no tools executed. See trace for reason.</span>}
                  </div>
                  <div className="text-xs mono bg-zinc-50 border rounded-xl p-3 space-y-1">
                    {result.tools_used.map((t: any) => (
                      <div key={t.tool_id}>
                        <span className="font-bold">{t.tool_id}</span>: {t.reason} <span className="text-zinc-500">{JSON.stringify(t.params)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Execution trace */}
          <Card className="overflow-hidden">
            <button onClick={() => setTraceOpen(!traceOpen)} className="w-full flex items-center justify-between px-5 py-3.5 bg-zinc-900 text-zinc-100">
              <span className="font-bold text-sm tracking-wide flex items-center gap-2">
                <Clock className="w-4 h-4" /> EXECUTION TRACE
              </span>
              <span className="flex items-center gap-2 text-xs mono">
                auditable • deterministic {traceOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              </span>
            </button>
            {traceOpen && (
              <div className="p-4 mono text-xs leading-relaxed space-y-1.5 max-h-[380px] overflow-auto bg-zinc-900 text-zinc-100 scrollbar-thin">
                {!result ? (
                  <div className="text-zinc-500">Awaiting analysis… The agent will emit task → validation → tool selection → execution → fusion → confidence.</div>
                ) : (
                  result.execution_trace.map((s: any) => (
                    <div key={s.step} className="flex gap-2">
                      <span className={cn(s.status === "ok" ? "text-emerald-400" : s.status === "warn" ? "text-amber-400" : "text-red-400")}>{s.status === "ok" ? "✓" : s.status === "warn" ? "⚠" : "✗"}</span>
                      <span className="text-zinc-500">[{String(s.step).padStart(2, "0")}]</span>
                      <span className="text-zinc-200">{s.action}</span>
                      <span className="text-zinc-400">— {s.detail}</span>
                      {s.tool && <span className="ml-auto px-1.5 py-0.5 rounded bg-zinc-800 border border-zinc-700">{s.tool}</span>}
                    </div>
                  ))
                )}
              </div>
            )}
          </Card>

          {/* Result */}
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle>RESULT</CardTitle>
              {result && <Badge className="bg-zinc-900 text-white mono">{result.task}</Badge>}
            </CardHeader>
            <div className="p-5 space-y-4">
              {!result ? (
                <div className="text-sm text-zinc-500">—</div>
              ) : (
                <>
                  <div>
                    <div className="text-xs font-bold tracking-widest text-zinc-500">ANSWER</div>
                    <div className="mt-2 text-sm leading-relaxed text-zinc-800 whitespace-pre-wrap">{result.answer}</div>
                  </div>
                  {result.caption && (
                    <div>
                      <div className="text-xs font-bold tracking-widest text-zinc-500">CAPTION</div>
                      <div className="mt-1 text-sm text-zinc-700 italic">{result.caption}</div>
                    </div>
                  )}
                  {result.change_description && (
                    <div>
                      <div className="text-xs font-bold tracking-widest text-zinc-500">CHANGE ANALYSIS</div>
                      <div className="mt-1 text-sm">
                        <span className="font-bold">{result.change_area_pct}% changed</span> — {result.change_description}
                      </div>
                    </div>
                  )}
                  {(result.fused_statement || result.fusion_evidence) && (
                    <div>
                      <div className="text-xs font-bold tracking-widest text-zinc-500">CROSS-MODAL FUSION</div>
                      <div className="mt-1 text-sm">{result.fused_statement || JSON.stringify(result.fusion_evidence)}</div>
                    </div>
                  )}
                </>
              )}
            </div>
          </Card>

          {/* Confidence */}
          <Card className="p-5">
            <div className="font-bold text-sm">
              CONFIDENCE <span className="text-xs font-normal text-zinc-500">— heuristic, not calibrated probability</span>
            </div>
            {!result ? (
              <div className="mt-3 text-sm text-zinc-500">No estimate yet.</div>
            ) : (
              <div className="mt-3 space-y-3">
                {[
                  ["Input quality", result.confidence.input_quality],
                  ["Model confidence", result.confidence.model_confidence],
                  ["Evidence agreement", result.confidence.evidence_agreement],
                  ...(result.confidence.cross_modal_support != null ? [["Cross-modal support", result.confidence.cross_modal_support] as const] : []),
                  ["Overall", result.confidence.overall],
                ].map(([label, val]) => (
                  <div key={label as string} className="flex items-center gap-2">
                    <div className="w-36 text-xs mono text-zinc-600">{label as string}</div>
                    <div className="flex-1 h-2.5 bg-zinc-100 rounded-full overflow-hidden">
                      <div className={cn("h-full", label === "Overall" ? "bg-emerald-600" : "bg-zinc-900")} style={{ width: `${(val as number) * 100}%` }} />
                    </div>
                    <div className="w-12 text-xs mono font-bold">{(val as number).toFixed(2)}</div>
                  </div>
                ))}
                <div className={cn("text-xs mono px-2 py-1 rounded-full inline-block", result.confidence.is_heuristic ? "text-amber-700 bg-amber-50 border border-amber-200" : "text-emerald-700 bg-emerald-50 border border-emerald-200")}>
                  {result.confidence.is_heuristic ? "Heuristic confidence" : "Model confidence"}
                </div>
                {result.refused && <div className="text-xs bg-red-50 border border-red-200 text-red-800 px-3 py-2 rounded-xl">Refused — see validation reason in trace. This is intentional trust behavior.</div>}
              </div>
            )}
          </Card>

          {/* Provenance */}
          <Card className="p-5">
            <div className="font-bold text-sm">PROVENANCE</div>
            {!result ? (
              <div className="text-sm text-zinc-500 mt-2">Models / tools / versions / parameters will appear after analysis.</div>
            ) : (
              <div className="mt-3 space-y-2 text-xs mono">
                <div className="flex flex-wrap gap-2">
                  {result.tools_used.map((t: any) => (
                    <span key={t.tool_id} className="px-2.5 py-1.5 rounded-full bg-zinc-900 text-white">
                      {t.tool_id}
                    </span>
                  ))}
                </div>
                <div className="bg-zinc-50 border rounded-xl p-3 space-y-1">
                  <div>report_id: {result.report_id}</div>
                  <div>timestamp: {result.timestamp}</div>
                  <div>is_heuristic: {String(result.is_heuristic)}</div>
                  <div>vlm: attempted={String(result.vlm_attempted)} loaded={String(result.vlm_loaded)} valid={String(result.vlm_generation_valid)}</div>
                  {result.fallback_reason && <div>fallback: {result.fallback_reason}</div>}
                </div>
                <div className="flex gap-2">
                  <Button onClick={() => downloadJSON(result)} className="flex-1 gap-2 text-xs">
                    <Download className="w-4 h-4" /> JSON Report
                  </Button>
                  <Button variant="outline" onClick={() => alert("PDF: JSON is primary per PS. For PDF, run python scripts/generate_pdf.py " + result.report_id + " or use browser Print → Save as PDF.")} className="flex-1 gap-2 text-xs">
                    <FileText className="w-4 h-4" /> PDF
                  </Button>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}

function downloadJSON(r: AnalysisResponse) {
  const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `SatQuery_report_${r.report_id}.json`;
  a.click();
}

// ---------------------------------------------------------------------------
// Evidence view with tabs, comparison slider, layer toggles
// ---------------------------------------------------------------------------
function EvidenceView({ result, tab, compare, setCompare }: any) {
  if (!result) return null;
  const hasChange = !!result.change_map_b64;
  const hasOptical = !!result.optical_overlay_b64;
  const hasSar = !!result.sar_overlay_b64;
  const hasFused = !!result.fused_overlay_b64;
  const hasGround = !!(result.grounding_boxes && result.grounding_boxes.length);

  const show = (t: string) => tab === "all" || tab === t;

  return (
    <div className="space-y-4">
      {/* Previews */}
      {result.previews && (
        <div className="grid grid-cols-2 gap-3">
          {result.previews.map((p: any, i: number) => (
            <div key={i} className="rounded-xl overflow-hidden border border-zinc-200">
              <img src={`data:image/png;base64,${p.preview_b64}`} alt={p.filename} className="w-full h-40 object-cover" />
              <div className="px-2 py-1.5 bg-zinc-50 mono text-[11px] truncate">
                {p.filename} • {p.meta.width}×{p.meta.height} • {p.meta.modality}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Before/after comparison for temporal */}
      {show("change") && hasChange && result.previews?.length >= 2 && (
        <div className="rounded-xl border border-zinc-200 overflow-hidden">
          <div className="px-3 py-2 bg-zinc-50 mono text-xs font-bold flex items-center justify-between">
            <span>BEFORE / AFTER COMPARISON</span>
            <span className="text-zinc-500">drag slider</span>
          </div>
          <div className="relative h-56 bg-zinc-900 overflow-hidden select-none">
            <img src={`data:image/png;base64,${result.previews[0].preview_b64}`} className="absolute inset-0 w-full h-full object-contain" alt="T1" />
            <div className="absolute inset-0 w-full h-full overflow-hidden" style={{ width: `${compare}%` }}>
              <img src={`data:image/png;base64,${result.previews[1].preview_b64}`} className="absolute inset-0 w-full h-full object-contain bg-zinc-900" alt="T2" />
            </div>
            <input type="range" min={0} max={100} value={compare} onChange={(e) => setCompare(Number(e.target.value))} className="absolute bottom-2 left-1/2 -translate-x-1/2 w-3/4" />
            <div className="absolute top-2 left-2 mono text-[10px] bg-black/70 text-white px-2 py-1 rounded-full">T1</div>
            <div className="absolute top-2 right-2 mono text-[10px] bg-white/90 text-zinc-900 px-2 py-1 rounded-full">T2</div>
          </div>
          <div className="px-3 py-2 text-xs text-zinc-600">{result.change_description}</div>
        </div>
      )}

      {/* Change map */}
      {show("change") && hasChange && (
        <div className="rounded-xl border border-zinc-200 overflow-hidden">
          <div className="px-3 py-2 bg-zinc-50 mono text-xs font-bold">CHANGE MAP (JET) + MASK (RED)</div>
          <div className="grid grid-cols-2 gap-0">
            <img src={`data:image/png;base64,${result.change_map_b64}`} className="w-full h-56 object-contain bg-zinc-900" alt="change" />
            <div className="relative bg-zinc-100 flex items-center justify-center">
              <img src={`data:image/png;base64,${result.previews[0].preview_b64}`} className="w-full h-56 object-contain opacity-60" alt="base" />
              <img src={`data:image/png;base64,${result.change_map_b64}`} className="absolute inset-0 w-full h-56 object-contain opacity-40 mix-blend-screen" alt="overlay" />
              <span className="absolute bottom-2 left-2 mono text-[10px] bg-red-600 text-white px-2 py-1 rounded-full">CHANGE OVERLAY</span>
            </div>
          </div>
        </div>
      )}

      {/* Optical / SAR */}
      {(show("optical") || show("sar")) && (hasOptical || hasSar) && (
        <div className="grid grid-cols-2 gap-3">
          {show("optical") && hasOptical && (
            <div className="rounded-xl border overflow-hidden">
              <div className="px-3 py-2 bg-sky-50 mono text-xs font-bold">OPTICAL — water regions</div>
              <div className="relative">
                <img src={`data:image/png;base64,${result.previews[0].preview_b64}`} className="w-full h-48 object-contain bg-zinc-50" alt="optical base" />
                <img src={`data:image/png;base64,${result.optical_overlay_b64}`} className="absolute inset-0 w-full h-48 object-contain opacity-60" alt="optical overlay" />
              </div>
              <div className="px-3 py-2 text-xs mono">{result.optical_evidence ? `water ${result.optical_evidence.water_pct}% • veg ${result.optical_evidence.veg_pct}% • built ${result.optical_evidence.built_pct}%` : ""}</div>
            </div>
          )}
          {show("sar") && hasSar && (
            <div className="rounded-xl border overflow-hidden">
              <div className="px-3 py-2 bg-amber-50 mono text-xs font-bold">SAR — water (low backscatter)</div>
              <div className="relative">
                <img src={`data:image/png;base64,${result.previews[result.previews.length - 1].preview_b64}`} className="w-full h-48 object-contain bg-zinc-900" alt="sar base" />
                <img src={`data:image/png;base64,${result.sar_overlay_b64}`} className="absolute inset-0 w-full h-48 object-contain opacity-60" alt="sar overlay" />
              </div>
              <div className="px-3 py-2 text-xs mono">{result.sar_evidence ? `water ${result.sar_evidence.water_pct}% • built ${result.sar_evidence.built_pct}% • CV ${result.sar_evidence.speckle_cv}` : ""}</div>
            </div>
          )}
        </div>
      )}

      {/* Fused */}
      {show("fusion") && hasFused && (
        <div className="rounded-xl border border-emerald-200 overflow-hidden">
          <div className="px-3 py-2 bg-emerald-50 mono text-xs font-bold">
            FUSED — high-confidence agreement (intersection) • IoU {result.fusion_evidence?.mean_iou ?? ""} • {result.fusion_evidence?.agreement_label ?? ""}
          </div>
          <div className="relative">
            <img src={`data:image/png;base64,${result.previews[0].preview_b64}`} className="w-full h-56 object-contain bg-zinc-50" alt="fused base" />
            <img src={`data:image/png;base64,${result.fused_overlay_b64}`} className="absolute inset-0 w-full h-56 object-contain opacity-70" alt="fused" />
          </div>
          <div className="px-3 py-2 text-xs">{result.fusion_evidence?.corroboration || result.fused_statement || ""}</div>
        </div>
      )}

      {/* Grounding */}
      {show("ground") && hasGround && (
        <GroundingView result={result} />
      )}

      {!hasChange && !hasOptical && !hasSar && !hasFused && !hasGround && (
        <div className="text-sm text-zinc-500 text-center py-6">No visual evidence for this task (e.g., pure VQA). Trace and answer contain the evidence.</div>
      )}
    </div>
  );
}

function GroundingView({ result }: any) {
  const meta = result.previews[0].meta;
  const boxes = result.grounding_boxes;
  return (
    <div className="rounded-xl border overflow-hidden">
      <div className="px-3 py-2 bg-violet-50 mono text-xs font-bold">GROUNDING — {boxes.length} box(es)</div>
      <div className="relative w-full flex justify-center bg-zinc-50 p-2">
        <div className="relative inline-block">
          <img src={`data:image/png;base64,${result.previews[0].preview_b64}`} className="max-h-[320px] object-contain" alt="ground" id="groundImg" />
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {boxes.map((b: any, i: number) => {
              // scale assuming preview is rendered at native-ish size; approximate
              const scaleX = 100 / meta.width;
              const scaleY = 100 / meta.height;
              return (
                <g key={i}>
                  <rect x={`${(b.x * scaleX)}%`} y={`${(b.y * scaleY)}%`} width={`${(b.w * scaleX)}%`} height={`${(b.h * scaleY)}%`} fill="none" stroke="#7c3aed" strokeWidth={2} rx={4} />
                  <text x={`${(b.x * scaleX)}%`} y={`${(b.y * scaleY)}%`} dy={-4} fill="#7c3aed" fontSize={10} fontWeight={800} fontFamily="monospace">
                    {b.label} {b.confidence}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
      <div className="px-3 py-2 text-xs mono">Boxes: {boxes.map((b: any) => `${b.label} [${b.x},${b.y},${b.w},${b.h}] conf ${b.confidence}`).join(" • ")}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Placeholder views for other nav items — styled but honest
// ---------------------------------------------------------------------------
function DatasetsView() {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-6">
      <h1 className="text-2xl font-extrabold">Datasets</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {[
          { name: "EuroSAT RGB", role: "RS adaptation (verified 1500, 64×64, 10 classes)", path: "datasets/raw/eurosat/EuroSAT", status: "present", count: "150/class" },
          { name: "BigEarthNet / reBEN v2", role: "Primary per PS (590k, 120×120, S1+S2)", path: "datasets/raw/reben", status: "manifest + multimodal loader", count: "100-pair subset verified" },
          { name: "VRSBench", role: "Caption/Grounding/VQA eval", path: "HF xiang709/VRSBench", status: "scaffold", count: "placeholder" },
          { name: "RSVQA", role: "VQA training augment", path: "rsvqa.sylvainlobry.com", status: "scaffold", count: "placeholder" },
          { name: "CDVQA", role: "Change VQA eval (SECOND-based)", path: "github YZHJessica/CDVQA", status: "scaffold", count: "2968 pairs" },
          { name: "SEN1-2", role: "Cross-modal pretraining", path: "mediatum", status: "placeholder", count: "—" },
        ].map((d) => (
          <Card key={d.name} className="p-4">
            <div className="font-bold">{d.name}</div>
            <div className="text-xs text-zinc-500 mt-1">{d.role}</div>
            <div className="text-xs mono mt-2 bg-zinc-50 border rounded-lg px-2 py-1">{d.path}</div>
            <div className="mt-2 flex items-center gap-2">
              <Badge className={cn(d.status === "present" ? "bg-emerald-600 text-white" : "bg-zinc-900 text-white")}>{d.status}</Badge>
              <span className="text-xs mono text-zinc-500">{d.count}</span>
            </div>
          </Card>
        ))}
      </div>
      <Card className="p-4">
        <div className="font-bold text-sm">How to prepare</div>
        <pre className="mt-2 text-xs mono bg-zinc-900 text-zinc-100 p-3 rounded-xl overflow-auto">
          python scripts/prepare_demo_data.py{`\n`}python scripts/download_eurosat.py --subset 1500{`\n`}python training/dataset_loader.py{`\n`}python training/train_lora_real.py --dataset eurosat --subset 1500 --dry_run
        </pre>
      </Card>
    </div>
  );
}
function ModelsView({ registry }: any) {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-6">
      <h1 className="text-2xl font-extrabold">Models</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {(registry?.tools ?? []).map((t: any) => (
          <Card key={t.id} className="p-4">
            <div className="font-bold mono">{t.id}</div>
            <div className="text-sm font-semibold">{t.name}</div>
            <div className="text-xs text-zinc-500 mt-1">{t.capability?.join(" • ")} • {t.modality?.join("/")}</div>
            <div className="text-xs mono bg-zinc-50 border rounded-lg px-2 py-1 mt-2">base: {t.base_model}</div>
            <Badge className="mt-2 bg-emerald-600 text-white">{t.status}</Badge>
          </Card>
        ))}
      </div>
      <Card className="p-4">
        <div className="text-sm font-bold">Model abstraction</div>
        <p className="text-xs text-zinc-600 mt-1">
          App uses <span className="mono">packages/models/adapter.py::ModelAdapter</span> — orchestrator asks &quot;which model provides capability X?&quot; not &quot;call Qwen&quot;. Swappable: heuristic → BLIP LoRA → Qwen2-VL QLoRA → cloud model without rewriting the app. See <span className="mono">docs/MODEL_ABSTRACTION.md</span>.
        </p>
      </Card>
    </div>
  );
}
function ToolsView({ registry }: any) {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-6">
      <h1 className="text-2xl font-extrabold">Tools</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {[
          { id: "satquery-rs-vlm", caps: ["VQA", "Captioning", "Grounding"], inputs: "1 image", out: "answer, caption, boxes" },
          { id: "change-detector", caps: ["temporal change"], inputs: "T1, T2", out: "change map, mask, % , description" },
          { id: "optical-analyzer", caps: ["water", "built", "veg"], inputs: "optical", out: "masks, overlays" },
          { id: "sar-analyzer", caps: ["water", "built"], inputs: "SAR", out: "backscatter stats, masks" },
          { id: "fusion-engine", caps: ["IoU/Dice", "agreement"], inputs: "optical_res + sar_res", out: "fused masks, corroboration" },
        ].map((t) => (
          <Card key={t.id} className="p-4">
            <div className="font-bold mono">{t.id}</div>
            <div className="text-xs mt-1">{t.caps.join(" • ")}</div>
            <div className="text-xs mono mt-2">inputs: {t.inputs} → outputs: {t.out}</div>
            <div className="text-xs mono mt-2 bg-zinc-50 border rounded-lg px-2 py-1">module: tools/{t.id.replace(/-/g, "_")}.py</div>
          </Card>
        ))}
      </div>
      <Card className="p-4">
        <div className="text-sm font-bold">Tool contract</div>
        <p className="text-xs text-zinc-600 mt-1">
          Every tool implements <span className="mono">packages/tools/base.py::SpecialistTool</span> (spec, execute, health, provenance, input validation). Registry selects by capability. See <span className="mono">docs/TOOL_ABSTRACTION.md</span>.
        </p>
      </Card>
    </div>
  );
}
function RunsView({ recent, setNav }: any) {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-4">
      <h1 className="text-2xl font-extrabold">Runs</h1>
      {recent.length === 0 ? <div className="text-sm text-zinc-500">No runs yet. Go to Analysis and run a demo.</div> : recent.map((r: AnalysisResponse) => (
        <Card key={r.report_id} className="p-4">
          <div className="flex items-center justify-between">
            <div className="font-bold mono text-sm">{r.report_id}</div>
            <Badge className={cn(r.refused ? "bg-red-600 text-white" : "bg-emerald-600 text-white")}>{r.refused ? "refused" : r.task}</Badge>
          </div>
          <div className="text-sm mt-1">{r.query}</div>
          <div className="text-xs mono text-zinc-500 mt-1">{r.timestamp} • {r.tools_used.length} tools • overall {r.confidence.overall.toFixed(2)}</div>
        </Card>
      ))}
      <Button variant="outline" onClick={() => setNav("analysis")}>Go to Analysis</Button>
    </div>
  );
}
function ReportsView({ recent }: any) {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-4">
      <h1 className="text-2xl font-extrabold">Reports</h1>
      <p className="text-sm text-zinc-500">Each analysis saves <span className="mono">reports/report_&lt;id&gt;.json</span> with query, metas, task, tools, params, result, confidence, trace — plus PDF via <span className="mono">scripts/generate_pdf.py</span>.</p>
      {recent.length === 0 ? <div className="text-sm text-zinc-500 mt-4">No reports in this session. Run an analysis.</div> : recent.map((r: AnalysisResponse) => (
        <Card key={r.report_id} className="p-4">
          <div className="font-mono text-sm font-bold">{r.report_id}.json</div>
          <div className="text-xs mono bg-zinc-50 border rounded-lg p-2 mt-2 overflow-auto">{JSON.stringify({ query: r.query, task: r.task, tools: r.tools_used.map((t: any) => t.tool_id), confidence: r.confidence }, null, 2).slice(0, 800)}</div>
          <Button className="mt-3" onClick={() => downloadJSON(r)}><Download className="w-4 h-4 mr-2" />Download JSON</Button>
        </Card>
      ))}
    </div>
  );
}
function HealthView({ health, registry }: any) {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-4">
      <h1 className="text-2xl font-extrabold">System Health</h1>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="font-bold flex items-center gap-2"><Activity className="w-4 h-4" /> Backend</div>
          <div className="text-sm mono mt-2">{JSON.stringify(health ?? { status: "checking" }, null, 2)}</div>
        </Card>
        <Card className="p-4">
          <div className="font-bold flex items-center gap-2"><Layers className="w-4 h-4" /> Registry</div>
          <div className="text-sm mono mt-2">{registry ? `${registry.tools?.length} tools` : "loading"}</div>
        </Card>
        <Card className="p-4">
          <div className="font-bold flex items-center gap-2"><ShieldCheck className="w-4 h-4" /> Observability</div>
          <div className="text-xs mt-2">Every analysis has run_id, request_id, timestamp, input metadata, query, plan, tools, models, params, latency, errors, warnings, outputs, confidence, provenance. See <span className="mono">packages/observability/logger.py</span>.</div>
        </Card>
      </div>
      <Card className="p-4">
        <div className="font-bold text-sm">Latest health probe</div>
        <pre className="text-xs mono bg-zinc-900 text-zinc-100 p-3 rounded-xl overflow-auto mt-2">{JSON.stringify({ health, registry: registry?.tools?.map((t: any) => t.id) }, null, 2)}</pre>
      </Card>
    </div>
  );
}
function SettingsView() {
  return (
    <div className="max-w-[1480px] mx-auto px-6 py-8 space-y-4">
      <h1 className="text-2xl font-extrabold">Settings</h1>
      <Card className="p-4">
        <div className="font-bold text-sm">Architecture</div>
        <p className="text-xs text-zinc-600 mt-1">This frontend is <span className="mono">apps/web</span> (Next.js 14, TypeScript, Tailwind, shadcn/ui, Lucide, TanStack Query, Zod). Backend is FastAPI at <span className="mono">localhost:8000</span>. The original single-file UI remains at <span className="mono">frontend/dist/index.html</span> served by FastAPI for offline fallback.</p>
      </Card>
      <Card className="p-4">
        <div className="font-bold text-sm">What&apos;s modular now</div>
        <ul className="text-xs mono mt-2 space-y-1 list-disc pl-4">
          <li>packages/core/schemas.py — typed InputImage, RasterMetadata, AnalysisRequest/Plan/Result</li>
          <li>packages/models/adapter.py — ModelAdapter (Qwen/BLIP/heuristic swappable)</li>
          <li>packages/tools/base.py — SpecialistTool contract + registry by capability</li>
          <li>packages/agent/planner.py — query → intents → task → validated plan</li>
          <li>packages/geospatial/metadata.py — CRS, tiling, GSD foundations</li>
          <li>packages/observability/logger.py — structured logs with run_id</li>
          <li>apps/web — production dashboard (this app)</li>
          <li>apps/api — future extracted API (currently backend/app is wrapper)</li>
        </ul>
      </Card>
    </div>
  );
}
