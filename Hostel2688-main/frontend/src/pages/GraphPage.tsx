import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Loader2,
  AlertCircle,
  Send,
  Zap,
  Activity,
  Eye,
  ChevronDown,
  ChevronUp,
  Info,
  SkipForward,
  Square,
  Brain,
} from "lucide-react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { api } from "../utils/api";

function isWebGLAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}

class GraphErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: "" };
  }
  static getDerivedStateFromError(err: Error) {
    return { hasError: true, error: err.message || "3D render failed" };
  }
  componentDidCatch(err: Error) {
    console.warn("[GraphPage] ErrorBoundary caught:", err);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#07070c]">
          <AlertCircle className="w-8 h-8 text-yellow-500/70" />
          <p className="text-[#8B95A5] text-sm font-medium">
            3D visualization could not render
          </p>
          <p className="text-[#4A5568] text-xs max-w-xs text-center">
            WebGL error: {this.state.error}. Try a Chromium-based browser with
            hardware acceleration enabled.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: "" })}
            className="mt-2 px-4 py-1.5 bg-[#00C896]/10 border border-[#00C896]/20 text-[#00C896] text-xs rounded-lg hover:bg-[#00C896]/20 transition-colors"
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

async function loadStaticGraph(head: number): Promise<ClusterData | null> {
  try {
    const r = await fetch(`/graph/gstar_head${head}.json`);
    if (!r.ok) return null;
    const raw = await r.json();

    const clusterMap: Record<string, number[]> = raw.clusters || {};
    const cids = Object.keys(clusterMap)
      .map(Number)
      .sort((a, b) => a - b);
    const N: number = raw.N || 3072;

    const labels = new Array(N).fill(0);
    for (const [cid, members] of Object.entries(clusterMap)) {
      for (const nid of members) {
        if (nid < N) labels[nid] = Number(cid);
      }
    }

    const rawEdges = (raw.edges || []).slice(0, 800);
    const edges = rawEdges.map((e: number[]) => ({
      source: e[0],
      target: e[1],
      weight: +(e[2] || 0.1),
      same_cluster: labels[e[0]] === labels[e[1]] && labels[e[0]] > 0,
    }));

    const nodeSet = new Set<number>();
    for (const e of edges) {
      nodeSet.add(e.source);
      nodeSet.add(e.target);
    }
    const hubSet = new Set<number>((raw.hubs || []).map((h: number[]) => h[0]));
    for (const h of hubSet) nodeSet.add(h);

    const picked = Array.from(nodeSet).slice(0, 200);
    const pickedSet = new Set(picked);

    const degMap: Record<number, number> = {};
    for (const e of edges) {
      degMap[e.source] = (degMap[e.source] || 0) + 1;
      degMap[e.target] = (degMap[e.target] || 0) + 1;
    }

    const nodes = picked.map((id) => ({
      id,
      cluster: labels[id] ?? 0,
      degree: degMap[id] || 0,
      is_hub: hubSet.has(id),
    }));
    const filteredEdges = edges.filter(
      (e: any) => pickedSet.has(e.source) && pickedSet.has(e.target),
    );

    const clusters: ClusterMeta[] = cids.map((cid) => {
      const members = clusterMap[String(cid)] || [];
      return {
        cluster_id: cid,
        neuron_count: members.length,
        avg_out_degree: 0,
        avg_in_degree: 0,
        internal_edges: 0,
        internal_weight: 0,
        hub_neurons: members
          .filter((n) => hubSet.has(n))
          .slice(0, 3)
          .map((n) => ({ neuron: n, degree: degMap[n] || 0 })),
        label: cid === 0 ? "Isolated" : `Cluster ${cid}`,
      };
    });

    const hist = (raw.out_deg_hist || []).map((y: number, i: number) => ({
      x: +(i * (raw.threshold || 0.07)).toFixed(4),
      y,
    }));

    return {
      n_neurons: N,
      n_display_nodes: nodes.length,
      n_display_edges: filteredEdges.length,
      n_total_edges: raw.n_edges || filteredEdges.length,
      num_clusters: raw.n_clusters || cids.length,
      modularity: raw.modularity || 0,
      beta: raw.threshold || 0.07,
      nodes,
      edges: filteredEdges,
      clusters,
      histogram: hist,
    };
  } catch (err) {
    console.warn("[GraphPage] static fallback failed:", err);
    return null;
  }
}

interface ClusterMeta {
  cluster_id: number;
  neuron_count: number;
  avg_out_degree: number;
  avg_in_degree: number;
  internal_edges: number;
  internal_weight: number;
  hub_neurons: Array<{ neuron: number; degree: number }>;
  label: string | null;
  label_confidence?: number;
  top_trigger_words?: string[];
}
interface ClusterData {
  n_neurons: number;
  n_display_nodes: number;
  n_display_edges: number;
  n_total_edges: number;
  num_clusters: number;
  modularity: number;
  beta: number;
  beta_effective?: number;
  nodes: Array<{
    id: number;
    cluster: number;
    degree: number;
    is_hub: boolean;
  }>;
  edges: Array<{
    source: number;
    target: number;
    weight: number;
    same_cluster: boolean;
  }>;
  clusters: ClusterMeta[];
  histogram: Array<{ x: number; y: number }>;
}
interface TokenAct {
  token_idx: number;
  byte: number;
  char: string;
  cluster_activations: Array<{
    cluster_id: number;
    activation: number;
    normalized: number;
    active_neurons: number;
  }>;
  node_activations?: Record<string, number>;
}
interface ActResult {
  input_chars: string[];
  num_tokens: number;
  per_token: TokenAct[];
  cumulative_cluster_activations: Array<{
    cluster_id: number;
    normalized: number;
  }>;
  node_activations: Record<string, number>;
}

const CC = [
  "#3a3f4b",
  "#8B5CF6",
  "#3B82F6",
  "#10B981",
  "#F59E0B",
  "#EF4444",
  "#EC4899",
  "#06B6D4",
  "#84CC16",
  "#F97316",
  "#6366F1",
  "#14B8A6",
  "#E879F9",
  "#FB923C",
  "#A78BFA",
  "#34D399",
  "#FBBF24",
  "#F472B6",
  "#22D3EE",
  "#A3E635",
];

const ci = (cid: number) => (cid <= 0 ? 0 : ((cid - 1) % (CC.length - 1)) + 1);

function hexToRgb(hex: string): [number, number, number] {
  const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return r
    ? [parseInt(r[1], 16), parseInt(r[2], 16), parseInt(r[3], 16)]
    : [128, 128, 128];
}

function Hist({
  data,
  beta,
}: {
  data: Array<{ x: number; y: number }>;
  beta: number;
}) {
  if (!data?.length) return null;
  const my = Math.max(...data.map((d) => d.y), 1);
  const mx = Math.max(...data.map((d) => d.x)),
    mn = Math.min(...data.map((d) => d.x));
  const bp = mx > mn ? ((beta - mn) / (mx - mn)) * 100 : 50;
  return (
    <div className="mt-1.5">
      <div className="text-[9px] text-gray-600 mb-0.5">
        G* weight distribution
      </div>
      <div className="relative h-8 bg-gray-900/40 rounded overflow-hidden border border-gray-800/40">
        <div className="absolute inset-0 flex items-end">
          {data.map((d, i) => (
            <div
              key={i}
              className="flex-1"
              style={{
                height: `${(d.y / my) * 100}%`,
                backgroundColor:
                  d.x >= beta ? "rgba(139,92,246,0.5)" : "rgba(75,85,99,0.3)",
              }}
            />
          ))}
        </div>
        <div
          className="absolute top-0 bottom-0 w-px bg-red-500/70"
          style={{ left: `${Math.min(Math.max(bp, 0), 100)}%` }}
        />
      </div>
    </div>
  );
}

function Pill({
  c,
  act,
  blink,
  open,
  dimmed,
  selected,
  onClick,
}: {
  c: ClusterMeta;
  act: number;
  blink: boolean;
  open: boolean;
  dimmed: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  const color = CC[ci(c.cluster_id)];
  const effectiveAct = dimmed ? 0 : act;
  const effectiveBlink = dimmed ? false : blink;
  const isGlowing = selected || effectiveBlink || effectiveAct > 0.3;
  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-lg border transition-all duration-500"
      style={{
        borderColor: isGlowing ? color : "#1f2937",
        backgroundColor: isGlowing
          ? `${color}18`
          : dimmed
            ? "#0a0a0f"
            : "#111827",
        boxShadow: selected
          ? `0 0 20px ${color}60, 0 0 40px ${color}20`
          : effectiveBlink
            ? `0 0 14px ${color}50`
            : effectiveAct > 0.3
              ? `0 0 8px ${color}30`
              : "none",
        opacity: dimmed ? 0.15 : 1,
        transform: isGlowing ? "scale(1.02)" : "scale(1)",
      }}
    >
      <div className="p-1.5 flex items-center gap-1.5">
        <div className="relative flex-shrink-0">
          <div
            className="w-2.5 h-2.5 rounded-full"
            style={{ backgroundColor: dimmed ? "#1a1a2e" : color }}
          />
          {(effectiveBlink || selected) && (
            <div
              className="absolute inset-0 rounded-full animate-ping"
              style={{ backgroundColor: color, opacity: 0.35 }}
            />
          )}
        </div>
        <span
          className={`text-[10px] font-medium truncate flex-1 ${dimmed ? "text-gray-700" : "text-gray-300"}`}
        >
          {c.label || `Cluster ${c.cluster_id}`}
        </span>
        <span className="text-[8px] text-gray-600">{c.neuron_count}n</span>
        {effectiveAct > 0.05 && (
          <span className="text-[9px] font-mono" style={{ color }}>
            {(effectiveAct * 100) | 0}%
          </span>
        )}
        {open ? (
          <ChevronUp size={9} className="text-gray-600" />
        ) : (
          <ChevronDown size={9} className="text-gray-600" />
        )}
      </div>
      {effectiveAct > 0.05 && (
        <div className="mx-1.5 mb-1 h-0.5 rounded-full bg-gray-800 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ backgroundColor: color, width: `${effectiveAct * 100}%` }}
          />
        </div>
      )}
      {open && !dimmed && (
        <div className="px-1.5 pb-1.5 space-y-1">
          <div className="grid grid-cols-2 gap-1 text-[8px]">
            <div className="p-1 bg-gray-900/40 rounded">
              <span className="text-gray-600">Out° </span>
              {c.avg_out_degree}
            </div>
            <div className="p-1 bg-gray-900/40 rounded">
              <span className="text-gray-600">In° </span>
              {c.avg_in_degree}
            </div>
          </div>
          {c.label_confidence != null && c.label_confidence > 0 && (
            <div className="text-[7px] text-gray-500">
              Confidence: {(c.label_confidence * 100).toFixed(0)}%
            </div>
          )}
          {c.hub_neurons.length > 0 && (
            <div className="p-1 bg-gray-900/40 rounded flex gap-0.5 flex-wrap text-[8px]">
              {c.hub_neurons.map((h) => (
                <span
                  key={h.neuron}
                  className="px-0.5 rounded bg-gray-800 font-mono text-gray-500"
                >
                  #{h.neuron}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function GraphPage() {
  const [cd, setCd] = useState<ClusterData | null>(null);
  const [ar, setAr] = useState<ActResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [activating, setActivating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [head, setHead] = useState(0);
  const [beta, setBeta] = useState(0.1);
  const [layer, setLayer] = useState(-1);
  const [text, setText] = useState("");
  const [edges, setEdges] = useState(true);
  const [expCl, setExpCl] = useState<number | null>(null);

  // hlCl = cluster highlighted by click (non-inference mode)
  const [hlCl, setHlCl] = useState<number | null>(null);

  // Playback
  const [pIdx, setPIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const pTimer = useRef<any>(null);
  const gRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const bTimer = useRef<any>(null);

  const PLAYBACK_DELAY = 400; // ms per token

  // cleanup WebGL on unmount
  useEffect(() => {
    return () => {
      const fg = gRef.current;
      if (fg) {
        // ForceGraph3D exposes the Three.js renderer via internal methods
        try {
          const renderer = fg.renderer?.();
          if (renderer) {
            renderer.dispose();
            renderer.forceContextLoss();
            const gl = renderer.getContext();
            const ext = gl?.getExtension("WEBGL_lose_context");
            if (ext) ext.loseContext();
          }
        } catch {
          // Best-effort cleanup
        }
        // Also stop the force simulation
        try {
          fg.pauseAnimation?.();
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  const load = useCallback(async (h: number, b: number) => {
    setLoading(true);
    setError(null);
    setAr(null);
    setPIdx(-1);
    setPlaying(false);
    setHlCl(null);
    try {
      const r = await api.get("/graph/clusters/french", {
        params: { head: h, beta: b, max_nodes: 800 },
        timeout: 60000,
      });
      setCd(r.data);
      hasZoomed.current = false;
    } catch (e: any) {
      console.warn("[GraphPage] Backend failed, trying static fallback...");
      // Try static gstar JSON as fallback
      const staticData = await loadStaticGraph(h);
      if (staticData) {
        setCd(staticData);
        hasZoomed.current = false;
        setError(null); // Clear error — static data loaded OK
      } else {
        setError(
          e?.response?.data?.detail ||
            e?.message ||
            "Backend unavailable and no static data",
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(head, beta);
  }, [head]); // eslint-disable-line

  const hasZoomed = useRef(false);
  const doZoomToFit = useCallback(() => {
    if (hasZoomed.current) return;
    const fg = gRef.current;
    if (!fg) return;
    fg.zoomToFit(800, 80); // generous padding to show entire brain
    hasZoomed.current = true;
  }, []);
  const handleEngineStop = useCallback(() => {
    doZoomToFit();
  }, [doZoomToFit]);
  useEffect(() => {
    if (!cd || hasZoomed.current) return;
    const t = setTimeout(() => doZoomToFit(), 2500);
    return () => clearTimeout(t);
  }, [cd, doZoomToFit]);

  useEffect(() => {
    if (bTimer.current) clearTimeout(bTimer.current);
    bTimer.current = setTimeout(() => load(head, beta), 700);
    return () => clearTimeout(bTimer.current);
  }, [beta]); // eslint-disable-line

  const run = useCallback(async () => {
    if (!text.trim() || !cd) return;
    setActivating(true);
    setPIdx(-1);
    setPlaying(false);
    setHlCl(null);
    try {
      const r = await api.post("/graph/activate", {
        text,
        model_name: "french",
        head,
        layer,
      });
      setAr(r.data);
      // Auto-start playback
      setPIdx(0);
      setPlaying(true);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Failed");
    } finally {
      setActivating(false);
    }
  }, [text, head, layer, cd]);

  useEffect(() => {
    if (pTimer.current) clearInterval(pTimer.current);
    if (playing && ar && pIdx >= 0) {
      pTimer.current = setInterval(() => {
        setPIdx((p) => {
          if (p >= ar.num_tokens - 1) {
            setPlaying(false);
            return p;
          }
          return p + 1;
        });
      }, PLAYBACK_DELAY);
    }
    return () => {
      if (pTimer.current) clearInterval(pTimer.current);
    };
  }, [playing, ar]); // eslint-disable-line

  const curTok =
    pIdx >= 0 && ar && pIdx < ar.per_token.length ? ar.per_token[pIdx] : null;
  const curMap: Record<number, number> = {};
  const blinkIds = new Set<number>();
  if (curTok) {
    // Sort activations to find the top cluster
    const sorted = [...curTok.cluster_activations]
      .filter((ca) => ca.cluster_id > 0 && ca.activation > 0)
      .sort((a, b) => b.activation - a.activation);
    const topAct = sorted.length > 0 ? sorted[0].activation : 1;
    for (const ca of curTok.cluster_activations) {
      curMap[ca.cluster_id] = ca.normalized;
      // Only blink if this cluster has >= 40% of the top cluster's activation
      if (ca.activation >= topAct * 0.4 && ca.activation > 0)
        blinkIds.add(ca.cluster_id);
    }
  }

  const nodeClMap = useMemo(() => {
    const m: Record<number, number> = {};
    if (cd) for (const n of cd.nodes) m[n.id] = n.cluster;
    return m;
  }, [cd]);

  const activeClusters = useMemo(() => {
    const s = new Set<number>();
    if (!curTok || !cd) return s;
    // Sort cluster activations descending, pick only top clusters that are truly firing
    const sorted = [...curTok.cluster_activations]
      .filter((ca) => ca.cluster_id > 0 && ca.activation > 0)
      .sort((a, b) => b.activation - a.activation);
    if (sorted.length === 0) return s;
    // Take top cluster, then include others only if they have >= 30% of top's activation
    const topAct = sorted[0].activation;
    for (const ca of sorted) {
      if (ca.activation >= topAct * 0.3) {
        s.add(ca.cluster_id);
      }
    }
    return s;
  }, [curTok, cd, nodeClMap]);

  const isPlayback = pIdx >= 0 && !!curTok;

  const graphData = useMemo(() => {
    if (!cd) return { nodes: [], links: [] };
    const nodeActs = ar?.node_activations || {};
    const nodes = cd.nodes.map((n) => ({
      ...n,
      activation: nodeActs[String(n.id)] ? Number(nodeActs[String(n.id)]) : 0,
    }));
    const links = edges
      ? cd.edges.map((e) => ({
          source:
            typeof e.source === "object" ? (e.source as any).id : e.source,
          target:
            typeof e.target === "object" ? (e.target as any).id : e.target,
          weight: e.weight,
          same_cluster: e.same_cluster,
        }))
      : [];
    return { nodes, links };
  }, [cd, ar, edges]);

  const nodeThreeObject = useCallback(
    (node: any) => {
      const cluster = node.cluster || 0;
      const isHub = node.is_hub;
      const activation = node.activation || 0;
      const isHl = hlCl !== null && cluster === hlCl;
      const color = CC[ci(cluster)];
      const [r, g, b] = hexToRgb(color);

      // Cluster blinking from per-token data
      const isBlinking = blinkIds.has(cluster);
      const blinkBoost = isBlinking ? 1.8 : 1;

      const baseSize = isHub ? 3 : 1.5;
      const actBoost = activation > 0 ? 1 + activation * 2.5 : 0;
      const hlBoost = isHl ? 1.4 : 1;
      const size = (baseSize + actBoost) * hlBoost * blinkBoost;

      const dim = hlCl !== null && !isHl ? 0.12 : 1.0;
      const emissiveStr = Math.max(
        activation > 0.1 ? activation * 0.7 : 0,
        isHl ? 0.25 : 0,
        isBlinking ? 0.6 : 0,
      );

      const geo = new THREE.SphereGeometry(size, 10, 10);
      const mat = new THREE.MeshLambertMaterial({
        color: new THREE.Color(
          (r / 255) * dim,
          (g / 255) * dim,
          (b / 255) * dim,
        ),
        emissive: new THREE.Color(
          (r / 255) * emissiveStr,
          (g / 255) * emissiveStr,
          (b / 255) * emissiveStr,
        ),
        transparent: dim < 1,
        opacity: dim < 1 ? 0.25 : 1,
      });
      return new THREE.Mesh(geo, mat);
    },
    [hlCl, blinkIds],
  );

  const linkColor = useCallback(
    (link: any) => {
      if (!edges) return "rgba(0,0,0,0)";
      const sid =
        typeof link.source === "object" ? link.source.id : link.source;
      if (link.same_cluster) {
        const sc = nodeClMap[sid] ?? 0;
        const dimmed = hlCl !== null && sc !== hlCl;
        return dimmed ? "rgba(40,40,55,0.04)" : `${CC[ci(sc)]}44`;
      }
      return hlCl !== null ? "rgba(40,40,55,0.03)" : "rgba(120,120,150,0.12)";
    },
    [nodeClMap, hlCl, edges],
  );

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      run();
    }
  };

  const handleClusterClick = useCallback(
    (cid: number) => {
      if (isPlayback) {
        // During playback, clicking a cluster stops it and selects that cluster
        setPlaying(false);
        setPIdx(-1);
        setAr(null);
        setHlCl(cid);
        setExpCl((prev) => (prev === cid ? null : cid));
      } else {
        // Non-playback: toggle cluster highlight
        setHlCl((prev) => (prev === cid ? null : cid));
        setExpCl((prev) => (prev === cid ? null : cid));
      }
    },
    [isPlayback],
  );

  return (
    <div
      className="h-full flex flex-row"
      style={{ height: "100%", minHeight: 0 }}
    >
      {/* LEFT — graph + controls */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <div className="px-5 pt-3 pb-0.5 flex-shrink-0">
          <div className="flex items-center gap-2">
            <Brain size={20} className="text-bdh-accent" />
            <h1 className="text-xl font-bold">
              <span className="gradient-text">Neural Graph</span> Explorer
            </h1>
          </div>
          <p className="text-gray-600 text-[11px]">
            G*=D<sub>x</sub>E neuron interaction graph · Louvain clusters ·
            French model
            {isPlayback && (
              <span className="text-bdh-accent ml-2">
                ● LIVE — watching the brain think
              </span>
            )}
          </p>
        </div>

        {/* Controls */}
        <div className="px-5 py-1.5 flex flex-wrap items-center gap-2 flex-shrink-0">
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-gray-500 uppercase">Head</span>
            {[0, 1, 2, 3].map((h) => (
              <button
                key={h}
                onClick={() => setHead(h)}
                className={`w-6 h-6 rounded text-[10px] font-mono font-bold ${head === h ? "bg-bdh-accent text-white" : "bg-gray-800 text-gray-500 hover:bg-gray-700"}`}
              >
                {h + 1}
              </button>
            ))}
          </div>
          <div className="w-px h-4 bg-gray-800" />
          <div className="flex items-center gap-1 flex-1 max-w-[200px]">
            <span className="text-[9px] text-gray-500">β</span>
            <input
              type="range"
              min={0.01}
              max={0.5}
              step={0.005}
              value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value))}
              className="flex-1 h-1 appearance-none rounded-full bg-gray-800
                [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3
                [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-bdh-accent [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <span className="text-[9px] font-mono text-bdh-accent w-8 text-right">
              {beta.toFixed(3)}
            </span>
          </div>
          <div className="w-px h-4 bg-gray-800" />
          <button
            onClick={() => setEdges(!edges)}
            className={`p-1 rounded ${edges ? "bg-gray-700 text-gray-200" : "bg-gray-800/50 text-gray-600"}`}
          >
            <Eye size={13} />
          </button>
          <button
            onClick={() => {
              gRef.current?.zoomToFit(600, 80);
            }}
            className="p-1 rounded bg-gray-800/50 text-gray-500 hover:bg-gray-700"
          >
            <RotateCcw size={13} />
          </button>
          {hlCl !== null && !isPlayback && (
            <button
              onClick={() => setHlCl(null)}
              className="px-2 py-0.5 rounded text-[9px] bg-gray-700 text-gray-300 hover:bg-gray-600"
            >
              Show All
            </button>
          )}
        </div>

        {error && (
          <div className="mx-5 mb-1 p-1.5 bg-red-500/10 border border-red-500/30 rounded text-red-300 text-[10px] flex items-center gap-1 flex-shrink-0">
            <AlertCircle size={12} />
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-400"
            >
              ✕
            </button>
          </div>
        )}

        {cd &&
          cd.beta_effective != null &&
          Math.abs(cd.beta_effective - beta) > 0.001 && (
            <div className="mx-5 mb-1 p-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded text-yellow-300 text-[10px] flex items-center gap-1 flex-shrink-0">
              <Info size={12} />
              <span>
                β auto-adjusted: {beta.toFixed(3)} →{" "}
                {cd.beta_effective.toFixed(3)} for visibility
              </span>
              <button
                onClick={() => setBeta(cd.beta_effective!)}
                className="ml-auto text-yellow-400 text-[9px] underline"
              >
                Apply
              </button>
            </div>
          )}

        {/* 3D Graph */}
        <div
          ref={containerRef}
          className="flex-1 relative mx-5 mb-1.5 rounded-xl overflow-hidden border border-gray-800/40 bg-[#07070c]"
        >
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <Loader2 className="w-7 h-7 text-bdh-accent animate-spin" />
            </div>
          ) : !isWebGLAvailable() ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
              <AlertCircle className="w-8 h-8 text-yellow-500/70" />
              <p className="text-[#8B95A5] text-sm font-medium">
                WebGL is not available
              </p>
              <p className="text-[#4A5568] text-xs max-w-xs text-center">
                The 3D brain visualization requires WebGL. Enable hardware
                acceleration in your browser settings or try a different
                browser.
              </p>
            </div>
          ) : graphData.nodes.length > 0 ? (
            <GraphErrorBoundary>
              <ForceGraph3D
                ref={gRef}
                graphData={graphData}
                nodeThreeObject={nodeThreeObject}
                nodeThreeObjectExtend={false}
                linkColor={linkColor}
                linkOpacity={0.25}
                linkWidth={0.4}
                backgroundColor="#07070c"
                enableNodeDrag={true}
                enableNavigationControls={true}
                controlType="orbit"
                nodeLabel={(n: any) =>
                  `#${n.id} · Cluster ${n.cluster} · Deg ${n.degree}${n.is_hub ? " (HUB)" : ""}`
                }
                onNodeClick={(n: any) =>
                  setHlCl((prev) => (prev === n.cluster ? null : n.cluster))
                }
                onBackgroundClick={() => setHlCl(null)}
                onEngineStop={handleEngineStop}
                showNavInfo={false}
                d3AlphaDecay={0.025}
                d3VelocityDecay={0.3}
                warmupTicks={60}
                cooldownTicks={180}
                width={undefined}
                height={undefined}
              />
            </GraphErrorBoundary>
          ) : (
            !loading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                <AlertCircle className="w-8 h-8 text-[#4A5568]" />
                <p className="text-[#8B95A5] text-sm font-medium">
                  {error || "No graph data available"}
                </p>
                <p className="text-[#4A5568] text-xs max-w-xs text-center">
                  The Graph Brain requires the backend to compute neuron
                  clusters. Start it with:{" "}
                  <code className="text-[#00C896] text-[11px]">
                    uvicorn backend.main:app --reload
                  </code>
                </p>
                <button
                  onClick={() => load(head, beta)}
                  className="mt-2 px-4 py-1.5 bg-[#00C896]/10 border border-[#00C896]/20 text-[#00C896] text-xs rounded-lg hover:bg-[#00C896]/20 transition-colors"
                >
                  Retry
                </button>
              </div>
            )
          )}

          {/* Overlay stats */}
          {cd && !loading && (
            <div className="absolute top-2 left-2 flex gap-1 pointer-events-none">
              <div className="px-1.5 py-0.5 rounded bg-black/60 text-[8px] font-mono text-gray-500">
                {cd.n_display_nodes}n·{cd.n_display_edges}e
              </div>
              <div className="px-1.5 py-0.5 rounded bg-black/60 text-[8px] font-mono text-bdh-accent/80">
                {cd.num_clusters}cl Q={cd.modularity.toFixed(3)}
              </div>
            </div>
          )}

          {/* Token bar during playback */}
          {curTok && ar && (
            <div className="absolute bottom-1.5 left-2 right-2 pointer-events-none">
              <div className="px-2 py-1.5 rounded-lg bg-black/80 backdrop-blur flex items-center gap-0.5 overflow-x-auto">
                <span className="text-[7px] text-gray-600 mr-1 flex-shrink-0 uppercase">
                  Token
                </span>
                {ar.input_chars.map((ch, i) => (
                  <span
                    key={i}
                    className={`px-0.5 text-[10px] font-mono rounded flex-shrink-0 transition-all duration-300 ${
                      i === pIdx
                        ? "bg-bdh-accent text-white px-1.5 py-0.5 scale-110"
                        : i < pIdx
                          ? "text-gray-500"
                          : "text-gray-700"
                    }`}
                  >
                    {ch}
                  </span>
                ))}
                {(() => {
                  const total = cd?.nodes.length ?? 0;
                  const na = ar?.node_activations || {};
                  const firing =
                    total > 0
                      ? cd!.nodes.filter((n) => (na[String(n.id)] ?? 0) > 0)
                          .length
                      : 0;
                  const pct =
                    total > 0 ? ((1 - firing / total) * 100).toFixed(0) : "—";
                  return (
                    <span className="ml-auto text-[7px] text-bdh-accent/70 flex-shrink-0 whitespace-nowrap">
                      {firing}/{total} fire · {pct}% sparse
                    </span>
                  );
                })()}
              </div>
            </div>
          )}
        </div>

        {/* Inference input */}
        <div className="px-5 pb-3 flex-shrink-0">
          <div className="glass-card p-2">
            <div className="flex items-center gap-1 mb-1">
              <Zap size={11} className="text-bdh-accent" />
              <span className="text-[9px] font-semibold text-gray-400">
                Live Inference — type text, watch the brain think
              </span>
              {ar && (
                <div className="ml-auto flex gap-0.5">
                  <button
                    onClick={() => {
                      setPIdx(0);
                      setPlaying(true);
                    }}
                    className="p-0.5 rounded bg-gray-800 text-gray-500 hover:text-gray-300"
                  >
                    <RotateCcw size={9} />
                  </button>
                  <button
                    onClick={() => setPlaying(!playing)}
                    className={`p-0.5 rounded ${playing ? "bg-bdh-accent/20 text-bdh-accent" : "bg-gray-800 text-gray-500"}`}
                  >
                    {playing ? <Pause size={9} /> : <Play size={9} />}
                  </button>
                  <button
                    onClick={() =>
                      setPIdx((p) => Math.min(p + 1, (ar?.num_tokens || 1) - 1))
                    }
                    className="p-0.5 rounded bg-gray-800 text-gray-500 hover:text-gray-300"
                  >
                    <SkipForward size={9} />
                  </button>
                  <button
                    onClick={() => {
                      setPlaying(false);
                      setPIdx(-1);
                      setAr(null);
                      setHlCl(null);
                    }}
                    className="p-0.5 rounded bg-gray-800 text-gray-500 hover:text-gray-300"
                  >
                    <Square size={9} />
                  </button>
                </div>
              )}
            </div>
            <div className="flex gap-1.5">
              <input
                type="text"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={onKey}
                placeholder="e.g. The price in euros was 50 francs"
                className="input-field flex-1 text-[10px] !py-1.5"
              />
              <select
                value={layer}
                onChange={(e) => setLayer(+e.target.value)}
                className="px-1 rounded bg-gray-800 border border-gray-700 text-[8px] text-gray-400"
              >
                <option value={-1}>All</option>
                {[0, 1, 2, 3, 4, 5, 6, 7].map((l) => (
                  <option key={l} value={l}>
                    L{l}
                  </option>
                ))}
              </select>
              <button
                onClick={run}
                disabled={activating || !text.trim()}
                className="btn-primary flex items-center gap-1 text-[9px] disabled:opacity-40 !py-1 !px-2.5"
              >
                {activating ? (
                  <Loader2 size={11} className="animate-spin" />
                ) : (
                  <Send size={11} />
                )}{" "}
                Run
              </button>
            </div>
            <div className="mt-1 flex flex-wrap gap-1">
              {[
                "The price in euros was 50 francs",
                "Germany and France signed the treaty",
                "Le dollar américain",
                "The European Parliament",
                "le changement climatique menace",
              ].map((ex) => (
                <button
                  key={ex}
                  onClick={() => setText(ex)}
                  className="px-1 py-0.5 text-[7px] bg-gray-800/40 hover:bg-gray-700 rounded text-gray-600 hover:text-gray-400 truncate max-w-[150px]"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT SIDEBAR */}
      <aside className="w-64 border-l border-gray-800/50 bg-gray-900/30 flex flex-col overflow-hidden flex-shrink-0">
        <div className="p-2 border-b border-gray-800/50 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="text-[9px] font-bold text-gray-400 uppercase tracking-wider flex items-center gap-1">
              <Activity size={9} className="text-bdh-accent" /> Clusters
            </h2>
            {cd && (
              <span className="text-[7px] font-mono text-gray-700">
                {cd.n_neurons.toLocaleString()}n
              </span>
            )}
          </div>
          {curTok && (
            <div className="mt-1 px-1.5 py-1 rounded bg-bdh-accent/10 border border-bdh-accent/20">
              <div className="text-[7px] text-bdh-accent/60 uppercase">
                Token {curTok.token_idx + 1}/{ar?.num_tokens}
              </div>
              <div className="text-sm font-mono text-white">
                '{curTok.char}'
              </div>
              <div className="text-[7px] text-gray-500 mt-0.5">
                {activeClusters.size} cluster
                {activeClusters.size !== 1 ? "s" : ""} active
              </div>
            </div>
          )}
          {hlCl !== null && !isPlayback && (
            <div
              className="mt-1 px-1.5 py-1 rounded border border-gray-700"
              style={{
                borderColor: CC[ci(hlCl)] + "40",
                backgroundColor: CC[ci(hlCl)] + "08",
              }}
            >
              <div
                className="text-[7px] uppercase"
                style={{ color: CC[ci(hlCl)] }}
              >
                Selected Cluster
              </div>
              <div className="text-sm font-mono text-white">
                {cd?.clusters.find((c) => c.cluster_id === hlCl)?.label ||
                  `Cluster ${hlCl}`}
              </div>
            </div>
          )}
        </div>
        {cd && (
          <div className="px-2 flex-shrink-0">
            <Hist data={cd.histogram} beta={beta} />
          </div>
        )}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
          {cd?.clusters
            .filter((c) => c.cluster_id > 0)
            .map((c) => (
              <Pill
                key={c.cluster_id}
                c={c}
                act={
                  curMap[c.cluster_id] ??
                  (ar
                    ? (ar.cumulative_cluster_activations.find(
                        (x) => x.cluster_id === c.cluster_id,
                      )?.normalized ?? 0)
                    : 0)
                }
                blink={blinkIds.has(c.cluster_id)}
                open={expCl === c.cluster_id}
                selected={hlCl === c.cluster_id && !isPlayback}
                dimmed={
                  (isPlayback &&
                    activeClusters.size > 0 &&
                    !activeClusters.has(c.cluster_id)) ||
                  (hlCl !== null && !isPlayback && hlCl !== c.cluster_id)
                }
                onClick={() => handleClusterClick(c.cluster_id)}
              />
            ))}
          {cd?.clusters.find((c) => c.cluster_id === 0) && (
            <div className="pt-0.5 border-t border-gray-800/40 text-[7px] text-gray-700 flex items-center gap-0.5">
              <Info size={7} />{" "}
              {cd.clusters.find((c) => c.cluster_id === 0)?.neuron_count}{" "}
              isolated
            </div>
          )}
        </div>
        <div className="p-2 border-t border-gray-800/50 bg-gray-950/50 flex-shrink-0">
          <p className="text-[8px] text-gray-600">
            Head {head + 1}/4 · β={beta.toFixed(3)}
            {cd?.beta_effective != null &&
              Math.abs(cd.beta_effective - beta) > 0.001 && (
                <span className="text-yellow-500">
                  {" "}
                  → eff={cd.beta_effective.toFixed(3)}
                </span>
              )}
          </p>
          <p className="text-[7px] text-gray-700 mt-0.5">
            Click cluster to isolate · Enter text to watch brain activate
          </p>
        </div>
      </aside>
    </div>
  );
}
