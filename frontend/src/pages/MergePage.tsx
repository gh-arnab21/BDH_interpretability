import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import {
  GitMerge,
  Check,
  Zap,
  Brain,
  BarChart3,
  FileText,
  AlertCircle,
  Terminal,
  Loader2,
  Activity,
  TrendingDown,
  Timer,
} from "lucide-react";
import { adaptMergeData } from "../utils/dataAdapters";

interface ModelInfo {
  name: string;
  flag: string;
  params: number;
  n_neurons: number;
  n_heads: number;
  n_layers: number;
  n_embd: number;
}
interface Heritage {
  model1_name: string;
  model2_name: string;
  neurons_per_head_original: number;
  neurons_per_head_merged: number;
  total_neurons_per_model: number;
  total_neurons_merged: number;
  ranges: Record<string, { start: number; end: number }>;
}
interface EvalResult {
  french_loss: number | null;
  portuguese_loss: number | null;
}
interface Sample {
  label: string;
  prompt: string;
  generated: string;
  french_generated?: string;
  portuguese_generated?: string;
  merged_generated?: string;
  finetuned_generated?: string;
}
interface HeritageProbeLayerData {
  french: LayerHeritage;
  portuguese: LayerHeritage;
}
interface HeritageProbeInput {
  layers: Record<string, HeritageProbeLayerData>;
  summary: {
    french_percentage: number;
    portuguese_percentage: number;
    dominant_heritage: string;
  };
}
interface HeritageProbeData {
  french_input: HeritageProbeInput;
  portuguese_input: HeritageProbeInput;
  summary: {
    french_input_french_pct: number;
    french_input_portuguese_pct: number;
    portuguese_input_french_pct: number;
    portuguese_input_portuguese_pct: number;
    routing_quality: number;
    clear_separation: boolean;
  };
}
interface FinetuneInfo {
  source_checkpoint: string;
  iters: number;
  lr: number;
  pre_loss: number;
  post_loss: number;
}
interface MergeData {
  heritage: Heritage;
  models: Record<string, ModelInfo>;
  evaluation: Record<string, EvalResult>;
  samples: Sample[];
  heritage_probe?: HeritageProbeData;
  finetune_info?: FinetuneInfo;
}
interface LayerHeritage {
  origin: string;
  active_count: number;
  total_count: number;
  activation_ratio: number;
}

const fmtP = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}K`;
const fmtN = (n: number) => n.toLocaleString();
const lossColor = (v: number | null) =>
  v === null
    ? "text-[#4A5568]"
    : v < 1.2
      ? "text-[#00C896]"
      : v < 2.0
        ? "text-amber-400"
        : "text-red-400";
const lossDisplay = (v: number | null) => (v === null ? "—" : v.toFixed(4));
const API_ORIGIN =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) || "";
const API = `${API_ORIGIN}/api/merge`;

interface EvolutionRecord {
  iteration: number;
  loss: number;
  sparsity: {
    text_preview: string;
    layers: Record<string, { x_sparsity: number; y_sparsity: number }>;
    mean_x_sp: number;
    mean_y_sp: number;
  }[];
}

function TrainingEvolution() {
  const [frData, setFrData] = useState<EvolutionRecord[]>([]);
  const [ptData, setPtData] = useState<EvolutionRecord[]>([]);
  const [tab, setTab] = useState<"loss" | "sparsity">("loss");

  useEffect(() => {
    fetch("/evolution/evolution_french.json")
      .then((r) => r.json())
      .then(setFrData)
      .catch(() => {});
    fetch("/evolution/evolution_portuguese.json")
      .then((r) => r.json())
      .then(setPtData)
      .catch(() => {});
  }, []);

  const ready = frData.length > 0 && ptData.length > 0;
  if (!ready) return null;

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-[#8B95A5]" />
          <h2 className="text-lg font-semibold text-[#E2E8F0]">
            Training Evolution
          </h2>
        </div>
        <div className="flex gap-1">
          {(["loss", "sparsity"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className="px-3 py-1 rounded text-xs font-medium transition-all"
              style={{
                background:
                  tab === t ? "rgba(0,200,150,0.12)" : "rgba(255,255,255,0.03)",
                color: tab === t ? "#00C896" : "#6B7280",
                border: `1px solid ${tab === t ? "rgba(0,200,150,0.2)" : "rgba(255,255,255,0.06)"}`,
              }}
            >
              {t === "loss" ? "Loss Curve" : "Sparsity"}
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-[#4A5568] mb-4">
        How each specialist model trained before being merged.
      </p>
      {tab === "loss" ? (
        <EvolutionChart frData={frData} ptData={ptData} mode="loss" />
      ) : (
        <EvolutionChart frData={frData} ptData={ptData} mode="sparsity" />
      )}
      <div className="flex items-center gap-5 mt-3 text-[10px] text-[#6B7280]">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-0.5 rounded"
            style={{ background: "#00C896" }}
          />
          French
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block w-3 h-0.5 rounded"
            style={{ background: "#38BDF8" }}
          />
          Portuguese
        </span>
      </div>
    </motion.div>
  );
}

function EvolutionChart({
  frData,
  ptData,
  mode,
}: {
  frData: EvolutionRecord[];
  ptData: EvolutionRecord[];
  mode: "loss" | "sparsity";
}) {
  const W = 700,
    H = 200,
    PAD = { t: 10, r: 10, b: 25, l: 50 };
  const chartW = W - PAD.l - PAD.r;
  const chartH = H - PAD.t - PAD.b;

  const getValue = (r: EvolutionRecord): number => {
    if (mode === "loss") return Math.log10(Math.max(r.loss, 0.01));
    // average x_sparsity across layers
    if (!r.sparsity || r.sparsity.length === 0) return 0;
    return r.sparsity[0].mean_x_sp ?? 0;
  };

  const allVals = [...frData, ...ptData].map(getValue);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;

  const maxIter = Math.max(
    frData[frData.length - 1]?.iteration ?? 1,
    ptData[ptData.length - 1]?.iteration ?? 1,
  );

  const toPath = (data: EvolutionRecord[]) => {
    return data
      .map((r, i) => {
        const x = PAD.l + (r.iteration / maxIter) * chartW;
        const y = PAD.t + chartH - ((getValue(r) - minV) / range) * chartH;
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  };

  // Y-axis labels
  const yTicks = 4;
  const yLabels = Array.from({ length: yTicks + 1 }, (_, i) => {
    const v = minV + (range * i) / yTicks;
    if (mode === "loss") return Math.pow(10, v).toFixed(v > 2 ? 0 : 2);
    return (v * 100).toFixed(0) + "%";
  });

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      style={{ maxHeight: 220 }}
    >
      {/* Grid lines */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const y = PAD.t + chartH - (chartH * i) / yTicks;
        return (
          <g key={i}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={y}
              y2={y}
              stroke="rgba(255,255,255,0.04)"
              strokeWidth={0.5}
            />
            <text
              x={PAD.l - 4}
              y={y + 3}
              textAnchor="end"
              fill="#4A5568"
              fontSize={8}
              fontFamily="monospace"
            >
              {yLabels[i]}
            </text>
          </g>
        );
      })}
      {/* X-axis labels */}
      {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
        const x = PAD.l + pct * chartW;
        const iter = Math.round(pct * maxIter);
        return (
          <text
            key={pct}
            x={x}
            y={H - 4}
            textAnchor="middle"
            fill="#4A5568"
            fontSize={8}
            fontFamily="monospace"
          >
            {iter >= 1000 ? `${(iter / 1000).toFixed(0)}k` : iter}
          </text>
        );
      })}
      {/* French line */}
      <path
        d={toPath(frData)}
        fill="none"
        stroke="#00C896"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.8}
      />
      {/* Portuguese line */}
      <path
        d={toPath(ptData)}
        fill="none"
        stroke="#38BDF8"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.8}
      />
    </svg>
  );
}

export function MergePage() {
  const [data, setData] = useState<MergeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [backendAvailable, setBackendAvailable] = useState(false);

  useEffect(() => {
    // Try merge_data.json first (richer format with all models), fall back to merge_eval.json
    fetch("/merge/merge_data.json")
      .then((r) => {
        if (!r.ok)
          return fetch("/merge/merge_eval.json").then((r2) => r2.json());
        return r.json();
      })
      .then((raw) => {
        setData(adaptMergeData(raw) as unknown as MergeData);
        setLoading(false);
      })
      .catch(() => setLoading(false));
    fetch(`${API_ORIGIN}/health`)
      .then(async (r) => {
        if (!r.ok) return;
        try {
          const j = await r.json();
          if (j && j.status === "healthy") setBackendAvailable(true);
        } catch {
          // Response wasn't JSON (e.g. nginx SPA fallback HTML) — not a real backend
        }
      })
      .catch(() => {});
  }, []);

  if (loading)
    return (
      <div
        className="min-h-screen flex items-center justify-center"
        style={{ background: "#070D12" }}
      >
        <Loader2 className="w-8 h-8 animate-spin text-[#8B95A5]" />
      </div>
    );

  const d = data;
  const h = d?.heritage;
  const models = d?.models;
  const ev = d?.evaluation;
  const samples = d?.samples;
  const hasEval = ev && Object.values(ev).some((e) => e.french_loss !== null);
  const hasFT = ev && "finetuned" in ev;
  const probeData = d?.heritage_probe;
  const ftInfo = d?.finetune_info;

  return (
    <div
      className="max-w-7xl mx-auto px-4 py-8 space-y-10"
      style={{ background: "#070D12", minHeight: "100vh" }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <GitMerge className="w-8 h-8 text-[#00C896]" />
          <h1 className="text-3xl font-bold text-[#E2E8F0]">Model Merging</h1>
        </div>
        <p className="text-[#8B95A5] max-w-3xl">
          BDH's sparse, modular architecture enables direct concatenation of
          specialist models.
          {hasFT
            ? " After a brief fine-tuning step, the merged model handles both languages with near-specialist quality."
            : " Each specialist's neurons operate independently in the merged model."}
        </p>
      </motion.div>

      {h && models && (
        <MergeDiagram heritage={h} models={models} hasFT={hasFT} />
      )}
      <TrainingEvolution />
      {models && ev && <ModelCards models={models} evaluation={ev} />}
      {hasEval && <LossComparison evaluation={ev!} hasFT={!!hasFT} />}
      {ftInfo && <FinetuneInfoPanel info={ftInfo} />}
      {samples && samples.length > 0 && <SampleGenerations samples={samples} />}
      {h && <HeritageMap heritage={h} />}
      {probeData && h && (
        <PrecomputedHeritageProbe probe={probeData} heritage={h} />
      )}
      <LiveGeneration backendAvailable={backendAvailable} />
      <InsightPanel hasFT={!!hasFT} />
    </div>
  );
}

function MergeDiagram({
  heritage: h,
  models,
  hasFT,
}: {
  heritage: Heritage;
  models: Record<string, ModelInfo>;
  hasFT?: boolean;
}) {
  const [step, setStep] = useState(0);
  const maxStep = hasFT ? 4 : 3;
  useEffect(() => {
    const t = setInterval(() => setStep((s) => (s + 1) % maxStep), 1500);
    return () => clearInterval(t);
  }, [maxStep]);

  const steps = [
    {
      label: h.model1_name.charAt(0).toUpperCase() + h.model1_name.slice(1),
      sub: "Specialist",
      icon: "FR",
    },
    {
      label: h.model2_name.charAt(0).toUpperCase() + h.model2_name.slice(1),
      sub: "Specialist",
      icon: "PT",
    },
    { label: "Merge", sub: "Concatenate N", icon: "M" },
    ...(hasFT
      ? [{ label: "Fine-tune", sub: "Adapt routing", icon: "FT" }]
      : []),
    { label: "Polyglot", sub: "Both languages", icon: "P" },
  ];

  const fr = models[h.model1_name];
  const pt = models[h.model2_name];
  const mg = models.finetuned || models.merged;

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.1 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-lg font-semibold">MERGE PROCESS</h2>
      </div>
      <div className="flex items-center justify-between mb-8">
        {steps.map((s, i) => (
          <React.Fragment key={i}>
            <div className="flex flex-col items-center">
              <motion.div
                className={`w-14 h-14 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-all duration-300
                ${i <= step ? "border-[#00C896]/50 bg-[#00C896]/15 text-[#00C896]" : "border-white/[0.12] text-[#4A5568]"}`}
                animate={i === step ? { scale: [1, 1.1, 1] } : {}}
                transition={{ repeat: Infinity, duration: 1.5 }}
              >
                {i < step ? <Check className="w-5 h-5" /> : s.icon}
              </motion.div>
              <span
                className={`text-xs mt-1 ${i <= step ? "text-[#00C896]" : "text-[#4A5568]"}`}
              >
                {s.label}
              </span>
              <span className="text-[10px] text-[#4A5568]">{s.sub}</span>
            </div>
            {i < steps.length - 1 && (
              <div
                className={`flex-1 h-0.5 mx-2 ${i < step ? "bg-[#00C896]/50" : "bg-white/10"}`}
              />
            )}
          </React.Fragment>
        ))}
      </div>
      {/* Visual */}
      <div className="grid grid-cols-3 gap-4 items-center">
        <motion.div
          className="border border-white/10 rounded-lg p-4"
          animate={{ opacity: step >= 0 ? 1 : 0.3 }}
        >
          <div className="flex items-center gap-2 text-sm mb-2">
            <span className="text-[#00C896] font-mono">FR</span>
            <span className="text-[#00C896] font-semibold">{fr?.name}</span>
          </div>
          <div className="text-xs text-[#8B95A5]">
            N/head:{" "}
            <span className="text-[#E2E8F0] font-mono">
              {fmtN(fr?.n_neurons || 0)}
            </span>
          </div>
          <div className="flex gap-0.5 mt-2">
            {Array(12)
              .fill(0)
              .map((_, i) => (
                <div key={i} className="w-3 h-5 rounded-sm bg-[#00C896]/40" />
              ))}
          </div>
        </motion.div>
        <motion.div
          className={`border rounded-lg p-4 text-center ${step >= 3 ? "border-white/[0.12] bg-white/[0.03]" : "border-white/10"}`}
          animate={{ opacity: step >= 2 ? 1 : 0.2 }}
          transition={{ duration: 0.5 }}
        >
          {step >= 3 && mg && (
            <>
              <div className="flex items-center justify-center gap-2 mb-1">
                <span className="font-semibold text-[#E2E8F0]">{mg.name}</span>
              </div>
              <div className="text-xs text-[#8B95A5]">
                N/head:{" "}
                <span className="font-mono text-[#E2E8F0]">
                  {fmtN(mg.n_neurons)}
                </span>
              </div>
              <div className="flex gap-0.5 mt-2 justify-center flex-wrap">
                {Array(12)
                  .fill(0)
                  .map((_, i) => (
                    <div
                      key={i}
                      className="w-3 h-5 rounded-sm bg-[#00C896]/40"
                    />
                  ))}
                {Array(12)
                  .fill(0)
                  .map((_, i) => (
                    <div
                      key={i + 12}
                      className="w-3 h-5 rounded-sm bg-sky-500/40"
                    />
                  ))}
              </div>
              <div className="flex gap-3 justify-center mt-1 text-[10px] text-[#4A5568]">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm bg-[#00C896]" /> French
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-sm bg-sky-500" /> Portuguese
                </span>
              </div>
            </>
          )}
        </motion.div>
        <motion.div
          className="border border-white/10 rounded-lg p-4"
          animate={{ opacity: step >= 1 ? 1 : 0.3 }}
        >
          <div className="flex items-center gap-2 text-sm mb-2">
            <span className="text-sky-400 font-mono">PT</span>
            <span className="text-sky-300 font-semibold">{pt?.name}</span>
          </div>
          <div className="text-xs text-[#8B95A5]">
            N/head:{" "}
            <span className="text-[#E2E8F0] font-mono">
              {fmtN(pt?.n_neurons || 0)}
            </span>
          </div>
          <div className="flex gap-0.5 mt-2">
            {Array(12)
              .fill(0)
              .map((_, i) => (
                <div key={i} className="w-3 h-5 rounded-sm bg-sky-500/40" />
              ))}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

function ModelCards({
  models,
  evaluation,
}: {
  models: Record<string, ModelInfo>;
  evaluation: Record<string, EvalResult>;
}) {
  const order = Object.keys(models);
  const colors: Record<string, string> = {
    french: "cyan",
    portuguese: "emerald",
    merged: "amber",
    finetuned: "purple",
  };
  return (
    <div
      className={`grid grid-cols-1 gap-4 ${order.length <= 3 ? "md:grid-cols-3" : "md:grid-cols-2 lg:grid-cols-4"}`}
    >
      {order.map((k, i) => {
        const m = models[k];
        const ev = evaluation[k];
        const _c = colors[k] || "zinc";
        void _c;
        return (
          <motion.div
            key={k}
            className={`glass-card p-5 ${k === "finetuned" ? "" : ""}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="font-semibold text-[#E2E8F0]">{m.name}</span>
            </div>
            <div className="space-y-1.5 text-sm">
              <Row label="Parameters" value={fmtP(m.params)} />
              <Row label="Neurons/Head" value={fmtN(m.n_neurons)} />
              <Row
                label="Layers x Heads"
                value={`${m.n_layers} x ${m.n_heads}`}
              />
              <Row label="Embedding dim" value={String(m.n_embd)} />
              {ev &&
                (ev.french_loss !== null || ev.portuguese_loss !== null) && (
                  <>
                    <div className="border-t border-white/[0.06] my-2" />
                    <Row
                      label="French loss"
                      value={lossDisplay(ev.french_loss)}
                      valueClass={lossColor(ev.french_loss)}
                    />
                    <Row
                      label="Portuguese loss"
                      value={lossDisplay(ev.portuguese_loss)}
                      valueClass={lossColor(ev.portuguese_loss)}
                    />
                  </>
                )}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
function Row({
  label,
  value,
  valueClass,
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex justify-between">
      <span className="text-[#8B95A5]">{label}</span>
      <span className={`font-mono ${valueClass || "text-[#E2E8F0]"}`}>
        {value}
      </span>
    </div>
  );
}

function LossComparison({
  evaluation,
  hasFT,
}: {
  evaluation: Record<string, EvalResult>;
  hasFT: boolean;
}) {
  const order = Object.keys(evaluation);
  const flags: Record<string, string> = {
    french: "FR",
    portuguese: "PT",
    merged: "MG",
    finetuned: "FT",
  };
  const rowColors: Record<string, string> = {
    merged: "",
    finetuned: "",
  };
  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.2 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">LOSS COMPARISON</h2>
        <span className="text-xs text-[#4A5568] ml-2">
          NEXT-BYTE PREDICTION LOSS (LOWER = BETTER)
        </span>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[#4A5568] border-b border-white/[0.06]">
            <th className="text-left pb-2 w-1/3">Model</th>
            <th className="text-right pb-2 text-[#CBD5E0]">French Loss</th>
            <th className="text-right pb-2 text-[#CBD5E0]">Portuguese Loss</th>
          </tr>
        </thead>
        <tbody>
          {order.map((k) => {
            const ev = evaluation[k];
            return (
              <tr
                key={k}
                className={`border-b border-white/[0.06] ${rowColors[k] || ""}`}
              >
                <td className="py-3 flex items-center gap-2">
                  <span className="text-xs">{flags[k] || k}</span>
                  <span className="font-medium">
                    {k === "finetuned"
                      ? "Merged (fine-tuned)"
                      : k === "merged"
                        ? "Merged (zero-shot)"
                        : k.charAt(0).toUpperCase() + k.slice(1)}
                  </span>
                </td>
                <td
                  className={`text-right font-mono ${lossColor(ev.french_loss)}`}
                >
                  {lossDisplay(ev.french_loss)}
                </td>
                <td
                  className={`text-right font-mono ${lossColor(ev.portuguese_loss)}`}
                >
                  {lossDisplay(ev.portuguese_loss)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="mt-4 p-3 bg-white/[0.03] rounded-lg text-sm text-[#8B95A5] flex items-start gap-2">
        {hasFT ? (
          <span>
            The zero-shot merge shows degraded loss because averaging embeddings
            blurs the routing signal. After just ~500 iterations of fine-tuning
            on mixed data, the model{" "}
            <b className="text-[#E2E8F0]">
              recovers near-specialist quality on both languages
            </b>{" "}
            — proving BDH's neuron spaces are truly composable.
          </span>
        ) : (
          <span>
            Each specialist excels at its own language. The merged model handles{" "}
            <b className="text-[#E2E8F0]">both languages</b> after concatenating
            neuron spaces.
          </span>
        )}
      </div>
    </motion.div>
  );
}

function SampleGenerations({ samples }: { samples: Sample[] }) {
  const [selected, setSelected] = useState(0);
  const s = samples[selected];
  const hasFT = !!s?.finetuned_generated;

  const gens = [
    { key: "french_generated", label: "French Specialist", color: "cyan" },
    {
      key: "portuguese_generated",
      label: "Portuguese Specialist",
      color: "emerald",
    },
    { key: "merged_generated", label: "Merged (zero-shot)", color: "amber" },
    ...(hasFT
      ? [
          {
            key: "finetuned_generated",
            label: "Merged (fine-tuned)",
            color: "purple",
          },
        ]
      : []),
  ];

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <FileText className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">SAMPLE GENERATIONS</h2>
        <span className="text-xs text-[#4A5568] ml-2">
          SAME PROMPT THROUGH ALL MODELS
        </span>
      </div>
      <div className="flex gap-2 mb-4 flex-wrap">
        {samples.map((sample, i) => (
          <button
            key={i}
            onClick={() => setSelected(i)}
            className={`px-3 py-1 text-xs rounded-full border transition-all ${
              i === selected
                ? "border-[#00C896]/50 bg-[#00C896]/15 text-[#00C896]"
                : "border-white/10 text-[#4A5568] hover:border-white/20"
            }`}
          >
            {sample.label}
          </button>
        ))}
      </div>
      <div className="space-y-3">
        {gens.map((g) => {
          const text = (s as any)[g.key];
          if (!text) return null;
          return (
            <div
              key={g.key}
              className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06]"
            >
              <span className="text-xs font-medium text-[#CBD5E0] mb-1 block">
                {g.label}
              </span>
              <p className="font-mono text-sm text-[#CBD5E0] break-all">
                <span className="text-[#CBD5E0]">{s.prompt}</span>
                <span className="text-[#8B95A5]">
                  {text.slice(s.prompt.length)}
                </span>
              </p>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}

function HeritageMap({ heritage: h }: { heritage: Heritage }) {
  const segs = 120;
  const split = segs / 2;
  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.4 }}
    >
      <div className="flex items-center gap-2 mb-3">
        <AlertCircle className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">NEURON HERITAGE MAP</h2>
      </div>
      <p className="text-sm text-[#8B95A5] mb-3">
        Each neuron traces back to exactly one specialist.{" "}
        {h.model1_name.charAt(0).toUpperCase() + h.model1_name.slice(1)}{" "}
        neurons: 0–{fmtN(h.ranges[h.model1_name]?.end || 0)},{" "}
        {h.model2_name.charAt(0).toUpperCase() + h.model2_name.slice(1)}{" "}
        neurons: {fmtN(h.ranges[h.model2_name]?.start || 0)}–
        {fmtN(h.ranges[h.model2_name]?.end || 0)}.
      </p>
      <div className="flex gap-[1px] my-3" style={{ height: 24 }}>
        {Array(segs)
          .fill(0)
          .map((_, i) => (
            <div
              key={i}
              className={`flex-1 rounded-sm ${i < split ? "bg-[#00C896]/70" : "bg-sky-500/70"}`}
              style={{ opacity: 0.5 + Math.random() * 0.5 }}
            />
          ))}
      </div>
      <div className="flex justify-between text-[10px] text-[#4A5568] font-mono mb-4">
        <span>0</span>
        <span>← {h.model1_name} →</span>
        <span>{fmtN(h.neurons_per_head_original)}</span>
        <span>← {h.model2_name} →</span>
        <span>{fmtN(h.neurons_per_head_merged)}</span>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {[
          {
            l: `${h.model1_name.toUpperCase()} NEURONS`,
            v: fmtN(h.total_neurons_per_model),
            c: "cyan",
          },
          {
            l: `${h.model2_name.toUpperCase()} NEURONS`,
            v: fmtN(h.total_neurons_per_model),
            c: "emerald",
          },
          { l: "MERGED TOTAL", v: fmtN(h.total_neurons_merged), c: "purple" },
        ].map((x, i) => (
          <div
            key={i}
            className="text-center border border-white/[0.06] rounded-lg py-3"
          >
            <div className="text-xl font-bold text-[#E2E8F0] font-mono">
              {x.v}
            </div>
            <div className="text-[10px] text-[#4A5568]">{x.l}</div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

function LiveGeneration({ backendAvailable }: { backendAvailable: boolean }) {
  const [prompt, setPrompt] = useState(
    "Le commerce international est essentiel pour",
  );
  const [gens, setGens] = useState<Record<string, string>>({});
  const [generating, setGenerating] = useState(false);

  const go = useCallback(async () => {
    if (!prompt.trim()) return;
    setGenerating(true);
    try {
      const r = await fetch(`${API}/side-by-side`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_tokens: 60 }),
      });
      if (r.ok) {
        const d = await r.json();
        setGens(d.generations);
      } else {
        setGens({
          _error:
            "Backend unavailable – live generation requires a running API server.",
        });
      }
    } catch {
      setGens({
        _error:
          "Backend unavailable – live generation requires a running API server.",
      });
    }
    setGenerating(false);
  }, [prompt]);

  if (!backendAvailable) return null;

  const colors: Record<string, string> = {
    french_specialist: "cyan",
    french: "cyan",
    portuguese_specialist: "emerald",
    portuguese: "emerald",
    merged_polyglot: "amber",
    merged: "amber",
    merged_finetuned: "purple",
    finetuned: "purple",
  };
  const labels: Record<string, string> = {
    french_specialist: "French Specialist",
    french: "French Specialist",
    portuguese_specialist: "Portuguese Specialist",
    portuguese: "Portuguese Specialist",
    merged_polyglot: "Merged (zero-shot)",
    merged: "Merged (zero-shot)",
    merged_finetuned: "Merged (fine-tuned)",
    finetuned: "Merged (fine-tuned)",
  };

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.6 }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Zap className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">LIVE GENERATION</h2>
        <span className="text-[10px] bg-[#00C896]/10 text-[#00C896] border border-[#00C896]/20 px-2 py-0.5 rounded-full">
          LIVE
        </span>
      </div>
      <p className="text-sm text-[#8B95A5] mb-4">
        Same prompt through all models. Compare specialist quality with merged
        and fine-tuned output.
      </p>
      <div className="flex gap-2 mb-4">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
          className="flex-1 bg-[#0B1216] border border-white/10 rounded-lg px-3 py-2 text-sm text-[#E2E8F0] focus:outline-none focus:border-[#00C896]/50"
        />
        <button
          onClick={go}
          disabled={generating}
          className="px-4 py-2 bg-white/5 border border-white/\[0.12\] rounded-lg text-[#E2E8F0] text-sm font-medium hover:bg-white/10 disabled:opacity-50 flex items-center gap-2"
        >
          {generating ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Zap className="w-4 h-4" />
          )}{" "}
          Generate
        </button>
      </div>
      {Object.keys(gens).length > 0 && (
        <div className="space-y-2">
          {"_error" in gens ? (
            <div className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06] text-sm text-[#8B95A5]">
              <Terminal className="w-4 h-4 inline mr-2" />
              {gens._error}
            </div>
          ) : (
            Object.entries(gens).map(([name, text]) => {
              const _c2 = colors[name] || "zinc";
              void _c2;
              return (
                <div
                  key={name}
                  className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06]"
                >
                  <span className="text-xs font-medium text-[#CBD5E0] mb-1 block">
                    {labels[name] || name}
                  </span>
                  <p className="font-mono text-sm text-[#CBD5E0] break-all">
                    <span className="text-[#CBD5E0]">{prompt}</span>
                    <span className="text-[#8B95A5]">{text}</span>
                  </p>
                </div>
              );
            })
          )}
        </div>
      )}
    </motion.div>
  );
}

function FinetuneInfoPanel({ info }: { info: FinetuneInfo }) {
  const reduction = ((1 - info.post_loss / info.pre_loss) * 100).toFixed(1);
  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.25 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <TrendingDown className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">FINE-TUNING RESULTS</h2>
        <span className="text-xs text-[#4A5568] ml-2">
          POST-MERGE ADAPTATION
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        {[
          { label: "Iterations", value: String(info.iters) },
          {
            label: "Learning Rate",
            value: info.lr.toExponential(1),
          },
          {
            label: "Pre-FT Loss",
            value: info.pre_loss.toFixed(4),
          },
          {
            label: "Post-FT Loss",
            value: info.post_loss.toFixed(4),
          },
        ].map((item, i) => (
          <div
            key={i}
            className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06] text-center"
          >
            <div className="text-lg font-bold font-mono text-[#E2E8F0]">
              {item.value}
            </div>
            <div className="text-[10px] text-[#4A5568]">{item.label}</div>
          </div>
        ))}
      </div>
      {/* Loss reduction bar */}
      <div className="bg-[#0B1216]/50 rounded-lg p-4 border border-white/[0.06]">
        <div className="flex justify-between text-xs text-[#8B95A5] mb-2">
          <span>
            Pre-finetune:{" "}
            <span className="text-[#8B95A5] font-mono">
              {info.pre_loss.toFixed(4)}
            </span>
          </span>
          <span>
            Post-finetune:{" "}
            <span className="text-[#CBD5E0] font-mono">
              {info.post_loss.toFixed(4)}
            </span>
          </span>
        </div>
        <div className="relative h-6 bg-white/5 rounded-full overflow-hidden">
          <motion.div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-[#00C896]/60 to-[#00C896]/60 rounded-full"
            initial={{ width: "100%" }}
            animate={{ width: `${(info.post_loss / info.pre_loss) * 100}%` }}
            transition={{ duration: 1.5, ease: "easeOut" }}
          />
          <div className="absolute inset-0 flex items-center justify-center text-xs font-bold text-[#E2E8F0]">
            {reduction}% reduction
          </div>
        </div>
        <div className="text-center mt-2 text-xs text-[#4A5568]">
          <Timer className="w-3 h-3 inline mr-1" />
          {info.iters} iterations of mixed-language fine-tuning on CPU
        </div>
      </div>
    </motion.div>
  );
}

function PrecomputedHeritageProbe({
  probe,
  heritage,
}: {
  probe: HeritageProbeData;
  heritage: Heritage;
}) {
  const [selected, setSelected] = useState<"french_input" | "portuguese_input">(
    "french_input",
  );
  const inputData = probe[selected];
  const s = probe.summary;

  const frName =
    heritage.model1_name.charAt(0).toUpperCase() +
    heritage.model1_name.slice(1);
  const ptName =
    heritage.model2_name.charAt(0).toUpperCase() +
    heritage.model2_name.slice(1);

  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.45 }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-5 h-5 text-[#8B95A5]" />
        <h2 className="text-lg font-semibold">HERITAGE PROBE</h2>
        <span className="text-[10px] bg-[#00C896]/10 text-[#00C896] border border-[#00C896]/20 px-2 py-0.5 rounded-full">
          PRECOMPUTED
        </span>
      </div>
      <p className="text-sm text-[#8B95A5] mb-4">
        Neuron activation patterns when feeding French vs Portuguese text
        through the fine-tuned merged model. Shows which neuron bank
        (French-origin vs Portuguese-origin) activates for each input language.
      </p>

      {/* Input selector */}
      <div className="flex gap-2 mb-4">
        {[
          {
            key: "french_input" as const,
            label: `${frName} Input`,
          },
          {
            key: "portuguese_input" as const,
            label: `${ptName} Input`,
          },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setSelected(tab.key)}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
              selected === tab.key
                ? "border-[#00C896]/50 bg-[#00C896]/15 text-[#00C896]"
                : "border-white/10 text-[#4A5568] hover:border-white/20"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Summary bar */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-[#8B95A5] mb-1">
          <span className="text-[#CBD5E0]">
            {frName} neurons: {inputData.summary.french_percentage.toFixed(1)}%
          </span>
          <span className="text-[#E2E8F0]">
            {ptName} neurons:{" "}
            {inputData.summary.portuguese_percentage.toFixed(1)}%
          </span>
        </div>
        <div className="flex h-8 rounded-lg overflow-hidden">
          <motion.div
            className="bg-[#00C896]/70 flex items-center justify-center text-xs font-bold text-[#E2E8F0]"
            key={`fr-${selected}`}
            initial={{ width: "50%" }}
            animate={{ width: `${inputData.summary.french_percentage}%` }}
            transition={{ duration: 0.5 }}
          >
            {inputData.summary.french_percentage > 15 &&
              `${inputData.summary.french_percentage.toFixed(0)}%`}
          </motion.div>
          <motion.div
            className="bg-sky-500/70 flex items-center justify-center text-xs font-bold text-[#E2E8F0]"
            key={`pt-${selected}`}
            initial={{ width: "50%" }}
            animate={{ width: `${inputData.summary.portuguese_percentage}%` }}
            transition={{ duration: 0.5 }}
          >
            {inputData.summary.portuguese_percentage > 15 &&
              `${inputData.summary.portuguese_percentage.toFixed(0)}%`}
          </motion.div>
        </div>
        <div className="text-center mt-1 text-sm">
          Dominant:{" "}
          <span
            className={`font-semibold ${inputData.summary.dominant_heritage === heritage.model1_name ? "text-[#00C896]" : "text-[#E2E8F0]"}`}
          >
            {inputData.summary.dominant_heritage.charAt(0).toUpperCase() +
              inputData.summary.dominant_heritage.slice(1)}
          </span>
        </div>
      </div>

      {/* Per-layer breakdown */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2 mb-4">
        {Object.entries(inputData.layers).map(([idx, ld]) => {
          const fr = ld.french,
            pt = ld.portuguese;
          const t = fr.activation_ratio + pt.activation_ratio;
          const w = t > 0 ? (fr.activation_ratio / t) * 100 : 50;
          return (
            <div
              key={idx}
              className="bg-[#0B1216]/50 rounded-lg p-2 border border-white/[0.06]"
            >
              <div className="text-[10px] text-[#4A5568] mb-1 text-center">
                Layer {idx}
              </div>
              <div className="flex h-3 rounded overflow-hidden mb-1">
                <div className="bg-[#00C896]/70" style={{ width: `${w}%` }} />
                <div
                  className="bg-sky-500/70"
                  style={{ width: `${100 - w}%` }}
                />
              </div>
              <div className="flex justify-between text-[9px] text-[#4A5568]">
                <span>{fr.active_count}</span>
                <span>{pt.active_count}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Routing quality summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06] text-center">
          <div className="text-xs text-[#4A5568] mb-1">
            {frName} Input → {frName} Neurons
          </div>
          <div
            className={`text-lg font-bold font-mono ${s.french_input_french_pct > 55 ? "text-[#00C896]" : "text-[#8B95A5]"}`}
          >
            {s.french_input_french_pct.toFixed(1)}%
          </div>
        </div>
        <div className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06] text-center">
          <div className="text-xs text-[#4A5568] mb-1">
            {ptName} Input → {ptName} Neurons
          </div>
          <div
            className={`text-lg font-bold font-mono ${s.portuguese_input_portuguese_pct > 55 ? "text-sky-400" : "text-[#8B95A5]"}`}
          >
            {s.portuguese_input_portuguese_pct.toFixed(1)}%
          </div>
        </div>
        <div className="bg-[#0B1216]/50 rounded-lg p-3 border border-white/[0.06] text-center">
          <div className="text-xs text-[#4A5568] mb-1">Routing Quality</div>
          <div
            className={`text-lg font-bold font-mono ${s.routing_quality > 60 ? "text-[#00C896]" : s.routing_quality > 50 ? "text-[#CBD5E0]" : "text-[#8B95A5]"}`}
          >
            {s.routing_quality.toFixed(1)}%
          </div>
          {s.clear_separation ? (
            <div className="text-[10px] text-[#00C896] mt-1">
              ✓ Clear language separation
            </div>
          ) : (
            <div className="text-[10px] text-[#8B95A5] mt-1">
              Both banks co-activate — shared representations emerged
            </div>
          )}
        </div>
      </div>

      {!s.clear_separation && (
        <div className="mt-3 p-3 bg-white/[0.03] rounded-lg border border-white/[0.08] text-sm text-[#8B95A5]">
          <AlertCircle className="w-4 h-4 inline mr-1 text-[#8B95A5]" />
          <span>
            The fine-tuned model uses both neuron banks for both languages
            rather than strict routing. This is expected when fine-tuning on a
            small built-in dataset — the model learned shared representations
            across both banks. With larger training data, more distinct routing
            patterns may emerge.
          </span>
        </div>
      )}
    </motion.div>
  );
}

function InsightPanel({ hasFT }: { hasFT: boolean }) {
  return (
    <motion.div
      className="glass-card p-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.7 }}
    >
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-lg font-semibold">Why This Matters</h2>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-[#E2E8F0] font-semibold mb-2">
            Transformers Can't Do This
          </h3>
          <p className="text-sm text-[#8B95A5]">
            Transformer weights are densely interconnected. Concatenating two
            transformer models produces garbage. Any "merging" requires
            expensive fine-tuning, distillation, or careful weight interpolation
            — and even then, the merged model shares all its capacity between
            tasks.
          </p>
        </div>
        <div>
          <h3 className="text-[#00C896] font-semibold mb-2">
            BDH Does It Naturally
          </h3>
          <p className="text-sm text-[#8B95A5]">
            BDH's sparse, modular architecture means neurons operate
            independently. Concatenating two models stacks their neuron spaces
            perfectly.
            {hasFT
              ? " A brief fine-tuning step (~500 iterations) teaches the shared embeddings to route correctly to both neuron banks, restoring near-specialist quality."
              : " The shared embedding layer can be adapted with minimal fine-tuning to route to both banks."}
          </p>
        </div>
      </div>
      {hasFT && (
        <div className="mt-4 p-3 bg-white/[0.03] rounded-lg border border-white/[0.08] text-sm">
          <span className="text-[#E2E8F0] font-medium">The workflow:</span>
          <span className="text-[#CBD5E0]">
            {" "}
            Train specialists independently → concatenate neurons → fine-tune
            routing (~5 min) → polyglot model. This enables{" "}
            <b>modular AI development</b> impossible with transformers.
          </span>
        </div>
      )}
      {!hasFT && (
        <div className="mt-4 p-3 bg-white/[0.03] rounded-lg border border-white/[0.08] text-sm">
          <span className="text-[#E2E8F0] font-medium">Implication:</span>
          <span className="text-[#CBD5E0]">
            {" "}
            Train specialists for specific tasks, merge them freely. This
            enables <b>modular AI development</b> — a paradigm impossible with
            current transformer architectures.
          </span>
        </div>
      )}
    </motion.div>
  );
}
