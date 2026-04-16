import { useState, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Zap,
  Play,
  Pause,
  RotateCcw,
  Loader2,
  Layers,
  Send,
  Activity,
  BarChart3,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Brain,
  Network,
  Sparkles,
} from "lucide-react";
import { visualization } from "../utils/api";

interface Prediction {
  byte: number;
  char: string;
  prob: number;
}

interface SynapseData {
  sigma: number;
  delta: number;
}

interface WordEntry {
  word: string;
  byte_range: [number, number];
  synapses: Record<string, SynapseData>;
  gate_activity: number;
}

interface TrackedSynapse {
  id: string;
  neuron: number;
  final_sigma: number;
}

interface HeadData {
  tracked_synapses: TrackedSynapse[];
  words: WordEntry[];
}

interface HeadSummary {
  head: number;
  total_gate_activity: number;
  top_synapse: TrackedSynapse | null;
}

interface LayerSummary {
  layer: number;
  heads: HeadSummary[];
}

interface HebbianResponse {
  input_text: string;
  num_bytes: number;
  num_words: number;
  words: string[];
  model_config: {
    n_layer: number;
    n_head: number;
    n_neurons: number;
  };
  predictions: {
    before: Prediction[];
    after: Prediction[];
    prefix_text: string;
  };
  layer_summary: LayerSummary[];
  layer_data: Record<number, Record<number, HeadData>>;
  sparsity: Record<string, number>;
}

const HEAD_COLORS = ["#8b5cf6", "#f59e0b", "#06b6d4", "#ef4444"];
const LAYER_GRADIENT = [
  "#4f46e5",
  "#6366f1",
  "#7c3aed",
  "#8b5cf6",
  "#a855f7",
  "#c084fc",
  "#d8b4fe",
  "#e9d5ff",
];

const EXAMPLE_SENTENCES = [
  "le parlement européen a voté cette résolution importante",
  "la france et l'allemagne coopèrent dans de nombreux domaines",
  "le dollar et l'euro sont les principales monnaies mondiales",
  "il parle couramment français et anglais depuis toujours",
];

export function HebbianPage() {
  const [inputText, setInputText] = useState(EXAMPLE_SENTENCES[0]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentWord, setCurrentWord] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HebbianResponse | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number>(0);
  const [selectedHead, setSelectedHead] = useState<number>(0);
  const [showDelta, setShowDelta] = useState(true);
  const [expandedLayer, setExpandedLayer] = useState<number | null>(null);
  const [playbackSpeed, setPlaybackSpeed] = useState(1800); // ms per word

  const headData = useMemo((): HeadData | null => {
    if (!data) return null;
    const layerData = data.layer_data[selectedLayer];
    if (!layerData) return null;
    return layerData[selectedHead] ?? null;
  }, [data, selectedLayer, selectedHead]);

  const availableLayers = useMemo(() => {
    if (!data) return [];
    return Object.keys(data.layer_data)
      .map(Number)
      .sort((a, b) => a - b);
  }, [data]);

  const maxGate = useMemo(() => {
    if (!headData) return 1;
    return Math.max(...headData.words.map((w) => Math.abs(w.gate_activity)), 1);
  }, [headData]);

  const maxSigma = useMemo(() => {
    if (!headData) return 1;
    let max = 0;
    for (const w of headData.words) {
      for (const syn of Object.values(w.synapses)) {
        const v = showDelta ? Math.abs(syn.delta) : Math.abs(syn.sigma);
        if (v > max) max = v;
      }
    }
    return max || 1;
  }, [headData, showDelta]);

  const runAnalysis = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setCurrentWord(0);
    setIsPlaying(false);
    try {
      const res = await visualization.hebbianTrack(inputText.trim());
      setData(res.data);
      // Default to layer with highest gate activity
      if (res.data.layer_summary.length > 0) {
        let bestLayer = 0;
        let bestGate = 0;
        for (const ls of res.data.layer_summary) {
          const total = ls.heads.reduce(
            (s: number, h: HeadSummary) => s + h.total_gate_activity,
            0,
          );
          if (total > bestGate) {
            bestGate = total;
            bestLayer = ls.layer;
          }
        }
        setSelectedLayer(bestLayer);
        setSelectedHead(0);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to reach backend");
    } finally {
      setLoading(false);
    }
  }, [inputText]);

  useEffect(() => {
    if (!isPlaying || !headData) return;
    const interval = setInterval(() => {
      setCurrentWord((prev) => {
        if (prev >= headData.words.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, playbackSpeed);
    return () => clearInterval(interval);
  }, [isPlaying, headData, playbackSpeed]);

  const words = headData?.words ?? [];
  const trackedSynapses = headData?.tracked_synapses ?? [];

  return (
    <div className="min-h-screen p-8" style={{ background: '#070D12' }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Hebbian Learning</span> Dynamics
        </h1>
        <p className="text-[#8B95A5]">
          Visualize how σ(i,j) = y_sparse · x_sparse accumulates word-by-word
          during a single forward pass — the gate mechanism IS Hebbian
          co-activation.
        </p>
      </motion.div>

      {/* Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-6"
      >
        <label className="text-sm text-[#8B95A5] mb-2 block">
          Input sequence (French — Europarl FR trained model)
        </label>
        <div className="flex gap-3 mb-3">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runAnalysis()}
            className="input-field flex-1"
            placeholder="Type a French sentence..."
          />
          <button
            onClick={runAnalysis}
            disabled={loading || !inputText.trim()}
            className="btn-primary flex items-center gap-2 px-5"
          >
            {loading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Send size={18} />
            )}
            {loading ? "Running…" : "Analyze"}
          </button>
        </div>

        {/* Quick examples */}
        <div className="flex flex-wrap gap-2 mb-3">
          <span className="text-xs text-[#4A5568]">Try:</span>
          {EXAMPLE_SENTENCES.map((ex, i) => (
            <button
              key={i}
              onClick={() => setInputText(ex)}
              className="text-xs px-2 py-1 rounded bg-white/5 text-[#8B95A5] hover:bg-white/10 hover:text-[#E2E8F0] transition-all truncate max-w-[220px]"
            >
              {ex}
            </button>
          ))}
        </div>

        {error && (
          <div className="text-red-400 text-sm mb-3 p-2 bg-red-500/10 rounded">
            {error}
          </div>
        )}

        {data && (
          <>
            {/* Summary stats */}
            <div className="flex gap-6 mb-4 text-xs text-[#4A5568]">
              <span>
                <span className="text-white font-mono">{data.num_bytes}</span>{" "}
                bytes
              </span>
              <span>
                <span className="text-white font-mono">{data.num_words}</span>{" "}
                words
              </span>
              <span>
                <span className="text-white font-mono">
                  {data.model_config.n_layer}
                </span>{" "}
                layers
              </span>
              <span>
                <span className="text-white font-mono">
                  {data.model_config.n_head}
                </span>{" "}
                heads
              </span>
              <span>
                <span className="text-white font-mono">
                  {data.model_config.n_neurons}
                </span>{" "}
                neurons/head
              </span>
            </div>

            {/* Layer + Head selectors */}
            <div className="flex items-center gap-4 mb-4 flex-wrap">
              <div className="flex items-center gap-2">
                <Layers size={14} className="text-[#4A5568]" />
                <span className="text-xs text-[#4A5568]">Layer:</span>
                {availableLayers.map((l) => (
                  <button
                    key={l}
                    onClick={() => setSelectedLayer(l)}
                    className={`px-2.5 py-1 rounded-full text-xs font-mono transition-all ${
                      selectedLayer === l
                        ? "bg-bdh-accent text-[#E2E8F0]"
                        : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                    }`}
                  >
                    L{l}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs text-[#4A5568]">Head:</span>
                {Array.from(
                  { length: data.model_config.n_head },
                  (_, i) => i,
                ).map((h) => (
                  <button
                    key={h}
                    onClick={() => setSelectedHead(h)}
                    className={`px-2.5 py-1 rounded-full text-xs font-mono transition-all ${
                      selectedHead === h
                        ? "text-[#E2E8F0]"
                        : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                    }`}
                    style={
                      selectedHead === h
                        ? {
                            backgroundColor:
                              HEAD_COLORS[h % HEAD_COLORS.length],
                          }
                        : {}
                    }
                  >
                    H{h}
                  </button>
                ))}
              </div>

              {/* Sigma/Delta toggle */}
              <div className="flex items-center gap-2 ml-auto">
                <span className="text-xs text-[#4A5568]">View:</span>
                <button
                  onClick={() => setShowDelta(false)}
                  className={`px-2.5 py-1 rounded-full text-xs transition-all ${
                    !showDelta
                      ? "bg-bdh-accent text-[#E2E8F0]"
                      : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                  }`}
                >
                  Cumulative σ
                </button>
                <button
                  onClick={() => setShowDelta(true)}
                  className={`px-2.5 py-1 rounded-full text-xs transition-all ${
                    showDelta
                      ? "bg-bdh-accent text-[#E2E8F0]"
                      : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                  }`}
                >
                  Δσ per word
                </button>
              </div>
            </div>

            {/* Word timeline */}
            <div className="flex flex-wrap gap-1 mb-4">
              {words.map((w, idx) => {
                const isActive = idx === currentWord;
                const isPast = idx < currentWord;
                const gate = w.gate_activity;
                const intensity = Math.min(gate / maxGate, 1);
                return (
                  <motion.span
                    key={idx}
                    className={`px-2 py-1 rounded font-mono text-sm cursor-pointer transition-all ${
                      isActive
                        ? "bg-bdh-accent text-[#E2E8F0] scale-110 ring-2 ring-bdh-accent/50"
                        : isPast
                          ? "text-[#2A7FFF] border border-[#2A7FFF]/30"
                          : "bg-white/5 text-[#4A5568]"
                    }`}
                    style={
                      !isActive && isPast
                        ? {
                            backgroundColor: `rgba(139, 92, 246, ${0.1 + intensity * 0.4})`,
                          }
                        : {}
                    }
                    onClick={() => setCurrentWord(idx)}
                    animate={isActive ? { scale: [1, 1.05, 1] } : {}}
                    transition={{ duration: 0.3 }}
                    title={`Gate activity: ${gate.toFixed(2)}`}
                  >
                    {w.word}
                  </motion.span>
                );
              })}
            </div>

            {/* Playback controls */}
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="btn-primary flex items-center gap-2"
                disabled={!words.length}
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setCurrentWord(0);
                  setIsPlaying(false);
                }}
                className="btn-secondary flex items-center gap-2"
              >
                <RotateCcw size={18} />
                Reset
              </button>
              <input
                type="range"
                min={0}
                max={Math.max(0, words.length - 1)}
                value={currentWord}
                onChange={(e) => setCurrentWord(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-[#8B95A5] font-mono text-sm">
                {words.length > 0 ? currentWord + 1 : 0}/{words.length}
              </span>
            </div>

            {/* Speed control */}
            <div className="flex items-center gap-3 mt-3">
              <span className="text-xs text-[#4A5568]">Speed:</span>
              <button
                onClick={() => setPlaybackSpeed(2800)}
                className={`px-2 py-0.5 rounded text-xs transition-all ${
                  playbackSpeed === 2800
                    ? "bg-bdh-accent text-[#E2E8F0]"
                    : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                }`}
              >
                Slow
              </button>
              <button
                onClick={() => setPlaybackSpeed(1800)}
                className={`px-2 py-0.5 rounded text-xs transition-all ${
                  playbackSpeed === 1800
                    ? "bg-bdh-accent text-[#E2E8F0]"
                    : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                }`}
              >
                Normal
              </button>
              <button
                onClick={() => setPlaybackSpeed(1000)}
                className={`px-2 py-0.5 rounded text-xs transition-all ${
                  playbackSpeed === 1000
                    ? "bg-bdh-accent text-[#E2E8F0]"
                    : "bg-white/5 text-[#8B95A5] hover:bg-white/10"
                }`}
              >
                Fast
              </button>
              <span className="text-[10px] text-[#4A5568] font-mono">
                {(playbackSpeed / 1000).toFixed(1)}s/word
              </span>
            </div>
          </>
        )}
      </motion.div>

      {/* Main panels */}
      {data && headData && (
        <div className="grid lg:grid-cols-2 gap-6 mb-6">
          {/* LEFT: σ Timeline */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
              <Activity size={20} className="text-bdh-accent" />
              {showDelta ? "Δσ" : "σ"} Word Timeline
              <span className="text-xs text-[#4A5568] font-normal ml-1">
                L{selectedLayer} H{selectedHead}
              </span>
            </h3>
            <p className="text-xs text-[#4A5568] mb-4">
              {showDelta
                ? "Per-word Hebbian increment: how much σ grows at each word"
                : "Cumulative Hebbian σ = Σ y_sparse · x_sparse up to each word"}
            </p>

            {/* Tracked synapse legend */}
            <div className="flex flex-wrap gap-2 mb-3">
              {trackedSynapses.map((syn, i) => (
                <span
                  key={syn.id}
                  className="text-[10px] font-mono px-2 py-1 rounded"
                  style={{
                    backgroundColor: `${HEAD_COLORS[i % HEAD_COLORS.length]}20`,
                    color: HEAD_COLORS[i % HEAD_COLORS.length],
                    borderLeft: `3px solid ${HEAD_COLORS[i % HEAD_COLORS.length]}`,
                  }}
                >
                  {syn.id} (final σ={syn.final_sigma.toFixed(1)})
                </span>
              ))}
            </div>

            {/* Word bars */}
            <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1">
              <AnimatePresence mode="popLayout">
                {words.map((w, wi) => {
                  const isActive = wi === currentWord;
                  const isPast = wi <= currentWord;
                  return (
                    <motion.div
                      key={wi}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{
                        opacity: isPast ? 1 : 0.3,
                        x: 0,
                      }}
                      className={`p-2 rounded-lg cursor-pointer transition-all ${
                        isActive
                          ? "bg-bdh-accent/15 ring-1 ring-bdh-accent/40"
                          : "bg-white/[0.04] hover:bg-white/[0.05]"
                      }`}
                      onClick={() => setCurrentWord(wi)}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className={`font-mono text-sm w-24 truncate ${
                            isActive ? "text-white font-bold" : "text-[#8B95A5]"
                          }`}
                        >
                          {w.word}
                        </span>
                        <span className="text-[10px] text-[#4A5568] font-mono">
                          gate={w.gate_activity.toFixed(1)}
                        </span>
                      </div>

                      {/* Per-synapse bars */}
                      <div className="space-y-0.5">
                        {trackedSynapses.map((syn, si) => {
                          const sd = w.synapses[syn.id];
                          if (!sd) return null;
                          const val = showDelta ? sd.delta : sd.sigma;
                          const pct =
                            maxSigma > 0 ? (Math.abs(val) / maxSigma) * 100 : 0;
                          const color = HEAD_COLORS[si % HEAD_COLORS.length];
                          return (
                            <div
                              key={syn.id}
                              className="flex items-center gap-1"
                            >
                              <span
                                className="text-[9px] font-mono w-8 text-right"
                                style={{ color }}
                              >
                                {syn.id.slice(0, 5)}
                              </span>
                              <div className="flex-1 h-2 bg-white/[0.08] rounded-full overflow-hidden">
                                <motion.div
                                  className="h-full rounded-full"
                                  style={{ backgroundColor: color }}
                                  initial={{ width: 0 }}
                                  animate={{
                                    width: isPast
                                      ? `${Math.min(pct, 100)}%`
                                      : "0%",
                                  }}
                                  transition={{ duration: 0.3 }}
                                />
                              </div>
                              <span
                                className="text-[9px] font-mono w-14 text-right"
                                style={{ color: isPast ? color : "#555" }}
                              >
                                {isPast ? val.toFixed(2) : "—"}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          </motion.div>

          {/* RIGHT: Prediction Shift + Layer Summary */}
          <div className="space-y-6">
            {/* Before/After Predictions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass-card p-6"
            >
              <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
                <BarChart3 size={20} className="text-[#00C896]" />
                Context-Driven Prediction Shift
              </h3>
              <p className="text-xs text-[#4A5568] mb-4">
                How the Hebbian gate reshapes predictions: top next-byte
                probabilities before vs after seeing the full sentence
              </p>

              <div className="grid grid-cols-2 gap-4">
                {/* Before */}
                <div>
                  <div className="text-xs text-[#4A5568] mb-2 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#4A5568]" />
                    Before (&ldquo;{data.predictions.prefix_text}&rdquo;)
                  </div>
                  <div className="space-y-1">
                    {data.predictions.before.slice(0, 8).map((p, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="font-mono text-[#8B95A5] w-8 text-center bg-white/5 rounded px-1">
                          {p.char}
                        </span>
                        <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#4A5568] rounded-full"
                            style={{ width: `${p.prob * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-[#4A5568] w-12 text-right">
                          {(p.prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* After */}
                <div>
                  <div className="text-xs text-[#4A5568] mb-2 flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-[#00C896]" />
                    After (full sentence)
                  </div>
                  <div className="space-y-1">
                    {data.predictions.after.slice(0, 8).map((p, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <span className="font-mono text-[#00C896] w-8 text-center bg-[#00C896]/10 rounded px-1">
                          {p.char}
                        </span>
                        <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-[#00C896] rounded-full"
                            style={{ width: `${p.prob * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-[#00C896] w-12 text-right">
                          {(p.prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-3 p-2 bg-white/\[0.03\] rounded text-xs text-[#4A5568]">
                <ArrowRight
                  size={12}
                  className="inline text-[#00C896] mr-1"
                />
                The gate = x_sparse · y_sparse shapes which neurons contribute
                to the next prediction. More context → more specific gates →
                sharper distribution.
              </div>
            </motion.div>

            {/* Layer-by-Layer σ Summary */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card p-6"
            >
              <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
                <Layers size={20} className="text-[#2A7FFF]" />
                Layer-by-Layer Gate Activity
              </h3>
              <p className="text-xs text-[#4A5568] mb-4">
                Total Hebbian gate magnitude per layer — where does the model
                concentrate its gating?
              </p>

              <div className="space-y-2">
                {data.layer_summary.map((ls) => {
                  const totalGate = ls.heads.reduce(
                    (s: number, h: HeadSummary) => s + h.total_gate_activity,
                    0,
                  );
                  const maxLayerGate = Math.max(
                    ...data.layer_summary.map((l) =>
                      l.heads.reduce(
                        (s: number, h: HeadSummary) =>
                          s + h.total_gate_activity,
                        0,
                      ),
                    ),
                  );
                  const pct =
                    maxLayerGate > 0 ? (totalGate / maxLayerGate) * 100 : 0;
                  const isExpanded = expandedLayer === ls.layer;
                  const isSelected = selectedLayer === ls.layer;

                  return (
                    <div key={ls.layer}>
                      <div
                        className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-all ${
                          isSelected
                            ? "bg-bdh-accent/10 ring-1 ring-bdh-accent/30"
                            : "hover:bg-white/[0.05]"
                        }`}
                        onClick={() => {
                          setSelectedLayer(ls.layer);
                          setExpandedLayer(isExpanded ? null : ls.layer);
                        }}
                      >
                        {isExpanded ? (
                          <ChevronDown size={12} className="text-[#4A5568]" />
                        ) : (
                          <ChevronRight size={12} className="text-[#4A5568]" />
                        )}
                        <span
                          className={`font-mono text-xs w-6 ${isSelected ? "text-[#E2E8F0]" : "text-[#8B95A5]"}`}
                        >
                          L{ls.layer}
                        </span>
                        <div className="flex-1 h-3 bg-white/5 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full rounded-full"
                            style={{
                              backgroundColor:
                                LAYER_GRADIENT[
                                  ls.layer % LAYER_GRADIENT.length
                                ],
                            }}
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{ duration: 0.5 }}
                          />
                        </div>
                        <span className="font-mono text-xs text-[#8B95A5] w-16 text-right">
                          {totalGate.toFixed(0)}
                        </span>
                      </div>

                      {/* Expanded: per-head breakdown */}
                      <AnimatePresence>
                        {isExpanded && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden ml-6 mt-1 space-y-1"
                          >
                            {ls.heads.map((h) => (
                              <div
                                key={h.head}
                                className={`flex items-center gap-2 p-1.5 rounded cursor-pointer ${
                                  selectedHead === h.head && isSelected
                                    ? "bg-white/[0.08]"
                                    : "hover:bg-white/[0.04]"
                                }`}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setSelectedLayer(ls.layer);
                                  setSelectedHead(h.head);
                                }}
                              >
                                <span
                                  className="w-4 h-4 rounded-full text-[8px] flex items-center justify-center"
                                  style={{
                                    backgroundColor: `${HEAD_COLORS[h.head % HEAD_COLORS.length]}30`,
                                    color:
                                      HEAD_COLORS[h.head % HEAD_COLORS.length],
                                  }}
                                >
                                  {h.head}
                                </span>
                                <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${totalGate > 0 ? (h.total_gate_activity / totalGate) * 100 : 0}%`,
                                      backgroundColor:
                                        HEAD_COLORS[
                                          h.head % HEAD_COLORS.length
                                        ],
                                    }}
                                  />
                                </div>
                                <span className="text-[10px] font-mono text-[#4A5568] w-14 text-right">
                                  {h.total_gate_activity.toFixed(0)}
                                </span>
                                {h.top_synapse && (
                                  <span className="text-[9px] text-[#4A5568] font-mono">
                                    top: {h.top_synapse.id}
                                  </span>
                                )}
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          </div>
        </div>
      )}

      {/* Gate Activity Heatmap */}
      {data && headData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6 mb-6"
        >
          <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
            <Zap size={20} className="text-amber-400" />
            Gate Activity by Word
          </h3>
          <p className="text-xs text-[#4A5568] mb-4">
            Total Σ(x_sparse · y_sparse) at each word position — brighter =
            stronger Hebbian co-activation
          </p>

          <div className="flex flex-wrap gap-1">
            {words.map((w, i) => {
              const intensity = Math.min(w.gate_activity / maxGate, 1);
              const isActive = i === currentWord;
              return (
                <motion.div
                  key={i}
                  className={`px-3 py-2 rounded-lg cursor-pointer text-sm font-mono transition-all ${
                    isActive ? "ring-2 ring-amber-400/60" : ""
                  }`}
                  style={{
                    backgroundColor: `rgba(245, 158, 11, ${0.05 + intensity * 0.5})`,
                    color:
                      intensity > 0.5
                        ? "#fff"
                        : intensity > 0.2
                          ? "#fbbf24"
                          : "#6b7280",
                  }}
                  onClick={() => setCurrentWord(i)}
                  whileHover={{ scale: 1.05 }}
                  title={`gate=${w.gate_activity.toFixed(2)}`}
                >
                  {w.word}
                  <div className="text-[9px] text-center mt-0.5 opacity-60">
                    {w.gate_activity.toFixed(0)}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Animated Hero (shown before analysis) */}
      {!data && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="space-y-6"
        >
          {/* Animated pipeline diagram */}
          <div className="glass-card p-8 relative overflow-hidden">
            <h3 className="text-xl font-bold mb-2 text-center relative z-10">
              <Sparkles size={18} className="inline text-amber-400 mr-2" />
              The BDH Hebbian Pipeline
            </h3>
            <p className="text-[#4A5568] text-sm text-center mb-8 relative z-10">
              How a single byte flows through the gated Hebbian architecture
            </p>

            {/* 4-stage animated pipeline */}
            <div className="relative z-10 flex items-center justify-center gap-2 md:gap-4 flex-wrap">
              {[
                {
                  label: "Encode",
                  formula: "x_sparse = ReLU(x·E)",
                  color: "#3b82f6",
                  icon: Brain,
                  desc: "Project to 8192-dim sparse space",
                },
                {
                  label: "Attend",
                  formula: "a* = Attn(x_sparse)·x",
                  color: "#22c55e",
                  icon: Network,
                  desc: "Linear attention co-activation",
                },
                {
                  label: "Gate",
                  formula: "gate = x·y (Hebbian)",
                  color: "#8b5cf6",
                  icon: Zap,
                  desc: "Fire together, gate together",
                },
                {
                  label: "Decode",
                  formula: "x_l = x + D·gate",
                  color: "#f59e0b",
                  icon: Layers,
                  desc: "Residual back to embed space",
                },
              ].map((stage, i) => (
                <div
                  key={stage.label}
                  className="flex items-center gap-2 md:gap-4"
                >
                  <motion.div
                    className="relative flex flex-col items-center"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + i * 0.15, duration: 0.5 }}
                  >
                    {/* Pulsing ring */}
                    <motion.div
                      className="absolute inset-0 rounded-2xl"
                      style={{ border: `2px solid ${stage.color}` }}
                      animate={{
                        scale: [1, 1.08, 1],
                        opacity: [0.3, 0.7, 0.3],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: i * 0.5,
                      }}
                    />
                    <div
                      className="w-36 md:w-44 p-4 rounded-2xl border"
                      style={{
                        borderColor: `${stage.color}40`,
                        background: `${stage.color}08`,
                      }}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <stage.icon size={16} style={{ color: stage.color }} />
                        <span
                          className="text-sm font-semibold"
                          style={{ color: stage.color }}
                        >
                          {i + 1}. {stage.label}
                        </span>
                      </div>
                      <div className="font-mono text-[11px] text-[#CBD5E0] mb-1">
                        {stage.formula}
                      </div>
                      <p className="text-[10px] text-[#4A5568]">{stage.desc}</p>
                    </div>
                  </motion.div>

                  {/* Animated arrow between stages */}
                  {i < 3 && (
                    <motion.div
                      className="hidden md:flex items-center"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 + i * 0.15 }}
                    >
                      <motion.div
                        animate={{ x: [0, 6, 0] }}
                        transition={{
                          duration: 1.5,
                          repeat: Infinity,
                          delay: i * 0.4,
                        }}
                      >
                        <ArrowRight size={18} className="text-[#4A5568]" />
                      </motion.div>
                    </motion.div>
                  )}
                </div>
              ))}
            </div>

            {/* Animated neuron firing visualization */}
            <div className="mt-10 relative z-10">
              <div className="flex items-center justify-center gap-1 mb-3">
                <Zap size={14} className="text-[#2A7FFF]" />
                <span className="text-xs text-[#4A5568]">
                  Simulated neuron co-activation
                </span>
              </div>
              <div className="flex justify-center gap-[2px] flex-wrap max-w-2xl mx-auto">
                {Array.from({ length: 80 }, (_, i) => {
                  const phase = (i * 0.3) % (2 * Math.PI);
                  return (
                    <motion.div
                      key={i}
                      className="w-3 h-8 rounded-sm"
                      animate={{
                        backgroundColor: [
                          "rgba(139, 92, 246, 0.05)",
                          i % 7 === 0 || i % 11 === 0
                            ? "rgba(139, 92, 246, 0.7)"
                            : "rgba(139, 92, 246, 0.15)",
                          "rgba(139, 92, 246, 0.05)",
                        ],
                        scaleY: [
                          0.3,
                          i % 7 === 0 || i % 11 === 0 ? 1 : 0.5,
                          0.3,
                        ],
                      }}
                      transition={{
                        duration: 2.5,
                        repeat: Infinity,
                        delay: phase * 0.3,
                        ease: "easeInOut",
                      }}
                    />
                  );
                })}
              </div>
              <p className="text-center text-[10px] text-[#4A5568] mt-2">
                ~5% of 8192 neurons fire per token — the gate amplifies
                correlated pairs
              </p>
            </div>
          </div>

          {/* Key insight cards */}
          <div className="grid md:grid-cols-3 gap-4">
            {[
              {
                icon: Brain,
                color: "#8b5cf6",
                title: "Hebbian Gating",
                text: 'gate = x_sparse · y_sparse implements "neurons that fire together, gate together" — a direct Hebbian outer product at each layer.',
              },
              {
                icon: Activity,
                color: "#06b6d4",
                title: "Cumulative σ",
                text: "σ(i,j) = Σ y[τ,i]·x[τ,j] accumulates through the sentence. Watch how context words strengthen specific synapse pairs.",
              },
              {
                icon: BarChart3,
                color: "#22c55e",
                title: "Prediction Shift",
                text: "See how the Hebbian gate reshapes the model's next-byte predictions as more context flows in — from generic to specific.",
              },
            ].map((card, i) => (
              <motion.div
                key={card.title}
                className="glass-card p-5 border-t-2"
                style={{ borderTopColor: card.color }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <card.icon size={18} style={{ color: card.color }} />
                  <span
                    className="font-semibold text-sm"
                    style={{ color: card.color }}
                  >
                    {card.title}
                  </span>
                </div>
                <p className="text-xs text-[#8B95A5] leading-relaxed">
                  {card.text}
                </p>
              </motion.div>
            ))}
          </div>

          {/* Bottom note */}
          <motion.div
            className="glass-card p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
          >
            <p className="text-[#4A5568] text-xs text-center">
              <span className="text-white font-medium">Note:</span> This
              visualizes inference-time Hebbian dynamics within a single forward
              pass. The model&apos;s weights are frozen — what changes is which
              neurons co-activate as context accumulates word by word.
            </p>
          </motion.div>
        </motion.div>
      )}

      {/* Compact Explainer (shown after analysis, collapsed) */}
      {data && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-4 mt-6"
        >
          <div className="flex items-center gap-2 mb-3">
            <Brain size={16} className="text-bdh-accent" />
            <span className="text-sm font-semibold text-[#CBD5E0]">
              Pipeline Recap
            </span>
            <div className="flex-1 h-px bg-white/5" />
          </div>
          <div className="flex gap-3 flex-wrap text-[10px] font-mono">
            <span className="px-2 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              1. x_sparse = ReLU(x·E)
            </span>
            <ArrowRight size={12} className="text-[#4A5568] self-center" />
            <span className="px-2 py-1 rounded bg-green-500/10 text-green-400 border border-green-500/20">
              2. a* = Attn(x_sparse)·x
            </span>
            <ArrowRight size={12} className="text-[#4A5568] self-center" />
            <span className="px-2 py-1 rounded bg-[#2A7FFF]/10 text-[#2A7FFF] border border-[#2A7FFF]/20">
              3. gate = x·y (Hebbian)
            </span>
            <ArrowRight size={12} className="text-[#4A5568] self-center" />
            <span className="px-2 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
              4. x_l = x + D·gate
            </span>
          </div>
          <p className="text-[10px] text-[#4A5568] mt-2">
            gate = x_sparse · y_sparse is the Hebbian signal. σ accumulates
            through the sentence. Weights are frozen — only co-activation
            patterns change.
          </p>
        </motion.div>
      )}
    </div>
  );
}
