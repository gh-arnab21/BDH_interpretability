import { useState, useEffect, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Zap,
  Play,
  Pause,
  RotateCcw,
  ChevronRight,
  Loader2,
  Layers,
  Send,
} from "lucide-react";
import { visualization } from "../utils/api";

/* ================================================================== */
/*  Types — matches backend /hebbian-track response                    */
/* ================================================================== */
interface SynapsePair {
  neuron_from: number;
  neuron_to: number;
  strength: number;
}
interface HebbianUpdate {
  token_idx: number;
  layer: number;
  head: number;
  pairs: SynapsePair[];
}
interface HebbianResponse {
  input_text: string;
  num_tokens: number;
  updates: HebbianUpdate[];
}

/* per-step aggregated view — all updates at a given token position */
interface StepData {
  tokenIdx: number;
  char: string;
  updates: HebbianUpdate[];
  totalPairs: number;
}

const HEAD_COLORS = ["#8b5cf6", "#f59e0b", "#06b6d4", "#ef4444"];

export function HebbianPage() {
  const [inputText, setInputText] = useState(
    "le parlement européen a voté aujourd'hui",
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HebbianResponse | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);

  /* ── Derive tokens (bytes) from input text ── */
  const tokens = useMemo(() => {
    const bytes = new TextEncoder().encode(inputText);
    return Array.from(bytes).map((b) =>
      b >= 32 && b < 127
        ? String.fromCharCode(b)
        : `\\x${b.toString(16).padStart(2, "0")}`,
    );
  }, [inputText]);

  /* ── Group updates by token_idx → StepData[] ── */
  const steps: StepData[] = useMemo(() => {
    if (!data) return [];
    const map = new Map<number, HebbianUpdate[]>();
    for (const u of data.updates) {
      if (selectedLayer !== null && u.layer !== selectedLayer) continue;
      const arr = map.get(u.token_idx) ?? [];
      arr.push(u);
      map.set(u.token_idx, arr);
    }
    // Build one step per token (some may have no updates)
    const result: StepData[] = [];
    for (let t = 1; t < data.num_tokens; t++) {
      const updates = map.get(t) ?? [];
      result.push({
        tokenIdx: t,
        char: t < tokens.length ? tokens[t] : "?",
        updates,
        totalPairs: updates.reduce((s, u) => s + u.pairs.length, 0),
      });
    }
    return result;
  }, [data, tokens, selectedLayer]);

  /* ── Available layers ── */
  const availableLayers = useMemo(() => {
    if (!data) return [];
    const set = new Set(data.updates.map((u) => u.layer));
    return [...set].sort((a, b) => a - b);
  }, [data]);

  /* ── Cumulative synapse matrix ── */
  /* Build a map of (from→to) accumulated strength up to currentStep */
  const cumulativeMatrix = useMemo(() => {
    if (!steps.length)
      return { cells: new Map<string, number>(), maxStrength: 0 };
    const cells = new Map<string, number>();
    let maxStrength = 0;
    for (let s = 0; s <= currentStep && s < steps.length; s++) {
      for (const u of steps[s].updates) {
        for (const p of u.pairs) {
          const key = `${p.neuron_from}_${p.neuron_to}`;
          const prev = cells.get(key) ?? 0;
          const next = prev + p.strength;
          cells.set(key, next);
          if (next > maxStrength) maxStrength = next;
        }
      }
    }
    return { cells, maxStrength };
  }, [steps, currentStep]);

  /* ── Top cumulative synapses for the matrix visualization ── */
  const topSynapses = useMemo(() => {
    const { cells } = cumulativeMatrix;
    const entries = [...cells.entries()]
      .map(([key, val]) => {
        const [from, to] = key.split("_").map(Number);
        return { from, to, strength: val };
      })
      .sort((a, b) => b.strength - a.strength);
    return entries.slice(0, 100);
  }, [cumulativeMatrix]);

  /* ── Fetch from model ── */
  const runAnalysis = useCallback(async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setCurrentStep(0);
    setIsPlaying(false);
    try {
      const res = await visualization.hebbianTrack(inputText.trim());
      setData(res.data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to reach backend");
    } finally {
      setLoading(false);
    }
  }, [inputText]);

  /* ── Playback ── */
  useEffect(() => {
    if (!isPlaying || !steps.length) return;
    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev >= steps.length - 1) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 400);
    return () => clearInterval(interval);
  }, [isPlaying, steps.length]);

  const currentStepData = steps[currentStep];
  const currentPairs =
    currentStepData?.updates.flatMap((u) =>
      u.pairs.map((p) => ({ ...p, layer: u.layer, head: u.head })),
    ) ?? [];
  /* sort by strength descending */
  currentPairs.sort((a, b) => b.strength - a.strength);
  const maxPairStrength = currentPairs[0]?.strength ?? 1;

  /* ── Summary stats ── */
  const totalUpdates = data?.updates.length ?? 0;
  const totalPairs = data?.updates.reduce((s, u) => s + u.pairs.length, 0) ?? 0;

  return (
    <div className="min-h-screen p-8">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold mb-2">
          <span className="gradient-text">Hebbian Learning</span> Animator
        </h1>
        <p className="text-gray-400">
          Watch memory form in real-time. "Neurons that fire together, wire
          together" — BDH implements this during inference, no backpropagation
          needed.
        </p>
      </motion.div>

      {/* ── Input + controls ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-6"
      >
        <label className="text-sm text-gray-400 mb-2 block">
          Input sequence (French — model was trained on Europarl FR)
        </label>
        <div className="flex gap-3 mb-4">
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

        {error && (
          <div className="text-red-400 text-sm mb-3 p-2 bg-red-500/10 rounded">
            {error}
          </div>
        )}

        {data && (
          <>
            {/* Summary stats */}
            <div className="flex gap-6 mb-4 text-xs text-gray-500">
              <span>
                <span className="text-white font-mono">{data.num_tokens}</span>{" "}
                bytes
              </span>
              <span>
                <span className="text-white font-mono">{totalUpdates}</span>{" "}
                update groups
              </span>
              <span>
                <span className="text-white font-mono">{totalPairs}</span>{" "}
                synapse pairs
              </span>
              <span>
                <span className="text-white font-mono">
                  {availableLayers.length}
                </span>{" "}
                layers active
              </span>
            </div>

            {/* Layer filter pills */}
            <div className="flex items-center gap-2 mb-4">
              <Layers size={14} className="text-gray-500" />
              <span className="text-xs text-gray-500">Layer:</span>
              <button
                onClick={() => setSelectedLayer(null)}
                className={`px-2.5 py-1 rounded-full text-xs font-mono transition-all ${
                  selectedLayer === null
                    ? "bg-bdh-accent text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              >
                All
              </button>
              {availableLayers.map((l) => (
                <button
                  key={l}
                  onClick={() => setSelectedLayer(l)}
                  className={`px-2.5 py-1 rounded-full text-xs font-mono transition-all ${
                    selectedLayer === l
                      ? "bg-bdh-accent text-white"
                      : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                  }`}
                >
                  L{l}
                </button>
              ))}
            </div>

            {/* Token timeline */}
            <div className="flex flex-wrap gap-1 mb-4">
              {tokens.map((tok, idx) => {
                const stepIdx = steps.findIndex((s) => s.tokenIdx === idx);
                const isActive = stepIdx === currentStep;
                const isPast = stepIdx >= 0 && stepIdx < currentStep;
                const hasPairs = stepIdx >= 0 && steps[stepIdx].totalPairs > 0;
                return (
                  <motion.span
                    key={idx}
                    className={`px-2 py-1 rounded font-mono text-sm cursor-pointer transition-all ${
                      isActive
                        ? "bg-bdh-accent text-white scale-110"
                        : isPast
                          ? hasPairs
                            ? "bg-violet-900/40 text-violet-300 border border-violet-500/30"
                            : "bg-gray-700 text-gray-300"
                          : "bg-gray-800 text-gray-500"
                    }`}
                    onClick={() => {
                      if (stepIdx >= 0) setCurrentStep(stepIdx);
                    }}
                    animate={isActive ? { scale: [1, 1.1, 1] } : {}}
                    transition={{ duration: 0.3 }}
                    title={`Byte ${idx}${hasPairs ? ` — ${steps[stepIdx].totalPairs} pairs` : ""}`}
                  >
                    {tok === " " ? "␣" : tok}
                  </motion.span>
                );
              })}
            </div>

            {/* Controls */}
            <div className="flex items-center gap-4">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="btn-primary flex items-center gap-2"
                disabled={!steps.length}
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setCurrentStep(0);
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
                max={Math.max(0, steps.length - 1)}
                value={currentStep}
                onChange={(e) => setCurrentStep(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-gray-400 font-mono text-sm">
                {steps.length > 0 ? currentStep + 1 : 0}/{steps.length}
              </span>
            </div>
          </>
        )}
      </motion.div>

      {/* ── Main panels ── */}
      {data && steps.length > 0 && (
        <div className="grid lg:grid-cols-2 gap-6">
          {/* ── LEFT: Synapse Updates for current step ── */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold mb-1 flex items-center gap-2">
              <Zap size={20} className="text-bdh-accent" />
              Synapse Updates
              <span className="text-xs text-gray-500 font-normal ml-1">
                byte {currentStepData?.tokenIdx ?? "?"} ("
                {currentStepData?.char}")
              </span>
            </h3>
            <p className="text-xs text-gray-600 mb-4">
              Co-active neuron pairs — strength = activation(prev) ×
              activation(curr)
            </p>

            {currentPairs.length === 0 ? (
              <div className="text-gray-600 text-sm py-8 text-center">
                No Hebbian pairs at this token position
                {selectedLayer !== null && " (try All layers)"}
              </div>
            ) : (
              <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
                <AnimatePresence mode="popLayout">
                  {currentPairs.slice(0, 25).map((pair, idx) => (
                    <motion.div
                      key={`${currentStep}-${pair.layer}-${pair.head}-${pair.neuron_from}-${pair.neuron_to}`}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ delay: idx * 0.02 }}
                      className="flex items-center gap-3 p-2.5 bg-gray-800/50 rounded-lg"
                    >
                      {/* Layer/Head badge */}
                      <span
                        className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                        style={{
                          backgroundColor: `${HEAD_COLORS[pair.head % HEAD_COLORS.length]}20`,
                          color: HEAD_COLORS[pair.head % HEAD_COLORS.length],
                        }}
                      >
                        L{pair.layer}H{pair.head}
                      </span>

                      {/* Neuron from */}
                      <div className="w-12 h-7 rounded bg-blue-500/15 border border-blue-500/30 flex items-center justify-center text-xs font-mono text-blue-400">
                        {pair.neuron_from}
                      </div>
                      <ChevronRight
                        size={14}
                        className="text-bdh-accent flex-shrink-0"
                      />
                      {/* Neuron to */}
                      <div className="w-12 h-7 rounded bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-xs font-mono text-emerald-400">
                        {pair.neuron_to}
                      </div>

                      {/* Strength bar */}
                      <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-bdh-accent to-purple-400"
                          initial={{ width: 0 }}
                          animate={{
                            width: `${(pair.strength / maxPairStrength) * 100}%`,
                          }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 font-mono w-16 text-right">
                        {pair.strength.toFixed(4)}
                      </span>
                    </motion.div>
                  ))}
                </AnimatePresence>
                {currentPairs.length > 25 && (
                  <div className="text-xs text-gray-600 text-center pt-1">
                    +{currentPairs.length - 25} more pairs
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center gap-4 mt-4 text-[10px] text-gray-600">
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded bg-blue-500/20 border border-blue-500/30" />{" "}
                prev neuron
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded bg-emerald-500/20 border border-emerald-500/30" />{" "}
                curr neuron
              </span>
              <span className="ml-auto">
                {currentPairs.length} pairs at this step
              </span>
            </div>
          </motion.div>

          {/* ── RIGHT: Cumulative synapse heat ── */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold mb-1">
              Cumulative Synapse Strength
            </h3>
            <p className="text-xs text-gray-600 mb-4">
              Top {Math.min(topSynapses.length, 100)} strongest accumulated
              connections up to step {currentStep + 1}
            </p>

            {topSynapses.length === 0 ? (
              <div className="text-gray-600 text-sm py-8 text-center">
                No synapses accumulated yet
              </div>
            ) : (
              <>
                {/* Top synapses as colored rows */}
                <div className="space-y-1 max-h-[380px] overflow-y-auto pr-1">
                  {topSynapses.slice(0, 30).map((syn, i) => {
                    const pct =
                      cumulativeMatrix.maxStrength > 0
                        ? (syn.strength / cumulativeMatrix.maxStrength) * 100
                        : 0;
                    return (
                      <motion.div
                        key={`${syn.from}-${syn.to}`}
                        className="flex items-center gap-2 text-xs"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: i * 0.02 }}
                      >
                        <span className="font-mono text-blue-400 w-12 text-right">
                          {syn.from}
                        </span>
                        <ChevronRight size={10} className="text-gray-600" />
                        <span className="font-mono text-emerald-400 w-12">
                          {syn.to}
                        </span>
                        <div className="flex-1 h-3 bg-gray-800 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full rounded-full"
                            style={{
                              background: `linear-gradient(90deg, rgba(139,92,246,0.6) 0%, rgba(139,92,246,1) ${pct}%)`,
                            }}
                            initial={{ width: 0 }}
                            animate={{ width: `${pct}%` }}
                            transition={{ duration: 0.4 }}
                          />
                        </div>
                        <span className="font-mono text-gray-400 w-14 text-right">
                          {syn.strength.toFixed(3)}
                        </span>
                      </motion.div>
                    );
                  })}
                </div>

                <div className="flex items-center justify-between text-xs mt-4">
                  <span className="text-gray-500">
                    {cumulativeMatrix.cells.size} unique connections
                  </span>
                  <div className="flex items-center gap-2">
                    <span className="text-gray-600">Weak</span>
                    <div className="w-20 h-2 rounded-full bg-gradient-to-r from-gray-800 via-bdh-accent/50 to-bdh-accent" />
                    <span className="text-gray-600">Strong</span>
                  </div>
                </div>
              </>
            )}
          </motion.div>
        </div>
      )}

      {/* ── Learning explainer ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-6 glass-card p-6"
      >
        <h3 className="text-lg font-semibold mb-4">
          Inference-Time Learning Demo
        </h3>

        <div className="grid md:grid-cols-3 gap-6">
          <div className="p-4 bg-gray-800/50 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">
              Step 1: Before Exposure
            </div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Q: "The capital of Xanadu is?"
              <br />
              A: <span className="text-red-400">[unknown/random]</span>
            </div>
          </div>

          <div className="p-4 bg-bdh-accent/10 border border-bdh-accent/30 rounded-lg">
            <div className="text-sm text-bdh-accent mb-2">
              Step 2: Single Exposure
            </div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Input: "The capital of Xanadu is Moonhaven."
              <br />
              <span className="text-bdh-accent">→ Hebbian update occurs</span>
            </div>
          </div>

          <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
            <div className="text-sm text-green-400 mb-2">
              Step 3: After Learning
            </div>
            <div className="p-3 bg-gray-900 rounded font-mono text-sm">
              Q: "The capital of Xanadu is?"
              <br />
              A: <span className="text-green-400">"Moonhaven"</span>
            </div>
          </div>
        </div>

        <p className="text-gray-400 mt-4">
          <span className="text-white font-medium">Key insight:</span> BDH
          learns new facts during inference through Hebbian synaptic updates. No
          gradient descent, no backpropagation, no fine-tuning. Transformers
          fundamentally cannot do this — they require retraining to learn new
          information.
        </p>
      </motion.div>
    </div>
  );
}
