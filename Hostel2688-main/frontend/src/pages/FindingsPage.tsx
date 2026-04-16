import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";
import * as d3 from "d3";
import { Play, Pause, RotateCcw, ChevronDown } from "lucide-react";
import { adaptMonoData, adaptMergeData } from "../utils/dataAdapters";

interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

interface SelectivityData {
  histogram: HistogramBin[];
  total_neurons: number;
  total_selective: number;
  mean_selectivity: number;
}

interface SynapseWord {
  word: string;
  byte_start: number;
  byte_end: number;
  is_concept: boolean;
  sigma: Record<string, number>;
  delta_sigma: Record<string, number>;
}

interface TrackingSentence {
  sentence: string;
  n_bytes: number;
  words: SynapseWord[];
}

interface SynapseInfo {
  id: string;
  label: string;
  layer: number;
  head: number;
  i: number;
  j: number;
  selectivity: number;
}

interface SynapseTracking {
  synapses: SynapseInfo[];
  sentences: TrackingSentence[];
}

interface CrossConcept {
  primary: string;
  secondary: string;
  distinctness_per_layer: number[];
}

interface MonoNeuron {
  layer: number;
  head: number;
  neuron: number;
  selectivity: number;
  mean_in: number;
  mean_out: number;
  p_value: number;
  per_word: number[];
}

interface ConceptData {
  concept: string;
  words: unknown[];
  similarity: Record<string, number>;
  shared_neurons: unknown[];
  model_info: unknown;
  monosemantic_neurons: MonoNeuron[];
}

interface PrecomputedData {
  model_info: { n_layers: number; n_heads: number; n_neurons: number };
  best_layer: number;
  concepts: Record<string, ConceptData>;
  cross_concept: CrossConcept[];
  selectivity: SelectivityData;
  synapse_tracking: Record<string, SynapseTracking>;
}

interface EvalResult {
  french_loss: number | null;
  portuguese_loss: number | null;
}

interface MergeData {
  heritage: {
    neurons_per_head_original: number;
    neurons_per_head_merged: number;
    [k: string]: unknown;
  };
  models: Record<
    string,
    { params: number; n_neurons: number; [k: string]: unknown }
  >;
  evaluation: Record<string, EvalResult>;
  heritage_probe: {
    summary: {
      french_input_french_pct: number;
      portuguese_input_portuguese_pct: number;
      routing_quality: number;
      clear_separation: boolean;
      [k: string]: unknown;
    };
  };
  finetune_info: {
    iters: number;
    lr: number;
    pre_loss: number;
    post_loss: number;
    [k: string]: unknown;
  };
}

const C = {
  french: "#3b82f6",
  portuguese: "#10b981",
  merged: "#8b5cf6",
  finetuned: "#f59e0b",
  accent: "#00C896",
  red: "#ef4444",
};

const CARD = "rounded-xl p-6" + " " + "card-interactive";

function AnimatedNumber({
  target,
  decimals = 0,
  suffix = "",
  prefix = "",
  duration = 1.2,
}: {
  target: number;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  duration?: number;
}) {
  const [current, setCurrent] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    const start = performance.now();
    function tick(now: number) {
      const t = Math.min((now - start) / (duration * 1000), 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setCurrent(target * ease);
      if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }, [inView, target, duration]);

  return (
    <span ref={ref}>
      {prefix}
      {current.toFixed(decimals)}
      {suffix}
    </span>
  );
}

function HeroCard({
  label,
  value,
  suffix,
  decimals,
}: {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  color?: string;
  icon?: React.ElementType;
  index?: number;
}) {
  return (
    <div className={CARD + " flex flex-col items-center text-center"}>
      <div className="text-3xl font-bold text-[#E2E8F0] mb-1">
        <AnimatedNumber
          target={value}
          decimals={decimals ?? 0}
          suffix={suffix ?? ""}
        />
      </div>
      <div className="text-xs text-[#8B95A5] uppercase tracking-wider">
        {label}
      </div>
    </div>
  );
}

function LossChart({ evaluation }: { evaluation: Record<string, EvalResult> }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const inView = useInView(wrapRef, { once: true, margin: "-60px" });

  const W = 520,
    H = 300;
  const M = { top: 30, right: 20, bottom: 60, left: 50 };

  const models = ["french", "portuguese", "merged", "finetuned"] as const;
  const langs = ["french_loss", "portuguese_loss"] as const;
  const langLabel: Record<string, string> = {
    french_loss: "French",
    portuguese_loss: "Portuguese",
  };
  const modelLabel: Record<string, string> = {
    french: "FR Specialist",
    portuguese: "PT Specialist",
    merged: "Merged",
    finetuned: "Finetuned",
  };

  const allVals = models.flatMap((m) =>
    langs.map((l) => Number(evaluation[m]?.[l as keyof EvalResult] ?? 0)),
  );
  const yMax = Math.max(...allVals) * 1.15;

  const x0 = d3
    .scaleBand()
    .domain([...models])
    .range([M.left, W - M.right])
    .paddingInner(0.25);
  const x1 = d3
    .scaleBand()
    .domain([...langs])
    .range([0, x0.bandwidth()])
    .padding(0.08);
  const y = d3
    .scaleLinear()
    .domain([0, yMax])
    .range([H - M.bottom, M.top]);

  return (
    <div ref={wrapRef}>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* Grid */}
        {y.ticks(5).map((t) => (
          <g key={t}>
            <line
              x1={M.left}
              x2={W - M.right}
              y1={y(t)}
              y2={y(t)}
              stroke="#374151"
              strokeWidth={0.5}
            />
            <text
              x={M.left - 6}
              y={y(t) + 4}
              textAnchor="end"
              fill="#9ca3af"
              fontSize={10}
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Bars */}
        {models.map((m) =>
          langs.map((l) => {
            const val = Number(evaluation[m]?.[l as keyof EvalResult] ?? 0);
            const bx = (x0(m) ?? 0) + (x1(l) ?? 0);
            const bw = x1.bandwidth();
            const bh = y(0) - y(val);
            const clr = l === "french_loss" ? C.french : C.portuguese;
            return (
              <g key={`${m}-${l}`}>
                <motion.rect
                  x={bx}
                  width={bw}
                  rx={3}
                  fill={clr}
                  fillOpacity={0.85}
                  initial={{ y: y(0), height: 0 }}
                  animate={
                    inView ? { y: y(val), height: bh } : { y: y(0), height: 0 }
                  }
                  transition={{ duration: 0.7, delay: 0.12 }}
                />
                <motion.text
                  x={bx + bw / 2}
                  textAnchor="middle"
                  fill="#e5e7eb"
                  fontSize={9}
                  fontWeight={600}
                  initial={{ y: y(0), opacity: 0 }}
                  animate={
                    inView
                      ? { y: y(val) - 5, opacity: 1 }
                      : { y: y(0), opacity: 0 }
                  }
                  transition={{ duration: 0.7, delay: 0.35 }}
                >
                  {val.toFixed(2)}
                </motion.text>
              </g>
            );
          }),
        )}

        {/* X labels */}
        {models.map((m) => (
          <text
            key={m}
            x={(x0(m) ?? 0) + x0.bandwidth() / 2}
            y={H - M.bottom + 18}
            textAnchor="middle"
            fill="#d1d5db"
            fontSize={11}
            fontWeight={500}
          >
            {modelLabel[m]}
          </text>
        ))}

        {/* Legend */}
        {langs.map((l, i) => (
          <g key={l} transform={`translate(${M.left + i * 120}, ${H - 14})`}>
            <rect
              width={12}
              height={12}
              rx={2}
              fill={l === "french_loss" ? C.french : C.portuguese}
              fillOpacity={0.85}
            />
            <text x={18} y={10} fill="#d1d5db" fontSize={11}>
              {langLabel[l]}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function FinetuneMeter({
  preLoss,
  postLoss,
  iters,
}: {
  preLoss: number;
  postLoss: number;
  iters: number;
}) {
  const reduction = ((preLoss - postLoss) / preLoss) * 100;

  return (
    <div className="space-y-5">
      <div className="text-center">
        <div className="text-3xl font-bold text-[#E2E8F0]">
          <AnimatedNumber target={reduction} decimals={1} suffix="%" />
        </div>
        <p className="text-[#4A5568] text-sm mt-1">Loss Reduction</p>
      </div>

      {[
        { label: "Before Fine-tuning", val: preLoss, color: "#6b7280" },
        {
          label: `After ${iters} iterations`,
          val: postLoss,
          color: C.french,
        },
      ].map((row) => (
        <div key={row.label}>
          <div className="flex justify-between text-xs text-[#8B95A5] mb-1">
            <span>{row.label}</span>
            <span className="text-white font-mono">{row.val.toFixed(4)}</span>
          </div>
          <div className="h-4 rounded bg-white/5 overflow-hidden">
            <div
              className="h-full rounded"
              style={{
                background: row.color,
                width: `${(row.val / 4) * 100}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function SelectivityHistogram({ histogram }: { histogram: HistogramBin[] }) {
  const W = 480,
    H = 240;
  const M = { top: 16, right: 12, bottom: 40, left: 44 };

  const x = d3
    .scaleLinear()
    .domain([0, 1])
    .range([M.left, W - M.right]);
  const yMax = d3.max(histogram, (d) => d.count) ?? 1;
  const y = d3
    .scaleLinear()
    .domain([0, yMax * 1.1])
    .range([H - M.bottom, M.top]);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {y.ticks(4).map((t) => (
        <g key={t}>
          <line
            x1={M.left}
            x2={W - M.right}
            y1={y(t)}
            y2={y(t)}
            stroke="#374151"
            strokeWidth={0.5}
          />
          <text
            x={M.left - 6}
            y={y(t) + 4}
            textAnchor="end"
            fill="#9ca3af"
            fontSize={9}
          >
            {t >= 1000 ? `${(t / 1000).toFixed(1)}k` : t}
          </text>
        </g>
      ))}

      {histogram.map((bin, i) => {
        const bx = x(bin.bin_start);
        const bw = x(bin.bin_end) - x(bin.bin_start) - 1;
        const bh = y(0) - y(bin.count);
        return (
          <motion.rect
            key={i}
            x={bx}
            width={Math.max(bw, 1)}
            rx={2}
            fill={C.accent}
            fillOpacity={0.75}
            initial={{ y: y(0), height: 0 }}
            whileInView={{ y: y(bin.count), height: bh }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.03 }}
          />
        );
      })}

      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
        <text
          key={v}
          x={x(v)}
          y={H - M.bottom + 16}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize={10}
        >
          {v.toFixed(2)}
        </text>
      ))}
      <text
        x={(M.left + W - M.right) / 2}
        y={H - 4}
        textAnchor="middle"
        fill="#6b7280"
        fontSize={10}
      >
        Selectivity Score
      </text>
    </svg>
  );
}

function RadialGauge({
  value,
  max,
  label,
  color,
  size = 170,
}: {
  value: number;
  max: number;
  label: string;
  color: string;
  size?: number;
}) {
  const r = size / 2 - 14;
  const circ = 2 * Math.PI * r;
  const pct = value / max;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke="#1f2937"
          strokeWidth={10}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={10}
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          whileInView={{ strokeDashoffset: circ * (1 - pct) }}
          viewport={{ once: true }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text
          x={size / 2}
          y={size / 2 - 6}
          textAnchor="middle"
          fill="white"
          fontSize={size / 5.5}
          fontWeight={700}
        >
          {(pct * 100).toFixed(1)}%
        </text>
        <text
          x={size / 2}
          y={size / 2 + 14}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize={size / 13}
        >
          {value.toLocaleString()} / {max.toLocaleString()}
        </text>
      </svg>
      <span className="text-[#8B95A5] text-xs mt-1">{label}</span>
    </div>
  );
}

function RoutingDonut({
  frPct,
  ptPct,
  quality,
}: {
  frPct: number;
  ptPct: number;
  quality: number;
}) {
  const size = 180,
    r = 68;
  const circ = 2 * Math.PI * r;
  const frArc = circ * (frPct / 100);
  const ptArc = circ * (ptPct / 100);

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={C.french}
          strokeWidth={14}
          strokeDasharray={`${frArc} ${circ - frArc}`}
          initial={{ strokeDashoffset: circ }}
          whileInView={{ strokeDashoffset: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1 }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={C.portuguese}
          strokeWidth={14}
          strokeDasharray={`${ptArc} ${circ - ptArc}`}
          strokeDashoffset={-frArc}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.5 }}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
        />
        <text
          x={size / 2}
          y={size / 2 - 4}
          textAnchor="middle"
          fill="white"
          fontSize={20}
          fontWeight={700}
        >
          {quality.toFixed(1)}%
        </text>
        <text
          x={size / 2}
          y={size / 2 + 14}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize={10}
        >
          routing quality
        </text>
      </svg>
      <div className="flex gap-5 mt-2 text-xs text-[#CBD5E0]">
        <span className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: C.french }}
          />
          FR {frPct.toFixed(1)}%
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: C.portuguese }}
          />
          PT {ptPct.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

function SigmaReplay({
  tracking,
}: {
  tracking: SynapseTracking;
  category?: string;
}) {
  const [sentIdx, setSentIdx] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const sent = tracking.sentences[sentIdx];
  const words = sent?.words ?? [];
  const mainSynId = tracking.synapses[0]?.id ?? "";

  const cumSigma = useMemo(() => {
    const res: number[] = [];
    let acc = 0;
    for (const w of words) {
      acc += w.delta_sigma[mainSynId] ?? 0;
      res.push(acc);
    }
    return res;
  }, [words, mainSynId]);

  const maxSigma = Math.max(...cumSigma, 1);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep((s) => {
          if (s >= words.length - 1) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      }, 550);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [playing, words.length]);

  const restart = useCallback(() => {
    setStep(0);
    setPlaying(true);
  }, []);

  const W = 440,
    H = 160;
  const M = { top: 12, right: 12, bottom: 30, left: 40 };
  const xS = d3
    .scaleLinear()
    .domain([0, Math.max(words.length - 1, 1)])
    .range([M.left, W - M.right]);
  const yS = d3
    .scaleLinear()
    .domain([0, maxSigma * 1.1])
    .range([H - M.bottom, M.top]);

  const visibleData = cumSigma.slice(0, step + 1);
  const pathD = d3
    .line<number>()
    .x((_, i) => xS(i))
    .y((d) => yS(d))
    .curve(d3.curveMonotoneX)(visibleData);

  return (
    <div className="space-y-3">
      {/* Controls */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <select
            value={sentIdx}
            onChange={(e) => {
              setSentIdx(Number(e.target.value));
              setStep(0);
              setPlaying(false);
            }}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm text-[#E2E8F0] appearance-none pr-8"
          >
            {tracking.sentences.map((s, i) => (
              <option key={i} value={i}>
                {s.sentence}
              </option>
            ))}
          </select>
          <ChevronDown
            size={14}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[#4A5568] pointer-events-none"
          />
        </div>
        <button
          onClick={() => setPlaying((p) => !p)}
          className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[#CBD5E0]"
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button
          onClick={restart}
          className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-[#CBD5E0]"
        >
          <RotateCcw size={14} />
        </button>
      </div>

      {/* Chart */}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {yS.ticks(3).map((t) => (
          <line
            key={t}
            x1={M.left}
            x2={W - M.right}
            y1={yS(t)}
            y2={yS(t)}
            stroke="#374151"
            strokeWidth={0.5}
          />
        ))}
        {pathD && (
          <motion.path
            d={pathD}
            fill="none"
            stroke={C.accent}
            strokeWidth={2.5}
            strokeLinecap="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.3 }}
          />
        )}
        {visibleData.length > 0 && (
          <motion.circle
            cx={xS(step)}
            cy={yS(visibleData[step] ?? 0)}
            r={5}
            fill={C.accent}
            stroke="white"
            strokeWidth={2}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 300 }}
          />
        )}
        {words.map((w, i) => (
          <text
            key={i}
            x={xS(i)}
            y={H - M.bottom + 16}
            textAnchor="middle"
            fill={i === step ? "white" : w.is_concept ? C.finetuned : "#6b7280"}
            fontSize={9}
            fontWeight={i === step ? 700 : 400}
          >
            {w.word.length > 8 ? w.word.slice(0, 7) + "\u2026" : w.word}
          </text>
        ))}
      </svg>

      {/* Value display */}
      <div className="text-center">
        <AnimatePresence mode="wait">
          <motion.span
            key={step}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="inline-block px-3 py-1 rounded-full bg-white/5 text-sm"
          >
            <span className="text-[#8B95A5]">&sigma; =</span>{" "}
            <span className="text-white font-mono font-bold">
              {cumSigma[step]?.toFixed(2)}
            </span>
            <span className="text-[#4A5568] ml-2">
              at &ldquo;{words[step]?.word}&rdquo;
            </span>
          </motion.span>
        </AnimatePresence>
      </div>
    </div>
  );
}

function DistinctnessChart({
  crossConcept,
  nLayers,
}: {
  crossConcept: CrossConcept[];
  nLayers: number;
}) {
  const nLegend = crossConcept.length;
  const legendH = nLegend * 16 + 8;
  const W = 480,
    H = 200 + legendH;
  const M = { top: 12, right: 20, bottom: 40 + legendH, left: 44 };
  const layers = Array.from({ length: nLayers }, (_, i) => i);

  const xS = d3
    .scaleLinear()
    .domain([0, nLayers - 1])
    .range([M.left, W - M.right]);
  const allVals = crossConcept.flatMap((c) => c.distinctness_per_layer);
  const yMax = Math.max(...allVals, 0.5) * 1.15;
  const yS = d3
    .scaleLinear()
    .domain([0, yMax])
    .range([H - M.bottom, M.top]);

  const palette = [
    C.french,
    C.portuguese,
    C.merged,
    C.finetuned,
    C.accent,
    C.red,
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {yS.ticks(4).map((t) => (
        <g key={t}>
          <line
            x1={M.left}
            x2={W - M.right}
            y1={yS(t)}
            y2={yS(t)}
            stroke="#374151"
            strokeWidth={0.5}
          />
          <text
            x={M.left - 6}
            y={yS(t) + 4}
            textAnchor="end"
            fill="#9ca3af"
            fontSize={9}
          >
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {crossConcept.map((cc, ci) => {
        const p =
          d3
            .line<number>()
            .x((_, i) => xS(i))
            .y((d) => yS(d))
            .curve(d3.curveMonotoneX)(cc.distinctness_per_layer) ?? "";
        return (
          <motion.path
            key={ci}
            d={p}
            fill="none"
            stroke={palette[ci % palette.length]}
            strokeWidth={2}
            strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            whileInView={{ pathLength: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1, delay: ci * 0.15 }}
          />
        );
      })}

      {layers.map((l) => (
        <text
          key={l}
          x={xS(l)}
          y={H - M.bottom + 18}
          textAnchor="middle"
          fill="#9ca3af"
          fontSize={10}
        >
          L{l}
        </text>
      ))}
      <text
        x={(M.left + W - M.right) / 2}
        y={H - 4}
        textAnchor="middle"
        fill="#6b7280"
        fontSize={10}
      >
        Layer
      </text>

      {crossConcept.map((cc, ci) => (
        <g
          key={`lg${ci}`}
          transform={`translate(${M.left}, ${H - legendH + 8 + ci * 16})`}
        >
          <line
            x1={0}
            x2={14}
            y1={0}
            y2={0}
            stroke={palette[ci % palette.length]}
            strokeWidth={2}
          />
          <text x={18} y={4} fill="#d1d5db" fontSize={9}>
            {cc.primary} vs {cc.secondary}
          </text>
        </g>
      ))}
    </svg>
  );
}

function NeuronHeatmap({
  concepts,
}: {
  concepts: Record<string, ConceptData>;
}) {
  if (!concepts || typeof concepts !== "object") return null;
  return (
    <div className="space-y-4">
      {Object.entries(concepts).map(([cat, data]) => {
        const neurons = (data.monosemantic_neurons ?? []).slice(0, 12);
        return (
          <div key={cat}>
            <div className="text-xs text-[#8B95A5] uppercase tracking-wider mb-1.5">
              {data.concept}
            </div>
            <div className="flex gap-1 flex-wrap">
              {neurons.map((n, i) => (
                <div key={i} className="relative group cursor-default">
                  <div
                    className="w-8 h-8 rounded flex items-center justify-center text-[8px] font-mono text-[#E2E8F0]/90"
                    style={{ background: d3.interpolateViridis(n.selectivity) }}
                  >
                    {n.selectivity.toFixed(1)}
                  </div>
                  <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 hidden group-hover:block bg-[#0B1216] border border-white/10 rounded px-2 py-1 text-[10px] whitespace-nowrap z-20 pointer-events-none">
                    L{n.layer} H{n.head} N{n.neuron}
                    <br />
                    sel: {n.selectivity.toFixed(3)} &bull; p=
                    {n.p_value.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
      <div className="flex items-center gap-2 mt-2">
        <span className="text-[10px] text-[#4A5568]">0.0</span>
        <div
          className="h-2 flex-1 rounded"
          style={{
            background: `linear-gradient(to right, ${d3.interpolateViridis(0)}, ${d3.interpolateViridis(0.5)}, ${d3.interpolateViridis(1)})`,
          }}
        />
        <span className="text-[10px] text-[#4A5568]">1.0</span>
        <span className="text-[10px] text-[#4A5568] ml-1">selectivity</span>
      </div>
    </div>
  );
}

function SectionHead({
  title,
  sub,
}: {
  icon?: React.ElementType;
  title: string;
  sub: string;
}) {
  return (
    <div className="mb-6">
      <h2 className="text-lg font-semibold text-[#E2E8F0] mb-1">{title}</h2>
      <p className="text-sm text-[#4A5568]">{sub}</p>
    </div>
  );
}

export function FindingsPage() {
  const [monoData, setMonoData] = useState<PrecomputedData | null>(null);
  const [mergeData, setMergeData] = useState<MergeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTrackCat, setActiveTrackCat] = useState("");

  useEffect(() => {
    Promise.all([
      fetch("/monosemanticity/precomputed.json").then((r) => {
        if (!r.ok) throw new Error(`precomputed.json: ${r.status}`);
        return r.json();
      }),
      fetch("/merge/merge_eval.json").then((r) => {
        if (!r.ok)
          return fetch("/merge/merge_data.json").then((r2) => {
            if (!r2.ok) throw new Error(`merge data: ${r2.status}`);
            return r2.json();
          });
        return r.json();
      }),
    ])
      .then(([monoRaw, mergeRaw]) => {
        setMonoData(adaptMonoData(monoRaw) as unknown as PrecomputedData);
        setMergeData(adaptMergeData(mergeRaw) as unknown as MergeData);
      })
      .catch((err) => console.error("FindingsPage data load failed:", err))
      .finally(() => setLoading(false));
  }, []);

  // Set default tracking category
  useEffect(() => {
    if (monoData && !activeTrackCat) {
      const cats = Object.keys(monoData.synapse_tracking ?? {});
      if (cats.length) setActiveTrackCat(cats[0]);
    }
  }, [monoData, activeTrackCat]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh] text-[#4A5568] text-sm">
        Loading…
      </div>
    );
  }

  if (!monoData || !mergeData) {
    return (
      <div className="text-center py-20 text-[#8B95A5]">
        Failed to load data.
      </div>
    );
  }

  const sel = monoData.selectivity ?? {
    histogram: [],
    total_neurons: 0,
    total_selective: 0,
    mean_selectivity: 0,
  };
  const ft = mergeData.finetune_info;
  const probe = mergeData.heritage_probe?.summary;
  const nCats = Object.keys(monoData.concepts ?? {}).length;
  const nLayers = monoData.model_info.n_layers;
  const trackingCats = Object.keys(monoData.synapse_tracking ?? {});

  return (
    <div
      className="max-w-6xl mx-auto space-y-16 pb-20"
      style={{ background: "#070D12", minHeight: "100vh" }}
    >
      {/* Title */}
      <div className="pt-4">
        <h1 className="text-2xl font-bold text-[#E2E8F0] mb-1">Findings</h1>
        <p className="text-[#4A5568] text-sm">
          Monosemanticity, heritage routing, and merge viability — measured on
          real checkpoints.
        </p>
      </div>

      {/* Hero Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <HeroCard label="Neurons Analyzed" value={sel.total_neurons} />
        <HeroCard label="Selective Neurons" value={sel.total_selective} />
        <HeroCard
          label="Mean Selectivity"
          value={sel.mean_selectivity}
          decimals={4}
        />
        <HeroCard label="Concept Categories" value={nCats} />
      </div>

      {/* Loss Landscape */}
      <section>
        <SectionHead
          title="Loss Landscape"
          sub="Cross-lingual loss comparison across model variants. Lower is better."
        />
        <div className={ft ? "grid md:grid-cols-5 gap-6" : "grid gap-6"}>
          <div className={`${CARD} ${ft ? "md:col-span-3" : ""}`}>
            <LossChart evaluation={mergeData.evaluation} />
          </div>
          {ft && (
            <div
              className={`${CARD} md:col-span-2 flex flex-col justify-center`}
            >
              <FinetuneMeter
                preLoss={ft.pre_loss}
                postLoss={ft.post_loss}
                iters={ft.iters}
              />
            </div>
          )}
        </div>
      </section>

      {/* Selectivity */}
      <section>
        <SectionHead
          title="Neuron Selectivity"
          sub="Distribution of selectivity scores. Selective neurons fire strongly for specific concepts."
        />
        <div className="grid md:grid-cols-5 gap-6">
          <div className={`${CARD} md:col-span-3`}>
            <SelectivityHistogram histogram={sel.histogram} />
          </div>
          <div
            className={`${CARD} md:col-span-2 flex items-center justify-center`}
          >
            <RadialGauge
              value={sel.total_selective}
              max={sel.total_neurons}
              label="Selective / Total Neurons"
              color={C.accent}
            />
          </div>
        </div>
      </section>

      {/* Heritage Routing */}
      {probe && (
        <section>
          <SectionHead
            title="Heritage Routing"
            sub="How well the merged model routes inputs to their parent-language neurons."
          />
          <div className="grid md:grid-cols-3 gap-6">
            <div className={`${CARD} flex items-center justify-center`}>
              <RoutingDonut
                frPct={probe.french_input_french_pct}
                ptPct={probe.portuguese_input_portuguese_pct}
                quality={probe.routing_quality}
              />
            </div>
            <div className={`${CARD} md:col-span-2`}>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {[
                    {
                      label: "FR \u2192 FR neurons",
                      value: probe.french_input_french_pct,
                      color: C.french,
                    },
                    {
                      label: "PT \u2192 PT neurons",
                      value: probe.portuguese_input_portuguese_pct,
                      color: C.portuguese,
                    },
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="flex justify-between text-xs text-[#8B95A5] mb-1">
                        <span>{item.label}</span>
                        <span className="text-white font-mono">
                          {item.value.toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                        <motion.div
                          className="h-full rounded-full"
                          style={{ background: item.color }}
                          initial={{ width: 0 }}
                          whileInView={{ width: `${item.value}%` }}
                          viewport={{ once: true }}
                          transition={{ duration: 0.8 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <div className="text-center pt-2">
                  <span
                    className={`inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium ${
                      probe.clear_separation
                        ? "bg-white/5 text-[#CBD5E0] border border-white/10"
                        : "bg-white/5 text-[#8B95A5] border border-white/10"
                    }`}
                  >
                    {probe.clear_separation
                      ? "Clear Separation Detected"
                      : "No Clear Separation \u2014 Needs More Training"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Sigma Replay */}
      {trackingCats.length > 0 && (
        <section>
          <SectionHead
            title="Synapse \u03C3-Curve Replay"
            sub="Cumulative sigma accumulated word-by-word as the model reads a sentence."
          />
          <div className={CARD}>
            <div className="flex gap-2 mb-4 flex-wrap">
              {trackingCats.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveTrackCat(cat)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                    activeTrackCat === cat
                      ? "bg-white/10 text-[#E2E8F0] border border-white/\[0.12\]"
                      : "bg-transparent text-[#8B95A5] border border-white/10 hover:text-[#E2E8F0]"
                  }`}
                >
                  {cat.charAt(0).toUpperCase() + cat.slice(1)}
                </button>
              ))}
            </div>
            {activeTrackCat && monoData.synapse_tracking?.[activeTrackCat] && (
              <SigmaReplay
                tracking={monoData.synapse_tracking[activeTrackCat]}
                category={activeTrackCat}
              />
            )}
          </div>
        </section>
      )}

      {/* Cross-Concept Distinctness */}
      <section>
        <SectionHead
          title="Cross-Concept Distinctness"
          sub="Per-layer distinctness between concept pairs. Higher values = stronger separation."
        />
        <div className={CARD}>
          <DistinctnessChart
            crossConcept={monoData.cross_concept}
            nLayers={nLayers}
          />
        </div>
      </section>

      {/* Neuron Heatmap */}
      <section>
        <SectionHead
          title="Monosemantic Neuron Map"
          sub="Top selective neurons per concept. Hover for layer/head/neuron details."
        />
        <div className={CARD}>
          <NeuronHeatmap concepts={monoData.concepts} />
        </div>
      </section>
    </div>
  );
}
