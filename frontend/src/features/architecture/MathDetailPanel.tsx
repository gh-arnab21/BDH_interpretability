import { useEffect, useRef, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X } from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface FrameData {
  token_idx: number;
  token_char: string;
  token_byte: number;
  layer: number;
  x_sparsity: number;
  y_sparsity: number;
  x_active_count?: number;
  y_active_count?: number;
  x_active: Array<{ indices: number[]; values: number[] }>;
  y_active: Array<{ indices: number[]; values: number[] }>;
  x_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  y_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  x_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
    histogram?: Array<{ start: number; end: number; count: number }>;
  };
  y_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
    histogram?: Array<{ start: number; end: number; count: number }>;
  };
  gating?: {
    x_only: number;
    y_only: number;
    both: number;
    survival_rate: number;
  };
  attention_weights?: Array<{
    token_idx: number;
    char: string;
    weight: number;
  }>;
  attention_stats?: {
    top_attended: Array<{ token_idx: number; char: string; weight: number }>;
  };
  embedding?: {
    byte_value: number;
    norm: number;
    mean: number;
    std: number;
    vector_ds?: number[];
    pre_ln_ds?: number[];
    pre_ln_norm?: number;
    pre_ln_mean?: number;
    pre_ln_std?: number;
  };
  x_activation_grid?: number[][];
  y_activation_grid?: number[][];
  hadamard_grid?: number[][];
  a_star_ds?: number[];
  a_star_norm?: number;
  decoder_ds?: number[];
  decoder_norm?: number;
  decoder_mean?: number;
  decoder_std?: number;
}

interface PlaybackData {
  input_text: string;
  input_chars: string[];
  num_layers: number;
  num_heads: number;
  neurons_per_head: number;
  embedding_dim?: number;
  total_neurons?: number;
  frames: FrameData[];
  predictions?: Array<Array<{ byte: number; char: string; prob: number }>>;
  rho_matrices?: Record<number, number[][]>;
}

interface MathDetailPanelProps {
  selectedBlock: number | null;
  onClose: () => void;
  frameData?: FrameData;
  playbackData?: PlaybackData;
  currentLayer: number;
}

function Tex({ math, display = false }: { math: string; display?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(math, ref.current, {
          displayMode: display,
          throwOnError: false,
          trust: true,
        });
      } catch {
        if (ref.current) ref.current.textContent = math;
      }
    }
  }, [math, display]);
  return <span ref={ref} />;
}

const BLOCK_META: Record<
  number,
  {
    name: string;
    color: string;
    accentBg: string;
  }
> = {
  0: {
    name: "Input Token",
    color: "#E2E8F0",
    accentBg: "rgba(226,232,240,0.06)",
  },
  1: {
    name: "Embedding",
    color: "#C4B5FD",
    accentBg: "rgba(196,181,253,0.06)",
  },
  2: { name: "LayerNorm", color: "#FCD34D", accentBg: "rgba(252,211,77,0.06)" },
  3: { name: "Linear Dₓ", color: "#FCD34D", accentBg: "rgba(252,211,77,0.06)" },
  4: { name: "ReLU (x)", color: "#FCA5A5", accentBg: "rgba(252,165,165,0.06)" },
  5: { name: "ρ Memory", color: "#67E8F9", accentBg: "rgba(103,232,249,0.06)" },
  6: {
    name: "a* Readout",
    color: "#67E8F9",
    accentBg: "rgba(103,232,249,0.06)",
  },
  7: { name: "Linear Dᵧ", color: "#FCD34D", accentBg: "rgba(252,211,77,0.06)" },
  8: { name: "ReLU (y)", color: "#FCA5A5", accentBg: "rgba(252,165,165,0.06)" },
  9: {
    name: "Hadamard x⊙y",
    color: "#22D3EE",
    accentBg: "rgba(34,211,238,0.06)",
  },
  10: {
    name: "Decoder D",
    color: "#6EE7B7",
    accentBg: "rgba(110,231,183,0.06)",
  },
  11: {
    name: "Residual ⊕",
    color: "#C4B5FD",
    accentBg: "rgba(196,181,253,0.06)",
  },
  12: { name: "Output", color: "#A78BFA", accentBg: "rgba(167,139,250,0.06)" },
};

export function MathDetailPanel({
  selectedBlock,
  onClose,
  frameData,
  playbackData,
  currentLayer,
}: MathDetailPanelProps) {
  const config = useMemo(
    () => ({
      d: playbackData?.embedding_dim ?? 256,
      n: playbackData?.neurons_per_head ?? 8192,
      h: playbackData?.num_heads ?? 4,
      total: playbackData?.total_neurons ?? 32768,
    }),
    [playbackData],
  );

  const predictions = useMemo(() => {
    if (!playbackData?.predictions || !frameData) return null;
    return playbackData.predictions[frameData.token_idx];
  }, [playbackData, frameData]);

  if (selectedBlock === null) return null;

  const meta = BLOCK_META[selectedBlock] ?? {
    name: `Step ${selectedBlock}`,
    color: "#E2E8F0",
    accentBg: "rgba(255,255,255,0.04)",
  };

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={selectedBlock}
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: 30 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
        className="w-[400px] flex-shrink-0 self-start sticky top-4"
      >
        <div
          className="rounded-xl overflow-hidden"
          style={{
            background: "#0C1219",
            border: `1px solid rgba(255,255,255,0.08)`,
          }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-5 py-3"
            style={{
              background: meta.accentBg,
              borderBottom: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            <div className="flex items-center gap-2">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: meta.color }}
              />
              <span
                className="font-semibold text-sm"
                style={{ color: meta.color }}
              >
                {meta.name}
              </span>
              <span className="text-xs text-[#4A5568] ml-1">
                Step {selectedBlock}
              </span>
            </div>
            <button
              onClick={onClose}
              className="p-1 rounded hover:bg-white/10 transition-colors"
            >
              <X size={16} className="text-[#6B7280]" />
            </button>
          </div>

          {/* Content */}
          <div
            className="p-5 space-y-5 max-h-[80vh] overflow-y-auto"
            style={{
              scrollbarWidth: "thin",
              scrollbarColor: "#2D3748 transparent",
            }}
          >
            {/* Section 1: The Math */}
            <Section title="Equation" color={meta.color}>
              <BlockEquation step={selectedBlock} config={config} />
            </Section>

            {/* Section 2: Live Numbers */}
            <Section title="Live Values" color={meta.color}>
              <BlockLiveData
                step={selectedBlock}
                frameData={frameData}
                playbackData={playbackData}
                config={config}
                predictions={predictions}
                currentLayer={currentLayer}
              />
            </Section>

            {/* Section 3: Interpretation */}
            <Section title="Interpretation" color={meta.color}>
              <BlockInterpretation step={selectedBlock} />
            </Section>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

function Section({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-1 h-4 rounded-full"
          style={{ background: color, opacity: 0.5 }}
        />
        <span className="text-xs font-semibold uppercase tracking-wider text-[#6B7280]">
          {title}
        </span>
      </div>
      <div
        className="rounded-lg p-3"
        style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.04)",
        }}
      >
        {children}
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
}: {
  label: string;
  value: string | number;
  unit?: string;
}) {
  return (
    <div className="flex items-baseline justify-between py-1">
      <span className="text-xs text-[#6B7280]">{label}</span>
      <span className="text-sm font-mono text-[#E2E8F0]">
        {value}
        {unit && <span className="text-[#4A5568] ml-0.5 text-xs">{unit}</span>}
      </span>
    </div>
  );
}

function MiniBarChart({
  data,
  maxVal,
  color,
  height = 36,
}: {
  data: Array<{ label: string; value: number }>;
  maxVal?: number;
  color: string;
  height?: number;
}) {
  const mx = maxVal ?? Math.max(0.001, ...data.map((d) => d.value));
  return (
    <div className="flex items-end gap-1" style={{ height }}>
      {data.map((d, i) => (
        <div key={i} className="flex flex-col items-center flex-1 min-w-0">
          <div
            className="w-full rounded-t"
            style={{
              height: `${Math.max(2, (d.value / mx) * height * 0.8)}px`,
              background: color,
              opacity: 0.8,
            }}
          />
          <span
            className="text-[9px] text-[#6B7280] truncate w-full text-center mt-0.5"
            title={d.label}
          >
            {d.label}
          </span>
        </div>
      ))}
    </div>
  );
}

function NeuronTable({
  neurons,
  color,
}: {
  neurons: Array<{ head: number; neuron: number; value: number }>;
  color: string;
}) {
  if (!neurons || neurons.length === 0) {
    return <span className="text-xs text-[#4A5568]">No active neurons</span>;
  }
  return (
    <div className="space-y-0.5">
      <div className="grid grid-cols-[40px_60px_1fr_60px] gap-1 text-[10px] text-[#4A5568] uppercase tracking-wider mb-1">
        <span>Head</span>
        <span>Neuron</span>
        <span>Activation</span>
        <span className="text-right">Value</span>
      </div>
      {neurons.slice(0, 8).map((n, i) => {
        const maxVal = neurons[0]?.value ?? 1;
        const barWidth = (n.value / maxVal) * 100;
        return (
          <div
            key={i}
            className="grid grid-cols-[40px_60px_1fr_60px] gap-1 items-center text-xs"
          >
            <span className="text-[#8B95A5] font-mono">H{n.head}</span>
            <span className="text-[#8B95A5] font-mono">#{n.neuron}</span>
            <div
              className="h-2 rounded-full overflow-hidden"
              style={{ background: "rgba(255,255,255,0.06)" }}
            >
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${barWidth}%`, background: color }}
              />
            </div>
            <span className="text-right text-[#E2E8F0] font-mono">
              {n.value.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function VectorViz({
  values,
  height = 18,
}: {
  values: number[];
  height?: number;
}) {
  if (!values || values.length === 0) return null;
  const maxAbs = Math.max(0.001, ...values.map((v) => Math.abs(v)));
  return (
    <div className="flex rounded overflow-hidden" style={{ height }}>
      {values.map((v, i) => {
        const norm = v / maxAbs;
        let bg: string;
        if (norm < 0) {
          const t = Math.min(1, -norm);
          bg = `rgb(${Math.round(20 + t * 10)}, ${Math.round(30 + t * 100)}, ${Math.round(60 + t * 195)})`;
        } else {
          const t = Math.min(1, norm);
          bg = `rgb(${Math.round(60 + t * 195)}, ${Math.round(30 + t * 60)}, ${Math.round(20)})`;
        }
        return (
          <div key={i} className="flex-1 min-w-0" style={{ background: bg }} />
        );
      })}
    </div>
  );
}

function BlockEquation({
  step,
  config,
}: {
  step: number;
  config: { d: number; n: number; h: number; total: number };
}) {
  const equations: Record<number, { main: string; detail?: string }> = {
    0: {
      main: "b_t = \\text{encode}(\\text{char}_t) \\in \\{0, 1, \\ldots, 255\\}",
      detail:
        "\\text{Byte-level tokenization: each character becomes its UTF-8 byte value}",
    },
    1: {
      main: `v^* = W_{\\text{emb}}[b_t] \\in \\mathbb{R}^{${config.d}}`,
      detail: `W_{\\text{emb}} \\in \\mathbb{R}^{256 \\times ${config.d}} \\quad \\text{(learnable lookup table)}`,
    },
    2: {
      main: "\\hat{v} = \\frac{v^* - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta",
      detail: `\\mu = \\frac{1}{${config.d}} \\sum_i v^*_i, \\quad \\sigma^2 = \\frac{1}{${config.d}} \\sum_i (v^*_i - \\mu)^2`,
    },
    3: {
      main: `x_{\\text{pre}} = \\hat{v} \\cdot E_x \\in \\mathbb{R}^{${config.h} \\times ${config.n}}`,
      detail: `E_x \\in \\mathbb{R}^{${config.d} \\times ${config.total}} \\quad \\text{(${config.d}→${config.total} expansion)}`,
    },
    4: {
      main: "x_{\\text{sparse}} = \\text{ReLU}(x_{\\text{pre}}) = \\max(0, \\, x_{\\text{pre}})",
      detail: `\\text{sparsity} = 1 - \\frac{|\\{i : x_i > 0\\}|}{${config.total}}`,
    },
    5: {
      main: "\\rho_l = \\sum_{\\tau \\leq t} \\, x_\\tau \\cdot x_\\tau^T",
      detail:
        "\\text{Outer-product attention: } O(T) \\text{ rank-1 updates, not } O(T^2)",
    },
    6: {
      main: `a^* = \\text{LN}(\\rho \\cdot v^*) \\in \\mathbb{R}^{${config.d}}`,
      detail:
        "\\text{Context-weighted readout — how much each past token contributes}",
    },
    7: {
      main: `y_{\\text{pre}} = a^* \\cdot E_y \\in \\mathbb{R}^{${config.h} \\times ${config.n}}`,
      detail: `E_y \\in \\mathbb{R}^{${config.d} \\times ${config.total}} \\quad \\text{(attention path expansion)}`,
    },
    8: {
      main: "y_{\\text{sparse}} = \\text{ReLU}(y_{\\text{pre}}) = \\max(0, \\, y_{\\text{pre}})",
      detail: "\\text{Independently sparse from } x_{\\text{sparse}}",
    },
    9: {
      main: "z = x_{\\text{sparse}} \\odot y_{\\text{sparse}}",
      detail:
        "\\text{survival\\_rate} = \\frac{|\\{i : x_i > 0 \\land y_i > 0\\}|}{|\\{i : x_i > 0\\}|}",
    },
    10: {
      main: `\\Delta v^* = z \\cdot D \\in \\mathbb{R}^{${config.d}}`,
      detail: `D \\in \\mathbb{R}^{${config.total} \\times ${config.d}} \\quad \\text{(${config.total}→${config.d} projection)}`,
    },
    11: {
      main: "v_{\\text{out}} = v^* + \\Delta v^*",
      detail: "\\text{Residual connection preserves input information}",
    },
    12: {
      main: "P(b_{t+1}) = \\text{softmax}(v_{\\text{out}} \\cdot W_{\\text{emb}}^T)",
      detail: "\\text{Weight-tied decoder: re-uses embedding matrix}",
    },
  };

  const eq = equations[step];
  if (!eq) return <span className="text-xs text-[#4A5568]">—</span>;

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto py-1">
        <Tex math={eq.main} display />
      </div>
      {eq.detail && (
        <div className="overflow-x-auto py-1 opacity-70">
          <Tex math={eq.detail} display />
        </div>
      )}
    </div>
  );
}

function BlockLiveData({
  step,
  frameData,
  playbackData,
  config,
  predictions,
  currentLayer,
}: {
  step: number;
  frameData?: FrameData;
  playbackData?: PlaybackData;
  config: { d: number; n: number; h: number; total: number };
  predictions: Array<{ byte: number; char: string; prob: number }> | null;
  currentLayer: number;
}) {
  if (!frameData) {
    return (
      <span className="text-xs text-[#4A5568]">
        Run inference to see live values.
      </span>
    );
  }

  switch (step) {
    case 0: // Input Token
      return (
        <div className="space-y-1">
          <Stat label="Character" value={`"${frameData.token_char}"`} />
          <Stat label="Byte value" value={frameData.token_byte} />
          <Stat label="Position" value={frameData.token_idx} />
          <Stat
            label="UTF-8 hex"
            value={`0x${frameData.token_byte.toString(16).toUpperCase().padStart(2, "0")}`}
          />
        </div>
      );

    case 1: {
      // Embedding
      const emb = frameData.embedding;
      if (!emb) return <NoData />;
      return (
        <div className="space-y-2">
          <Stat label="Byte → row index" value={emb.byte_value} />
          <Stat label="‖v*‖" value={emb.norm.toFixed(3)} />
          <Stat label="μ (mean)" value={emb.mean.toFixed(4)} />
          <Stat label="σ (std)" value={emb.std.toFixed(4)} />
          <Stat label="Dimensions" value={config.d} />
          {emb.vector_ds && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Embedding vector (64-bin downsample)
              </span>
              <VectorViz values={emb.vector_ds} />
            </div>
          )}
        </div>
      );
    }

    case 2: {
      // LayerNorm
      const emb = frameData.embedding;
      if (!emb) return <NoData />;
      return (
        <div className="space-y-2">
          <div className="text-xs text-[#8B95A5] mb-1">
            Before → After normalization
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-0.5">
                Raw
              </span>
              <Stat label="‖v‖" value={emb.pre_ln_norm?.toFixed(2) ?? "—"} />
              <Stat label="μ" value={emb.pre_ln_mean?.toFixed(5) ?? "—"} />
              <Stat label="σ" value={emb.pre_ln_std?.toFixed(5) ?? "—"} />
            </div>
            <div>
              <span className="text-[10px] text-[#FCD34D] block mb-0.5">
                Normalized
              </span>
              <Stat label="‖v̂‖" value={emb.norm.toFixed(2)} />
              <Stat label="μ" value={emb.mean.toFixed(5)} />
              <Stat label="σ" value={emb.std.toFixed(5)} />
            </div>
          </div>
          {emb.pre_ln_ds && emb.vector_ds && (
            <div className="space-y-1">
              <span className="text-[10px] text-[#6B7280]">
                Raw → Normalized
              </span>
              <VectorViz values={emb.pre_ln_ds} height={12} />
              <VectorViz values={emb.vector_ds} height={12} />
            </div>
          )}
          {emb.pre_ln_norm != null && (
            <Stat
              label="Norm reduction"
              value={`${((1 - emb.norm / emb.pre_ln_norm) * 100).toFixed(1)}%`}
            />
          )}
        </div>
      );
    }

    case 3: {
      // Linear Dx
      const pre = frameData.x_pre_relu;
      if (!pre) return <NoData />;
      return (
        <div className="space-y-2">
          <Stat
            label="Output dims"
            value={`${config.h} × ${config.n} = ${config.total}`}
          />
          <Stat label="Mean" value={pre.mean.toFixed(4)} />
          <Stat label="Std" value={pre.std.toFixed(4)} />
          <Stat
            label="Range"
            value={`[${pre.min.toFixed(2)}, ${pre.max.toFixed(2)}]`}
          />
          <Stat
            label="Positive"
            value={`${pre.positive_count} / ${pre.total}`}
          />
          <Stat
            label="Positive %"
            value={`${((pre.positive_count / pre.total) * 100).toFixed(1)}%`}
          />
          {pre.histogram && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Pre-activation distribution
              </span>
              <HistogramMini bins={pre.histogram} />
            </div>
          )}
        </div>
      );
    }

    case 4: {
      // ReLU x
      return (
        <div className="space-y-2">
          <Stat
            label="Sparsity"
            value={`${(frameData.x_sparsity * 100).toFixed(2)}%`}
          />
          <Stat
            label="Active neurons"
            value={`${frameData.x_active_count ?? "—"} / ${config.total}`}
          />
          <Stat
            label="Zeroed out"
            value={`${config.total - (frameData.x_active_count ?? 0)}`}
          />
          {/* Per-head breakdown */}
          {frameData.x_active && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Per-head active count
              </span>
              <div className="grid grid-cols-4 gap-1">
                {frameData.x_active.map((h, i) => (
                  <div
                    key={i}
                    className="text-center p-1 rounded"
                    style={{ background: "rgba(239,68,68,0.1)" }}
                  >
                    <div className="text-[10px] text-[#6B7280]">H{i}</div>
                    <div className="text-sm font-mono text-[#FCA5A5]">
                      {h.indices.length}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {frameData.x_top_neurons && frameData.x_top_neurons.length > 0 && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Strongest neurons
              </span>
              <NeuronTable neurons={frameData.x_top_neurons} color="#EF4444" />
            </div>
          )}
        </div>
      );
    }

    case 5: {
      // ρ Memory
      const rhoMatrix = playbackData?.rho_matrices?.[currentLayer];
      return (
        <div className="space-y-2">
          <Stat label="Layer" value={currentLayer} />
          <Stat label="Token position" value={frameData.token_idx} />
          <Stat
            label="Matrix size"
            value={
              rhoMatrix
                ? `${rhoMatrix.length}×${rhoMatrix[0]?.length ?? 0}`
                : "—"
            }
          />
          {frameData.attention_weights &&
            frameData.attention_weights.length > 0 && (
              <div>
                <span className="text-[10px] text-[#6B7280] block mb-1">
                  Attention distribution (head-averaged)
                </span>
                <MiniBarChart
                  data={frameData.attention_weights.map((w) => ({
                    label: w.char === " " ? "␣" : w.char,
                    value: w.weight,
                  }))}
                  color="#22D3EE"
                  height={48}
                />
              </div>
            )}
          {frameData.attention_stats?.top_attended && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Top attended tokens
              </span>
              {frameData.attention_stats.top_attended.map((t, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between py-0.5"
                >
                  <span className="text-xs font-mono text-[#67E8F9]">
                    "{t.char === " " ? "␣" : t.char}" (pos {t.token_idx})
                  </span>
                  <span className="text-xs font-mono text-[#E2E8F0]">
                    {(t.weight * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }

    case 6: {
      // a* Readout
      return (
        <div className="space-y-2">
          <Stat label="‖a*‖" value={frameData.a_star_norm?.toFixed(4) ?? "—"} />
          <Stat
            label="Context"
            value={
              frameData.token_idx === 0
                ? "No past tokens → a*≈0"
                : `${frameData.token_idx} past tokens`
            }
          />
          {frameData.a_star_ds && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                a* vector (64-bin downsample)
              </span>
              <VectorViz values={frameData.a_star_ds} />
            </div>
          )}
        </div>
      );
    }

    case 7: {
      // Linear Dy
      const pre = frameData.y_pre_relu;
      if (!pre) return <NoData />;
      return (
        <div className="space-y-2">
          <Stat
            label="Output dims"
            value={`${config.h} × ${config.n} = ${config.total}`}
          />
          <Stat label="Mean" value={pre.mean.toFixed(4)} />
          <Stat label="Std" value={pre.std.toFixed(4)} />
          <Stat
            label="Range"
            value={`[${pre.min.toFixed(2)}, ${pre.max.toFixed(2)}]`}
          />
          <Stat
            label="Positive"
            value={`${pre.positive_count} / ${pre.total}`}
          />
          <Stat
            label="Positive %"
            value={`${((pre.positive_count / pre.total) * 100).toFixed(1)}%`}
          />
          {pre.histogram && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Pre-activation distribution
              </span>
              <HistogramMini bins={pre.histogram} />
            </div>
          )}
        </div>
      );
    }

    case 8: {
      // ReLU y
      return (
        <div className="space-y-2">
          <Stat
            label="Sparsity"
            value={`${(frameData.y_sparsity * 100).toFixed(2)}%`}
          />
          <Stat
            label="Active neurons"
            value={`${frameData.y_active_count ?? "—"} / ${config.total}`}
          />
          <Stat
            label="Zeroed out"
            value={`${config.total - (frameData.y_active_count ?? 0)}`}
          />
          {frameData.y_active && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Per-head active count
              </span>
              <div className="grid grid-cols-4 gap-1">
                {frameData.y_active.map((h, i) => (
                  <div
                    key={i}
                    className="text-center p-1 rounded"
                    style={{ background: "rgba(16,185,129,0.1)" }}
                  >
                    <div className="text-[10px] text-[#6B7280]">H{i}</div>
                    <div className="text-sm font-mono text-[#6EE7B7]">
                      {h.indices.length}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {frameData.y_top_neurons && frameData.y_top_neurons.length > 0 && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Strongest neurons
              </span>
              <NeuronTable neurons={frameData.y_top_neurons} color="#10B981" />
            </div>
          )}
        </div>
      );
    }

    case 9: {
      // Hadamard
      const g = frameData.gating;
      if (!g) return <NoData />;
      const xTotal = frameData.x_active_count ?? 0;
      const yTotal = frameData.y_active_count ?? 0;
      return (
        <div className="space-y-2">
          <Stat
            label="Survival rate"
            value={`${(g.survival_rate * 100).toFixed(1)}%`}
          />
          <Stat label="Both active (pass gate)" value={g.both} />
          <Stat label="x-only (killed)" value={g.x_only} />
          <Stat label="y-only (killed)" value={g.y_only} />
          {/* Visual Venn-style breakdown */}
          <div>
            <span className="text-[10px] text-[#6B7280] block mb-1">
              Gate breakdown
            </span>
            <div className="flex items-center gap-1 h-5">
              <div
                className="h-full rounded-l"
                style={{
                  width: `${(g.x_only / Math.max(1, xTotal + yTotal)) * 100}%`,
                  background: "#EF4444",
                  opacity: 0.5,
                  minWidth: g.x_only > 0 ? "4px" : "0",
                }}
              />
              <div
                className="h-full"
                style={{
                  width: `${(g.both / Math.max(1, xTotal + yTotal)) * 100}%`,
                  background: "#22D3EE",
                  minWidth: g.both > 0 ? "4px" : "0",
                }}
              />
              <div
                className="h-full rounded-r"
                style={{
                  width: `${(g.y_only / Math.max(1, xTotal + yTotal)) * 100}%`,
                  background: "#10B981",
                  opacity: 0.5,
                  minWidth: g.y_only > 0 ? "4px" : "0",
                }}
              />
            </div>
            <div className="flex justify-between text-[9px] text-[#6B7280] mt-0.5">
              <span style={{ color: "#FCA5A5" }}>x-only</span>
              <span style={{ color: "#67E8F9" }}>both</span>
              <span style={{ color: "#6EE7B7" }}>y-only</span>
            </div>
          </div>
        </div>
      );
    }

    case 10: {
      // Decoder
      return (
        <div className="space-y-2">
          <Stat
            label="‖Δv*‖"
            value={frameData.decoder_norm?.toFixed(4) ?? "—"}
          />
          <Stat
            label="μ (mean)"
            value={frameData.decoder_mean?.toFixed(5) ?? "—"}
          />
          <Stat
            label="σ (std)"
            value={frameData.decoder_std?.toFixed(5) ?? "—"}
          />
          <Stat
            label="Input neurons (gated)"
            value={`${frameData.gating?.both ?? "—"}`}
          />
          <Stat label="Output dims" value={config.d} />
          {frameData.decoder_ds && (
            <div>
              <span className="text-[10px] text-[#6B7280] block mb-1">
                Δv* vector (64-bin downsample)
              </span>
              <VectorViz values={frameData.decoder_ds} />
            </div>
          )}
        </div>
      );
    }

    case 11: {
      // Residual
      const embNorm = frameData.embedding?.norm ?? 0;
      const decNorm = frameData.decoder_norm ?? 0;
      return (
        <div className="space-y-2">
          <Stat label="‖v*‖ (input)" value={embNorm.toFixed(3)} />
          <Stat label="‖Δv*‖ (update)" value={decNorm.toFixed(3)} />
          <Stat
            label="Update/Input ratio"
            value={
              embNorm > 0 ? `${((decNorm / embNorm) * 100).toFixed(1)}%` : "—"
            }
          />
          <div className="text-xs text-[#8B95A5] mt-1">
            The residual connection adds the layer's update to the original
            embedding. A small ratio means this layer makes a subtle refinement;
            a large ratio means a significant transformation.
          </div>
        </div>
      );
    }

    case 12: {
      // Output
      if (!predictions || predictions.length === 0) return <NoData />;
      return (
        <div className="space-y-2">
          <span className="text-[10px] text-[#6B7280] block">
            Top 5 next-byte predictions (softmax)
          </span>
          {predictions.slice(0, 5).map((p, i) => (
            <div key={i} className="flex items-center gap-2">
              <span
                className="w-5 h-5 rounded flex items-center justify-center text-xs font-bold"
                style={{
                  background: i === 0 ? "#7C3AED" : "rgba(255,255,255,0.06)",
                  color: i === 0 ? "#fff" : "#8B95A5",
                }}
              >
                {i + 1}
              </span>
              <span className="text-sm font-mono text-[#E2E8F0] w-10">
                "{p.char}"
              </span>
              <div
                className="flex-1 h-2 rounded-full overflow-hidden"
                style={{ background: "rgba(255,255,255,0.06)" }}
              >
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${p.prob * 100}%`,
                    background: i === 0 ? "#A78BFA" : "#4A5568",
                  }}
                />
              </div>
              <span className="text-xs font-mono text-[#8B95A5] w-16 text-right">
                {(p.prob * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </div>
      );
    }

    default:
      return <NoData />;
  }
}

function BlockInterpretation({ step }: { step: number }) {
  const interpretations: Record<number, string> = {
    0: "BDH uses byte-level tokenization — no subword vocabulary needed. Every UTF-8 byte (0–255) is a token. This means the model sees raw character data and must learn language structure from scratch, making it truly interpretable at the lowest level.",
    1: "The embedding table maps each of the 256 possible byte values to a dense vector. This is the only learnable lookup in the model. The embedding norm and distribution tell you how 'distinct' this byte's representation is from others.",
    2: "LayerNorm re-centers and re-scales the embedding so that each layer receives inputs with stable statistics. Watch how the norm and mean change — this prevents gradient explosion across the 6-layer stack.",
    3: "The linear expansion Dₓ projects from the compact embedding space (D=256) into the massively overcomplete neuron space (N=32,768). This 128× expansion is what enables monosemantic representations — each neuron can specialize.",
    4: "ReLU enforces ~95% sparsity, producing a monosemantic encoding where each surviving neuron corresponds to a specific byte-level feature. The sparsity ratio is the key BDH interpretability metric — higher sparsity = cleaner features.",
    5: "The ρ memory state accumulates outer-product attention updates: ρ += x·xᵀ. This is O(T) per token instead of O(T²), enabling unlimited context. Each row shows how the current token's query attends to all past keys.",
    6: "The a* readout multiplies the accumulated ρ matrix by the current embedding to produce a context-aware vector. When token_idx=0, there are no past tokens so a*≈0. For later tokens, a* captures what the model 'remembers'.",
    7: "Linear Dᵧ expands the attention readout into the same overcomplete neuron space as the x-path. This creates an independent sparse code from the attention-derived signal, which will be gated against the direct-path x-sparse.",
    8: "The y-path ReLU creates an independently sparse activation from the attention pathway. The y sparsity pattern encodes 'what the context suggests should be active' — different from the x-path's 'what the current token activates'.",
    9: "The Hadamard gate x⊙y is the key BDH mechanism: a neuron contributes to the output ONLY if both the direct path (x) AND the attention path (y) agree it should be active. This biological-like AND-gate dramatically increases effective sparsity.",
    10: "The decoder D projects the ultra-sparse gated signal (only ~1-5% of 32K neurons survive) back down to the embedding dimension. This compression forces the model to extract maximum information from the surviving neurons.",
    11: "The residual connection v* + Δv* adds the layer's update to the original signal. This ensures information flows even if a layer makes small updates, and enables gradient flow during training. After all 6 layers, the final v_out encodes the prediction.",
    12: "The output layer re-uses the embedding matrix (weight tying) to convert the final hidden state into a probability distribution over all 256 bytes. The highest-probability byte becomes the model's prediction for what comes next.",
  };

  const text = interpretations[step];
  if (!text) return <span className="text-xs text-[#4A5568]">—</span>;
  return <p className="text-xs text-[#8B95A5] leading-relaxed">{text}</p>;
}

function NoData() {
  return (
    <span className="text-xs text-[#4A5568]">
      No data available for this step. Click a different step or run inference.
    </span>
  );
}

function HistogramMini({
  bins,
}: {
  bins: Array<{ start: number; end: number; count: number }>;
}) {
  const maxCount = Math.max(1, ...bins.map((b) => b.count));
  return (
    <div className="flex items-end gap-px" style={{ height: 32 }}>
      {bins.map((bin, i) => {
        const h = (bin.count / maxCount) * 28;
        const mid = (bin.start + bin.end) / 2;
        return (
          <div
            key={i}
            className="flex-1 min-w-0 rounded-t"
            style={{
              height: `${Math.max(1, h)}px`,
              background: mid >= 0 ? "#F59E0B" : "#3B82F6",
              opacity: 0.8,
            }}
          />
        );
      })}
    </div>
  );
}
