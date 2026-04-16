import { useEffect, useState, useId, useMemo } from "react";
import { motion } from "framer-motion";

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
  attention?: number[];
  attention_stats?: {
    top_attended: Array<{ token_idx: number; char: string; weight: number }>;
  };
  attention_weights?: Array<{
    token_idx: number;
    char: string;
    weight: number;
  }>;
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

interface Props {
  frameData?: FrameData;
  playbackData?: PlaybackData;
  showTooltips?: boolean;
  currentLayer: number;
  isAnimating: boolean;
  currentStep?: number;
  onStepChange?: (step: number) => void;
  selectedBlock?: number | null;
  onBlockClick?: (step: number) => void;
}

const STEPS = [
  { id: 0, name: "Input Token", key: "input" },
  { id: 1, name: "Embedding", key: "embedding" },
  { id: 2, name: "LayerNorm", key: "layernorm" },
  { id: 3, name: "Linear Dₓ", key: "linear_dx" },
  { id: 4, name: "ReLU (x)", key: "relu_x" },
  { id: 5, name: "ρ Memory", key: "rho" },
  { id: 6, name: "a* Readout", key: "a_star" },
  { id: 7, name: "Linear Dᵧ", key: "linear_dy" },
  { id: 8, name: "ReLU (y)", key: "relu_y" },
  { id: 9, name: "Hadamard x⊙y", key: "hadamard" },
  { id: 10, name: "Decoder", key: "decoder" },
  { id: 11, name: "Residual", key: "residual" },
  { id: 12, name: "Output", key: "output" },
] as const;

function divergingColor(v: number, maxAbs: number): string {
  const norm = maxAbs > 0 ? v / maxAbs : 0;
  if (norm < 0) {
    const t = Math.min(1, -norm);
    return `rgb(${Math.round(20 + t * 10)}, ${Math.round(30 + t * 100)}, ${Math.round(60 + t * 195)})`;
  }
  const t = Math.min(1, norm);
  return `rgb(${Math.round(60 + t * 195)}, ${Math.round(30 + t * 60)}, ${Math.round(20)})`;
}

function rhoColor(v: number, maxAbs: number): string {
  if (maxAbs <= 0) return "#0F172A";
  const norm = v / maxAbs;
  if (norm <= 0) {
    const t = Math.min(1, Math.abs(norm));
    return `rgb(${Math.round(15 + t * 5)}, ${Math.round(20 + t * 10)}, ${Math.round(40 + t * 60)})`;
  }
  const t = Math.min(1, norm);
  return `rgb(${Math.round(15 + t * 20)}, ${Math.round(40 + t * 180)}, ${Math.round(60 + t * 195)})`;
}

function activationColor(
  v: number,
  maxVal: number,
  hue: "red" | "green" | "cyan" = "red",
): string {
  if (v <= 0) return "#0F172A";
  const t = Math.min(1, v / (maxVal || 0.001));
  switch (hue) {
    case "red":
      return `rgb(${Math.round(30 + t * 225)}, ${Math.round(15 + t * 50)}, ${Math.round(15 + t * 25)})`;
    case "green":
      return `rgb(${Math.round(15 + t * 25)}, ${Math.round(30 + t * 225)}, ${Math.round(25 + t * 105)})`;
    case "cyan":
      return `rgb(${Math.round(15 + t * 25)}, ${Math.round(30 + t * 190)}, ${Math.round(50 + t * 205)})`;
  }
}

export function BDHArchitectureDiagram({
  frameData,
  playbackData,
  currentLayer,
  isAnimating,
  currentStep: externalStep,
  onStepChange,
  selectedBlock,
  onBlockClick,
}: Props) {
  const [internalStep, setInternalStep] = useState(0);
  const [fillProgress, setFillProgress] = useState(1);
  const uniqueId = useId();
  const currentStep = externalStep ?? internalStep;

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

  // Get the ρ matrix for the current layer
  const rhoMatrix = useMemo(() => {
    if (!playbackData?.rho_matrices || frameData === undefined) return null;
    return playbackData.rho_matrices[currentLayer] ?? null;
  }, [playbackData, currentLayer, frameData]);

  useEffect(() => {
    // Always animate fillProgress 0→1 for gradual data reveal on every step change
    setFillProgress(0);
    const startTime = Date.now();
    const duration = 2000; // 2s gradual reveal
    const raf = { id: 0 };
    const tick = () => {
      const p = Math.min((Date.now() - startTime) / duration, 1);
      setFillProgress(p);
      if (p < 1) raf.id = requestAnimationFrame(tick);
    };
    raf.id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.id);
  }, [currentStep]);

  const isActive = (s: number) => currentStep >= s;
  const isCurrent = (s: number) => currentStep === s;
  const getProgress = (s: number) =>
    currentStep > s ? 1 : currentStep === s ? fillProgress : 0;

  /* ---- Layout constants ---- */
  const W = 900;
  const LX = 30; // left column x
  const LW = 350; // left column width
  const RX = 500; // right column x
  const RW = 360; // right column width
  const CX = W / 2; // center x

  return (
    <div className="flex gap-6">
      {/* LEFT SIDEBAR*/}
      <div className="w-48 flex-shrink-0">
        <div className="glass-card p-3 sticky top-4">
          <h3 className="text-sm font-semibold text-[#CBD5E0] mb-3">
            Architecture Flow
          </h3>
          <div className="space-y-1">
            {STEPS.map((step, idx) => (
              <button
                key={step.id}
                onClick={() =>
                  onStepChange ? onStepChange(idx) : setInternalStep(idx)
                }
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-sm transition-all ${
                  idx < currentStep
                    ? "bg-[#2A7FFF]/20 text-[#2A7FFF]"
                    : idx === currentStep
                      ? "bg-[#2A7FFF] text-white font-semibold"
                      : "bg-white/[0.03] text-[#4A5568] hover:bg-white/[0.08]"
                }`}
              >
                <div
                  className={`w-5 h-5 rounded flex items-center justify-center text-xs font-bold ${
                    idx < currentStep
                      ? "bg-[#2A7FFF] text-white"
                      : idx === currentStep
                        ? "bg-white text-[#2A7FFF]"
                        : "bg-white/10 text-[#8B95A5]"
                  }`}
                >
                  {idx}
                </div>
                <span className="truncate">{step.name}</span>
                {idx === currentStep && isAnimating && (
                  <div className="ml-auto w-8 h-1 bg-white/10 rounded overflow-hidden">
                    <div
                      className="h-full bg-[#2A7FFF] transition-all"
                      style={{ width: `${fillProgress * 100}%` }}
                    />
                  </div>
                )}
              </button>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-white/10">
            <div className="text-sm text-[#8B95A5] mb-1">Layer</div>
            <div className="text-lg font-bold text-[#2A7FFF]">
              {currentLayer + 1} / {playbackData?.num_layers ?? 6}
            </div>
          </div>
        </div>
      </div>

      {/* SVG ARCHITECTURE DIAGRAM*/}
      <div className="flex-1 min-w-0">
        <svg
          viewBox={`0 0 ${W} 1650`}
          className="w-full"
          style={{ maxHeight: "1500px" }}
        >
          <defs>
            <pattern
              id={`grid-${uniqueId}`}
              width="25"
              height="25"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 25 0 L 0 0 0 25"
                fill="none"
                stroke="#1F2937"
                strokeWidth="0.5"
                opacity="0.3"
              />
            </pattern>
            <linearGradient
              id={`gp-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#8B5CF6" />
              <stop offset="100%" stopColor="#6D28D9" />
            </linearGradient>
            <linearGradient
              id={`go-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#F59E0B" />
              <stop offset="100%" stopColor="#D97706" />
            </linearGradient>
            <linearGradient
              id={`gr-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#EF4444" />
              <stop offset="100%" stopColor="#DC2626" />
            </linearGradient>
            <linearGradient
              id={`gc-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#06B6D4" />
              <stop offset="100%" stopColor="#0891B2" />
            </linearGradient>
            <linearGradient
              id={`gg-${uniqueId}`}
              x1="0%"
              y1="0%"
              x2="0%"
              y2="100%"
            >
              <stop offset="0%" stopColor="#10B981" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
          </defs>

          <rect width="100%" height="100%" fill={`url(#grid-${uniqueId})`} />

          {/* TITLE*/}
          <text
            x={CX}
            y="28"
            textAnchor="middle"
            fill="#E5E7EB"
            fontSize="16"
            fontWeight="bold"
          >
            BDH Architecture — Layer {currentLayer + 1}
          </text>
          {frameData && (
            <text
              x={CX}
              y="48"
              textAnchor="middle"
              fill="#9CA3AF"
              fontSize="15"
            >
              Token: "<tspan fill="#F59E0B">{frameData.token_char}</tspan>"
              (byte {frameData.token_byte}, position {frameData.token_idx})
            </text>
          )}

          {/* EMBEDDING (y=65, h=115)*/}
          <g transform={`translate(${CX - 225}, 65)`}>
            <ArchBox
              width={450}
              height={115}
              title="Embedding"
              gradient={`url(#gp-${uniqueId})`}
              isActive={isActive(1)}
              isCurrent={isCurrent(1) && isAnimating}
              progress={getProgress(1)}
              isSelected={selectedBlock === 1}
              onClick={() => onBlockClick?.(1)}
            >
              {isActive(1) && frameData?.embedding ? (
                <g>
                  <text
                    x="25"
                    y="36"
                    fill="#C4B5FD"
                    fontSize="14"
                    fontFamily="monospace"
                  >
                    byte {frameData.embedding.byte_value} → v* ∈ ℝ{config.d}
                  </text>
                  {frameData.embedding.vector_ds ? (
                    <HeatmapStrip
                      values={frameData.embedding.vector_ds}
                      x={25}
                      y={44}
                      width={400}
                      height={26}
                      progress={getProgress(1)}
                    />
                  ) : (
                    <rect
                      x="25"
                      y="44"
                      width="400"
                      height="26"
                      fill="#1F2937"
                      rx="2"
                    />
                  )}
                  <text x="25" y="86" fill="#6B7280" fontSize="11">
                    ← negative (blue)
                  </text>
                  <text
                    x="425"
                    y="86"
                    textAnchor="end"
                    fill="#6B7280"
                    fontSize="11"
                  >
                    positive (red) →
                  </text>
                  <text
                    x="25"
                    y="102"
                    fill="#9CA3AF"
                    fontSize="12"
                    fontFamily="monospace"
                    opacity={Math.min(1, getProgress(1) * 2)}
                  >
                    ‖v*‖={frameData.embedding.norm.toFixed(2)} μ=
                    {frameData.embedding.mean.toFixed(3)} σ=
                    {frameData.embedding.std.toFixed(3)}
                  </text>
                </g>
              ) : (
                <text
                  x="225"
                  y="65"
                  textAnchor="middle"
                  fill="#9CA3AF"
                  fontSize="14"
                >
                  w_t → v* ∈ ℝ^{config.d}
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow x1={CX} y1={180} x2={CX} y2={212} active={isActive(1)} />

          {/* LAYERNORM (y=212, h=55)*/}
          <g transform={`translate(${CX - 205}, 212)`}>
            <ArchBox
              width={410}
              height={55}
              title="LayerNorm"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(2)}
              isCurrent={isCurrent(2) && isAnimating}
              progress={getProgress(2)}
              isSelected={selectedBlock === 2}
              onClick={() => onBlockClick?.(2)}
            >
              {isActive(2) &&
              frameData?.embedding?.pre_ln_ds &&
              frameData.embedding.vector_ds ? (
                <g>
                  {/* Before LN strip */}
                  <text
                    x="18"
                    y="30"
                    fill="#9CA3AF"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    Raw
                  </text>
                  <HeatmapStrip
                    values={frameData.embedding.pre_ln_ds}
                    x={42}
                    y={23}
                    width={148}
                    height={10}
                    progress={getProgress(2)}
                  />
                  {/* After LN strip */}
                  <text
                    x="200"
                    y="30"
                    fill="#FCD34D"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    LN'd
                  </text>
                  <HeatmapStrip
                    values={frameData.embedding.vector_ds}
                    x={230}
                    y={23}
                    width={150}
                    height={10}
                    progress={getProgress(2)}
                  />
                  {/* Stats */}
                  <text
                    x="205"
                    y="48"
                    textAnchor="middle"
                    fill="#6B7280"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    ‖v‖ {frameData.embedding.pre_ln_norm?.toFixed(1)} →{" "}
                    {frameData.embedding.norm.toFixed(1)} | μ{" "}
                    {frameData.embedding.pre_ln_mean?.toFixed(3)} →{" "}
                    {frameData.embedding.mean.toFixed(3)}
                  </text>
                </g>
              ) : (
                <text
                  x="205"
                  y="38"
                  textAnchor="middle"
                  fill="#FCD34D"
                  fontSize="13"
                  fontFamily="monospace"
                >
                  v* = (v* − μ) / σ
                </text>
              )}
            </ArchBox>
          </g>

          {/* BRANCH ARROWS*/}
          <FlowArrow x1={CX} y1={267} x2={CX} y2={280} active={isActive(2)} />
          {/* Left branch to x-path */}
          <path
            d={`M ${CX} 280 L ${CX} 288 L ${LX + LW / 2} 288 L ${LX + LW / 2} 310`}
            stroke={isActive(2) ? "#8B5CF6" : "#374151"}
            strokeWidth="2.5"
            fill="none"
            opacity={isActive(2) ? 1 : 0.15}
          />
          {isActive(2) && (
            <path
              d={`M ${CX} 280 L ${CX} 288 L ${LX + LW / 2} 288 L ${LX + LW / 2} 310`}
              stroke="#C4B5FD"
              strokeWidth="2.5"
              strokeDasharray="5 9"
              fill="none"
              opacity="0.4"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="0.8s"
                repeatCount="indefinite"
              />
            </path>
          )}
          {/* Right branch to attention */}
          <path
            d={`M ${CX} 280 L ${CX} 288 L ${RX + RW / 2} 288 L ${RX + RW / 2} 310`}
            stroke={isActive(5) ? "#06B6D4" : "#374151"}
            strokeWidth="2.5"
            fill="none"
            strokeDasharray="5 3"
            opacity={isActive(5) ? 1 : 0.15}
          />
          {isActive(5) && (
            <path
              d={`M ${CX} 280 L ${CX} 288 L ${RX + RW / 2} 288 L ${RX + RW / 2} 310`}
              stroke="#67E8F9"
              strokeWidth="2.5"
              strokeDasharray="5 9"
              fill="none"
              opacity="0.4"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="0.8s"
                repeatCount="indefinite"
              />
            </path>
          )}
          <text
            x={LX + LW / 2 + 30}
            y="305"
            fill={isActive(2) ? "#9CA3AF" : "#374151"}
            fontSize="12"
            opacity={isActive(2) ? 1 : 0.3}
          >
            to x-path
          </text>
          <text
            x={RX + RW / 2 - 80}
            y="305"
            fill={isActive(5) ? "#67E8F9" : "#374151"}
            fontSize="12"
            opacity={isActive(5) ? 1 : 0.3}
          >
            to attention (v*)
          </text>

          {/* LINEAR Dₓ (y=310, h=125)*/}
          <g transform={`translate(${LX}, 310)`}>
            <ArchBox
              width={LW}
              height={125}
              title="Linear Dₓ"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(3)}
              isCurrent={isCurrent(3) && isAnimating}
              progress={getProgress(3)}
              shape="trapezoid"
              isSelected={selectedBlock === 3}
              onClick={() => onBlockClick?.(3)}
            >
              {isActive(3) && frameData?.x_pre_relu?.histogram ? (
                <g>
                  <text
                    x={LW / 2}
                    y="34"
                    textAnchor="middle"
                    fill="#FCD34D"
                    fontSize="12"
                    fontFamily="monospace"
                  >
                    x = v* @ E ({config.d}→{config.n})
                  </text>
                  <HistogramViz
                    bins={frameData.x_pre_relu.histogram}
                    x={20}
                    y={42}
                    width={LW - 40}
                    height={44}
                    progress={getProgress(3)}
                  />
                  <text
                    x="20"
                    y="98"
                    fill="#9CA3AF"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    range [{frameData.x_pre_relu.min.toFixed(1)},{" "}
                    {frameData.x_pre_relu.max.toFixed(1)}]
                  </text>
                  <text
                    x="20"
                    y="112"
                    fill="#F59E0B"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    {frameData.x_pre_relu.positive_count}/
                    {frameData.x_pre_relu.total} positive (
                    {(
                      (frameData.x_pre_relu.positive_count /
                        frameData.x_pre_relu.total) *
                      100
                    ).toFixed(1)}
                    %)
                  </text>
                </g>
              ) : (
                <text
                  x={LW / 2}
                  y="65"
                  textAnchor="middle"
                  fill="#FCD34D"
                  fontSize="13"
                  fontFamily="monospace"
                >
                  x = v* @ E ({config.d}→{config.n})
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={LX + LW / 2}
            y1={435}
            x2={LX + LW / 2}
            y2={468}
            active={isActive(3)}
          />

          {/* RELU x (y=468, h=160)*/}
          <g transform={`translate(${LX}, 468)`}>
            <ArchBox
              width={LW}
              height={160}
              title="ReLU (x-path)"
              gradient={`url(#gr-${uniqueId})`}
              isActive={isActive(4)}
              isCurrent={isCurrent(4) && isAnimating}
              progress={getProgress(4)}
              isSelected={selectedBlock === 4}
              onClick={() => onBlockClick?.(4)}
            >
              {isActive(4) && frameData?.x_activation_grid ? (
                <g>
                  <text
                    x={LW / 2}
                    y="32"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="12"
                    fontFamily="monospace"
                  >
                    x_sparse = max(0, x)
                  </text>
                  <NeuronGrid
                    grid={frameData.x_activation_grid}
                    x={18}
                    y={40}
                    width={LW - 36}
                    height={52}
                    hue="red"
                    progress={getProgress(4)}
                  />
                  {/* Sparsity bar */}
                  <rect
                    x="18"
                    y="98"
                    width={LW - 36}
                    height="7"
                    fill="#1E293B"
                    rx="3"
                  />
                  <rect
                    x="18"
                    y="98"
                    width={
                      (LW - 36) * (1 - frameData.x_sparsity) * getProgress(4)
                    }
                    height="7"
                    fill="#EF4444"
                    rx="3"
                  />
                  <text
                    x={LW / 2}
                    y="118"
                    textAnchor="middle"
                    fill="#FFF"
                    fontSize="13"
                    fontWeight="bold"
                  >
                    {(frameData.x_sparsity * 100).toFixed(1)}% sparse
                  </text>
                  <text
                    x={LW / 2}
                    y="132"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="11"
                  >
                    {frameData.x_active_count ??
                      Math.round(
                        (1 - frameData.x_sparsity) * config.total,
                      )}{" "}
                    / {config.total} neurons active
                  </text>
                  {frameData.x_top_neurons &&
                    frameData.x_top_neurons.length > 0 && (
                      <text
                        x="18"
                        y="146"
                        fill="#F87171"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        strongest: [H{frameData.x_top_neurons[0]?.head},
                        {frameData.x_top_neurons[0]?.neuron}]=
                        {frameData.x_top_neurons[0]?.value.toFixed(2)}
                      </text>
                    )}
                </g>
              ) : isActive(4) && frameData ? (
                <g>
                  <text
                    x={LW / 2}
                    y="55"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="13"
                  >
                    Sparsity: {(frameData.x_sparsity * 100).toFixed(1)}%
                  </text>
                  <text
                    x={LW / 2}
                    y="75"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="12"
                  >
                    {frameData.x_active_count ?? "?"} / {config.total} active
                  </text>
                </g>
              ) : (
                <text
                  x={LW / 2}
                  y="70"
                  textAnchor="middle"
                  fill="#FCA5A5"
                  fontSize="13"
                >
                  x_sparse = ReLU(x) → ~95% zeros
                </text>
              )}
            </ArchBox>
            {/* x_l label */}
            <text
              x={LW / 2}
              y="178"
              textAnchor="middle"
              fill={isActive(4) ? "#C4B5FD" : "#374151"}
              fontSize="15"
              fontWeight="bold"
              opacity={isActive(4) ? 1 : 0.3}
            >
              x
              <tspan baselineShift="sub" fontSize="9">
                l
              </tspan>
            </text>
          </g>

          {/* Connection: x_l → ρ Memory (x as query) */}
          <path
            d={`M ${LX + LW / 2} 648 L ${LX + LW / 2} 668 L ${RX + RW / 2} 668 L ${RX + RW / 2} 575`}
            stroke={isActive(5) ? "#8B5CF6" : "#374151"}
            strokeWidth="2.5"
            fill="none"
            opacity={isActive(5) ? 1 : 0.1}
          />
          {isActive(5) && (
            <path
              d={`M ${LX + LW / 2} 648 L ${LX + LW / 2} 668 L ${RX + RW / 2} 668 L ${RX + RW / 2} 575`}
              stroke="#C4B5FD"
              strokeWidth="2.5"
              strokeDasharray="5 9"
              fill="none"
              opacity="0.4"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="1s"
                repeatCount="indefinite"
              />
            </path>
          )}
          <text
            x={CX}
            y="682"
            textAnchor="middle"
            fill={isActive(5) ? "#9CA3AF" : "#374151"}
            fontSize="12"
            opacity={isActive(5) ? 1 : 0.2}
          >
            x
            <tspan baselineShift="sub" fontSize="7">
              l
            </tspan>{" "}
            as query into ρ
          </text>

          {/* Connection: x_l down to Hadamard */}
          <path
            d={`M ${LX + LW / 2} 668 L ${LX + LW / 2} 1130 L ${CX - 65} 1130`}
            stroke={isActive(9) ? "#8B5CF6" : "#374151"}
            strokeWidth="2.5"
            fill="none"
            opacity={isActive(9) ? 1 : 0.1}
          />
          {isActive(9) && (
            <path
              d={`M ${LX + LW / 2} 668 L ${LX + LW / 2} 1130 L ${CX - 65} 1130`}
              stroke="#C4B5FD"
              strokeWidth="2.5"
              strokeDasharray="5 9"
              fill="none"
              opacity="0.3"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="1s"
                repeatCount="indefinite"
              />
            </path>
          )}

          {/* ρ MEMORY STATE (y=310, h=260)*/}
          <g transform={`translate(${RX}, 310)`}>
            <ArchBox
              width={RW}
              height={260}
              title="ρ Memory State"
              gradient={`url(#gc-${uniqueId})`}
              isActive={isActive(5)}
              isCurrent={isCurrent(5) && isAnimating}
              progress={getProgress(5)}
              isSelected={selectedBlock === 5}
              onClick={() => onBlockClick?.(5)}
            >
              <text
                x={RW / 2}
                y="30"
                textAnchor="middle"
                fill="#67E8F9"
                fontSize="12"
                fontFamily="monospace"
              >
                ρ = x
                <tspan baselineShift="sub" fontSize="7">
                  sparse
                </tspan>{" "}
                · x
                <tspan baselineShift="sub" fontSize="7">
                  sparse
                </tspan>
                <tspan baselineShift="super" fontSize="7">
                  T
                </tspan>{" "}
                (Q·K
                <tspan baselineShift="super" fontSize="7">
                  T
                </tspan>
                )
              </text>
              {isActive(5) && rhoMatrix && frameData ? (
                <RhoMatrixViz
                  matrix={rhoMatrix}
                  currentT={frameData.token_idx}
                  tokenChars={playbackData?.input_chars ?? []}
                  x={15}
                  y={38}
                  width={RW - 30}
                  height={170}
                  progress={getProgress(5)}
                />
              ) : (
                <text
                  x={RW / 2}
                  y="130"
                  textAnchor="middle"
                  fill="#67E8F9"
                  fontSize="12"
                >
                  Attention score matrix accumulates
                </text>
              )}
              {isActive(5) && rhoMatrix && frameData && (
                <text
                  x={RW / 2}
                  y="248"
                  textAnchor="middle"
                  fill="#6B7280"
                  fontSize="11"
                >
                  Row {frameData.token_idx} = how token "{frameData.token_char}"
                  attends to past · cols = keys · rows = queries
                </text>
              )}
              {!rhoMatrix && (
                <text
                  x={RW / 2}
                  y="150"
                  textAnchor="middle"
                  fill="#475569"
                  fontSize="11"
                >
                  Each row shows attention from token to all past tokens.
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={570}
            x2={RX + RW / 2}
            y2={605}
            active={isActive(5)}
          />

          {/* a* READOUT (y=605, h=95) */}
          <g transform={`translate(${RX}, 605)`}>
            <ArchBox
              width={RW}
              height={95}
              title="a* Readout"
              gradient={`url(#gc-${uniqueId})`}
              isActive={isActive(6)}
              isCurrent={isCurrent(6) && isAnimating}
              progress={getProgress(6)}
              isSelected={selectedBlock === 6}
              onClick={() => onBlockClick?.(6)}
            >
              <text
                x={RW / 2}
                y="30"
                textAnchor="middle"
                fill="#67E8F9"
                fontSize="12"
                fontFamily="monospace"
              >
                a* = LN(ρ · v*) → ℝ
                <tspan baselineShift="super" fontSize="7">
                  {config.d}
                </tspan>
              </text>
              {isActive(6) && frameData?.a_star_ds ? (
                <g>
                  <HeatmapStrip
                    values={frameData.a_star_ds}
                    x={18}
                    y={38}
                    width={RW - 36}
                    height={20}
                    progress={getProgress(6)}
                  />
                  <text
                    x="18"
                    y="72"
                    fill="#9CA3AF"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    ‖a*‖ = {frameData.a_star_norm?.toFixed(3) ?? "—"}
                  </text>
                  <text
                    x={RW - 18}
                    y="72"
                    textAnchor="end"
                    fill="#475569"
                    fontSize="11"
                  >
                    {frameData.token_idx === 0
                      ? "no past tokens → a*≈0"
                      : "weighted sum of past embeddings"}
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="55"
                  textAnchor="middle"
                  fill="#475569"
                  fontSize="11"
                >
                  Output of reading from ρ memory
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={700}
            x2={RX + RW / 2}
            y2={735}
            active={isActive(6)}
          />

          {/* LINEAR Dᵧ (y=735, h=125)*/}
          <g transform={`translate(${RX}, 735)`}>
            <ArchBox
              width={RW}
              height={125}
              title="Linear Dᵧ"
              gradient={`url(#go-${uniqueId})`}
              isActive={isActive(7)}
              isCurrent={isCurrent(7) && isAnimating}
              progress={getProgress(7)}
              shape="trapezoid"
              isSelected={selectedBlock === 7}
              onClick={() => onBlockClick?.(7)}
            >
              {isActive(7) && frameData?.y_pre_relu?.histogram ? (
                <g>
                  <text
                    x={RW / 2}
                    y="34"
                    textAnchor="middle"
                    fill="#FCD34D"
                    fontSize="12"
                    fontFamily="monospace"
                  >
                    y = a* @ Ev ({config.d}→{config.n})
                  </text>
                  <HistogramViz
                    bins={frameData.y_pre_relu.histogram}
                    x={20}
                    y={42}
                    width={RW - 40}
                    height={44}
                    progress={getProgress(7)}
                  />
                  <text
                    x="20"
                    y="98"
                    fill="#9CA3AF"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    range [{frameData.y_pre_relu.min.toFixed(1)},{" "}
                    {frameData.y_pre_relu.max.toFixed(1)}]
                  </text>
                  <text
                    x="20"
                    y="112"
                    fill="#F59E0B"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    {frameData.y_pre_relu.positive_count}/
                    {frameData.y_pre_relu.total} positive (
                    {(
                      (frameData.y_pre_relu.positive_count /
                        frameData.y_pre_relu.total) *
                      100
                    ).toFixed(1)}
                    %)
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="65"
                  textAnchor="middle"
                  fill="#FCD34D"
                  fontSize="13"
                  fontFamily="monospace"
                >
                  y = a* @ Ev ({config.d}→{config.n})
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={RX + RW / 2}
            y1={860}
            x2={RX + RW / 2}
            y2={895}
            active={isActive(7)}
          />

          {/* RELU y (y=895, h=160)*/}
          <g transform={`translate(${RX}, 895)`}>
            <ArchBox
              width={RW}
              height={160}
              title="ReLU (y-path)"
              gradient={`url(#gr-${uniqueId})`}
              isActive={isActive(8)}
              isCurrent={isCurrent(8) && isAnimating}
              progress={getProgress(8)}
              isSelected={selectedBlock === 8}
              onClick={() => onBlockClick?.(8)}
            >
              {isActive(8) && frameData?.y_activation_grid ? (
                <g>
                  <text
                    x={RW / 2}
                    y="32"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="12"
                    fontFamily="monospace"
                  >
                    y_sparse = max(0, y)
                  </text>
                  <NeuronGrid
                    grid={frameData.y_activation_grid}
                    x={18}
                    y={40}
                    width={RW - 36}
                    height={52}
                    hue="green"
                    progress={getProgress(8)}
                  />
                  <rect
                    x="18"
                    y="98"
                    width={RW - 36}
                    height="7"
                    fill="#1E293B"
                    rx="3"
                  />
                  <rect
                    x="18"
                    y="98"
                    width={
                      (RW - 36) * (1 - frameData.y_sparsity) * getProgress(8)
                    }
                    height="7"
                    fill="#10B981"
                    rx="3"
                  />
                  <text
                    x={RW / 2}
                    y="118"
                    textAnchor="middle"
                    fill="#FFF"
                    fontSize="13"
                    fontWeight="bold"
                  >
                    {(frameData.y_sparsity * 100).toFixed(1)}% sparse
                  </text>
                  <text
                    x={RW / 2}
                    y="132"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="11"
                  >
                    {frameData.y_active_count ??
                      Math.round(
                        (1 - frameData.y_sparsity) * config.total,
                      )}{" "}
                    / {config.total} neurons active
                  </text>
                  {frameData.y_top_neurons &&
                    frameData.y_top_neurons.length > 0 && (
                      <text
                        x="18"
                        y="146"
                        fill="#F87171"
                        fontSize="11"
                        fontFamily="monospace"
                      >
                        strongest: [H{frameData.y_top_neurons[0]?.head},
                        {frameData.y_top_neurons[0]?.neuron}]=
                        {frameData.y_top_neurons[0]?.value.toFixed(2)}
                      </text>
                    )}
                </g>
              ) : isActive(8) && frameData ? (
                <g>
                  <text
                    x={RW / 2}
                    y="55"
                    textAnchor="middle"
                    fill="#FCA5A5"
                    fontSize="13"
                  >
                    Sparsity: {(frameData.y_sparsity * 100).toFixed(1)}%
                  </text>
                  <text
                    x={RW / 2}
                    y="75"
                    textAnchor="middle"
                    fill="#9CA3AF"
                    fontSize="12"
                  >
                    {frameData.y_active_count ?? "?"} / {config.total} active
                  </text>
                </g>
              ) : (
                <text
                  x={RW / 2}
                  y="70"
                  textAnchor="middle"
                  fill="#FCA5A5"
                  fontSize="13"
                >
                  y_sparse = ReLU(y)
                </text>
              )}
            </ArchBox>
            {/* y_l label */}
            <text
              x={RW / 2}
              y="178"
              textAnchor="middle"
              fill={isActive(8) ? "#10B981" : "#374151"}
              fontSize="15"
              fontWeight="bold"
              opacity={isActive(8) ? 1 : 0.3}
            >
              y
              <tspan baselineShift="sub" fontSize="9">
                l
              </tspan>
            </text>
          </g>

          {/* Connection: y_l → Hadamard */}
          <path
            d={`M ${RX + RW / 2} 1078 L ${RX + RW / 2} 1130 L ${CX + 65} 1130`}
            stroke={isActive(9) ? "#10B981" : "#374151"}
            strokeWidth="2.5"
            fill="none"
            opacity={isActive(9) ? 1 : 0.1}
          />
          {isActive(9) && (
            <path
              d={`M ${RX + RW / 2} 1078 L ${RX + RW / 2} 1130 L ${CX + 65} 1130`}
              stroke="#6EE7B7"
              strokeWidth="2.5"
              strokeDasharray="5 9"
              fill="none"
              opacity="0.3"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="1s"
                repeatCount="indefinite"
              />
            </path>
          )}

          {/* HADAMARD (y=1105)*/}
          <g
            transform={`translate(${CX - 65}, 1105)`}
            onClick={(e) => {
              e.stopPropagation();
              onBlockClick?.(9);
            }}
            style={{ cursor: onBlockClick ? "pointer" : undefined }}
          >
            <motion.circle
              cx={65}
              cy={30}
              r={30}
              fill={isActive(9) ? "#164E63" : "#1F2937"}
              stroke={
                selectedBlock === 9
                  ? "#00C896"
                  : isActive(9)
                    ? "#06B6D4"
                    : "#374151"
              }
              strokeWidth={selectedBlock === 9 ? 3 : isActive(9) ? 2.5 : 1.2}
              strokeDasharray={
                isActive(9) || selectedBlock === 9 ? undefined : "6 4"
              }
              animate={
                isCurrent(9) && isAnimating ? { scale: [1, 1.05, 1] } : {}
              }
              transition={{ duration: 1, repeat: Infinity }}
            />
            <text
              x="65"
              y="36"
              textAnchor="middle"
              fill={isActive(9) ? "#67E8F9" : "#6B7280"}
              fontSize="24"
            >
              ⊙
            </text>
            <text
              x="65"
              y="75"
              textAnchor="middle"
              fill={isActive(9) ? "#E5E7EB" : "#4B5563"}
              fontSize="13"
              fontWeight="bold"
              opacity={isActive(9) ? 1 : 0.4}
            >
              x ⊙ y gating
            </text>
            {isActive(9) && frameData?.gating && (
              <g opacity={getProgress(9)}>
                <text
                  x="65"
                  y="90"
                  textAnchor="middle"
                  fill="#22D3EE"
                  fontSize="14"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  {(frameData.gating.survival_rate * 100).toFixed(0)}% survive
                </text>
                <text
                  x="65"
                  y="105"
                  textAnchor="middle"
                  fill="#9CA3AF"
                  fontSize="12"
                >
                  {frameData.gating.both} neurons pass both gates
                </text>
                <text
                  x="65"
                  y="118"
                  textAnchor="middle"
                  fill="#6B7280"
                  fontSize="11"
                >
                  x-only: {frameData.gating.x_only} | y-only:{" "}
                  {frameData.gating.y_only}
                </text>
              </g>
            )}
          </g>

          <FlowArrow x1={CX} y1={1230} x2={CX} y2={1265} active={isActive(9)} />

          {/* DECODER D (y=1265, h=180)*/}
          <g transform={`translate(${CX - 225}, 1265)`}>
            <ArchBox
              width={450}
              height={180}
              title="Decoder D"
              gradient={`url(#gg-${uniqueId})`}
              isActive={isActive(10)}
              isCurrent={isCurrent(10) && isAnimating}
              progress={getProgress(10)}
              shape="trapezoidInv"
              isSelected={selectedBlock === 10}
              onClick={() => onBlockClick?.(10)}
            >
              {isActive(10) && frameData?.hadamard_grid ? (
                <g>
                  <text
                    x="225"
                    y="32"
                    textAnchor="middle"
                    fill="#6EE7B7"
                    fontSize="12"
                    fontFamily="monospace"
                  >
                    (x⊙y) gated input → D ({config.n}→{config.d})
                  </text>
                  <text x="25" y="46" fill="#9CA3AF" fontSize="11">
                    Gated neurons per head:
                  </text>
                  <NeuronGrid
                    grid={frameData.hadamard_grid}
                    x={25}
                    y={50}
                    width={400}
                    height={44}
                    hue="cyan"
                    progress={getProgress(10)}
                  />
                  {frameData.decoder_ds ? (
                    <g>
                      <text x="25" y="108" fill="#6EE7B7" fontSize="11">
                        Δv* output vector:
                      </text>
                      <HeatmapStrip
                        values={frameData.decoder_ds}
                        x={25}
                        y={112}
                        width={400}
                        height={20}
                        progress={getProgress(10)}
                      />
                      <text
                        x="25"
                        y="146"
                        fill="#9CA3AF"
                        fontSize="11"
                        fontFamily="monospace"
                        opacity={Math.min(1, getProgress(10) * 2)}
                      >
                        ‖Δv*‖={frameData.decoder_norm?.toFixed(2)} μ=
                        {frameData.decoder_mean?.toFixed(4)} σ=
                        {frameData.decoder_std?.toFixed(4)}
                      </text>
                    </g>
                  ) : (
                    <text
                      x="225"
                      y="126"
                      textAnchor="middle"
                      fill="#6EE7B7"
                      fontSize="12"
                      fontFamily="monospace"
                    >
                      Δv* = (x⊙y) @ D → ℝ^{config.d}
                    </text>
                  )}
                  {frameData.gating && (
                    <text
                      x="225"
                      y="164"
                      textAnchor="middle"
                      fill="#475569"
                      fontSize="11"
                      opacity={Math.min(1, getProgress(10) * 1.5)}
                    >
                      {frameData.gating.both} active neurons → compressed to{" "}
                      {config.d}-dim update
                    </text>
                  )}
                </g>
              ) : (
                <text
                  x="225"
                  y="95"
                  textAnchor="middle"
                  fill="#6EE7B7"
                  fontSize="13"
                  fontFamily="monospace"
                >
                  Δv* = (x⊙y) @ D ({config.n}→{config.d})
                </text>
              )}
            </ArchBox>
          </g>

          <FlowArrow
            x1={CX}
            y1={1445}
            x2={CX}
            y2={1480}
            active={isActive(10)}
          />

          {/* RESIDUAL (y=1480)*/}
          <g
            transform={`translate(${CX - 28}, 1480)`}
            onClick={(e) => {
              e.stopPropagation();
              onBlockClick?.(11);
            }}
            style={{ cursor: onBlockClick ? "pointer" : undefined }}
          >
            <motion.circle
              cx={28}
              cy={22}
              r={22}
              fill={isActive(11) ? "#312E81" : "#1F2937"}
              stroke={
                selectedBlock === 11
                  ? "#00C896"
                  : isActive(11)
                    ? "#8B5CF6"
                    : "#374151"
              }
              strokeWidth={selectedBlock === 11 ? 3 : isActive(11) ? 2.5 : 1.2}
              strokeDasharray={
                isActive(11) || selectedBlock === 11 ? undefined : "6 4"
              }
              animate={
                isCurrent(11) && isAnimating ? { scale: [1, 1.05, 1] } : {}
              }
              transition={{ duration: 1, repeat: Infinity }}
            />
            <text
              x="28"
              y="28"
              textAnchor="middle"
              fill={isActive(11) ? "#C4B5FD" : "#6B7280"}
              fontSize="20"
              fontWeight="bold"
            >
              ⊕
            </text>
            <text
              x="28"
              y="56"
              textAnchor="middle"
              fill={isActive(11) ? "#9CA3AF" : "#374151"}
              fontSize="13"
              opacity={isActive(11) ? 1 : 0.35}
            >
              v* + Δv*
            </text>
          </g>

          {/* Skip connection line */}
          <path
            d={`M ${CX} 280 L ${W - 32} 280 L ${W - 32} 1502 L ${CX + 28} 1502`}
            stroke={isActive(11) ? "#8B5CF6" : "#374151"}
            strokeWidth="2"
            fill="none"
            strokeDasharray="5 3"
            opacity={isActive(11) ? 0.5 : 0.08}
          />
          {isActive(11) && (
            <path
              d={`M ${CX} 280 L ${W - 32} 280 L ${W - 32} 1502 L ${CX + 28} 1502`}
              stroke="#C4B5FD"
              strokeWidth="2"
              strokeDasharray="4 10"
              fill="none"
              opacity="0.3"
            >
              <animate
                attributeName="stroke-dashoffset"
                from="14"
                to="0"
                dur="1.2s"
                repeatCount="indefinite"
              />
            </path>
          )}

          {/* OUTPUT PREDICTIONS*/}
          {isActive(12) && predictions && (
            <g
              transform={`translate(${CX - 270}, 1570)`}
              onClick={(e) => {
                e.stopPropagation();
                onBlockClick?.(12);
              }}
              style={{ cursor: onBlockClick ? "pointer" : undefined }}
            >
              <text
                x="270"
                y="0"
                textAnchor="middle"
                fill="#F3F4F6"
                fontSize="14"
                fontWeight="bold"
              >
                Next Token Predictions
              </text>
              <g transform="translate(0, 16)">
                {predictions.slice(0, 5).map((p, i) => {
                  const cardProgress = Math.max(
                    0,
                    Math.min(1, (getProgress(12) * 5 - i) / 1.5),
                  );
                  const cardW = i === 0 ? 115 : 100;
                  const cardH = i === 0 ? 52 : 42;
                  const gap = 8;
                  // Calculate x offset: first card is wider
                  let xOff = 0;
                  for (let j = 0; j < i; j++)
                    xOff += (j === 0 ? 115 : 100) + gap;
                  return (
                    <g
                      key={i}
                      transform={`translate(${xOff}, ${i === 0 ? 0 : 5})`}
                      opacity={cardProgress}
                    >
                      <rect
                        x="0"
                        y="0"
                        width={cardW}
                        height={cardH}
                        rx="8"
                        fill={i === 0 ? "#7C3AED" : "#1F2937"}
                        stroke={i === 0 ? "#A78BFA" : "#374151"}
                        strokeWidth={i === 0 ? 2 : 1}
                      />
                      {i === 0 && (
                        <text
                          x={cardW / 2}
                          y="-4"
                          textAnchor="middle"
                          fill="#A78BFA"
                          fontSize="11"
                          fontWeight="bold"
                        >
                          TOP PREDICTION
                        </text>
                      )}
                      <text
                        x={cardW / 2}
                        y={i === 0 ? 22 : 18}
                        textAnchor="middle"
                        fill="#FFF"
                        fontSize={i === 0 ? 18 : 14}
                        fontWeight="bold"
                      >
                        "{p.char}"
                      </text>
                      <text
                        x={cardW / 2}
                        y={i === 0 ? 40 : 34}
                        textAnchor="middle"
                        fill={i === 0 ? "#E9D5FF" : "#9CA3AF"}
                        fontSize={i === 0 ? 12 : 10}
                        fontFamily="monospace"
                      >
                        {(p.prob * 100).toFixed(1)}%
                      </text>
                    </g>
                  );
                })}
              </g>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}

function RhoMatrixViz({
  matrix,
  currentT,
  tokenChars,
  x,
  y,
  width,
  height,
  progress = 1,
}: {
  matrix: number[][];
  currentT: number;
  tokenChars: string[];
  x: number;
  y: number;
  width: number;
  height: number;
  progress?: number;
}) {
  const T = Math.min(currentT + 1, matrix.length);
  if (T <= 0) return null;

  // Extract the sub-matrix up to the current token
  const subMatrix = matrix.slice(0, T).map((row) => row.slice(0, T));

  // Find max absolute value for normalizing colors
  const maxAbs = Math.max(
    0.001,
    ...subMatrix.flatMap((row) => row.map((v) => Math.abs(v))),
  );

  // Layout
  const labelW = Math.min(22, width * 0.1);
  const labelH = Math.min(14, height * 0.08);
  const gridW = width - labelW;
  const gridH = height - labelH;
  const cellW = gridW / T;
  const cellH = gridH / T;
  const showLabels = T <= 20 && cellW > 8;

  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Background */}
      <rect
        x={labelW}
        y={0}
        width={gridW}
        height={gridH}
        fill="#0F172A"
        rx="3"
      />

      {/* Matrix cells — reveal progressively row by row */}
      {subMatrix.map((row, i) => {
        // Row-by-row reveal: each row fades in based on progress
        const rowProgress = Math.max(
          0,
          Math.min(1, (T * progress - i) / Math.max(1, T * 0.2)),
        );
        return row.map((val, j) => {
          // Only lower triangle has values (causal mask with diagonal=-1)
          const isAboveDiag = j >= i;
          return (
            <rect
              key={`${i}-${j}`}
              x={labelW + j * cellW + 0.3}
              y={i * cellH + 0.3}
              width={Math.max(1, cellW - 0.6)}
              height={Math.max(1, cellH - 0.6)}
              fill={
                isAboveDiag
                  ? "#0A0F1A"
                  : rowProgress > 0
                    ? rhoColor(val, maxAbs)
                    : "#0A0F1A"
              }
              opacity={isAboveDiag ? 1 : rowProgress}
              rx="0.5"
            />
          );
        });
      })}

      {/* Highlight current token's row (the new update) */}
      {currentT < T && (
        <rect
          x={labelW}
          y={currentT * cellH}
          width={gridW}
          height={cellH}
          fill="none"
          stroke="#F59E0B"
          strokeWidth="1.5"
          rx="1"
        />
      )}

      {/* Row labels (token chars) — left axis */}
      {showLabels &&
        tokenChars.slice(0, T).map((c, i) => (
          <text
            key={`rl-${i}`}
            x={labelW - 3}
            y={i * cellH + cellH / 2 + 3}
            textAnchor="end"
            fill={i === currentT ? "#F59E0B" : "#6B7280"}
            fontSize={Math.min(8, cellH - 1)}
            fontFamily="monospace"
            fontWeight={i === currentT ? "bold" : "normal"}
          >
            {c === " " ? "␣" : c.length > 1 ? "·" : c}
          </text>
        ))}

      {/* Column labels (token chars) — top axis */}
      {showLabels &&
        tokenChars.slice(0, T).map((c, j) => (
          <text
            key={`cl-${j}`}
            x={labelW + j * cellW + cellW / 2}
            y={gridH + labelH - 2}
            textAnchor="middle"
            fill="#6B7280"
            fontSize={Math.min(7, cellW - 1)}
            fontFamily="monospace"
          >
            {c === " " ? "␣" : c.length > 1 ? "·" : c}
          </text>
        ))}

      {/* Border */}
      <rect
        x={labelW}
        y={0}
        width={gridW}
        height={gridH}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="3"
      />

      {/* Legend */}
      <text x={labelW} y={gridH + labelH + 10} fill="#6B7280" fontSize="11">
        dark = low attn
      </text>
      <text
        x={labelW + gridW}
        y={gridH + labelH + 10}
        textAnchor="end"
        fill="#22D3EE"
        fontSize="11"
      >
        bright cyan = high attn
      </text>
      <rect
        x={labelW + gridW - 4}
        y={gridH + labelH + 4}
        width="3"
        height="3"
        fill="#22D3EE"
        rx="0.5"
      />
    </g>
  );
}

function HeatmapStrip({
  values,
  x,
  y,
  width,
  height,
  progress = 1,
}: {
  values: number[];
  x: number;
  y: number;
  width: number;
  height: number;
  progress?: number;
}) {
  const cellW = width / values.length;
  const maxAbs = Math.max(0.001, ...values.map((v) => Math.abs(v)));
  const revealCount = Math.ceil(values.length * Math.min(1, progress));
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width={width} height={height} fill="#0F172A" rx="2" />
      {values.slice(0, revealCount).map((v, i) => (
        <rect
          key={i}
          x={i * cellW}
          width={cellW}
          height={height}
          fill={divergingColor(v, maxAbs)}
          rx="0.5"
        />
      ))}
      {/* Sweep cursor at the reveal front */}
      {progress < 1 && revealCount > 0 && (
        <rect
          x={revealCount * cellW - 2}
          y={0}
          width={3}
          height={height}
          fill="#A78BFA"
          opacity={0.7}
          rx="1"
        />
      )}
      <rect
        width={width}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
    </g>
  );
}

function HistogramViz({
  bins,
  x,
  y,
  width,
  height,
  progress = 1,
}: {
  bins: Array<{ start: number; end: number; count: number }>;
  x: number;
  y: number;
  width: number;
  height: number;
  progress?: number;
}) {
  const maxCount = Math.max(1, ...bins.map((b) => b.count));
  const barW = width / bins.length;

  // Find the precise zero-crossing pixel position
  const straddleIdx = bins.findIndex((b) => b.start < 0 && b.end > 0);
  const firstPositiveIdx = bins.findIndex((b) => b.start >= 0);
  // Interpolate zero within the straddling bin, or use first positive bin edge
  const zeroPixel =
    straddleIdx >= 0
      ? (straddleIdx +
          -bins[straddleIdx].start /
            (bins[straddleIdx].end - bins[straddleIdx].start)) *
        barW
      : firstPositiveIdx >= 0
        ? firstPositiveIdx * barW
        : -1;

  // Ease-out for smoother bar growth
  const easedProgress = 1 - Math.pow(1 - Math.min(1, progress), 3);

  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width={width} height={height} fill="#0F172A" rx="2" />
      {bins.map((bin, i) => {
        const fullBarH = (bin.count / maxCount) * (height - 2);
        const barH = fullBarH * easedProgress;
        // Color by midpoint: bins centered in positive range → orange
        const midpoint = (bin.start + bin.end) / 2;
        const isPositive = midpoint >= 0;
        return (
          <rect
            key={i}
            x={i * barW + 0.5}
            y={height - barH - 1}
            width={barW - 1}
            height={barH}
            fill={isPositive ? "#F59E0B" : "#3B82F6"}
            opacity={0.85}
            rx="0.5"
          />
        );
      })}
      {zeroPixel >= 0 && (
        <line
          x1={zeroPixel}
          y1={0}
          x2={zeroPixel}
          y2={height}
          stroke="#EF4444"
          strokeWidth="1.5"
          strokeDasharray="2 2"
        />
      )}
      <rect
        width={width}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
    </g>
  );
}

function NeuronGrid({
  grid,
  x,
  y,
  width,
  height,
  hue = "red",
  progress = 1,
}: {
  grid: number[][];
  x: number;
  y: number;
  width: number;
  height: number;
  hue?: "red" | "green" | "cyan";
  progress?: number;
}) {
  const numHeads = grid.length;
  const bins = grid[0]?.length || 0;
  if (bins === 0) return null;
  const maxVal = Math.max(0.001, ...grid.flat());
  const labelW = 20;
  const cellW = (width - labelW) / bins;
  const cellH = height / numHeads;
  const totalCells = numHeads * bins;
  // Cascade wavefront: cells illuminate in a sweep
  const wavefront = totalCells * Math.min(1, progress);
  const edgeWidth = Math.max(3, totalCells * 0.12);
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect
        x={labelW}
        width={width - labelW}
        height={height}
        fill="#0F172A"
        rx="2"
      />
      {grid.map((row, h) => (
        <g key={h}>
          <text
            x="0"
            y={h * cellH + cellH / 2 + 3}
            fill="#9CA3AF"
            fontSize="11"
            fontFamily="monospace"
            opacity={progress >= (h + 1) / numHeads ? 1 : 0.3}
          >
            H{h}
          </text>
          {row.map((val, b) => {
            const cellIdx = h * bins + b;
            const cellOpacity = Math.max(
              0,
              Math.min(1, (wavefront - cellIdx) / edgeWidth),
            );
            return (
              <rect
                key={b}
                x={labelW + b * cellW}
                y={h * cellH}
                width={Math.max(0.5, cellW - 0.3)}
                height={cellH - 0.5}
                fill={
                  cellOpacity > 0
                    ? activationColor(val, maxVal, hue)
                    : "#0F172A"
                }
                opacity={cellOpacity}
              />
            );
          })}
        </g>
      ))}
      <rect
        x={labelW}
        width={width - labelW}
        height={height}
        fill="none"
        stroke="#374151"
        strokeWidth="0.5"
        rx="2"
      />
    </g>
  );
}

interface ArchBoxProps {
  width: number;
  height: number;
  title: string;
  gradient: string;
  isActive: boolean;
  isCurrent: boolean;
  progress: number;
  shape?: "rect" | "trapezoid" | "trapezoidInv";
  children?: React.ReactNode;
  isSelected?: boolean;
  onClick?: () => void;
}

function ArchBox({
  width,
  height,
  title,
  gradient,
  isActive,
  isCurrent,
  progress,
  shape = "rect",
  children,
  isSelected,
  onClick,
}: ArchBoxProps) {
  const offset = 15;
  const getPath = () => {
    switch (shape) {
      case "trapezoid":
        return `M 0 0 L ${width} 0 L ${width - offset} ${height} L ${offset} ${height} Z`;
      case "trapezoidInv":
        return `M ${offset} 0 L ${width - offset} 0 L ${width} ${height} L 0 ${height} Z`;
      default:
        return `M 0 0 L ${width} 0 L ${width} ${height} L 0 ${height} Z`;
    }
  };
  const fillHeight = height * progress;
  const fillY = height - fillHeight;
  const clipId = `clip-${title.replace(/\s+/g, "-")}-${Math.random().toString(36).substr(2, 5)}`;

  return (
    <g
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      style={{ cursor: onClick ? "pointer" : undefined }}
    >
      <path d={getPath()} fill="#111827" stroke="#1F2937" strokeWidth="1" />
      <defs>
        <clipPath id={clipId}>
          <rect x="0" y={fillY} width={width} height={fillHeight} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        <motion.path
          d={getPath()}
          fill={isActive ? gradient : "#1F2937"}
          opacity={isActive ? 0.3 : 0}
          animate={isCurrent ? { opacity: [0.2, 0.4, 0.2] } : {}}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
      </g>
      <path
        d={getPath()}
        fill="none"
        stroke={isSelected ? "#00C896" : isActive ? "#8B5CF6" : "#374151"}
        strokeWidth={isSelected ? 3 : isActive ? 2 : 1}
        strokeDasharray={isActive || isSelected ? undefined : "6 4"}
      />
      {/* Selected glow */}
      {isSelected && (
        <path
          d={getPath()}
          fill="none"
          stroke="#00C896"
          strokeWidth="1"
          opacity="0.3"
          strokeDasharray="4 4"
        >
          <animate
            attributeName="stroke-dashoffset"
            from="8"
            to="0"
            dur="0.6s"
            repeatCount="indefinite"
          />
        </path>
      )}
      {/* Title bar */}
      <rect
        x="1"
        y="1"
        width={width - 2}
        height="26"
        fill="#0D1117"
        opacity="0.7"
        rx="1"
      />
      <line
        x1="1"
        y1="27"
        x2={width - 1}
        y2="27"
        stroke={isSelected ? "#00C896" : isActive ? "#6D28D9" : "#374151"}
        strokeWidth="1"
        opacity={isSelected ? 0.8 : isActive ? 0.6 : 0.3}
      />
      {/* Computing shimmer — animated scan line when current step */}
      {isCurrent && (
        <g>
          <rect
            x="2"
            y="28"
            width={width - 4}
            height="3"
            rx="1.5"
            fill="#A78BFA"
            opacity="0.7"
          >
            <animate
              attributeName="y"
              from="28"
              to={String(height - 3)}
              dur="1.2s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.7;0.2;0.7"
              dur="1.2s"
              repeatCount="indefinite"
            />
          </rect>
          <rect
            x="0"
            y="0"
            width={width}
            height={height}
            fill="#8B5CF6"
            opacity="0.05"
            rx="4"
          >
            <animate
              attributeName="opacity"
              values="0.02;0.08;0.02"
              dur="2s"
              repeatCount="indefinite"
            />
          </rect>
        </g>
      )}
      <text
        x={width / 2}
        y="19"
        textAnchor="middle"
        fill={isSelected ? "#00C896" : isActive ? "#F3F4F6" : "#6B7280"}
        fontSize="15"
        fontWeight="bold"
      >
        {title}
      </text>
      <g opacity={isActive ? 1 : 0.35}>{children}</g>
    </g>
  );
}

function FlowArrow({
  x1,
  y1,
  x2,
  y2,
  active,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  active: boolean;
}) {
  // Arrowhead triangle (pointing in direction of travel)
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const headLen = 7;
  const headW = 4.5;
  const tipX = x2;
  const tipY = y2;
  const baseX = tipX - ux * headLen;
  const baseY = tipY - uy * headLen;
  const lx = baseX + uy * headW;
  const ly = baseY - ux * headW;
  const rx = baseX - uy * headW;
  const ry = baseY + ux * headW;

  return (
    <g>
      <line
        x1={x1}
        y1={y1}
        x2={baseX}
        y2={baseY}
        stroke={active ? "#8B5CF6" : "#374151"}
        strokeWidth={active ? 2.5 : 1.2}
        opacity={active ? 1 : 0.25}
      />
      <polygon
        points={`${tipX},${tipY} ${lx},${ly} ${rx},${ry}`}
        fill={active ? "#8B5CF6" : "#374151"}
        opacity={active ? 1 : 0.25}
      />
      {active && (
        <>
          <line
            x1={x1}
            y1={y1}
            x2={baseX}
            y2={baseY}
            stroke="#C4B5FD"
            strokeWidth="2.5"
            strokeDasharray="5 9"
            opacity="0.5"
          >
            <animate
              attributeName="stroke-dashoffset"
              from="14"
              to="0"
              dur="0.9s"
              repeatCount="indefinite"
            />
          </line>
          <circle r="3.5" fill="#A78BFA" opacity="0.85">
            <animate
              attributeName="cx"
              from={String(x1)}
              to={String(x2)}
              dur="1s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="cy"
              from={String(y1)}
              to={String(y2)}
              dur="1s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="r"
              values="2.5;4.5;2.5"
              dur="1s"
              repeatCount="indefinite"
            />
            <animate
              attributeName="opacity"
              values="0.9;0.3;0.9"
              dur="1s"
              repeatCount="indefinite"
            />
          </circle>
        </>
      )}
    </g>
  );
}
