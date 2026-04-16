import { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, RefreshCw, AlertCircle } from "lucide-react";
import { api } from "../utils/api";


interface TokenSparsity {
  token_idx: number;
  char: string;
  x_sparsity: number;
  y_sparsity: number;
  combined_sparsity: number;
  x_active: number;
  y_active: number;
}

interface LayerSparsity {
  layer: number;
  x_sparsity: number;
  y_sparsity: number;
  combined: number;
}

interface TransformerRef {
  activation_rate: number;
  sparsity: number;
  source: string;
  note: string;
}

interface SparsityData {
  input_text: string;
  model_name: string;
  total_neurons: number;
  active_neurons: number;
  overall_sparsity: number;
  per_layer: LayerSparsity[];
  per_token: TokenSparsity[];
  active_indices_sample: number[];
  transformer_reference: TransformerRef;
}


const FALLBACK: SparsityData = (() => {
  const rng = (s: number) => {
    const x = Math.sin(s) * 10000;
    return x - Math.floor(x);
  };
  const arr = new Array(400).fill(-1);
  const activeCount = Math.round(400 * 0.052);
  const positions = new Set<number>();
  let seed = 42;
  while (positions.size < activeCount) {
    positions.add(Math.floor(rng(seed++) * 400));
  }
  positions.forEach((p) => (arr[p] = p));

  return {
    input_text: "(precomputed over 15 sentences)",
    model_name: "french",
    total_neurons: 32768,
    active_neurons: 1687,
    overall_sparsity: 0.9485,
    per_layer: [
      { layer: 0, x_sparsity: 0.9537, y_sparsity: 0.9928, combined: 0.9733 },
      { layer: 1, x_sparsity: 0.957, y_sparsity: 0.9947, combined: 0.9759 },
      { layer: 2, x_sparsity: 0.9528, y_sparsity: 0.9929, combined: 0.9729 },
      { layer: 3, x_sparsity: 0.9462, y_sparsity: 0.9898, combined: 0.968 },
      { layer: 4, x_sparsity: 0.9415, y_sparsity: 0.9869, combined: 0.9642 },
      { layer: 5, x_sparsity: 0.9398, y_sparsity: 0.9853, combined: 0.9626 },
    ],
    per_token: [],
    active_indices_sample: arr,
    transformer_reference: {
      activation_rate: 0.92,
      sparsity: 0.08,
      source: "Anthropic / Elhage et al., 2022-2023",
      note: "Dense transformer MLP layers typically show 80-95% neuron activation rates.",
    },
  };
})();


const DEFAULT_TEXTS: Record<string, string> = {
  french: "Le Parlement européen a adopté la résolution.",
  portuguese: "O Parlamento Europeu adotou a resolução.",
  merged: "The European Parliament adopted the resolution.",
};

export function SparsityPage() {
  const [modelName, setModelName] = useState("french");
  const [inputText, setInputText] = useState(DEFAULT_TEXTS["french"]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [data, setData] = useState<SparsityData | null>(null);
  const [hoveredToken, setHoveredToken] = useState<number | null>(null);
  const [animKey, setAnimKey] = useState(0);

  const handleModelChange = useCallback(
    (newModel: string) => {
      setModelName(newModel);
      // Update text to match language if user hasn't customized it
      const currentDefault = DEFAULT_TEXTS[modelName];
      if (inputText === currentDefault || inputText === "") {
        setInputText(DEFAULT_TEXTS[newModel] ?? DEFAULT_TEXTS["merged"]);
      }
    },
    [modelName, inputText],
  );

  const handleAnalyze = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const r = await api.post("/sparsity/analyze", {
        text: inputText,
        model_name: modelName,
      });
      setData(r.data);
      setIsLive(true);
      setAnimKey((k) => k + 1);
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail || err?.message || "Backend unavailable";
      setError(msg + " — showing precomputed data");
      setData(FALLBACK);
      setIsLive(false);
      setAnimKey((k) => k + 1);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, modelName]);

  const d = data;

  /* Transformer grid — deterministic scatter */
  const transformerGrid = useMemo(() => {
    const arr = new Array(400).fill(true);
    const inactiveCount = Math.round(
      400 * (d?.transformer_reference.sparsity ?? 0.08),
    );
    const rng = (s: number) => {
      const x = Math.sin(s) * 10000;
      return x - Math.floor(x);
    };
    const positions = new Set<number>();
    let seed = 77;
    while (positions.size < inactiveCount) {
      positions.add(Math.floor(rng(seed++) * 400));
    }
    positions.forEach((p) => (arr[p] = false));
    return arr;
  }, [d?.transformer_reference.sparsity]);

  /* Per-layer bar scaling */
  const layerMax = useMemo(() => {
    if (!d) return 1;
    return Math.max(...d.per_layer.map((l) => 1 - l.combined), 0.1);
  }, [d]);

  return (
    <div className="min-h-screen p-8" style={{ background: "#070D12" }}>
      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mb-4 p-3 rounded-lg text-[#8B95A5] flex items-center gap-2 text-sm"
            style={{
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            <AlertCircle size={14} />
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-1">
          <h1 className="text-2xl font-bold text-[#E2E8F0]">
            Sparse Brain <span className="text-[#00C896]">Comparator</span>
          </h1>
          {d && (
            <span
              className={`px-2 py-0.5 rounded text-xs font-medium ${
                isLive ? "text-[#00C896]" : "text-[#6B7280]"
              }`}
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              {isLive ? "Measured" : "Precomputed"}
            </span>
          )}
        </div>
        <p className="text-[#8B95A5] text-sm">
          Compare measured sparsity in BDH with known activation density in
          standard Transformers.
        </p>
      </div>

      {/* Input bar */}
      <div className="card-interactive p-5 mb-6">
        <div className="flex gap-3 items-center">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to analyze..."
            className="input-field flex-1"
            onKeyDown={(e) => e.key === "Enter" && handleAnalyze()}
          />
          <select
            value={modelName}
            onChange={(e) => handleModelChange(e.target.value)}
            className="input-field w-36 text-sm"
          >
            <option value="french">French</option>
            <option value="portuguese">Portuguese</option>
            <option value="merged">Merged</option>
          </select>
          <button
            onClick={handleAnalyze}
            className="btn-primary flex items-center gap-2 whitespace-nowrap"
            disabled={isLoading}
          >
            {isLoading ? (
              <RefreshCw className="animate-spin" size={16} />
            ) : (
              <Zap size={16} />
            )}
            Analyze
          </button>
        </div>
        {isLoading && (
          <p className="text-[#6B7280] text-xs mt-2">
            Running inference through BDH layers…
          </p>
        )}
      </div>

      {/* Side-by-side comparison */}
      <div className="grid md:grid-cols-5 gap-6 mb-6">
        {/* BDH card — primary measurement */}
        <div className="card-interactive p-6 md:col-span-3">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2.5">
              <div>
                <h2 className="text-base font-semibold text-[#E2E8F0]">BDH</h2>
                <p className="text-[#6B7280] text-xs">
                  Post-transformer architecture
                </p>
              </div>
            </div>
            {d && (
              <span
                className="text-[10px] px-2 py-0.5 rounded text-[#00C896]"
                style={{
                  background: "rgba(0,200,150,0.08)",
                  border: "1px solid rgba(0,200,150,0.15)",
                }}
              >
                {isLive ? "live inference" : "precomputed"}
              </span>
            )}
          </div>

          {/* Headline number */}
          <div className="mb-4">
            <div className="flex justify-between items-baseline mb-1.5">
              <span className="text-[#8B95A5] text-sm">Measured sparsity</span>
              <span className="text-[#E2E8F0] font-semibold text-lg tabular-nums">
                {d ? `${(d.overall_sparsity * 100).toFixed(1)}%` : "—"}
                {d && (
                  <span className="text-[#6B7280] text-xs font-normal ml-1.5">
                    neurons silent
                  </span>
                )}
              </span>
            </div>
            <div
              className="h-1.5 rounded-full overflow-hidden"
              style={{ background: "rgba(255,255,255,0.04)" }}
            >
              <motion.div
                key={`bdh-bar-${animKey}`}
                className="h-full rounded-full"
                style={{ background: "rgba(0,200,150,0.65)" }}
                initial={{ width: 0 }}
                animate={{
                  width: d ? `${d.overall_sparsity * 100}%` : "0%",
                }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Neuron grid */}
          <div className="mb-4">
            <p className="text-[#6B7280] text-xs mb-2">
              Neuron activations (400-neuron sample)
            </p>
            <div
              className="grid gap-[2px]"
              style={{ gridTemplateColumns: "repeat(20, 1fr)" }}
            >
              {(d?.active_indices_sample ?? new Array(400).fill(-1)).map(
                (v, i) => {
                  const isActive = v >= 0;
                  return (
                    <motion.div
                      key={`bdh-n-${i}`}
                      className="aspect-square rounded-[2px]"
                      initial={{ opacity: 0 }}
                      animate={{
                        opacity: 1,
                        background: isActive
                          ? "rgba(0,200,150,0.5)"
                          : "rgba(255,255,255,0.025)",
                      }}
                      transition={{
                        delay: isActive ? 0.3 + i * 0.002 : 0.05 + i * 0.001,
                        duration: 0.25,
                      }}
                    />
                  );
                },
              )}
            </div>
          </div>

          {/* Count */}
          <div
            className="text-center p-3 rounded-lg"
            style={{ background: "rgba(255,255,255,0.02)" }}
          >
            <div className="text-xl font-bold text-[#E2E8F0]">
              {d ? d.active_neurons.toLocaleString() : "—"}
            </div>
            <div className="text-[#8B95A5] text-xs">active neurons</div>
            <div className="text-[#6B7280] text-[10px] mt-0.5">
              out of {d?.total_neurons.toLocaleString() ?? "—"}
            </div>
          </div>
        </div>

        {/* Transformer card — reference baseline */}
        <div
          className="md:col-span-2 p-6 rounded-xl"
          style={{
            background: "rgba(255,255,255,0.015)",
            border: "1px dashed rgba(255,255,255,0.08)",
          }}
        >
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2.5">
              <div
                className="w-7 h-7 rounded flex items-center justify-center text-xs text-[#6B7280]"
                style={{ background: "rgba(255,255,255,0.04)" }}
              >
                T
              </div>
              <div>
                <h2 className="text-sm font-semibold text-[#8B95A5]">
                  Dense Transformer
                </h2>
                <p className="text-[#4A5568] text-xs">Reference baseline</p>
              </div>
            </div>
            <span
              className="text-[10px] px-2 py-0.5 rounded text-[#6B7280]"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              reference
            </span>
          </div>

          {/* Headline number */}
          <div className="mb-4">
            <div className="flex justify-between items-baseline mb-1.5">
              <span className="text-[#6B7280] text-sm">Typical sparsity</span>
              <span className="text-[#8B95A5] font-semibold text-base tabular-nums">
                {d
                  ? `${(d.transformer_reference.sparsity * 100).toFixed(0)}%`
                  : "—"}
              </span>
            </div>
            <div
              className="h-1.5 rounded-full overflow-hidden"
              style={{ background: "rgba(255,255,255,0.04)" }}
            >
              <motion.div
                key={`tf-bar-${animKey}`}
                className="h-full rounded-full"
                style={{ background: "rgba(107,114,128,0.5)" }}
                initial={{ width: 0 }}
                animate={{
                  width: d
                    ? `${d.transformer_reference.sparsity * 100}%`
                    : "0%",
                }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Neuron grid */}
          <div className="mb-4">
            <p className="text-[#6B7280] text-xs mb-2">
              Neuron activations (400-neuron sample)
            </p>
            <div
              className="grid gap-[2px]"
              style={{ gridTemplateColumns: "repeat(20, 1fr)" }}
            >
              {transformerGrid.map((active, i) => (
                <motion.div
                  key={`tf-n-${i}`}
                  className="aspect-square rounded-sm"
                  initial={{ opacity: 0 }}
                  animate={{
                    opacity: 1,
                    background: active
                      ? "rgba(107,114,128,0.3)"
                      : "rgba(255,255,255,0.02)",
                  }}
                  transition={{
                    delay: 0.1 + i * 0.001,
                    duration: 0.15,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Count */}
          <div
            className="text-center p-3 rounded-lg"
            style={{ background: "rgba(255,255,255,0.02)" }}
          >
            <div className="text-xl font-bold text-[#E2E8F0]">
              {d
                ? Math.round(
                    d.total_neurons * d.transformer_reference.activation_rate,
                  ).toLocaleString()
                : "—"}
            </div>
            <div className="text-[#8B95A5] text-xs">active neurons (est.)</div>
            <div className="text-[#6B7280] text-[10px] mt-0.5">
              out of {d?.total_neurons.toLocaleString() ?? "—"}
            </div>
          </div>

          {/* Citation */}
          {d && (
            <p className="text-[#4A5568] text-[10px] mt-3 leading-relaxed">
              Source: {d.transformer_reference.source}
            </p>
          )}
        </div>
      </div>

      {/* Per-Layer Breakdown */}
      {d && d.per_layer.length > 0 && (
        <div className="card-interactive p-6 mb-6">
          <h3 className="text-sm font-semibold text-[#E2E8F0] mb-4">
            Per-Layer Activation Rate
          </h3>
          <div className="space-y-2.5">
            {d.per_layer.map((l, i) => {
              const activationRate = 1 - l.combined;
              const barWidth = (activationRate / layerMax) * 100;
              return (
                <div key={l.layer} className="flex items-center gap-3">
                  <span className="text-[#6B7280] text-xs w-14 shrink-0">
                    Layer {l.layer}
                  </span>
                  <div className="flex-1 relative">
                    <div
                      className="h-3 rounded overflow-hidden"
                      style={{ background: "rgba(255,255,255,0.025)" }}
                    >
                      <motion.div
                        key={`layer-${i}-${animKey}`}
                        className="h-full rounded"
                        style={{ background: "rgba(0,200,150,0.55)" }}
                        initial={{ width: 0 }}
                        animate={{ width: `${barWidth}%` }}
                        transition={{
                          delay: 0.3 + i * 0.08,
                          duration: 0.6,
                          ease: "easeOut",
                        }}
                      />
                    </div>
                    {/* Transformer reference line */}
                    <div
                      className="absolute top-0 h-full border-l border-dashed"
                      style={{
                        left: `${
                          (d.transformer_reference.activation_rate / layerMax) *
                          100
                        }%`,
                        borderColor: "rgba(107,114,128,0.5)",
                      }}
                    />
                  </div>
                  <span className="text-[#8B95A5] text-xs w-14 text-right shrink-0">
                    {(activationRate * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
          <div className="flex items-center gap-4 mt-3 text-[10px] text-[#6B7280]">
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-2 rounded-sm"
                style={{ background: "rgba(0,200,150,0.55)" }}
              />
              BDH activation rate
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block w-3 h-0 border-t border-dashed"
                style={{ borderColor: "rgba(107,114,128,0.6)" }}
              />
              Transformer reference (~
              {(d.transformer_reference.activation_rate * 100).toFixed(0)}%)
            </span>
          </div>
        </div>
      )}

      {/* Per-Token Timeline */}
      {d && d.per_token.length > 0 && (
        <div className="card-interactive p-6 mb-6">
          <h3 className="text-sm font-semibold text-[#E2E8F0] mb-1">
            Per-Token Sparsity
          </h3>
          <p className="text-[#6B7280] text-xs mb-4">
            How many neurons fire for each character of the input.
          </p>

          <div className="overflow-x-auto">
            <div
              className="flex items-end gap-[3px]"
              style={{ minHeight: 120 }}
            >
              {d.per_token.map((t, i) => {
                const activationRate = 1 - t.combined_sparsity;
                const maxRate = Math.max(
                  ...d.per_token.map((tk) => 1 - tk.combined_sparsity),
                  0.01,
                );
                const barH = Math.max((activationRate / maxRate) * 100, 2);
                const isHovered = hoveredToken === i;

                return (
                  <div
                    key={i}
                    className="flex flex-col items-center cursor-default"
                    style={{ minWidth: 22 }}
                    onMouseEnter={() => setHoveredToken(i)}
                    onMouseLeave={() => setHoveredToken(null)}
                  >
                    {/* Tooltip */}
                    <AnimatePresence>
                      {isHovered && (
                        <motion.div
                          initial={{ opacity: 0, y: 4 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          className="mb-1 text-center whitespace-nowrap"
                        >
                          <span className="text-[9px] text-[#8B95A5]">
                            {(t.combined_sparsity * 100).toFixed(1)}%
                          </span>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {/* Bar */}
                    <motion.div
                      key={`tok-${i}-${animKey}`}
                      className="w-4 rounded-t"
                      style={{
                        background: isHovered
                          ? "rgba(0,200,150,0.65)"
                          : "rgba(0,200,150,0.4)",
                        opacity: 1,
                      }}
                      initial={{ height: 0 }}
                      animate={{ height: barH }}
                      transition={{
                        delay: 0.4 + i * 0.03,
                        duration: 0.4,
                        ease: "easeOut",
                      }}
                    />

                    {/* Char label */}
                    <span
                      className="mt-1.5 text-[10px] font-mono select-none"
                      style={{
                        color: isHovered ? "#E2E8F0" : "#4A5568",
                      }}
                    >
                      {t.char === " " ? "·" : t.char}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Hovered detail */}
          <AnimatePresence>
            {hoveredToken !== null && d.per_token[hoveredToken] && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mt-3 flex gap-6 text-xs text-[#8B95A5]"
              >
                <span>
                  Token:{" "}
                  <span className="text-[#E2E8F0] font-mono">
                    &quot;{d.per_token[hoveredToken].char}&quot;
                  </span>
                </span>
                <span>
                  Active (x):{" "}
                  <span className="text-[#00C896]">
                    {d.per_token[hoveredToken].x_active.toLocaleString()}
                  </span>
                </span>
                <span>
                  Active (y):{" "}
                  <span className="text-[#00C896]">
                    {d.per_token[hoveredToken].y_active.toLocaleString()}
                  </span>
                </span>
                <span>
                  Sparsity:{" "}
                  <span className="text-white">
                    {(
                      d.per_token[hoveredToken].combined_sparsity * 100
                    ).toFixed(2)}
                    %
                  </span>
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Insight */}
      <div className="card-interactive p-6">
        <h3 className="text-sm font-semibold mb-3 text-[#E2E8F0]">
          Key Insight
        </h3>
        <p className="text-[#8B95A5] text-sm leading-relaxed">
          BDH achieves ~95% sparsity <em>architecturally</em> — through ReLU
          after expansion (D→N), not through regularization or pruning. Each
          active neuron carries meaningful, interpretable signal. Transformer
          MLPs, by contrast, show 80–95% activation rates (Elhage et al., 2022),
          making per-neuron interpretation impractical.
        </p>
        {d && (
          <p className="text-[#4A5568] text-[10px] mt-3">
            Transformer reference: {d.transformer_reference.source} —{" "}
            {d.transformer_reference.note}
          </p>
        )}
      </div>
    </div>
  );
}
