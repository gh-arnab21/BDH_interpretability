import React, { useState, useMemo, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

interface StepData {
  id: number;
  title: string;
  subtitle: string;
  difficulty: "Easy" | "Medium" | "Hard";
  description: string[];
  theory: { heading: string; content: string }[];
  keyInsight: string;
  code: string;
}

const STEPS: StepData[] = [
  {
    id: 1,
    title: "Byte Embedding",
    subtitle: "Raw bytes to continuous vectors",
    difficulty: "Easy",
    description: [
      "BDH reads raw UTF-8 bytes — no tokenizer, no subword encoding. Each byte (0–255) maps to a learnable 192-dimensional vector through an embedding lookup table.",
      "The input tensor transforms from (B, T) → (B, T, D), then gets unsqueezed to (B, 1, T, D) so every attention head receives the same initial representation.",
      'The character "é" becomes two bytes [195, 169], giving the model fine-grained access to every character in every language without any tokenizer bias.',
    ],
    theory: [
      {
        heading: "Why bytes over tokens?",
        content:
          "Tokenizers like BPE create a fixed vocabulary from training data, biasing the model toward languages and patterns seen during tokenizer training. Byte-level input is universal — every UTF-8 string, every language, every symbol is already supported.",
      },
      {
        heading: "Embedding dimensions",
        content:
          "With vocab_size = 256 and n_embd = 192, the embedding table is a 256 × 192 matrix. Each of the 256 possible byte values gets its own unique 192-dimensional vector, learned during training.",
      },
      {
        heading: "Multi-head broadcast",
        content:
          "After embedding, the tensor is unsqueezed from (B,T,D) to (B,1,T,D). The size-1 dimension broadcasts across all attention heads, so every head starts from the same embedding before applying its own encoder.",
      },
    ],
    keyInsight:
      "Byte-level embedding means BDH can process any language, any script, any encoding — without needing a language-specific tokenizer.",
    code: `self.embed = nn.Embedding(256, D)   # D = 192
self.pos_emb = nn.Embedding(4096, D)

# In forward():
pos = torch.arange(0, T, device=idx.device)
x = self.embed(idx) + self.pos_emb(pos)
x = x.unsqueeze(1)        # → (B, 1, T, D)
x = self.ln(x)            # Initial LayerNorm`,
  },
  {
    id: 2,
    title: "Sparse Encoding",
    subtitle: "D→N expansion + ReLU sparsification",
    difficulty: "Medium",
    description: [
      "Each attention head owns an encoder matrix that projects from D = 192 to N = 3,072 neuron dimensions — a 16× expansion into an overcomplete representation.",
      "After expansion, ReLU zeros out all negative activations. This typically leaves only ~5% of neurons active (sparsity ≈ 95%), creating a sparse distributed code where each neuron can specialize.",
      "Some neurons learn to fire for French text, others for numbers, others for punctuation. The sparsity enforces clean separation between concepts.",
    ],
    theory: [
      {
        heading: "Overcomplete representations",
        content:
          "With N >> D, there are far more neurons than input dimensions. This overcomplete basis lets the model represent inputs as combinations of many specific features rather than few entangled ones.",
      },
      {
        heading: "ReLU as a sparsifier",
        content:
          "ReLU(x) = max(0, x) sets all negative values to zero. In practice, roughly 95% of neurons end up at zero for any given input. The model learns which neurons should fire for which patterns through backpropagation.",
      },
      {
        heading: "Sparsity enables interpretability",
        content:
          'Dense representations entangle features — one neuron might encode "French + noun + capitalized." Sparse overcomplete representations let each neuron encode a single concept, making the model naturally interpretable.',
      },
    ],
    keyInsight:
      "The D→N expansion with ReLU is where interpretability begins — each of the 3,072 neurons per head can specialize for one concept.",
    code: `# Encoder matrix per head: (nh, D, N) = (4, 192, 3072)
enc = self.decoder_x  # shared across layers

# Project to neuron space
x_latent = x @ enc          # (B, nh, T, N)

# Sparsify — ~95% of activations become zero
x_sparse = F.relu(x_latent) # (B, nh, T, N)`,
  },
  {
    id: 3,
    title: "Rotary Position Embedding",
    subtitle: "Position-dependent rotation in neuron space",
    difficulty: "Medium",
    description: [
      "RoPE encodes positional information by rotating pairs of dimensions in the sparse representation. Each dimension pair rotates at a different frequency, creating a unique position signature for every token.",
      "The frequency for dimension i is: freq_i = 1 / (θ^(i/N)) where θ = 2¹⁶. Lower dimensions oscillate quickly (capturing local patterns), higher dimensions oscillate slowly (capturing global structure).",
      "When computing attention, the relative position between two tokens is captured entirely by the angle between their rotated vectors — no additional position parameters needed.",
    ],
    theory: [
      {
        heading: "Rotation mechanics",
        content:
          "For a token at position t, each pair of dimensions (2i, 2i+1) is rotated by angle t × freq_i: v·cos(θ) + v_rot·sin(θ). This preserves vector magnitude while encoding position.",
      },
      {
        heading: "Relative position from rotation",
        content:
          "When computing Q·Kᵀ, the angle between rotated Q at position p and rotated K at position q depends only on (p − q). This gives position-aware attention without absolute position embeddings.",
      },
      {
        heading: "Frequency spectrum",
        content:
          "With N = 3,072 dimensions and θ = 65,536, the frequencies span many octaves. High-frequency components capture local character-level proximity; low-frequency components capture long-range document structure.",
      },
    ],
    keyInsight:
      "RoPE gives BDH position awareness through rotation — no additional parameters. Relative position is encoded in the angle between any two token vectors.",
    code: `def get_freqs(n, theta, dtype):
    return 1.0 / (theta ** (arange(0, n) / n)) / (2 * pi)

# Position-dependent phases
r_phases = arange(0, T).view(1,1,-1,1) * self.freqs

# Apply rotation
QR = self.rope(r_phases, Q)   # rotated queries
KR = QR                        # self-attention`,
  },
  {
    id: 4,
    title: "Linear Attention",
    subtitle: "Causal co-activation in sparse space",
    difficulty: "Hard",
    description: [
      "BDH computes attention directly in the sparse neuron space. Rotated queries and keys produce attention scores: scores = QR · KRᵀ, then a causal mask ensures each token only looks at the past.",
      "Unlike standard transformers, there is no softmax. The raw co-activation scores are directly interpretable — they represent how strongly two sparse activation patterns overlap.",
      'Values V stay in embedding space (D = 192), so the attention output captures "what context contributes" in the original representation.',
    ],
    theory: [
      {
        heading: "No softmax — raw co-activation",
        content:
          "Standard transformers normalize attention into a probability distribution with softmax. BDH skips this: scores = (QR @ KR.T).tril(). The raw scores are Hebbian co-activations — when two neurons fire together, the score is high.",
      },
      {
        heading: "Causal masking",
        content:
          ".tril(diagonal=−1) masks the upper triangle to zero, preventing future-token leakage. Position t can only attend to positions 0 through t−1, making BDH autoregressive for generation.",
      },
      {
        heading: "Effective linear complexity",
        content:
          "Because the sparse representations have ~95% zeros, the effective attention computation is far cheaper than dense O(T²). Only the ~5% non-zero neurons contribute meaningfully to dot products.",
      },
    ],
    keyInsight:
      'The attention matrix IS a Hebbian co-activation map — "neurons that fire together, wire together" implemented as linear algebra, with no softmax to obscure the raw signals.',
    code: `# In Attention.forward():
scores = (QR @ KR.mT)        # (B, nh, T, T)
scores = scores.tril(-1)     # causal mask

# V is in embedding space, not neuron space
output = scores @ V           # (B, nh, T, D)`,
  },
  {
    id: 5,
    title: "Value Encoding",
    subtitle: "Second sparse path for attended context",
    difficulty: "Medium",
    description: [
      "The attention output yKV lives in embedding space (D = 192). A second encoder matrix projects it back to sparse neuron space: D → N = 3,072 per head.",
      'A second ReLU creates another sparse representation. This y_sparse captures "which neurons are relevant based on what attention gathered from context."',
      'Now we have two independently sparse vectors: x_sparse ("what features this token has") and y_sparse ("what features the context provides").',
    ],
    theory: [
      {
        heading: "Why a second encoding?",
        content:
          "x_sparse encodes the token itself. y_sparse encodes its context. These carry fundamentally different information and must be independently sparsified so the gating step can combine them meaningfully.",
      },
      {
        heading: "Separate encoder matrices",
        content:
          "enc_v is a distinct parameter from enc. This allows the value path to learn different sparse features than the input path. The two encoders develop complementary, specialized representations.",
      },
      {
        heading: "Dual sparsity",
        content:
          "After value encoding, we have two independent sparse vectors. A neuron active in x_sparse may be silent in y_sparse, and vice versa. Their intersection (next step) creates extremely selective activation.",
      },
    ],
    keyInsight:
      'The second encoder creates a parallel sparse representation from context: x_sparse says "what I am," y_sparse says "what my neighbors say I should produce."',
    code: `# After attention + LayerNorm
yKV = self.ln(yKV)            # (B, nh, T, D)

# Second encoding: D → N
y_latent = yKV @ enc_v         # (B, nh, T, N)
y_sparse = F.relu(y_latent)    # second sparsification`,
  },
  {
    id: 6,
    title: "Sparse Gating",
    subtitle: "Double sparsity through element-wise multiplication",
    difficulty: "Easy",
    description: [
      "The core operation: xy = x_sparse × y_sparse. Element-wise multiplication means a neuron must be active in BOTH the input encoding AND the value encoding to survive.",
      "If x_sparse[i] = 0.8 but y_sparse[i] = 0, then xy[i] = 0 — the neuron is silenced. This creates dramatically higher sparsity than either path alone.",
      "In V2, a persistent ρ buffer modulates the gate: xy × (1 + ρ). The ρ is an exponential moving average of historical gate activity — frequently co-active neurons get amplified over time.",
    ],
    theory: [
      {
        heading: "Gating as feature selection",
        content:
          "Multiplication acts as a soft AND gate: output is non-zero only where both inputs agree. This ensures only contextually-appropriate features pass to the decoder. After gating, ~99% of neurons are zero.",
      },
      {
        heading: "ρ buffer — Hebbian memory (V2)",
        content:
          "ρ accumulates which neurons have historically been co-active: ρ ← decay · ρ + (1−decay) · mean(gate). The (1+ρ) modulation amplifies frequently-used pathways — a form of synaptic long-term potentiation.",
      },
      {
        heading: "Interpretability from gating",
        content:
          "After gating, surviving neurons are doubly validated: the token activates them AND the context confirms them. This creates extremely clean, monosemantic activations where each active neuron has a clear, singular meaning.",
      },
    ],
    keyInsight:
      'Gating is BDH\'s key mechanism — requiring agreement between "what I am" and "what context says" lets only the most relevant neurons survive.',
    code: `# Element-wise gating
xy_sparse = x_sparse * y_sparse   # (B, nh, T, N)

# V2: ρ modulation (Hebbian memory)
if per_layer_encoders:
    rho = self.rho[layer_idx]      # (nh, N)
    xy_sparse *= (1.0 + rho)

xy_sparse = self.drop(xy_sparse)`,
  },
  {
    id: 7,
    title: "Decode + Residual",
    subtitle: "N→D compression and skip connection",
    difficulty: "Medium",
    description: [
      "The gated sparse representation is reshaped by concatenating all heads: (B, nh, T, N) → (B, 1, T, nh×N). Then an encoder matrix projects from nh × N = 12,288 back down to D = 192.",
      "LayerNorm normalizes the decoded output, and a residual connection adds it to the original input: x = LN(x + y). The layer only needs to learn what to ADD to the representation.",
      "This output becomes the input to the next layer, and the encode → attend → gate → decode cycle repeats.",
    ],
    theory: [
      {
        heading: "Head concatenation",
        content:
          "Each of the 4 heads independently processes N = 3,072 neurons. Reshaping from (nh, T, N) to (T, nh×N) concatenates all heads' sparse outputs into one 12,288-dimensional vector.",
      },
      {
        heading: "Decoder projection",
        content:
          "The encoder matrix (nh×N, D) = (12288, 192) compresses the wide sparse code back to embedding space. The model learns which combinations of sparse activations should produce which embedding updates.",
      },
      {
        heading: "Residual learning",
        content:
          "x = LN(x + y) means the layer learns a residual: what to add. Without the skip connection, gradients vanish in deep networks. With it, information flows freely and each layer can focus on refinement.",
      },
    ],
    keyInsight:
      "The encoder reverses the expansion, compressing 12,288 sparse dimensions back to 192 dense dimensions. The residual connection preserves the original signal.",
    code: `# Reshape: (B, nh, T, N) → (B, 1, T, nh*N)
concat = xy_sparse.transpose(1, 2)
           .reshape(B, 1, T, N * nh)

# Decode: nh*N → D
yMLP = concat @ dec    # (B, 1, T, D)
y = self.ln(yMLP)

# Residual connection
x = self.ln(x + y)`,
  },
  {
    id: 8,
    title: "Full BDH Layer",
    subtitle: "Complete processing pipeline",
    difficulty: "Hard",
    description: [
      "A complete BDH layer chains all previous steps: Encode → Sparsify → Attend → Value-Encode → Gate → Decode → Residual. This entire cycle repeats n_layer = 6 times.",
      "After all layers, the output is squeezed back to (B, T, D) and projected through lm_head (D → vocab_size = 256) to predict the next byte.",
      "Cross-entropy loss between predictions and targets drives learning — encoders, decoders, attention, and embeddings all co-adapt through backpropagation.",
    ],
    theory: [
      {
        heading: "Layer specialization",
        content:
          "With 6 stacked layers, each processes progressively abstract representations. Early layers tend to capture character-level patterns (byte digraphs, encoding sequences). Later layers capture semantic patterns (word identity, language detection).",
      },
      {
        heading: "Parameter count",
        content:
          "Shared weights: decoder_x (4 × 192 × 3072 ≈ 2.4M), decoder_y (same ≈ 2.4M), encoder (12288 × 192 ≈ 2.4M), plus pos_emb and lm_head. Total: ~7.96M parameters — extremely compact.",
      },
      {
        heading: "Output projection",
        content:
          "lm_head (D=192, vocab=256) maps the final representation to byte probabilities. Training minimizes cross-entropy: −log P(correct_byte), sending gradients through all layers simultaneously.",
      },
    ],
    keyInsight:
      "BDH's power comes from stacking sparse layers. Each layer is individually interpretable — we can read which neurons fire and why — yet together they form a capable multilingual model.",
    code: `x = self.embed(idx).unsqueeze(1)
x = self.ln(x)

for layer_idx in range(n_layer):
    enc, enc_v, dec = get_params(layer_idx)
    x_sparse  = F.relu(x @ enc)           # encode
    yKV       = self.attn(x_sparse, x_sparse, x)
    y_sparse  = F.relu(self.ln(yKV) @ enc_v)
    xy_sparse = x_sparse * y_sparse        # gate
    y = self.ln(xy_sparse.reshape(…) @ dec)
    x = self.ln(x + y)                     # residual

logits = x.squeeze(1) @ self.lm_head`,
  },
];

const DIFF_COLORS: Record<
  string,
  { bg: string; text: string; border: string }
> = {
  Easy: {
    bg: "bg-[#00C896]/10",
    text: "text-[#00C896]",
    border: "border-[#00C896]/20",
  },
  Medium: {
    bg: "bg-amber-900/40",
    text: "text-amber-400",
    border: "border-amber-800/50",
  },
  Hard: {
    bg: "bg-rose-900/40",
    text: "text-rose-400",
    border: "border-rose-800/50",
  },
};


/* ---------- 1. Byte Embedding -------------------------------- */
const ByteEmbeddingViz: React.FC = () => {
  const bytes = [
    { val: 66, ch: "B" },
    { val: 111, ch: "o" },
    { val: 110, ch: "n" },
    { val: 106, ch: "j" },
    { val: 111, ch: "o" },
    { val: 117, ch: "u" },
    { val: 114, ch: "r" },
  ];
  const rand = useMemo(() => seededRandom(42), []);
  const vectors = useMemo(
    () => bytes.map(() => Array.from({ length: 16 }, () => rand() * 2 - 1)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  return (
    <div className="flex flex-col items-center gap-6">
      <div className="text-xs text-[#4A5568] tracking-wider">
        "Bonjour" → 7 bytes → 7 embedding vectors (showing 16 of 256 dims)
      </div>
      <div className="flex items-start gap-3 overflow-x-auto pb-2">
        {bytes.map((b, i) => (
          <div
            key={i}
            className="flex flex-col items-center gap-2 min-w-[60px]"
          >
            <div className="flex flex-col items-center">
              <span className="text-lg font-mono text-white font-semibold">
                {b.ch}
              </span>
              <span className="text-[10px] text-[#4A5568] font-mono">
                {b.val}
              </span>
            </div>
            <div className="w-px h-3 bg-white/10" />
            <div className="flex gap-px">
              {vectors[i].map((v, j) => (
                <div
                  key={j}
                  className="w-[3px] rounded-sm"
                  style={{
                    height: `${Math.abs(v) * 28 + 4}px`,
                    backgroundColor:
                      v > 0
                        ? `rgba(52, 211, 153, ${Math.abs(v) * 0.7 + 0.3})`
                        : `rgba(244, 63, 94, ${Math.abs(v) * 0.7 + 0.3})`,
                    marginTop: v > 0 ? `${(1 - Math.abs(v)) * 14}px` : "0px",
                  }}
                />
              ))}
            </div>
            <span className="text-[9px] text-[#4A5568] font-mono">D=256</span>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-4 text-[10px] text-[#4A5568]">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-[#00C896]/60" /> positive
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-rose-400/60" /> negative
        </span>
      </div>
    </div>
  );
};

/* ---------- 2. Sparse Encoding ------------------------------- */
const SparseEncodingViz: React.FC = () => {
  const [showRelu, setShowRelu] = useState(false);
  const rand = useMemo(() => seededRandom(123), []);
  const bars = useMemo(
    () => Array.from({ length: 48 }, () => rand() * 2 - 1),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const activeCount = bars.filter((v) => v > 0).length;
  const sparsity = (((bars.length - activeCount) / bars.length) * 100).toFixed(
    0,
  );

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setShowRelu(false)}
          className={`px-3 py-1 text-xs rounded transition-colors ${
            !showRelu
              ? "bg-white/10 text-[#E2E8F0]"
              : "text-[#4A5568] hover:text-[#CBD5E0]"
          }`}
        >
          Before ReLU
        </button>
        <button
          onClick={() => setShowRelu(true)}
          className={`px-3 py-1 text-xs rounded transition-colors ${
            showRelu
              ? "bg-[#00C896]/10 text-[#00C896]"
              : "text-[#4A5568] hover:text-[#CBD5E0]"
          }`}
        >
          After ReLU
        </button>
        {showRelu && (
          <span className="text-xs text-[#00C896] font-mono ml-2">
            sparsity: {sparsity}%
          </span>
        )}
      </div>
      <div className="text-[10px] text-[#4A5568] tracking-wider">
        Showing 48 of 8,192 neurons per head
      </div>
      <div className="relative flex items-end gap-[2px] h-32 px-4">
        <div className="absolute bottom-1/2 left-0 right-0 h-px bg-white/5" />
        {bars.map((v, i) => {
          const value = showRelu ? Math.max(0, v) : v;
          const isZeroed = showRelu && v < 0;
          const absVal = Math.abs(value);
          const height = absVal * 56;
          const isPositive = value >= 0;

          return (
            <div
              key={i}
              className="relative flex flex-col items-center"
              style={{ height: "128px", justifyContent: "center" }}
            >
              <div
                className="w-[5px] rounded-sm transition-all duration-300"
                style={{
                  height: `${isZeroed ? 2 : height + 2}px`,
                  backgroundColor: isZeroed
                    ? "rgb(63, 63, 70)"
                    : isPositive
                      ? `rgba(52, 211, 153, ${absVal * 0.6 + 0.4})`
                      : `rgba(161, 161, 170, ${absVal * 0.6 + 0.3})`,
                  [isPositive ? "marginTop" : "marginBottom"]: "auto",
                  alignSelf: isPositive ? "flex-end" : "flex-start",
                  opacity: isZeroed ? 0.3 : 1,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-2 text-[10px] text-[#4A5568]">
        <span className="font-mono">D=256</span>
        <span>→</span>
        <span className="font-mono">N=8,192</span>
        <span>→</span>
        <span className="font-mono">ReLU</span>
        <span>→</span>
        <span className="font-mono text-[#00C896]">~5% active</span>
      </div>
    </div>
  );
};

/* ---------- 3. RoPE ------------------------------------------ */
const RoPEViz: React.FC = () => {
  const [position, setPosition] = useState(3);
  const freqs = [0.5, 0.15, 0.05];
  const colors = ["#34d399", "#60a5fa", "#fbbf24"];
  const labels = [
    "dim 0–1 (fast)",
    "dim 100–101 (mid)",
    "dim 8000–8001 (slow)",
  ];
  const W = 480;
  const H = 120;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-1">
        Position-dependent phase rotation at three frequencies
      </div>
      <svg width={W} height={H} className="overflow-visible">
        {Array.from({ length: 9 }, (_, i) => (
          <line
            key={`g${i}`}
            x1={i * (W / 8)}
            y1={0}
            x2={i * (W / 8)}
            y2={H}
            stroke="rgb(39,39,42)"
            strokeWidth={1}
          />
        ))}
        <line
          x1={0}
          y1={H / 2}
          x2={W}
          y2={H / 2}
          stroke="rgb(63,63,70)"
          strokeWidth={1}
        />

        {freqs.map((f, fi) => {
          const points = Array.from({ length: W }, (_, px) => {
            const t = (px / W) * 8;
            const y = H / 2 - Math.sin(t * f * Math.PI * 2) * (H / 2 - 8);
            return `${px},${y}`;
          }).join(" ");
          return (
            <polyline
              key={fi}
              points={points}
              fill="none"
              stroke={colors[fi]}
              strokeWidth={1.5}
              opacity={0.7}
            />
          );
        })}

        <line
          x1={position * (W / 8)}
          y1={0}
          x2={position * (W / 8)}
          y2={H}
          stroke="white"
          strokeWidth={1.5}
          strokeDasharray="3,3"
        />
        {freqs.map((f, fi) => {
          const t = position;
          const y = H / 2 - Math.sin(t * f * Math.PI * 2) * (H / 2 - 8);
          return (
            <circle
              key={fi}
              cx={position * (W / 8)}
              cy={y}
              r={4}
              fill={colors[fi]}
              stroke="white"
              strokeWidth={1}
            />
          );
        })}

        {Array.from({ length: 9 }, (_, i) => (
          <text
            key={`l${i}`}
            x={i * (W / 8)}
            y={H + 14}
            textAnchor="middle"
            className="text-[9px] fill-[#4A5568] font-mono"
          >
            t={i}
          </text>
        ))}
      </svg>

      <input
        type="range"
        min={0}
        max={8}
        step={1}
        value={position}
        onChange={(e) => setPosition(Number(e.target.value))}
        className="w-48 accent-[#00C896]"
      />
      <div className="text-[10px] text-[#8B95A5] font-mono">
        position = {position}
      </div>

      <div className="flex gap-4 mt-1">
        {labels.map((l, i) => (
          <span
            key={i}
            className="flex items-center gap-1 text-[10px] text-[#4A5568]"
          >
            <span
              className="w-3 h-[2px] rounded"
              style={{ backgroundColor: colors[i] }}
            />
            {l}
          </span>
        ))}
      </div>
    </div>
  );
};

/* ---------- 4. Linear Attention ------------------------------ */
const LinearAttentionViz: React.FC = () => {
  const [hovered, setHovered] = useState<number | null>(null);
  const tokens = ["B", "o", "n", "j", "o", "u"];
  const n = tokens.length;
  const rand = useMemo(() => seededRandom(77), []);
  const weights = useMemo(
    () =>
      Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (j < i ? rand() : 0)),
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const cellSize = 44;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-1">
        Causal attention matrix — lower triangle only (tril)
      </div>
      <div className="flex items-start gap-2">
        <div
          className="flex flex-col items-end justify-start pt-3"
          style={{ marginTop: cellSize + 4 }}
        >
          {tokens.map((t, i) => (
            <div
              key={i}
              className={`font-mono text-xs flex items-center justify-center transition-colors ${
                hovered === i ? "text-[#00C896]" : "text-[#4A5568]"
              }`}
              style={{ height: cellSize }}
            >
              {t}
            </div>
          ))}
        </div>

        <div>
          <div className="flex" style={{ paddingLeft: 0 }}>
            {tokens.map((t, i) => (
              <div
                key={i}
                className="font-mono text-xs text-[#4A5568] flex items-center justify-center"
                style={{ width: cellSize, height: cellSize * 0.6 }}
              >
                {t}
              </div>
            ))}
          </div>

          <div className="border border-white/[0.06] rounded">
            {weights.map((row, i) => (
              <div key={i} className="flex">
                {row.map((w, j) => {
                  const isCausal = j < i;
                  const isHighlighted = hovered === i;
                  return (
                    <div
                      key={j}
                      className="border border-white/[0.06] flex items-center justify-center text-[9px] font-mono transition-all cursor-default"
                      style={{
                        width: cellSize,
                        height: cellSize,
                        backgroundColor: isCausal
                          ? `rgba(52, 211, 153, ${w * (isHighlighted ? 1 : 0.65)})`
                          : i === j
                            ? "rgba(63, 63, 70, 0.3)"
                            : "rgba(24, 24, 27, 0.5)",
                      }}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                    >
                      {isCausal ? (
                        <span
                          className={
                            isHighlighted
                              ? "text-[#E2E8F0]"
                              : "text-[#00C896]/80/60"
                          }
                        >
                          {w.toFixed(2)}
                        </span>
                      ) : i === j ? (
                        <span className="text-[#4A5568]">—</span>
                      ) : (
                        <span className="text-[#374151]">0</span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4 text-[10px] text-[#4A5568] mt-1">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-[#00C896]/50" />
          attends (j &lt; i)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-white/[0.03] border border-white/10" />
          masked (j ≥ i)
        </span>
        <span className="text-[#4A5568]">hover a row to highlight</span>
      </div>
    </div>
  );
};

/* ---------- 5. Value Encoding -------------------------------- */
const ValueEncodingViz: React.FC = () => {
  const steps = [
    {
      label: "yKV",
      dims: "D=256",
      desc: "Attention output",
      color: "border-sky-700 bg-sky-900/20",
    },
    {
      label: "LN",
      dims: "",
      desc: "LayerNorm",
      color: "border-white/10 bg-white/[0.04]",
    },
    {
      label: "@ enc_v",
      dims: "D→N",
      desc: "Second encoder",
      color: "border-amber-700 bg-amber-900/20",
    },
    {
      label: "ReLU",
      dims: "",
      desc: "Sparsify",
      color: "border-[#00C896]/30 bg-[#00C896]/8",
    },
    {
      label: "y_sparse",
      dims: "N=8192",
      desc: "Context features",
      color: "border-[#00C896]/40 bg-[#00C896]/8",
    },
  ];

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-1">
        Second encoding path: attended values → sparse neuron space
      </div>
      <div className="flex items-center gap-2 flex-wrap justify-center">
        {steps.map((s, i) => (
          <React.Fragment key={i}>
            <div
              className={`border rounded-lg px-4 py-3 text-center ${s.color}`}
            >
              <div className="text-sm font-mono text-[#E2E8F0]">{s.label}</div>
              {s.dims && (
                <div className="text-[10px] text-[#8B95A5] font-mono mt-0.5">
                  {s.dims}
                </div>
              )}
              <div className="text-[10px] text-[#4A5568] mt-1">{s.desc}</div>
            </div>
            {i < steps.length - 1 && (
              <span className="text-[#4A5568] text-lg">→</span>
            )}
          </React.Fragment>
        ))}
      </div>
      <div className="text-[10px] text-[#4A5568] mt-1">
        Creates the second independent sparse representation needed for gating
      </div>
    </div>
  );
};

/* ---------- 6. Sparse Gating --------------------------------- */
const SparseGatingViz: React.FC = () => {
  const rand = useMemo(() => seededRandom(55), []);
  const n = 14;
  const xSparse = useMemo(
    () =>
      Array.from({ length: n }, () =>
        rand() > 0.6 ? +(rand() * 0.8 + 0.2).toFixed(1) : 0,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const ySparse = useMemo(
    () =>
      Array.from({ length: n }, () =>
        rand() > 0.55 ? +(rand() * 0.8 + 0.2).toFixed(1) : 0,
      ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  const gated = xSparse.map((x, i) => +(x * ySparse[i]).toFixed(2));

  const xActive = xSparse.filter((v) => v > 0).length;
  const yActive = ySparse.filter((v) => v > 0).length;
  const gatedActive = gated.filter((v) => v > 0).length;

  const BarRow = ({
    values,
    color,
    label,
    count,
  }: {
    values: number[];
    color: string;
    label: string;
    count: number;
  }) => (
    <div className="flex items-center gap-3">
      <div className="w-20 text-right">
        <span className="text-xs text-[#8B95A5] font-mono">{label}</span>
        <span className="text-[10px] text-[#4A5568] ml-1">
          ({count}/{n})
        </span>
      </div>
      <div className="flex gap-[3px]">
        {values.map((v, i) => (
          <div
            key={i}
            className="w-6 h-8 rounded-sm flex items-end justify-center transition-all"
            style={{
              backgroundColor: v > 0 ? color : "rgb(39, 39, 42)",
              opacity: v > 0 ? v * 0.6 + 0.4 : 0.4,
            }}
          >
            {v > 0 && (
              <span className="text-[7px] text-[#E2E8F0]/80 font-mono mb-0.5">
                {v.toFixed(1)}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-2">
        Element-wise multiplication: both must be active to survive
      </div>
      <BarRow
        values={xSparse}
        color="rgba(52, 211, 153, 0.7)"
        label="x_sparse"
        count={xActive}
      />
      <div className="text-[#4A5568] text-lg font-mono ml-20">×</div>
      <BarRow
        values={ySparse}
        color="rgba(96, 165, 250, 0.7)"
        label="y_sparse"
        count={yActive}
      />
      <div className="text-[#4A5568] text-lg font-mono ml-20">=</div>
      <BarRow
        values={gated}
        color="rgba(251, 191, 36, 0.7)"
        label="gated"
        count={gatedActive}
      />
      <div className="text-[10px] text-[#4A5568] mt-2">
        {xActive} active × {yActive} active → only{" "}
        <span className="text-amber-400 font-semibold">
          {gatedActive} survive
        </span>{" "}
        the gate
      </div>
    </div>
  );
};

/* ---------- 7. Decode + Residual ----------------------------- */
const DecodeResidualViz: React.FC = () => (
  <div className="flex flex-col items-center gap-4">
    <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-1">
      Compress sparse code back to embedding space + skip connection
    </div>

    <div className="relative flex items-center gap-2 max-w-xl mx-auto flex-wrap justify-center">
      {/* Input */}
      <div className="border border-white/10 rounded px-3 py-2 bg-white/[0.04] text-center min-w-[56px]">
        <div className="text-xs font-mono text-[#CBD5E0]">x</div>
        <div className="text-[9px] text-[#4A5568]">D=256</div>
      </div>

      <span className="text-[#4A5568] text-sm">→</span>

      {/* Gated sparse */}
      <div className="border border-amber-800/50 rounded px-3 py-2 bg-amber-900/20 text-center">
        <div className="text-xs font-mono text-amber-300">xy_sparse</div>
        <div className="text-[9px] text-[#4A5568]">nh×N = 32,768</div>
      </div>

      <span className="text-[#4A5568] text-sm">→</span>

      {/* Decoder */}
      <div className="border border-white/[0.12] rounded px-3 py-2 bg-white/[0.05] text-center">
        <div className="text-xs font-mono text-[#CBD5E0]">@ dec</div>
        <div className="text-[9px] text-[#4A5568]">N→D</div>
      </div>

      <span className="text-[#4A5568] text-sm">→</span>

      {/* LN */}
      <div className="border border-white/10 rounded px-2 py-2 bg-white/[0.04] text-center">
        <div className="text-xs font-mono text-[#CBD5E0]">LN</div>
      </div>

      <span className="text-[#4A5568] text-sm font-mono">+x</span>

      <span className="text-[#4A5568] text-sm">→</span>

      {/* Output */}
      <div className="border border-[#00C896]/30 rounded px-3 py-2 bg-[#00C896]/8 text-center">
        <div className="text-xs font-mono text-[#00C896]">output</div>
        <div className="text-[9px] text-[#4A5568]">D=256</div>
      </div>
    </div>

    <div className="flex items-center gap-2 text-[10px] text-[#4A5568] mt-2">
      <span className="font-mono">32,768</span>
      <span>→ 128× compression →</span>
      <span className="font-mono text-[#00C896]">256</span>
      <span className="text-[#374151] mx-2">|</span>
      <span className="text-[#4A5568]">
        + residual skip keeps gradient flow
      </span>
    </div>
  </div>
);

/* ---------- 8. Full BDH Layer -------------------------------- */
const FullLayerViz: React.FC = () => {
  const pipeline = [
    {
      label: "Embed",
      dim: "D",
      color: "border-white/[0.12] bg-white/[0.04]",
      text: "text-[#CBD5E0]",
    },
    {
      label: "Encode",
      dim: "D→N",
      color: "border-sky-700 bg-sky-900/20",
      text: "text-sky-300",
    },
    {
      label: "ReLU",
      dim: "N",
      color: "border-[#00C896]/30 bg-[#00C896]/8",
      text: "text-[#00C896]",
    },
    {
      label: "Attend",
      dim: "T×T",
      color: "border-[#2A7FFF]/30 bg-[#2A7FFF]/10",
      text: "text-[#2A7FFF]",
    },
    {
      label: "Enc V",
      dim: "D→N",
      color: "border-amber-700 bg-amber-900/20",
      text: "text-amber-300",
    },
    {
      label: "Gate",
      dim: "N",
      color: "border-amber-600 bg-amber-900/30",
      text: "text-amber-300",
    },
    {
      label: "Decode",
      dim: "N→D",
      color: "border-[#00C896]/40 bg-[#00C896]/8",
      text: "text-[#00C896]",
    },
  ];

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="text-[10px] text-[#4A5568] tracking-wider uppercase mb-1">
        Complete BDH layer pipeline — repeated 6 times
      </div>

      <div className="flex items-center gap-1.5 flex-wrap justify-center">
        {pipeline.map((p, i) => (
          <React.Fragment key={i}>
            <div
              className={`border rounded-lg px-3 py-2 text-center ${p.color}`}
            >
              <div className={`text-xs font-mono font-medium ${p.text}`}>
                {p.label}
              </div>
              <div className="text-[9px] text-[#4A5568] font-mono mt-0.5">
                {p.dim}
              </div>
            </div>
            {i < pipeline.length - 1 && (
              <span className="text-[#374151] text-sm">→</span>
            )}
          </React.Fragment>
        ))}
        <span className="text-[#4A5568] text-sm ml-1">+ res</span>
      </div>

      <div className="flex items-center gap-2 mt-2">
        <span className="text-[10px] text-[#4A5568]">×6 layers:</span>
        {Array.from({ length: 6 }, (_, i) => (
          <div
            key={i}
            className="w-8 h-5 rounded border border-white/10 bg-white/[0.05] flex items-center justify-center"
          >
            <span className="text-[9px] font-mono text-[#8B95A5]">L{i}</span>
          </div>
        ))}
        <span className="text-[#374151] text-sm ml-1">→</span>
        <div className="border border-[#00C896]/30 rounded px-2 py-0.5 bg-[#00C896]/8">
          <span className="text-[10px] font-mono text-[#00C896]">lm_head</span>
        </div>
      </div>

      <div className="text-[10px] text-[#4A5568] mt-1">
        Early layers → syntax &amp; phonetics · Later layers → semantics &amp;
        language identity
      </div>
    </div>
  );
};

const vizComponents: Record<number, React.FC> = {
  1: ByteEmbeddingViz,
  2: SparseEncodingViz,
  3: RoPEViz,
  4: LinearAttentionViz,
  5: ValueEncodingViz,
  6: SparseGatingViz,
  7: DecodeResidualViz,
  8: FullLayerViz,
};

const StepVisualization: React.FC<{ stepId: number }> = ({ stepId }) => {
  const Viz = vizComponents[stepId];
  return Viz ? <Viz /> : null;
};

export function LearnBDHPage() {
  const [currentStep, setCurrentStep] = useState(0);
  const [activeTab, setActiveTab] = useState<"description" | "theory">(
    "description",
  );

  const step = STEPS[currentStep];
  const diff = DIFF_COLORS[step.difficulty];

  const goNext = useCallback(() => {
    if (currentStep < STEPS.length - 1) {
      setCurrentStep((s) => s + 1);
      setActiveTab("description");
    }
  }, [currentStep]);

  const goPrev = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep((s) => s - 1);
      setActiveTab("description");
    }
  }, [currentStep]);

  return (
    <div className="flex h-full min-h-0" style={{ background: "#070D12" }}>
      {/* Sidebar */}
      <aside className="w-72 shrink-0 border-r border-white/[0.06] bg-[#0B1216] flex flex-col">
        <div className="px-6 pt-6 pb-4">
          <h2 className="text-sm font-semibold text-[#E2E8F0] tracking-tight">
            Learn BDH
          </h2>
          <p className="text-[11px] text-[#4A5568] mt-0.5">
            8 architecture steps
          </p>
        </div>

        <nav className="flex-1 overflow-y-auto px-3 pb-4">
          {STEPS.map((s, i) => {
            const isActive = i === currentStep;
            return (
              <button
                key={s.id}
                onClick={() => {
                  setCurrentStep(i);
                  setActiveTab("description");
                }}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg mb-0.5 text-left transition-colors ${
                  isActive ? "bg-white/[0.06]" : "hover:bg-white/[0.04]"
                }`}
              >
                <span
                  className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-medium border transition-colors ${
                    isActive
                      ? "bg-[#00C896] border-[#00C896] text-[#E2E8F0]"
                      : "border-white/10 text-[#4A5568] bg-[#0B1216]"
                  }`}
                >
                  {s.id}
                </span>
                <span
                  className={`text-sm truncate transition-colors ${
                    isActive ? "text-white font-medium" : "text-[#8B95A5]"
                  }`}
                >
                  {s.title}
                </span>
              </button>
            );
          })}
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={step.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
            className="max-w-4xl mx-auto px-8 py-8"
          >
            {/* Header */}
            <div className="flex items-center gap-3 mb-1">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-[#00C896]/10 text-[#00C896] border border-[#00C896]/20">
                {String(step.id).padStart(2, "0")}/
                {String(STEPS.length).padStart(2, "0")}
              </span>
              <h1 className="text-2xl font-semibold text-[#E2E8F0] tracking-tight">
                {step.title}
              </h1>
              <span
                className={`ml-auto inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${diff.bg} ${diff.text} ${diff.border}`}
              >
                {step.difficulty}
              </span>
            </div>
            <p className="text-sm text-[#4A5568] mb-6">{step.subtitle}</p>

            {/* Tabs */}
            <div className="flex gap-1 mb-6 border-b border-white/[0.06] pb-px">
              {(["description", "theory"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-sm capitalize rounded-t transition-colors ${
                    activeTab === tab
                      ? "bg-white/[0.05] text-[#E2E8F0] border-b-2 border-[#00C896]"
                      : "text-[#4A5568] hover:text-[#CBD5E0]"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab content */}
            {activeTab === "description" ? (
              <div>
                {/* Visualization card */}
                <div className="border border-white/[0.06] rounded-xl bg-[#0B1216]/60 p-6 mb-8">
                  <span className="text-[10px] uppercase tracking-widest text-[#4A5568] block mb-5">
                    Visualization
                  </span>
                  <StepVisualization stepId={step.id} />
                </div>

                {/* Description */}
                <div className="space-y-3 mb-8">
                  {step.description.map((p, i) => (
                    <p
                      key={i}
                      className="text-sm text-[#CBD5E0] leading-relaxed"
                    >
                      {p}
                    </p>
                  ))}
                </div>

                {/* Key insight */}
                <div className="border-l-2 border-[#00C896]/40 pl-4 py-2 bg-[#00C896]/5 rounded-r">
                  <span className="text-[10px] uppercase tracking-widest text-[#00C896]/70 block mb-1">
                    Key Insight
                  </span>
                  <p className="text-sm text-[#00C896]/80/80 leading-relaxed">
                    {step.keyInsight}
                  </p>
                </div>
              </div>
            ) : (
              <div>
                {/* Theory sections */}
                <div className="space-y-6 mb-8">
                  {step.theory.map((t, i) => (
                    <div key={i}>
                      <h3 className="text-sm font-medium text-[#E2E8F0] mb-1.5">
                        {t.heading}
                      </h3>
                      <p className="text-sm text-[#8B95A5] leading-relaxed">
                        {t.content}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Code snippet */}
                <div className="border border-white/[0.06] rounded-xl overflow-hidden">
                  <div className="flex items-center gap-2 px-4 py-2 bg-[#0B1216]/80 border-b border-white/[0.06]">
                    <span className="text-[10px] uppercase tracking-widest text-[#4A5568]">
                      Source — bdh.py
                    </span>
                  </div>
                  <pre className="px-5 py-4 text-[13px] leading-relaxed font-mono text-[#CBD5E0] overflow-x-auto bg-[#070D12]/60">
                    {step.code}
                  </pre>
                </div>
              </div>
            )}

            {/* Navigation */}
            <div className="flex items-center justify-between mt-10 pt-6 border-t border-white/[0.06]/60">
              <button
                onClick={goPrev}
                disabled={currentStep === 0}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                  currentStep === 0
                    ? "text-[#374151] cursor-not-allowed"
                    : "text-[#8B95A5] hover:text-[#E2E8F0] hover:bg-white/[0.05]"
                }`}
              >
                <ChevronLeft size={16} />
                {currentStep > 0 ? STEPS[currentStep - 1].title : "Previous"}
              </button>

              <div className="flex gap-1.5">
                {STEPS.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setCurrentStep(i);
                      setActiveTab("description");
                    }}
                    className={`w-2 h-2 rounded-full transition-colors ${
                      i === currentStep
                        ? "bg-[#00C896]"
                        : "bg-white/10 hover:bg-white/20"
                    }`}
                  />
                ))}
              </div>

              <button
                onClick={goNext}
                disabled={currentStep === STEPS.length - 1}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                  currentStep === STEPS.length - 1
                    ? "text-[#374151] cursor-not-allowed"
                    : "text-[#8B95A5] hover:text-[#E2E8F0] hover:bg-white/[0.05]"
                }`}
              >
                {currentStep < STEPS.length - 1
                  ? STEPS[currentStep + 1].title
                  : "Next"}
                <ChevronRight size={16} />
              </button>
            </div>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
