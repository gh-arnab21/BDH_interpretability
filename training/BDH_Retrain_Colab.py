#!/usr/bin/env python3
"""
BDH Retrain for Monosemanticity — Google Colab Script
=====================================================

This is a SELF-CONTAINED script you run in Google Colab to retrain the BDH
model with architectural changes that fix the monosemanticity limitations:

CHANGES FROM ORIGINAL:
  1. PER-LAYER ENCODERS  — each layer gets its own E, E_v, D matrices
     (original shared a single encoder across all layers, killing cross-concept
      distinctness because every layer produces identical fingerprints)
  2. PERSISTENT ρ BUFFER — register_buffer('rho') that accumulates Hebbian
     σ = y_sparse · x_sparse across training examples with EMA decay
     (original had no persistent memory — only within-forward-pass gating)
  3. LARGER DATASET      — uses full Europarl en-fr (~2M sentence pairs)

After training, download these files:
  - checkpoints/french_v2/checkpoint_best.pt  → paste into your local
    checkpoints/french/french_best.pt

Then locally re-run:
  python scripts/precompute_monosemanticity.py

HOW TO RUN IN COLAB:
  1. Upload this file to Colab (or paste cells)
  2. Run each section in order
  3. Training takes ~2-4 hours on T4/A100
  4. Download the checkpoint_best.pt file
  5. Replace your local checkpoints/french/french_best.pt with it
  6. Re-run precompute_monosemanticity.py locally

=====================================================
"""

# %% [markdown]
# # 🐉 BDH V2 Retrain — Per-Layer Encoders + Persistent ρ
#
# Run all cells in order. Estimated time: ~2-4h on T4 GPU.

# %% Cell 1: Setup & GPU check
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("torch")
install("numpy")
install("tqdm")
install("pyyaml")
install("requests")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
print(f"Using: {DEVICE}, {DTYPE}")


# %% Cell 2: Download Europarl en-fr data
import os, tarfile, requests
from pathlib import Path
from tqdm import tqdm
import numpy as np

DATA_DIR = Path("data/en-fr")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EUROPARL_URL = "https://www.statmt.org/europarl/v7/fr-en.tgz"
ARCHIVE = DATA_DIR / "europarl.tgz"

if not (DATA_DIR / "train.bin").exists():
    # Download
    if not ARCHIVE.exists():
        print("📥 Downloading Europarl en-fr (~180MB)...")
        resp = requests.get(EUROPARL_URL, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get('content-length', 0))
        with open(ARCHIVE, 'wb') as f:
            with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Extract
    print("📦 Extracting...")
    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        tar.extractall(DATA_DIR, filter='data' if hasattr(tarfile, 'data_filter') else None)
    
    # Find files
    en_file = fr_file = None
    for f in DATA_DIR.rglob("*"):
        if f.name.endswith(".en") and f.is_file():
            en_file = f
        elif f.name.endswith(".fr") and f.is_file():
            fr_file = f
    
    assert en_file and fr_file, f"Could not find .en/.fr files in {DATA_DIR}"
    print(f"  Found: {en_file.name}, {fr_file.name}")
    
    # Convert to BDH byte format
    print("🔄 Converting to BDH byte format...")
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    en_lines = fr_lines = None
    for enc in encodings:
        try:
            en_lines = open(en_file, encoding=enc).readlines()
            fr_lines = open(fr_file, encoding=enc).readlines()
            break
        except UnicodeDecodeError:
            continue
    
    assert en_lines and fr_lines, "Could not read corpus files"
    print(f"  {len(en_lines):,} EN lines, {len(fr_lines):,} FR lines")
    
    # Build byte stream with language markers
    all_bytes = bytearray()
    count = 0
    for en, fr in tqdm(zip(en_lines, fr_lines), total=min(len(en_lines), len(fr_lines)), desc="  Encoding"):
        en_s, fr_s = en.strip(), fr.strip()
        if not en_s or not fr_s or len(en_s) > 800 or len(fr_s) > 800:
            continue
        line = f"<F:en>{en_s}<T:fr>{fr_s}"
        all_bytes.extend(line.encode('utf-8', errors='replace'))
        count += 1
    
    # Split 90/10
    split = int(len(all_bytes) * 0.9)
    train_bytes = np.frombuffer(bytes(all_bytes[:split]), dtype=np.uint8)
    val_bytes   = np.frombuffer(bytes(all_bytes[split:]), dtype=np.uint8)
    
    train_file = DATA_DIR / "train.bin"
    val_file   = DATA_DIR / "val.bin"
    train_bytes.tofile(str(train_file))
    val_bytes.tofile(str(val_file))
    
    print(f"✅ {count:,} pairs → train={len(train_bytes)/1e6:.1f}MB, val={len(val_bytes)/1e6:.1f}MB")
else:
    print("✅ Data already exists, skipping download")
    train_file = DATA_DIR / "train.bin"
    val_file   = DATA_DIR / "val.bin"
    print(f"  Train: {train_file.stat().st_size/1e6:.1f}MB, Val: {val_file.stat().st_size/1e6:.1f}MB")


# %% Cell 3: MODEL ARCHITECTURE — BDH V2 with per-layer encoders + persistent ρ
import math
import dataclasses
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    """Configuration for BDH V2 model."""
    n_layer: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256
    # V2: ρ EMA decay factor (0 = no persistence, 0.99 = long memory)
    rho_decay: float = 0.99
    rho_max: float = 5.0        # Maximum ρ magnitude (prevents runaway)

    @property
    def n_neurons(self) -> int:
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head

    @property
    def total_neurons(self) -> int:
        return self.n_neurons * self.n_head


@dataclasses.dataclass
class ExtractionConfig:
    capture_sparse_activations: bool = True
    capture_attention_patterns: bool = True
    capture_pre_relu: bool = True
    capture_layer_outputs: bool = True
    capture_residuals: bool = False
    layers_to_capture: Optional[List[int]] = None

    def should_capture_layer(self, layer_idx: int) -> bool:
        if self.layers_to_capture is None:
            return True
        return layer_idx in self.layers_to_capture


class ExtractionBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.x_sparse: Dict[int, torch.Tensor] = {}
        self.y_sparse: Dict[int, torch.Tensor] = {}
        self.x_pre_relu: Dict[int, torch.Tensor] = {}
        self.y_pre_relu: Dict[int, torch.Tensor] = {}
        self.attention_scores: Dict[int, torch.Tensor] = {}
        self.attention_output: Dict[int, torch.Tensor] = {}
        self.layer_outputs: Dict[int, torch.Tensor] = {}
        self.residuals: Dict[int, torch.Tensor] = {}
        self.pre_layernorm: Optional[torch.Tensor] = None
        self.input_tokens: Optional[torch.Tensor] = None
        self.final_output: Optional[torch.Tensor] = None

    def get_sparsity_stats(self) -> Dict[str, float]:
        stats = {}
        for layer_idx, x_sparse in self.x_sparse.items():
            total = x_sparse.numel()
            nonzero = (x_sparse > 0).sum().item()
            stats[f"layer_{layer_idx}_x_sparsity"] = 1 - (nonzero / total)
        for layer_idx, y_sparse in self.y_sparse.items():
            total = y_sparse.numel()
            nonzero = (y_sparse > 0).sum().item()
            stats[f"layer_{layer_idx}_y_sparsity"] = 1 - (nonzero / total)
        all_x = torch.cat([x.flatten() for x in self.x_sparse.values()])
        all_y = torch.cat([y.flatten() for y in self.y_sparse.values()])
        stats["overall_x_sparsity"] = 1 - ((all_x > 0).sum().item() / all_x.numel())
        stats["overall_y_sparsity"] = 1 - ((all_y > 0).sum().item() / all_y.numel())
        stats["overall_sparsity"] = 1 - (
            ((all_x > 0).sum().item() + (all_y > 0).sum().item()) /
            (all_x.numel() + all_y.numel())
        )
        return stats

    def get_active_neuron_indices(self, layer: int, threshold: float = 0.0) -> Dict[str, torch.Tensor]:
        result = {}
        if layer in self.x_sparse:
            result["x_active"] = (self.x_sparse[layer] > threshold).any(dim=(0, 2))
        if layer in self.y_sparse:
            result["y_active"] = (self.y_sparse[layer] > threshold).any(dim=(0, 2))
        return result

    def to_dict(self, detach: bool = True, cpu: bool = True) -> Dict[str, Any]:
        def proc(t):
            if detach: t = t.detach()
            if cpu: t = t.cpu()
            return t
        return {
            "x_sparse": {k: proc(v) for k, v in self.x_sparse.items()},
            "y_sparse": {k: proc(v) for k, v in self.y_sparse.items()},
            "x_pre_relu": {k: proc(v) for k, v in self.x_pre_relu.items()},
            "y_pre_relu": {k: proc(v) for k, v in self.y_pre_relu.items()},
            "attention_scores": {k: proc(v) for k, v in self.attention_scores.items()},
            "attention_output": {k: proc(v) for k, v in self.attention_output.items()},
            "layer_outputs": {k: proc(v) for k, v in self.layer_outputs.items()},
            "input_tokens": proc(self.input_tokens) if self.input_tokens is not None else None,
            "final_output": proc(self.final_output) if self.final_output is not None else None,
        }


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """Linear attention with RoPE — operates in sparse neuron space."""
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        N = config.n_neurons
        self.register_buffer(
            "freqs",
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        c, s = Attention.phases_cos_sin(phases)
        return (v * c).to(v.dtype) + (v_rot * s).to(v.dtype)

    def forward(self, Q, K, V, return_scores=False):
        assert K is Q
        _, _, T, _ = Q.size()
        r = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r, Q)
        KR = QR
        scores = (QR @ KR.mT).tril(diagonal=-1)
        output = scores @ V
        if return_scores:
            return output, scores
        return output


class BDH(nn.Module):
    """
    BDH V2 — Per-layer encoders + persistent ρ buffer.

    KEY CHANGES FROM V1:
    ────────────────────
    1. self.encoders    = ParameterList of n_layer × (nh, D, N) matrices
       self.encoders_v  = ParameterList of n_layer × (nh, D, N)
       self.decoders    = ParameterList of n_layer × (nh*N, D)
       → Each layer learns its OWN projection to neuron space.
       → This means L0 and L7 have different neuron specializations.
       → Cross-concept distinctness WILL improve because different layers
         can develop different neuron populations for different concepts.

    2. self.rho = register_buffer of (n_layer, nh, N) — persistent Hebbian
       accumulator. Updated with EMA: ρ = decay * ρ + (1-decay) * mean(gate).
       → During forward pass, the gate is MODULATED by ρ:
         gate = x_sparse * y_sparse * (1 + rho)
       → Neurons that frequently co-fire across examples get amplified.
       → This breaks the "inference-only" limitation — ρ carries context
         across training examples, exactly like Hebbian long-term potentiation.

    3. Attention module is SHARED across layers (same as V1) — lightweight.
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config

        nh = config.n_head
        D  = config.n_embd
        N  = config.n_neurons
        nL = config.n_layer

        # ── V2: Per-layer encoder/decoder (ParameterList) ──
        self.encoders   = nn.ParameterList([
            nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
            for _ in range(nL)
        ])
        self.encoders_v = nn.ParameterList([
            nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
            for _ in range(nL)
        ])
        self.decoders   = nn.ParameterList([
            nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))
            for _ in range(nL)
        ])

        # ── V2: Persistent ρ buffer (not a parameter — updated manually) ──
        self.register_buffer('rho', torch.zeros(nL, nh, N))

        # Attention (shared across layers — it's position-dependent, not layer-specific)
        self.attn = Attention(config)

        # LayerNorm, embedding, output
        self.ln      = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed   = nn.Embedding(config.vocab_size, D)
        self.lm_head = nn.Parameter(torch.zeros(D, config.vocab_size).normal_(std=0.02))
        self.drop    = nn.Dropout(config.dropout)

        # Extraction state
        self._extraction_config: Optional[ExtractionConfig] = None
        self._extraction_buffer: Optional[ExtractionBuffer] = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @contextmanager
    def extraction_mode(self, config: Optional[ExtractionConfig] = None):
        self._extraction_config = config or ExtractionConfig()
        self._extraction_buffer = ExtractionBuffer()
        try:
            yield self._extraction_buffer
        finally:
            self._extraction_config = None

    def get_extraction_buffer(self):
        return self._extraction_buffer

    def forward(self, idx, targets=None):
        C = self.config
        extracting = self._extraction_config is not None
        buffer = self._extraction_buffer

        B, T = idx.size()
        D  = C.n_embd
        nh = C.n_head
        N  = C.n_neurons

        if extracting and buffer:
            buffer.input_tokens = idx.clone()

        x = self.embed(idx).unsqueeze(1)  # (B, 1, T, D)

        if extracting and buffer:
            buffer.pre_layernorm = x.clone()

        x = self.ln(x)

        for layer_idx in range(C.n_layer):
            should_capture = (
                extracting and buffer and
                self._extraction_config.should_capture_layer(layer_idx)
            ) if extracting else False

            # ── V2: Use per-layer encoder ──
            encoder   = self.encoders[layer_idx]    # (nh, D, N)
            encoder_v = self.encoders_v[layer_idx]  # (nh, D, N)
            decoder   = self.decoders[layer_idx]    # (nh*N, D)

            # ENCODE: D → N
            x_latent = x @ encoder  # (B, nh, T, N)

            if should_capture and self._extraction_config.capture_pre_relu:
                buffer.x_pre_relu[layer_idx] = x_latent.clone()

            # SPARSIFY
            x_sparse = F.relu(x_latent)

            if should_capture and self._extraction_config.capture_sparse_activations:
                buffer.x_sparse[layer_idx] = x_sparse.clone()

            # ATTEND
            if should_capture and self._extraction_config.capture_attention_patterns:
                yKV, scores = self.attn(Q=x_sparse, K=x_sparse, V=x, return_scores=True)
                buffer.attention_scores[layer_idx] = scores.clone()
            else:
                yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)

            yKV = self.ln(yKV)

            if should_capture:
                buffer.attention_output[layer_idx] = yKV.clone()

            # ENCODE V: D → N
            y_latent = yKV @ encoder_v

            if should_capture and self._extraction_config.capture_pre_relu:
                buffer.y_pre_relu[layer_idx] = y_latent.clone()

            # SPARSIFY V
            y_sparse = F.relu(y_latent)

            if should_capture and self._extraction_config.capture_sparse_activations:
                buffer.y_sparse[layer_idx] = y_sparse.clone()

            # ── GATE (Hebbian) ──
            gate_raw = x_sparse * y_sparse  # (B, nh, T, N)

            # ── V2: Update ρ with EMA from RAW gate BEFORE amplification ──
            # This prevents the positive feedback loop that caused ρ explosion.
            if self.training and not extracting:
                with torch.no_grad():
                    # Mean raw gate across batch and time: (B, nh, T, N) → (nh, N)
                    gate_mean = gate_raw.detach().mean(dim=(0, 2))
                    self.rho[layer_idx] = (
                        C.rho_decay * self.rho[layer_idx] +
                        (1.0 - C.rho_decay) * gate_mean
                    )
                    # Clamp to prevent runaway even if raw gate has large values
                    self.rho[layer_idx].clamp_(-C.rho_max, C.rho_max)

            # ── V2: Modulate gate by persistent ρ ──
            # ρ[layer] shape: (nh, N) → broadcast to (1, nh, 1, N)
            rho_layer = self.rho[layer_idx].unsqueeze(0).unsqueeze(2)  # (1, nh, 1, N)
            gate = gate_raw * (1.0 + rho_layer)  # amplify frequently co-active neurons

            gate = self.drop(gate)

            # DECODE: N → D
            yMLP = (
                gate.transpose(1, 2).reshape(B, 1, T, N * nh) @ decoder
            )
            y = self.ln(yMLP)

            if should_capture and self._extraction_config.capture_residuals:
                buffer.residuals[layer_idx] = y.clone()

            x = self.ln(x + y)

            if should_capture and self._extraction_config.capture_layer_outputs:
                buffer.layer_outputs[layer_idx] = x.clone()

        # Project to vocab
        logits = x.view(B, T, D) @ self.lm_head

        if extracting and buffer:
            buffer.final_output = logits.clone()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            # Guard against NaN/Inf logits (from corrupted weights)
            if torch.isnan(logits).any() or torch.isinf(logits).all():
                # Fall back to uniform sampling
                idx_next = torch.randint(0, logits.size(-1), (idx.size(0), 1), device=idx.device)
                idx = torch.cat((idx, idx_next), dim=1)
                continue
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            # Clamp to avoid multinomial CUDA errors on near-zero probabilities
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_graph_topology(self, threshold=0.01):
        """Extract graph topology — uses layer 0 encoder for compatibility."""
        with torch.no_grad():
            nh = self.config.n_head
            N  = self.config.n_neurons
            D  = self.config.n_embd
            encoder = self.encoders[0]
            decoder = self.decoders[0]
            decoder_reshaped = decoder.view(nh, N, D)
            G = torch.bmm(decoder_reshaped, encoder)
            G_abs = G.abs()
            adjacency = (G_abs > threshold).float()
            edges_per_head = adjacency.sum(dim=(1, 2)).tolist()
            total_edges = sum(edges_per_head)
            max_possible = nh * N * N
            density = total_edges / max_possible
            out_degree = adjacency.sum(dim=2)
            in_degree = adjacency.sum(dim=1)
            return {
                "adjacency": G.cpu().numpy(),
                "binary_adjacency": adjacency.cpu().numpy(),
                "edges_per_head": edges_per_head,
                "total_edges": int(total_edges),
                "density": density,
                "out_degree": out_degree.cpu().numpy(),
                "in_degree": in_degree.cpu().numpy(),
                "n_heads": nh,
                "n_neurons_per_head": N,
            }


def create_model(
    n_layer=8, n_embd=256, n_head=4, mlp_multiplier=128,
    dropout=0.1, vocab_size=256, rho_decay=0.99, rho_max=5.0,
):
    config = BDHConfig(
        n_layer=n_layer, n_embd=n_embd, n_head=n_head,
        mlp_internal_dim_multiplier=mlp_multiplier,
        dropout=dropout, vocab_size=vocab_size,
        rho_decay=rho_decay, rho_max=rho_max,
    )
    return BDH(config)


def load_model(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        # Add defaults for missing keys (backward compat)
        if "rho_decay" not in cfg:
            cfg["rho_decay"] = 0.99
        if "rho_max" not in cfg:
            cfg["rho_max"] = 5.0
        # Filter out extra keys that aren't BDHConfig fields
        valid_keys = {f.name for f in dataclasses.fields(BDHConfig)}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        config = BDHConfig(**cfg)
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        stripped = {}
        for k, v in state_dict.items():
            stripped[k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k] = v
        state_dict = stripped
        enc_shape = state_dict.get("encoders.0", state_dict.get("encoder")).shape
        n_head = enc_shape[0]
        n_embd = enc_shape[1]
        N      = enc_shape[2]
        mlp_multiplier = (N * n_head) // n_embd
        n_layer = sum(1 for k in state_dict if k.startswith("encoders.")) or 8
        config = BDHConfig(
            n_layer=n_layer, n_embd=n_embd, n_head=n_head,
            mlp_internal_dim_multiplier=mlp_multiplier,
        )

    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k] = v

    model = BDH(config)
    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()
    return model


# %% Cell 4: Dataset
class ByteDataset:
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.block_size = block_size
        self.length = len(self.data)
        print(f"  Loaded {data_path}: {self.length:,} bytes ({self.length/1e6:.1f} MB)")

    def __len__(self):
        return self.length - self.block_size - 1

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix
        ])
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


# %% Cell 5: Training loop
import time, json
from contextlib import nullcontext
from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    # Data
    train_data: str = "data/en-fr/train.bin"
    val_data: str   = "data/en-fr/val.bin"
    # Model architecture — MUST stay consistent!
    n_layer: int = 8
    n_embd: int  = 256
    n_head: int  = 4
    mlp_multiplier: int = 128
    dropout: float = 0.1
    vocab_size: int = 256
    rho_decay: float = 0.99    # V2: ρ EMA decay
    rho_max: float = 5.0         # V2: ρ clamp ceiling
    # Training — tuned for ~2.5h on A100 40GB
    batch_size: int = 8          # Fits A100 40GB (compile disabled)
    block_size: int = 256        # Good context length for translation
    max_iters: int  = 8000       # ~2.5h on A100
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 600
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 8   # Effective batch = 8×8 = 64
    # Logging
    log_interval: int  = 100
    eval_interval: int = 500
    save_interval: int = 2000
    eval_iters: int    = 50
    # Output
    output_dir: str = "checkpoints"
    run_name: str   = "french_v2"


def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_ds, val_ds, cfg, ctx):
    model.eval()
    losses = {}
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        total = 0.0
        for _ in range(cfg.eval_iters):
            x, y = ds.get_batch(cfg.batch_size, DEVICE)
            with ctx:
                _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / cfg.eval_iters
    model.train()
    return losses


def save_ckpt(model, optimizer, cfg, iteration, losses, out_dir, is_best=False):
    ckpt = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "mlp_internal_dim_multiplier": cfg.mlp_multiplier,
            "dropout": cfg.dropout,
            "vocab_size": cfg.vocab_size,
            "rho_decay": cfg.rho_decay,
            "rho_max": cfg.rho_max,
            "per_layer_encoders": True,  # V2 flag
        },
        "training_config": asdict(cfg),
        "losses": losses,
    }
    torch.save(ckpt, out_dir / f"checkpoint_{iteration:06d}.pt")
    torch.save(ckpt, out_dir / "checkpoint_latest.pt")
    if is_best:
        torch.save(ckpt, out_dir / "checkpoint_best.pt")
        print(f"  🏆 New best model! Val loss: {losses['val']:.4f}")


def train():
    cfg = TrainConfig()
    print("=" * 60)
    print("🐉 BDH V2 Training — Per-Layer Encoders + Persistent ρ")
    print("=" * 60)
    print(f"  Layers: {cfg.n_layer}, Embd: {cfg.n_embd}, Heads: {cfg.n_head}")
    print(f"  Neurons/head: {cfg.n_embd * cfg.mlp_multiplier // cfg.n_head}")
    print(f"  ρ decay: {cfg.rho_decay}, max: {cfg.rho_max}")
    print(f"  Max iters: {cfg.max_iters}")
    print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print("=" * 60)

    # Precision context
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    ptdtype = dtype_map[DTYPE]
    ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if DEVICE == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler(device=DEVICE, enabled=(DTYPE == "float16"))

    # Output dir
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Data
    print("\n📂 Loading data...")
    train_ds = ByteDataset(cfg.train_data, cfg.block_size)
    val_ds   = ByteDataset(cfg.val_data,   cfg.block_size)

    # Model
    print("\n🧠 Creating BDH V2 model...")
    model = create_model(
        n_layer=cfg.n_layer, n_embd=cfg.n_embd, n_head=cfg.n_head,
        mlp_multiplier=cfg.mlp_multiplier, dropout=cfg.dropout,
        vocab_size=cfg.vocab_size, rho_decay=cfg.rho_decay,
        rho_max=cfg.rho_max,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  ρ buffer shape: {model.rho.shape}")

    # Skip torch.compile — it materializes extra buffers causing OOM on 40GB GPUs
    compile_ok = False
    print("  ⚠ torch.compile disabled (saves ~5GB VRAM)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    # Training
    print("\n🚀 Starting training...")
    model.train()
    best_val = float("inf")
    running_loss = 0.0
    nan_recovery_count = 0
    max_nan_recoveries = 5  # Stop after this many NaN rollbacks
    t0 = time.time()
    x, y = train_ds.get_batch(cfg.batch_size, DEVICE)

    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps
            x, y = train_ds.get_batch(cfg.batch_size, DEVICE)
            scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        current_loss = loss.item() * cfg.gradient_accumulation_steps

        # ── NaN detection & recovery ──
        if math.isnan(current_loss) or math.isinf(current_loss):
            nan_recovery_count += 1
            print(f"  ⚠ NaN/Inf loss at iter {it}! Recovery attempt {nan_recovery_count}/{max_nan_recoveries}")
            if nan_recovery_count > max_nan_recoveries:
                print(f"  ❌ Too many NaN recoveries. Stopping training.")
                print(f"  ℹ Your best checkpoint is still valid at: {out_dir / 'checkpoint_best.pt'}")
                break
            best_ckpt = out_dir / "checkpoint_best.pt"
            if best_ckpt.exists():
                ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                model.train()  # Ensure training mode after state_dict load
                print(f"  ✅ Restored from best checkpoint (iter {ckpt.get('iteration', '?')})")
            else:
                print(f"  ❌ No checkpoint to restore! Stopping.")
                break
            running_loss = 0.0
            t0 = time.time()
            continue

        running_loss += current_loss

        # Logging
        if it % cfg.log_interval == 0:
            avg = running_loss / cfg.log_interval if it > 0 else running_loss
            elapsed = time.time() - t0
            tps = (cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps * cfg.log_interval) / elapsed if it > 0 else 0
            rho_norm = model.rho.norm().item() if not compile_ok else 0
            print(f"  iter {it:6d} | loss {avg:.4f} | lr {lr:.2e} | {tps:.0f} tok/s | ρ‖={rho_norm:.1f}")
            running_loss = 0.0
            t0 = time.time()

        # Eval
        if it > 0 and it % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_ds, val_ds, cfg, ctx)
            print(f"  📊 Eval@{it}: train={losses['train']:.4f}, val={losses['val']:.4f}")
            is_best = losses["val"] < best_val
            if is_best:
                best_val = losses["val"]
            if it % cfg.save_interval == 0 or is_best:
                save_ckpt(model, optimizer, cfg, it, losses, out_dir, is_best)

    # Final
    print("\n✅ Training complete!")
    losses = estimate_loss(model, train_ds, val_ds, cfg, ctx)
    save_ckpt(model, optimizer, cfg, cfg.max_iters, losses, out_dir, losses["val"] < best_val)

    # Quick test
    print("\n📝 Test generation:")
    model.eval()
    prompt_text = "<F:en>The European Parliament<T:fr>"
    prompt = torch.tensor([list(prompt_text.encode('utf-8'))], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=80, top_k=5)
    print(f"  {bytes(out[0].cpu().tolist()).decode('utf-8', errors='replace')}")

    # Sparsity check
    test_text = "le parlement européen a voté cette résolution"
    tokens = torch.tensor([list(test_text.encode('utf-8'))], dtype=torch.long, device=DEVICE)
    ext_cfg = ExtractionConfig(capture_sparse_activations=True, capture_attention_patterns=False)
    with torch.no_grad():
        with model.extraction_mode(ext_cfg) as buf:
            model(tokens)
            stats = buf.get_sparsity_stats()
    print(f"\n📊 Sparsity: x={stats['overall_x_sparsity']:.1%}, y={stats['overall_y_sparsity']:.1%}")
    print(f"   ρ buffer norm: {model.rho.norm().item():.2f}")
    print(f"   ρ per-layer norms: {[f'{model.rho[l].norm().item():.2f}' for l in range(cfg.n_layer)]}")

    return model


# %% Cell 6: RUN TRAINING
model = train()


# %% Cell 7: Download checkpoint
print("\n" + "=" * 60)
print("📥 FILES TO DOWNLOAD")
print("=" * 60)

ckpt_dir = Path("checkpoints/french_v2")
best = ckpt_dir / "checkpoint_best.pt"
latest = ckpt_dir / "checkpoint_latest.pt"

if best.exists():
    print(f"  ✅ {best} ({best.stat().st_size/1e6:.1f} MB)")
else:
    print(f"  ⚠ No best checkpoint, using latest")
    best = latest

print(f"""
WHAT TO DO:
  1. Download: checkpoints/french_v2/checkpoint_best.pt
  2. Copy it to your local project:
     → checkpoints/french/french_best.pt  (replace the old one)
  3. Then locally run:
     python scripts/precompute_monosemanticity.py
  4. The precompute script will automatically use the new model
     and generate updated monosemanticity data.

The V2 model has:
  - Per-layer encoders (each layer specializes differently)
  - Persistent ρ buffer (Hebbian long-term potentiation)
  - Same architecture shape (8L × 4H × 8192N) so all existing
    backend/frontend code works WITHOUT changes.
""")

# If running in Colab, offer download:
try:
    from google.colab import files
    files.download(str(best))
    print("✅ Download started!")
except ImportError:
    print("(Not in Colab — copy the file manually)")
