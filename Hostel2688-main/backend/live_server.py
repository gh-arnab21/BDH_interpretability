#!/usr/bin/env python3
"""Standalone BDH inference server. Usage: python live_server.py --model checkpoints/french/french_best.pt"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# PyTorch
import torch
import torch.nn.functional as F
@dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 192
    dropout: float = 0.0  # No dropout during inference
    n_head: int = 4
    mlp_dim_mult: int = 64           # N = D * mlp_dim_mult / n_head
    vocab_size: int = 256
    block_size: int = 4096            # max sequence length (pos_emb size)

    @property
    def n_neurons(self) -> int:
        return self.n_embd * self.mlp_dim_mult // self.n_head

    @property
    def total_neurons(self) -> int:
        return self.n_neurons * self.n_head


def get_freqs(n: int, theta: float = 2**16) -> torch.Tensor:
    def quantize(t, q=2):
        return (t / q).floor() * q
    return (1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=torch.float32)) / n)) / (2 * math.pi))


class BDH(torch.nn.Module):
    """
    BDH model for inference — matches the new checkpoint format:
      decoder_x (nh, D, N) — expands D→N for the x path
      decoder_y (nh, D, N) — expands D→N for the y path
      encoder   (nh*N, D)  — compresses N→D
      pos_emb   — learned positional embeddings
      rope_freqs — RoPE frequency buffer
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh, D, N = config.n_head, config.n_embd, config.n_neurons

        # Sparse projections (checkpoint naming convention)
        self.decoder_x = torch.nn.Parameter(torch.zeros((nh, D, N)))
        self.decoder_y = torch.nn.Parameter(torch.zeros((nh, D, N)))
        self.encoder = torch.nn.Parameter(torch.zeros((nh * N, D)))

        # Normalization, embedding, output projection
        self.ln = torch.nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.embed = torch.nn.Embedding(config.vocab_size, D)
        self.pos_emb = torch.nn.Embedding(config.block_size, D)
        self.lm_head = torch.nn.Parameter(torch.zeros((D, config.vocab_size)))

        # RoPE frequencies
        self.register_buffer("rope_freqs", get_freqs(N).view(1, 1, 1, N))

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding."""
        _, _, T, _ = x.size()
        r_phases = (
            torch.arange(0, T, device=self.rope_freqs.device,
                         dtype=self.rope_freqs.dtype)
            .view(1, 1, -1, 1) * self.rope_freqs
        )
        phases = (r_phases % 1) * (2 * math.pi)
        v_rot = torch.stack((-x[..., 1::2], x[..., ::2]),
                            dim=-1).view(*x.size())
        return (x * torch.cos(phases)).to(x.dtype) + (v_rot * torch.sin(phases)).to(x.dtype)

    def forward_with_extraction(self, idx: torch.Tensor) -> tuple:
        """Forward pass that captures all activations for visualization."""
        C = self.config
        B, T = idx.size()
        D, nh, N = C.n_embd, C.n_head, C.n_neurons

        extractions = {
            "x_sparse": [],
            "y_sparse": [],
            "attention_scores": [],
        }

        # Embed with positional information
        positions = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_emb(positions)   # (B, T, D)
        x = x.unsqueeze(1)                               # (B, 1, T, D)
        x = self.ln(x)

        for layer_idx in range(C.n_layer):
            # Expand to neuron space  D → N  (per head)
            x_latent = x @ self.decoder_x                 # (B, nh, T, N)
            x_sparse = F.relu(x_latent)
            extractions["x_sparse"].append(x_sparse.detach().cpu())

            # RoPE + causal linear attention
            QR = self._rope(x_sparse)
            scores = (QR @ QR.mT).tril(diagonal=-1)
            yKV = scores @ x                              # (B, nh, T, D)
            extractions["attention_scores"].append(scores.detach().cpu())

            # Value path  D → N
            yKV = self.ln(yKV)
            y_latent = yKV @ self.decoder_y
            y_sparse = F.relu(y_latent)
            extractions["y_sparse"].append(y_sparse.detach().cpu())

            # Gating + decode  N → D
            xy_sparse = x_sparse * y_sparse
            yMLP = xy_sparse.transpose(1, 2).reshape(
                B, 1, T, N * nh) @ self.encoder
            x = self.ln(x + self.ln(yMLP))

        logits = x.view(B, T, D) @ self.lm_head
        return logits, extractions

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 5):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(
                1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self.forward_with_extraction(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


_LEGACY_KEY_MAP = {
    "encoder":   "decoder_x",     # V1 encoder (nh, D, N) → new decoder_x
    "encoder_v": "decoder_y",     # V1 encoder_v           → new decoder_y
    "decoder":   "encoder",       # V1 decoder (nh*N, D)   → new encoder
    "attn.freqs": "rope_freqs",   # V1 RoPE buffer         → new rope_freqs
}


def _remap_state_dict(sd: dict) -> dict:
    """Translate legacy V1 key names to new checkpoint convention."""
    # Detect if remapping is needed (new format already has 'decoder_x')
    if "decoder_x" in sd:
        return sd
    new = {}
    for k, v in sd.items():
        new[_LEGACY_KEY_MAP.get(k, k)] = v
    return new


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple:
    """Load a BDH checkpoint (supports both old V1 and new format)."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        cfg = checkpoint["config"]
        # Handle both config key conventions
        mult = cfg.get("mlp_dim_mult",
                       cfg.get("mlp_internal_dim_multiplier", 64))
        config = BDHConfig(
            n_layer=cfg.get("n_layer", 6),
            n_embd=cfg.get("n_embd", 192),
            n_head=cfg.get("n_head", 4),
            mlp_dim_mult=mult,
            vocab_size=cfg.get("vocab_size", 256),
        )
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        # Infer from weights — new format has decoder_x, old has encoder
        if "decoder_x" in state_dict:
            shape = state_dict["decoder_x"].shape          # (nh, D, N)
        elif "encoder" in state_dict and state_dict["encoder"].dim() == 3:
            shape = state_dict["encoder"].shape             # legacy (nh, D, N)
        else:
            raise ValueError("Cannot infer config from checkpoint keys")
        config = BDHConfig(
            n_layer=6,
            n_embd=shape[1],
            n_head=shape[0],
            mlp_dim_mult=(shape[2] * shape[0] // shape[1]),
        )

    # Detect block_size from pos_emb if present
    state_dict = _remap_state_dict(state_dict)
    if "pos_emb.weight" in state_dict:
        config.block_size = state_dict["pos_emb.weight"].shape[0]

    # Strip _orig_mod. prefix from torch.compile
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model = BDH(config)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(
        f"Loaded: {config.n_layer}L, {config.n_embd}D, {config.n_head}H, N={config.n_neurons}")
    return model, config
app = FastAPI(
    title="BDH Live Inference",
    description="Real-time inference and visualization for BDH models"
)

# CORS - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL = None
CONFIG = None
DEVICE = "cpu"


# Request/Response models
class InferenceRequest(BaseModel):
    text: str
    include_attention: bool = False


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0
@app.get("/")
def root():
    return {
        "service": "BDH Live Inference",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "config": asdict(CONFIG) if CONFIG else None,
    }


@app.post("/api/inference/run")
def run_inference(request: InferenceRequest):
    """
    Main endpoint: Run any text through the model and get activations.

    This is what powers the live visualization!
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.text

    # Tokenize (byte-level)
    tokens = torch.tensor([list(text.encode('utf-8'))],
                          dtype=torch.long, device=DEVICE)
    T = tokens.shape[1]

    if T > 1024:
        raise HTTPException(
            status_code=400, detail="Text too long (max 1024 bytes)")

    # Run inference with extraction
    with torch.no_grad():
        logits, extractions = MODEL.forward_with_extraction(tokens)

    # Build response
    frames = []
    sparsity_by_layer = []

    for layer_idx in range(CONFIG.n_layer):
        x_sparse = extractions["x_sparse"][layer_idx][0]  # (nh, T, N)
        y_sparse = extractions["y_sparse"][layer_idx][0]

        # Compute sparsity
        x_sparsity = 1 - ((x_sparse > 0).sum().item() / x_sparse.numel())
        y_sparsity = 1 - ((y_sparse > 0).sum().item() / y_sparse.numel())
        sparsity_by_layer.append((x_sparsity + y_sparsity) / 2)

        # Build frames for each token
        for t in range(T):
            token_byte = tokens[0, t].item()
            try:
                token_char = chr(
                    token_byte) if 32 <= token_byte < 127 else f"\\x{token_byte:02x}"
            except:
                token_char = f"\\x{token_byte:02x}"

            x_t = x_sparse[:, t, :]  # (nh, N)
            y_t = y_sparse[:, t, :]

            # Only store non-zero activations (sparse!)
            x_active = []
            y_active = []

            for h in range(CONFIG.n_head):
                x_nz = (x_t[h] > 0).nonzero().squeeze(-1)
                y_nz = (y_t[h] > 0).nonzero().squeeze(-1)

                # Limit to top 100 for efficiency
                if x_nz.numel() > 100:
                    vals = x_t[h][x_nz]
                    top_idx = vals.argsort(descending=True)[:100]
                    x_nz = x_nz[top_idx]

                if y_nz.numel() > 100:
                    vals = y_t[h][y_nz]
                    top_idx = vals.argsort(descending=True)[:100]
                    y_nz = y_nz[top_idx]

                x_active.append({
                    "indices": x_nz.tolist() if x_nz.numel() > 0 else [],
                    "values": [round(v, 4) for v in x_t[h][x_nz].tolist()] if x_nz.numel() > 0 else [],
                })
                y_active.append({
                    "indices": y_nz.tolist() if y_nz.numel() > 0 else [],
                    "values": [round(v, 4) for v in y_t[h][y_nz].tolist()] if y_nz.numel() > 0 else [],
                })

            frames.append({
                "token_idx": t,
                "token_byte": token_byte,
                "token_char": token_char,
                "layer": layer_idx,
                "x_active": x_active,
                "y_active": y_active,
                "x_sparsity": round(1 - ((x_t > 0).sum().item() / x_t.numel()), 4),
                "y_sparsity": round(1 - ((y_t > 0).sum().item() / y_t.numel()), 4),
            })

    # Build character list
    input_chars = []
    for byte in tokens[0].cpu().tolist():
        try:
            char = chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}"
        except:
            char = f"\\x{byte:02x}"
        input_chars.append(char)

    return {
        "input_text": text,
        "input_tokens": tokens[0].cpu().tolist(),
        "input_chars": input_chars,
        "num_layers": CONFIG.n_layer,
        "num_heads": CONFIG.n_head,
        "neurons_per_head": CONFIG.n_neurons,
        "frames": frames,
        "overall_sparsity": round(sum(sparsity_by_layer) / len(sparsity_by_layer), 4),
        "sparsity_by_layer": [round(s, 4) for s in sparsity_by_layer],
    }


@app.post("/api/inference/generate")
def generate_text(request: GenerateRequest):
    """Generate text continuation from a prompt."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Tokenize
    tokens = torch.tensor(
        [list(request.prompt.encode('utf-8'))], dtype=torch.long, device=DEVICE)

    # Generate
    with torch.no_grad():
        output = MODEL.generate(
            tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    # Decode
    full_bytes = bytes(output[0].cpu().tolist())
    full_text = full_bytes.decode('utf-8', errors='backslashreplace')

    return {
        "prompt": request.prompt,
        "generated": full_text[len(request.prompt):],
        "full_text": full_text,
    }


@app.get("/api/model/info")
def model_info():
    """Get information about the loaded model."""
    if CONFIG is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "config": asdict(CONFIG),
        "total_neurons": CONFIG.total_neurons,
        "device": DEVICE,
    }
def main():
    global MODEL, CONFIG, DEVICE

    parser = argparse.ArgumentParser(description="BDH Live Inference Server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    DEVICE = args.device

    print("=" * 60)
    print("🐉 BDH Live Inference Server")
    print("=" * 60)

    # Load model
    MODEL, CONFIG = load_model(args.model, DEVICE)

    print(f"\nServer starting on http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
