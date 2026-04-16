# Copyright 2025 Pathway Technology, Inc.
# Modified for BDH Interpretability Suite with extraction hooks

"""
BDH (Baby Dragon Hatchling) model architecture with extraction hooks
for interpretability analysis.

Based on: https://github.com/pathwaycom/bdh
"""

import dataclasses
import math
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    """Configuration for BDH model."""
    n_layer: int = 6
    n_embd: int = 256  # D: embedding dimension
    dropout: float = 0.1
    n_head: int = 4  # nh: number of attention heads
    mlp_internal_dim_multiplier: int = 128  # N = D * this / n_head
    vocab_size: int = 256  # UTF-8 bytes
    # V2 additions
    per_layer_encoders: bool = False  # True = V2 per-layer encoder/decoder
    rho_decay: float = 0.99  # EMA decay for persistent ρ buffer (0 = disabled)
    rho_max: float = 5.0  # Maximum ρ magnitude (prevents runaway)
    
    @property
    def n_neurons(self) -> int:
        """Total number of neurons (n) per head."""
        return self.n_embd * self.mlp_internal_dim_multiplier // self.n_head
    
    @property
    def total_neurons(self) -> int:
        """Total neurons across all heads."""
        return self.n_neurons * self.n_head


@dataclasses.dataclass
class ExtractionConfig:
    """Configuration for what to extract during forward pass."""
    capture_sparse_activations: bool = True
    capture_attention_patterns: bool = True
    capture_pre_relu: bool = True
    capture_layer_outputs: bool = True
    capture_residuals: bool = False
    layers_to_capture: Optional[List[int]] = None  # None = all layers
    
    def should_capture_layer(self, layer_idx: int) -> bool:
        if self.layers_to_capture is None:
            return True
        return layer_idx in self.layers_to_capture


class ExtractionBuffer:
    """Buffer to store extracted activations during forward pass."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.x_sparse: Dict[int, torch.Tensor] = {}  # layer -> sparse x activations
        self.y_sparse: Dict[int, torch.Tensor] = {}  # layer -> sparse y activations
        self.x_pre_relu: Dict[int, torch.Tensor] = {}  # layer -> pre-ReLU x
        self.y_pre_relu: Dict[int, torch.Tensor] = {}  # layer -> pre-ReLU y
        self.attention_scores: Dict[int, torch.Tensor] = {}  # layer -> attention
        self.attention_output: Dict[int, torch.Tensor] = {}  # layer -> a* (attention output after LN)
        self.layer_outputs: Dict[int, torch.Tensor] = {}  # layer -> output
        self.residuals: Dict[int, torch.Tensor] = {}  # layer -> residual
        self.pre_layernorm: Optional[torch.Tensor] = None  # raw embedding before initial LN
        self.input_tokens: Optional[torch.Tensor] = None
        self.final_output: Optional[torch.Tensor] = None
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Compute sparsity statistics across all captured layers."""
        stats = {}
        
        for layer_idx, x_sparse in self.x_sparse.items():
            total = x_sparse.numel()
            nonzero = (x_sparse > 0).sum().item()
            stats[f"layer_{layer_idx}_x_sparsity"] = 1 - (nonzero / total)
        
        for layer_idx, y_sparse in self.y_sparse.items():
            total = y_sparse.numel()
            nonzero = (y_sparse > 0).sum().item()
            stats[f"layer_{layer_idx}_y_sparsity"] = 1 - (nonzero / total)
        
        # Overall sparsity
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
        """Get indices of active neurons for a specific layer."""
        result = {}
        
        if layer in self.x_sparse:
            x = self.x_sparse[layer]
            # x shape: (B, nh, T, N)
            # For each head, find which neurons are active
            result["x_active"] = (x > threshold).any(dim=(0, 2))  # (nh, N)
        
        if layer in self.y_sparse:
            y = self.y_sparse[layer]
            result["y_active"] = (y > threshold).any(dim=(0, 2))  # (nh, N)
        
        return result
    
    def to_dict(self, detach: bool = True, cpu: bool = True) -> Dict[str, Any]:
        """Convert buffer to dictionary for JSON export."""
        def process_tensor(t: torch.Tensor) -> torch.Tensor:
            if detach:
                t = t.detach()
            if cpu:
                t = t.cpu()
            return t
        
        return {
            "x_sparse": {k: process_tensor(v) for k, v in self.x_sparse.items()},
            "y_sparse": {k: process_tensor(v) for k, v in self.y_sparse.items()},
            "x_pre_relu": {k: process_tensor(v) for k, v in self.x_pre_relu.items()},
            "y_pre_relu": {k: process_tensor(v) for k, v in self.y_pre_relu.items()},
            "attention_scores": {k: process_tensor(v) for k, v in self.attention_scores.items()},
            "attention_output": {k: process_tensor(v) for k, v in self.attention_output.items()},
            "layer_outputs": {k: process_tensor(v) for k, v in self.layer_outputs.items()},
            "input_tokens": process_tensor(self.input_tokens) if self.input_tokens is not None else None,
            "final_output": process_tensor(self.final_output) if self.final_output is not None else None,
        }


def get_freqs(n: int, theta: float, dtype: torch.dtype) -> torch.Tensor:
    """Generate RoPE frequency tensor."""
    def quantize(t, q=2):
        return (t / q).floor() * q
    
    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(nn.Module):
    """
    Linear attention with RoPE (Rotary Position Embedding).
    
    Key difference from transformer attention:
    - Operates in sparse neuron space (N dimensions), not embedding space (D)
    - Linear complexity O(T) instead of O(T²) due to structure
    - No softmax - values remain interpretable
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        N = config.n_neurons
        
        # RoPE frequencies - registered as buffer (not parameter)
        self.register_buffer(
            "freqs",
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )
    
    @staticmethod
    def phases_cos_sin(phases: torch.Tensor):
        """Compute cos and sin of phases for RoPE."""
        phases = (phases % 1) * (2 * math.pi)
        return torch.cos(phases), torch.sin(phases)
    
    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding to input tensor."""
        # Rotate pairs of dimensions
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        return_scores: bool = False
    ):
        """
        Compute linear attention.
        
        Args:
            Q: Query tensor (B, nh, T, N) - in sparse neuron space
            K: Key tensor (B, nh, T, N) - same as Q for self-attention
            V: Value tensor (B, 1, T, D) - in embedding space
            return_scores: Whether to return attention scores
        
        Returns:
            Output tensor and optionally attention scores
        """
        assert self.freqs.dtype == torch.float32
        assert K is Q  # Self-attention
        
        _, _, T, _ = Q.size()
        
        # Compute position-dependent phases
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE
        QR = self.rope(r_phases, Q)
        KR = QR  # Same as Q for self-attention
        
        # Compute attention scores with causal mask
        # This is the Hebbian co-activation matrix!
        scores = (QR @ KR.mT).tril(diagonal=-1)  # (B, nh, T, T)
        
        output = scores @ V
        
        if return_scores:
            return output, scores
        return output


class BDH(nn.Module):
    """
    Baby Dragon Hatchling - A biologically-inspired language model.
    
    Key architectural features:
    - Sparse activations via ReLU after expansion (D -> N)
    - Hebbian learning through attention co-activation
    - Scale-free graph topology in weight matrices
    - Linear attention complexity O(T)
    
    The model processes through layers:
    1. Embed tokens to D dimensions
    2. For each layer:
       a. Encode: project D -> N (expand to neuron space)
       b. Sparsify: ReLU (this creates ~95% sparsity)
       c. Attend: linear attention in sparse space
       d. Encode V: project attended values D -> N
       e. Sparsify V: ReLU
       f. Gate: element-wise multiply (x_sparse * y_sparse)
       g. Decode: project N -> D (back to embedding space)
       h. Residual: add to input
    3. Project to vocabulary for prediction
    """
    
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        
        nh = config.n_head
        D = config.n_embd
        N = config.n_neurons
        nL = config.n_layer
        
        # Encoder/decoder projections — V1 (shared) or V2 (per-layer)
        if config.per_layer_encoders:
            # V2: Each layer gets its own encoder/decoder
            self.encoders = nn.ParameterList([
                nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
                for _ in range(nL)
            ])
            self.encoders_v = nn.ParameterList([
                nn.Parameter(torch.zeros(nh, D, N).normal_(std=0.02))
                for _ in range(nL)
            ])
            self.decoders = nn.ParameterList([
                nn.Parameter(torch.zeros(nh * N, D).normal_(std=0.02))
                for _ in range(nL)
            ])
            # Persistent ρ buffer — Hebbian EMA accumulator
            self.register_buffer('rho', torch.zeros(nL, nh, N))
        else:
            # V1: Single shared encoder/decoder across all layers
            self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
            self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
            self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        
        # Attention module
        self.attn = Attention(config)
        
        # Layer normalization (no learnable parameters)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Token embedding and output projection
        self.embed = nn.Embedding(config.vocab_size, D)
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))
        
        # Dropout for regularization
        self.drop = nn.Dropout(config.dropout)
        
        # Extraction state
        self._extraction_config: Optional[ExtractionConfig] = None
        self._extraction_buffer: Optional[ExtractionBuffer] = None
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with small normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @contextmanager
    def extraction_mode(self, config: Optional[ExtractionConfig] = None):
        """
        Context manager for extraction mode.
        
        Usage:
            with model.extraction_mode() as buffer:
                logits, loss = model(tokens)
                sparsity = buffer.get_sparsity_stats()
        """
        self._extraction_config = config or ExtractionConfig()
        self._extraction_buffer = ExtractionBuffer()
        
        try:
            yield self._extraction_buffer
        finally:
            self._extraction_config = None
            # Keep buffer for access after context
    
    def get_extraction_buffer(self) -> Optional[ExtractionBuffer]:
        """Get the current extraction buffer."""
        return self._extraction_buffer
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with optional extraction.
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation (B, T)
        
        Returns:
            logits: Output logits (B, T, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        C = self.config
        # ── Thread-safety: snapshot extraction state into locals ──
        # Another thread may exit extraction_mode() and set self._extraction_config = None
        # while this forward() is still running. Cache into local vars to prevent that.
        ext_config = self._extraction_config
        extracting = ext_config is not None
        buffer = self._extraction_buffer if extracting else None
        
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = C.n_neurons
        
        # Store input tokens if extracting
        if extracting and buffer:
            buffer.input_tokens = idx.clone()
        
        # Embed tokens: (B, T) -> (B, T, D) -> (B, 1, T, D)
        x = self.embed(idx).unsqueeze(1)
        
        # Capture raw embedding before LayerNorm
        if extracting and buffer:
            buffer.pre_layernorm = x.clone()
        
        # Initial layer norm
        x = self.ln(x)
        
        # Process through layers
        for layer_idx in range(C.n_layer):
            should_capture = (
                extracting and 
                buffer and 
                ext_config.should_capture_layer(layer_idx)
            ) if extracting else False
            
            # Get encoder/decoder for this layer (V1 shared or V2 per-layer)
            if C.per_layer_encoders:
                enc   = self.encoders[layer_idx]
                enc_v = self.encoders_v[layer_idx]
                dec   = self.decoders[layer_idx]
            else:
                enc   = self.encoder
                enc_v = self.encoder_v
                dec   = self.decoder
            
            # === ENCODE: D -> N ===
            # Project to high-dimensional neuron space
            x_latent = x @ enc  # (B, nh, T, N)
            
            if extracting and buffer and ext_config.capture_pre_relu:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.x_pre_relu[layer_idx] = x_latent.clone()
            
            # === SPARSIFY: ReLU ===
            # THIS IS WHERE THE MAGIC HAPPENS
            # ~95% of activations become zero here
            x_sparse = F.relu(x_latent)  # (B, nh, T, N)
            
            if extracting and buffer and ext_config.capture_sparse_activations:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.x_sparse[layer_idx] = x_sparse.clone()
            
            # === ATTEND ===
            # Linear attention in sparse neuron space
            if extracting and buffer and ext_config.capture_attention_patterns:
                if ext_config.should_capture_layer(layer_idx):
                    yKV, scores = self.attn(Q=x_sparse, K=x_sparse, V=x, return_scores=True)
                    buffer.attention_scores[layer_idx] = scores.clone()
                else:
                    yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            else:
                yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            
            yKV = self.ln(yKV)
            
            # Capture attention output (a*) for interpretability
            if should_capture:
                buffer.attention_output[layer_idx] = yKV.clone()
            
            # === ENCODE V: D -> N ===
            y_latent = yKV @ enc_v  # (B, nh, T, N)
            
            if extracting and buffer and ext_config.capture_pre_relu:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.y_pre_relu[layer_idx] = y_latent.clone()
            
            # === SPARSIFY V ===
            y_sparse = F.relu(y_latent)  # (B, nh, T, N)
            
            if extracting and buffer and ext_config.capture_sparse_activations:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.y_sparse[layer_idx] = y_sparse.clone()
            
            # === GATE ===
            # Element-wise multiplication creates sparse gating
            xy_sparse = x_sparse * y_sparse  # (B, nh, T, N)
            
            # V2: Update ρ from RAW gate, then modulate
            if C.per_layer_encoders and hasattr(self, 'rho'):
                # Update ρ with EMA from raw gate BEFORE amplification
                if self.training and not extracting:
                    with torch.no_grad():
                        gate_mean = xy_sparse.detach().mean(dim=(0, 2))  # (nh, N)
                        self.rho[layer_idx] = (
                            C.rho_decay * self.rho[layer_idx] +
                            (1.0 - C.rho_decay) * gate_mean
                        )
                        self.rho[layer_idx].clamp_(-C.rho_max, C.rho_max)
                
                # Modulate gate by persistent ρ
                rho_layer = self.rho[layer_idx].unsqueeze(0).unsqueeze(2)  # (1, nh, 1, N)
                xy_sparse = xy_sparse * (1.0 + rho_layer)
            
            # Dropout for regularization
            xy_sparse = self.drop(xy_sparse)
            
            # === DECODE: N -> D ===
            # Project back to embedding dimension
            # Reshape: (B, nh, T, N) -> (B, 1, T, nh*N) -> (B, 1, T, D)
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ dec
            )
            y = self.ln(yMLP)
            
            # === RESIDUAL ===
            if extracting and buffer and ext_config.capture_residuals:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.residuals[layer_idx] = y.clone()
            
            x = self.ln(x + y)
            
            if extracting and buffer and ext_config.capture_layer_outputs:
                if ext_config.should_capture_layer(layer_idx):
                    buffer.layer_outputs[layer_idx] = x.clone()
        
        # Final projection to vocabulary
        logits = x.view(B, T, D) @ self.lm_head
        
        if extracting and buffer:
            buffer.final_output = logits.clone()
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def get_graph_topology(self, threshold: float = 0.01) -> Dict[str, Any]:
        """
        Extract graph topology from encoder/decoder weight matrices.
        
        The product E @ D forms a graph structure where:
        - Nodes are neurons
        - Edges are connection weights above threshold
        
        Returns:
            Dictionary with nodes, edges, and statistics
        """
        with torch.no_grad():
            nh = self.config.n_head
            N = self.config.n_neurons
            D = self.config.n_embd
            
            # Use layer 0 encoder/decoder (V2) or shared (V1)
            if self.config.per_layer_encoders:
                encoder = self.encoders[0]
                decoder = self.decoders[0]
            else:
                encoder = self.encoder
                decoder = self.decoder
            
            decoder_reshaped = decoder.view(nh, N, D)
            
            # For neuron-to-neuron graph: G = decoder @ encoder
            # Shape: (nh, N, D) @ (nh, D, N) -> (nh, N, N)
            G = torch.bmm(decoder_reshaped, encoder)  # (nh, N, N)
            
            # Apply threshold to get adjacency
            G_abs = G.abs()
            adjacency = (G_abs > threshold).float()
            
            # Compute statistics
            edges_per_head = adjacency.sum(dim=(1, 2)).tolist()
            total_edges = sum(edges_per_head)
            max_possible = nh * N * N
            density = total_edges / max_possible
            
            # Degree distribution (sum of connections per neuron)
            out_degree = adjacency.sum(dim=2)  # (nh, N)
            in_degree = adjacency.sum(dim=1)   # (nh, N)
            
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
    n_layer: int = 6,
    n_embd: int = 256,
    n_head: int = 4,
    mlp_multiplier: int = 128,
    dropout: float = 0.1,
    vocab_size: int = 256,
) -> BDH:
    """Factory function to create a BDH model with specified config."""
    config = BDHConfig(
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        mlp_internal_dim_multiplier=mlp_multiplier,
        dropout=dropout,
        vocab_size=vocab_size,
    )
    return BDH(config)


def load_model(checkpoint_path: str, device: str = "cpu") -> BDH:
    """Load a trained BDH model from checkpoint. Supports V1 (shared) and V2 (per-layer) models."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state_dict only
    if "config" in checkpoint:
        cfg_dict = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
        # Detect V2 from state dict keys even if config doesn't say so
        is_v2_sd = any(k.startswith("encoders.") for k in state_dict)
        if "per_layer_encoders" not in cfg_dict:
            cfg_dict["per_layer_encoders"] = is_v2_sd
        if "rho_decay" not in cfg_dict:
            cfg_dict["rho_decay"] = 0.99
        if "rho_max" not in cfg_dict:
            cfg_dict["rho_max"] = 5.0
        # Filter out any extra keys not in BDHConfig
        valid_keys = {f.name for f in dataclasses.fields(BDHConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        config = BDHConfig(**cfg_dict)
    else:
        state_dict = checkpoint
        # Strip _orig_mod. prefix from torch.compile() wrapped models
        stripped = {}
        for k, v in state_dict.items():
            new_k = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            stripped[new_k] = v
        state_dict = stripped

        # Detect V2 (per-layer) vs V1 (shared)
        is_v2 = "encoders.0" in state_dict
        
        if is_v2:
            encoder_shape = state_dict["encoders.0"].shape
            n_head = encoder_shape[0]
            n_embd = encoder_shape[1]
            N = encoder_shape[2]
            mlp_multiplier = (N * n_head) // n_embd
            n_layer = sum(1 for k in state_dict if k.startswith("encoders.") and k.endswith(".0") is False and "." in k)
            # Count encoder layers properly
            n_layer = max(int(k.split(".")[1]) for k in state_dict if k.startswith("encoders.")) + 1
        else:
            encoder_shape = state_dict["encoder"].shape
            n_head = encoder_shape[0]
            n_embd = encoder_shape[1]
            N = encoder_shape[2]
            mlp_multiplier = (N * n_head) // n_embd
            n_layer = 6  # Default for V1
        
        config = BDHConfig(
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=n_head,
            mlp_internal_dim_multiplier=mlp_multiplier,
            per_layer_encoders=is_v2,
        )
    
    # Strip _orig_mod. prefix in case it's in model_state_dict too
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k] = v
    
    model = BDH(config)
    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()
    
    return model
