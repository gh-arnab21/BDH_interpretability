#!/usr/bin/env python3
"""
JSON Playback Data Generator

Generates pre-computed activation data for frontend visualization.
This enables smooth 60fps animations without requiring a live backend,
which is crucial for:
1. Demo presentations
2. Offline browsing
3. Handling 32k+ neurons without physics simulation lag

The generated JSON includes:
- Token-by-token activation snapshots
- Layer-by-layer data flow
- Sparsity patterns at each stage
- Attention patterns
- Graph topology (pre-computed node positions)

Usage:
    python generate_playback.py --model checkpoints/french/french_best.pt \
                                --output frontend/public/playback/
"""

from bdh import BDH, BDHConfig, ExtractionConfig, load_model
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import math

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))


@dataclass
class PlaybackFrame:
    """A single frame of playback data."""
    token_idx: int
    token_char: str
    token_byte: int
    layer: int

    # Sparse activations (indices of active neurons only, for efficiency)
    # Per head: list of active neuron indices
    x_active_indices: List[List[int]]
    x_active_values: List[List[float]]  # Per head: corresponding values
    y_active_indices: List[List[int]]
    y_active_values: List[List[float]]

    # Sparsity stats
    x_sparsity: float
    y_sparsity: float

    # Attention pattern for this token (which previous tokens it attends to)
    attention_weights: Optional[List[float]] = None


@dataclass
class PlaybackSequence:
    """Complete playback data for one input sequence."""
    input_text: str
    input_tokens: List[int]
    input_chars: List[str]
    num_layers: int
    num_heads: int
    neurons_per_head: int
    frames: List[Dict]  # List of PlaybackFrame as dicts

    # Aggregate statistics
    overall_sparsity: float
    sparsity_by_layer: List[float]


def compute_graph_layout(
    adjacency: np.ndarray,
    n_iterations: int = 100,
    seed: int = 42
) -> np.ndarray:
    """
    Compute force-directed layout for graph visualization.

    Uses a simplified force-directed algorithm that can handle
    large graphs (32k+ nodes) by:
    1. Using sparse representation
    2. Limiting iterations
    3. Pre-computing for frontend playback

    Returns:
        Node positions as (N, 3) array for 3D visualization
    """
    np.random.seed(seed)
    n_nodes = adjacency.shape[0]

    # Initialize random positions
    positions = np.random.randn(n_nodes, 3).astype(np.float32)

    # Normalize adjacency for force computation
    adj_sparse = (np.abs(adjacency) > 0.01).astype(np.float32)
    degrees = adj_sparse.sum(axis=1, keepdims=True) + 1

    # Force-directed iterations
    for _ in range(n_iterations):
        # Repulsive forces (all pairs, approximated)
        # Use random sampling for large graphs
        if n_nodes > 1000:
            sample_idx = np.random.choice(n_nodes, 1000, replace=False)
            sample_pos = positions[sample_idx]
            diff = positions[:, np.newaxis, :] - sample_pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2, keepdims=True) + 0.1
            repulsion = (diff / (dist ** 2)).sum(axis=1) * 0.1
        else:
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2, keepdims=True) + 0.1
            repulsion = (diff / (dist ** 2)).sum(axis=1) * 0.1

        # Attractive forces (connected nodes only)
        attraction = np.zeros_like(positions)
        for i in range(n_nodes):
            neighbors = np.where(adj_sparse[i] > 0)[0]
            if len(neighbors) > 0:
                neighbor_pos = positions[neighbors]
                diff = neighbor_pos - positions[i]
                attraction[i] = diff.mean(axis=0) * 0.5

        # Update positions
        positions += repulsion + attraction

        # Center and normalize
        positions -= positions.mean(axis=0)
        max_dist = np.abs(positions).max()
        if max_dist > 0:
            positions /= max_dist

    return positions


class PlaybackGenerator:
    """Generate playback data for frontend visualization."""

    def __init__(
        self,
        model: BDH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.config = model.config

    def _extract_frame(
        self,
        tokens: torch.Tensor,
        token_idx: int,
        layer: int,
        buffer
    ) -> PlaybackFrame:
        """Extract a single playback frame."""

        # Get sparse activations for this layer
        x_sparse = buffer.x_sparse[layer][0]  # (nh, T, N)
        y_sparse = buffer.y_sparse[layer][0]  # (nh, T, N)

        # Get activations for specific token
        x_token = x_sparse[:, token_idx, :]  # (nh, N)
        y_token = y_sparse[:, token_idx, :]  # (nh, N)

        # Find active neurons (non-zero) per head
        x_active_indices = []
        x_active_values = []
        y_active_indices = []
        y_active_values = []

        for head in range(self.config.n_head):
            # X activations
            x_head = x_token[head].cpu().numpy()
            x_nonzero = np.where(x_head > 0)[0]
            x_active_indices.append(x_nonzero.tolist())
            x_active_values.append(x_head[x_nonzero].tolist())

            # Y activations
            y_head = y_token[head].cpu().numpy()
            y_nonzero = np.where(y_head > 0)[0]
            y_active_indices.append(y_nonzero.tolist())
            y_active_values.append(y_head[y_nonzero].tolist())

        # Compute sparsity
        x_total = x_token.numel()
        x_active = (x_token > 0).sum().item()
        y_total = y_token.numel()
        y_active = (y_token > 0).sum().item()

        # Get attention weights for this token (if available)
        attention_weights = None
        if layer in buffer.attention_scores:
            attn = buffer.attention_scores[layer][0]  # (nh, T, T)
            # Average across heads, get weights for this token
            attn_token = attn.mean(
                dim=0)[token_idx, :token_idx+1]  # (token_idx+1,)
            attention_weights = attn_token.cpu().tolist()

        # Get token character
        token_byte = tokens[0, token_idx].item()
        try:
            token_char = chr(
                token_byte) if 32 <= token_byte < 127 else f"\\x{token_byte:02x}"
        except:
            token_char = f"\\x{token_byte:02x}"

        return PlaybackFrame(
            token_idx=token_idx,
            token_char=token_char,
            token_byte=token_byte,
            layer=layer,
            x_active_indices=x_active_indices,
            x_active_values=x_active_values,
            y_active_indices=y_active_indices,
            y_active_values=y_active_values,
            x_sparsity=1 - (x_active / x_total),
            y_sparsity=1 - (y_active / y_total),
            attention_weights=attention_weights,
        )

    def generate_sequence_playback(
        self,
        text: str,
        include_attention: bool = True
    ) -> PlaybackSequence:
        """Generate complete playback data for a text sequence."""

        # Tokenize
        tokens = torch.tensor(
            [list(text.encode('utf-8'))],
            dtype=torch.long,
            device=self.device
        )

        T = tokens.shape[1]

        # Extract with full capture
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
            capture_attention_patterns=include_attention,
            capture_pre_relu=False,
            capture_layer_outputs=False,
        )

        frames = []
        sparsity_by_layer = []

        with torch.no_grad():
            with self.model.extraction_mode(extraction_config) as buffer:
                _, _ = self.model(tokens)

                # Generate frames for each layer and token
                for layer in range(self.config.n_layer):
                    layer_sparsities = []

                    for token_idx in range(T):
                        frame = self._extract_frame(
                            tokens, token_idx, layer, buffer)
                        frames.append(asdict(frame))
                        layer_sparsities.append(
                            (frame.x_sparsity + frame.y_sparsity) / 2)

                    sparsity_by_layer.append(np.mean(layer_sparsities))

        # Build character list
        input_chars = []
        for byte in tokens[0].cpu().tolist():
            try:
                char = chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}"
            except:
                char = f"\\x{byte:02x}"
            input_chars.append(char)

        return PlaybackSequence(
            input_text=text,
            input_tokens=tokens[0].cpu().tolist(),
            input_chars=input_chars,
            num_layers=self.config.n_layer,
            num_heads=self.config.n_head,
            neurons_per_head=self.config.n_neurons,
            frames=frames,
            overall_sparsity=np.mean(sparsity_by_layer),
            sparsity_by_layer=sparsity_by_layer,
        )

    def generate_graph_data(self, threshold: float = 0.01) -> Dict[str, Any]:
        """Generate pre-computed graph topology data."""

        print("📊 Extracting graph topology from weights...")

        topology = self.model.get_graph_topology(threshold=threshold)

        # Compute layout for each head
        print("🔧 Computing 3D layouts (this may take a moment)...")
        layouts = {}

        for head in tqdm(range(self.config.n_head), desc="   Computing layouts"):
            adj = topology["adjacency"][head]  # (N, N)
            positions = compute_graph_layout(adj, n_iterations=50)
            layouts[f"head_{head}"] = positions.tolist()

        return {
            "n_heads": self.config.n_head,
            "n_neurons": self.config.n_neurons,
            "threshold": threshold,
            "density": topology["density"],
            "edges_per_head": topology["edges_per_head"],
            "layouts": layouts,
            "out_degrees": topology["out_degree"].tolist(),
            "in_degrees": topology["in_degree"].tolist(),
        }


# Example texts for playback generation
EXAMPLE_TEXTS = {
    "europarl_french": {
        "text": "<F:en>The European Parliament supports the amendment.<T:fr>Le Parlement européen soutient l'amendement.",
        "description": "French translation example"
    },
    "europarl_portuguese": {
        "text": "<F:en>The Commission welcomes this proposal.<T:pt>A Comissão acolhe esta proposta.",
        "description": "Portuguese translation example"
    },
    "currencies": {
        "text": "The price is 100 euros, or approximately 110 US dollars.",
        "description": "Currency concepts for monosemanticity demo"
    },
    "countries": {
        "text": "France, Germany, and Poland voted in favor of the resolution.",
        "description": "Country concepts for monosemanticity demo"
    },
    "mixed_concepts": {
        "text": "In January, the European Central Bank set the euro exchange rate against the dollar.",
        "description": "Multiple concepts: temporal, institution, currencies"
    },
    "simple": {
        "text": "Hello world",
        "description": "Simple test sequence"
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate playback data for frontend")
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--examples", nargs="+", default=list(EXAMPLE_TEXTS.keys()),
                        help="Which examples to generate")
    parser.add_argument("--include-graph", action="store_true",
                        help="Include graph topology")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎬 BDH Playback Data Generator")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")
    print(f"Examples: {args.examples}")
    print("=" * 60)

    # Load model
    print("\n📂 Loading model...")
    model = load_model(args.model, args.device)
    print(
        f"   Config: {model.config.n_layer}L, {model.config.n_embd}D, {model.config.n_head}H")

    generator = PlaybackGenerator(model, args.device)

    # Generate playback for each example
    print("\n🎬 Generating playback sequences...")

    manifest = {
        "model_config": {
            "n_layer": model.config.n_layer,
            "n_embd": model.config.n_embd,
            "n_head": model.config.n_head,
            "n_neurons": model.config.n_neurons,
        },
        "sequences": {},
    }

    for example_name in args.examples:
        if example_name not in EXAMPLE_TEXTS:
            print(f"   Warning: Unknown example '{example_name}', skipping")
            continue

        example = EXAMPLE_TEXTS[example_name]
        print(f"\n   Generating: {example_name}")
        print(f"   Text: {example['text'][:60]}...")

        sequence = generator.generate_sequence_playback(example["text"])

        # Save sequence
        seq_path = output_dir / f"{example_name}.json"
        with open(seq_path, "w") as f:
            json.dump(asdict(sequence), f)

        manifest["sequences"][example_name] = {
            "file": f"{example_name}.json",
            "description": example["description"],
            "num_tokens": len(sequence.input_tokens),
            "overall_sparsity": sequence.overall_sparsity,
        }

        print(f"   Tokens: {len(sequence.input_tokens)}")
        print(f"   Sparsity: {sequence.overall_sparsity:.1%}")
        print(f"   Saved: {seq_path}")

    # Generate graph data if requested
    if args.include_graph:
        print("\n🔗 Generating graph topology data...")
        graph_data = generator.generate_graph_data()

        graph_path = output_dir / "graph_topology.json"
        with open(graph_path, "w") as f:
            json.dump(graph_data, f)

        manifest["graph"] = {
            "file": "graph_topology.json",
            "density": graph_data["density"],
        }
        print(f"   Saved: {graph_path}")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Playback data generation complete!")
    print("=" * 60)
    print(f"\nFiles generated in {output_dir}:")
    print(f"  - manifest.json (index file)")
    for name in manifest["sequences"]:
        print(f"  - {name}.json")
    if "graph" in manifest:
        print(f"  - graph_topology.json")

    print("\nTo use in frontend:")
    print(f"  Copy {output_dir}/* to frontend/public/playback/")


if __name__ == "__main__":
    main()
