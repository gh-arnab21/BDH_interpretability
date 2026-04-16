#!/usr/bin/env python3
"""
Pre-compute monosemanticity data for static frontend visualization.

Generates a single JSON file containing:
- Per-word x_sparse fingerprints for every curated category
- Cosine similarity matrices (per layer)
- Top-K neuron indices per word
- Shared neuron intersection data
- Cross-concept distinctness (Jaccard) for negative-control pairs
- "best_layer" — layer with peak avg within-concept similarity
- **NEW** Selectivity scores per neuron (mean_in / (mean_in + mean_out))
- **NEW** Mann-Whitney U test p-values for concept selectivity
- **NEW** Synapse tracking timeseries (token-by-token x_sparse for example sentences)
- **NEW** Selectivity histogram (distribution of neuron selectivities)

Usage:
    python scripts/precompute_monosemanticity.py \
        --model checkpoints/french/french_best.pt \
        --output frontend/public/monosemanticity/precomputed.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

# ── Resolve imports ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from bdh import BDH, BDHConfig, ExtractionConfig, load_model  # noqa: E402

# ── Curated categories (must match frontend PRESETS) ─────────────────
CATEGORIES: Dict[str, Dict[str, Any]] = {
    "currencies": {
        "name": "Currencies",
        "icon": "💰",
        "words": ["dollar", "euro", "franc", "yen"],
    },
    "countries": {
        "name": "Countries",
        "icon": "🌍",
        "words": ["france", "germany", "spain", "italy"],
    },
    "languages": {
        "name": "Languages",
        "icon": "🗣️",
        "words": ["anglais", "français", "espagnol", "allemand"],
    },
    "politics": {
        "name": "Politics",
        "icon": "⚖️",
        "words": ["parlement", "commission", "conseil", "vote"],
    },
}

# Cross-concept pairs to pre-compute (primary, secondary)
CROSS_PAIRS = [
    ("currencies", "countries"),
    ("currencies", "languages"),
    ("countries", "politics"),
    ("languages", "politics"),
]

# ── Example sentences for synapse tracking ──────────────────────────
# Each sentence embeds a concept word in natural context so we can
# track x_sparse neuron activations token-by-token.
TRACKING_SENTENCES: Dict[str, List[str]] = {
    "currencies": [
        "le dollar est la monnaie des États-Unis",
        "un euro vaut environ un dollar aujourd'hui",
        "le parlement a voté pour le budget en euro",
    ],
    "countries": [
        "la france est un pays européen important",
        "le spain a une culture riche et diversifiée",
        "le parlement de germany se réunit chaque semaine",
    ],
    "languages": [
        "il parle couramment français et anglais",
        "l'espagnol est parlé dans de nombreux pays",
        "le vote était en français au parlement",
    ],
    "politics": [
        "le parlement européen a voté ce matin",
        "la commission propose un nouveau budget",
        "le conseil a adopté cette résolution",
    ],
}


def extract_fingerprint(
    model: BDH,
    words: List[str],
    concept_name: str,
    device: str,
    n_layers: int,
    n_heads: int,
    n_neurons: int,
) -> Dict[str, Any]:
    """
    Run words through model and return a FingerprintResult-shaped dict.
    Matches the /neuron-fingerprint backend response exactly.
    """
    word_fingerprints: List[Dict] = []
    # raw_x[layer][head] = list of (N,) numpy arrays, one per word
    raw_x: Dict[int, Dict[int, list]] = {}

    for text in words:
        tokens = torch.tensor(
            [list(text.encode("utf-8"))],
            dtype=torch.long,
            device=device,
        )
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
            capture_attention_patterns=False,
        )

        layers_data: List[Dict] = []
        with torch.no_grad():
            with model.extraction_mode(extraction_config) as buffer:
                model(tokens)

                for layer_idx in sorted(buffer.x_sparse.keys()):
                    x = buffer.x_sparse[layer_idx][0]  # (nh, T, N)

                    heads_data: List[Dict] = []
                    for h in range(n_heads):
                        x_mean = x[h].mean(dim=0).cpu().numpy()  # (N,)

                        # Downsample to 64 bins
                        bins = 64
                        stride = max(1, n_neurons // bins)
                        x_ds = []
                        for b in range(bins):
                            start = b * stride
                            end = min(start + stride, n_neurons)
                            x_ds.append(float(x_mean[start:end].max()))

                        x_active = int((x_mean > 0).sum())

                        # Top-K neurons
                        top_k = 20
                        top_idx = np.argsort(x_mean)[-top_k:][::-1]
                        top_neurons = [
                            {"idx": int(i), "val": round(float(x_mean[i]), 5)}
                            for i in top_idx
                            if x_mean[i] > 0
                        ]

                        heads_data.append(
                            {
                                "head": h,
                                "x_ds": x_ds,
                                "x_active": x_active,
                                "top_neurons": top_neurons,
                            }
                        )

                        raw_x.setdefault(layer_idx, {}).setdefault(h, []).append(
                            x_mean
                        )

                    layers_data.append({"layer": layer_idx, "heads": heads_data})

        word_fingerprints.append({"word": text, "layers": layers_data})

    # ── Cosine similarity matrix (per layer, averaged across heads) ──
    n_words = len(words)
    similarity_by_layer: Dict[str, list] = {}

    for layer_idx in sorted(raw_x.keys()):
        sim_matrix = np.zeros((n_words, n_words))
        for h in range(n_heads):
            vecs = np.stack(raw_x[layer_idx][h])  # (n_words, N)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
            normed = vecs / norms
            cos = normed @ normed.T
            sim_matrix += cos
        sim_matrix /= n_heads
        similarity_by_layer[str(layer_idx)] = [
            [round(float(sim_matrix[i][j]), 4) for j in range(n_words)]
            for i in range(n_words)
        ]

    # ── Shared neurons ──
    shared_neurons: List[Dict] = []
    for layer_idx in sorted(raw_x.keys()):
        for h in range(n_heads):
            acts = np.stack(raw_x[layer_idx][h])  # (n_words, N)
            active_mask = acts > 0
            all_active = active_mask.all(axis=0)
            shared_idx = np.where(all_active)[0]

            if len(shared_idx) > 0:
                mean_vals = acts[:, shared_idx].mean(axis=0)
                sort_order = np.argsort(mean_vals)[::-1][:5]
                for rank in sort_order:
                    nidx = int(shared_idx[rank])
                    shared_neurons.append(
                        {
                            "layer": int(layer_idx),
                            "head": int(h),
                            "neuron": nidx,
                            "mean_activation": round(float(mean_vals[rank]), 5),
                            "active_in": n_words,
                            "per_word": [
                                round(float(acts[w, nidx]), 5)
                                for w in range(n_words)
                            ],
                        }
                    )

    shared_neurons.sort(key=lambda s: s["mean_activation"], reverse=True)

    return {
        "concept": concept_name,
        "words": word_fingerprints,
        "similarity": similarity_by_layer,
        "shared_neurons": shared_neurons[:40],
        "model_info": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_neurons": n_neurons,
        },
        "raw_x": raw_x,  # kept temporarily for selectivity computation
    }


def compute_best_layer(concepts: Dict[str, Any], n_layers: int) -> int:
    """
    Find the layer with the highest average within-concept cosine similarity.
    This is the "most monosemantic" layer for the narrative default.
    """
    layer_scores: Dict[int, List[float]] = {l: [] for l in range(n_layers)}
    for _cid, result in concepts.items():
        sim = result["similarity"]
        for layer_str, matrix in sim.items():
            layer_idx = int(layer_str)
            n = len(matrix)
            total = sum(
                matrix[i][j] for i in range(n) for j in range(n) if i != j
            )
            count = n * (n - 1) if n > 1 else 1
            layer_scores[layer_idx].append(total / count)

    avg_scores = {l: np.mean(v) for l, v in layer_scores.items() if v}
    return int(max(avg_scores, key=avg_scores.get))


def compute_cross_concept(
    concepts: Dict[str, Any], n_layers: int, n_heads: int
) -> List[Dict]:
    """
    For each CROSS_PAIR, compute per-layer Jaccard distinctness between
    top-neuron sets of the two concepts.
    """
    cross_results: List[Dict] = []
    for primary_id, secondary_id in CROSS_PAIRS:
        p_result = concepts.get(primary_id)
        s_result = concepts.get(secondary_id)
        if not p_result or not s_result:
            continue

        distinctness_per_layer: List[float] = []
        for l in range(n_layers):
            p_neurons = set()
            for w in p_result["words"]:
                layer = next((la for la in w["layers"] if la["layer"] == l), None)
                if layer:
                    for h in layer["heads"]:
                        for n in h["top_neurons"]:
                            p_neurons.add(f"{h['head']}_{n['idx']}")

            s_neurons = set()
            for w in s_result["words"]:
                layer = next((la for la in w["layers"] if la["layer"] == l), None)
                if layer:
                    for h in layer["heads"]:
                        for n in h["top_neurons"]:
                            s_neurons.add(f"{h['head']}_{n['idx']}")

            intersection = len(p_neurons & s_neurons)
            union = len(p_neurons | s_neurons)
            distinctness_per_layer.append(
                round(1 - intersection / union, 4) if union > 0 else 1.0
            )

        # Strip raw_x from secondary result before embedding
        s_clean = {k: v for k, v in s_result.items() if k != "raw_x"}
        cross_results.append(
            {
                "primary": primary_id,
                "secondary": secondary_id,
                "distinctness_per_layer": distinctness_per_layer,
                "secondary_result": s_clean,
            }
        )

    return cross_results


def compute_selectivity(
    concepts: Dict[str, Any],
    best_layer: int,
    n_heads: int,
    n_neurons: int,
) -> Dict[str, Any]:
    """
    For each concept, compute neuron selectivity scores using all other
    concepts as the out-of-concept contrast.

    Selectivity = mean_in / (mean_in + mean_out)
    where mean_in = avg activation for words in this concept,
          mean_out = avg activation for words in all other concepts.

    Also computes Mann-Whitney U test p-values for top neurons.
    Returns per-concept monosemantic_neurons list and global histogram.

    IMPORTANT: The histogram includes ALL neurons (not just selective ones)
    to show the full distribution — a healthy BDH model should have a
    rightward tail, while a standard MLP would cluster near 0.5.
    """
    from scipy.stats import mannwhitneyu

    # First, collect per-concept activation vectors at best_layer
    concept_vecs: Dict[str, Dict[int, np.ndarray]] = {}  # cid → {head → (n_words, N)}
    for cid, result in concepts.items():
        raw_x = result.get("raw_x", {})
        if best_layer not in raw_x:
            continue
        head_vecs = {}
        for h in range(n_heads):
            if h in raw_x[best_layer]:
                head_vecs[h] = np.stack(raw_x[best_layer][h])  # (n_words, N)
        concept_vecs[cid] = head_vecs

    # Collect selectivity for ALL neurons across all heads (for histogram)
    all_selectivities: List[float] = []
    # Only the selective ones (for per-concept tables)
    per_concept_results: Dict[str, List[Dict]] = {}

    for target_cid in concepts.keys():
        if target_cid not in concept_vecs:
            per_concept_results[target_cid] = []
            continue

        target_words = concepts[target_cid]["words"]
        word_names = [w["word"] for w in target_words]

        monosemantic_neurons: List[Dict] = []

        for h in range(n_heads):
            if h not in concept_vecs[target_cid]:
                continue

            in_acts = concept_vecs[target_cid][h]  # (n_in, N)
            n_in = in_acts.shape[0]

            # Collect out-of-concept activations
            out_list = []
            for other_cid, other_vecs in concept_vecs.items():
                if other_cid == target_cid:
                    continue
                if h in other_vecs:
                    out_list.append(other_vecs[h])

            if not out_list:
                continue
            out_acts = np.concatenate(out_list, axis=0)  # (n_out, N)

            mean_in = in_acts.mean(axis=0)   # (N,)
            mean_out = out_acts.mean(axis=0)  # (N,)

            denom = mean_in + mean_out + 1e-10
            selectivity = mean_in / denom  # (N,)

            # ── Add ALL neurons to histogram (not just >0.5) ──
            # Only include neurons that have any activation at all
            # (dead neurons with 0/0 aren't informative)
            active_mask = (mean_in + mean_out) > 1e-8
            active_sel = selectivity[active_mask]
            all_selectivities.extend(active_sel.tolist())

            # ── For the per-concept table, keep only selective ones ──
            selective_idx = np.where((selectivity > 0.6) & active_mask)[0]

            for nidx in selective_idx:
                sel_score = float(selectivity[nidx])

                # Mann-Whitney U test for this neuron
                in_vals = in_acts[:, nidx]
                out_vals = out_acts[:, nidx]

                try:
                    if np.std(in_vals) > 0 or np.std(out_vals) > 0:
                        stat, pval = mannwhitneyu(
                            in_vals, out_vals, alternative="greater"
                        )
                    else:
                        pval = 1.0
                except ValueError:
                    pval = 1.0

                per_word_vals = [
                    round(float(in_acts[w, nidx]), 5) for w in range(n_in)
                ]

                monosemantic_neurons.append(
                    {
                        "layer": best_layer,
                        "head": h,
                        "neuron": int(nidx),
                        "selectivity": round(sel_score, 4),
                        "mean_in": round(float(mean_in[nidx]), 5),
                        "mean_out": round(float(mean_out[nidx]), 5),
                        "p_value": round(float(pval), 8),
                        "per_word": per_word_vals,
                    }
                )

        # Sort by selectivity descending
        monosemantic_neurons.sort(key=lambda n: n["selectivity"], reverse=True)
        per_concept_results[target_cid] = monosemantic_neurons[:30]

    # Build selectivity histogram (20 bins from 0.0 to 1.0)
    # This includes ALL active neurons — the shape of this distribution
    # is the key evidence: a rightward tail = monosemantic population
    n_bins = 20
    hist_counts, bin_edges = np.histogram(
        all_selectivities, bins=n_bins, range=(0.0, 1.0)
    )
    histogram = [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i + 1]), 2),
            "count": int(hist_counts[i]),
        }
        for i in range(n_bins)
    ]

    total_neurons = len(all_selectivities)
    selective_count = sum(1 for s in all_selectivities if s > 0.6)

    return {
        "per_concept": per_concept_results,
        "histogram": histogram,
        "total_neurons": total_neurons,
        "total_selective": selective_count,
        "mean_selectivity": round(float(np.mean(all_selectivities)), 4)
        if all_selectivities
        else 0.0,
    }


def _split_sentence_to_words(sentence: str) -> List[Tuple[str, int, int]]:
    """
    Split a sentence into words with their byte-range positions.
    Returns: [(word, byte_start, byte_end), ...]
    """
    words: List[Tuple[str, int, int]] = []
    byte_pos = 0
    for word in sentence.split(" "):
        word_bytes = len(word.encode("utf-8"))
        words.append((word, byte_pos, byte_pos + word_bytes))
        byte_pos += word_bytes + 1  # +1 for the space byte
    return words


def compute_synapse_tracking(
    model: BDH,
    device: str,
    best_layer: int,
    n_heads: int,
    n_neurons: int,
    concepts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For each concept, discover concept-selective synapses and track their
    σ(i,j) = Σ_{τ≤t} y_sparse[τ,i] · x_sparse[τ,j] values word-by-word.

    Strategy:
    1. Run all sentences for this concept through the model.
    2. For each layer, compute Δσ (change in σ diagonal) at each word.
    3. Find neurons where Δσ is consistently large at concept words and
       small at non-concept words → these are monosemantic synapses.
    4. Pick the layer with the best concept-selective synapses.
    5. Build the word-level timeline for the frontend.
    """
    tracking_data: Dict[str, Any] = {}

    for cid, sentences in TRACKING_SENTENCES.items():
        concept_result = concepts.get(cid)
        if not concept_result:
            continue

        # Get the concept's words for labelling
        concept_words = set()
        for cat_id, cat in CATEGORIES.items():
            if cat_id == cid:
                concept_words = {w.lower() for w in cat["words"]}
                break

        # ── Phase A: Run all sentences, collect per-word Δσ for every neuron ──
        # at each layer to discover which neurons are concept-selective
        n_layers = model.config.n_layer
        # per_layer_deltas[layer][head] = { neuron_idx: [(delta, is_concept), ...] }
        per_layer_deltas: Dict[int, Dict[int, Dict[int, List[Tuple[float, bool]]]]] = {}

        # Also store raw sentence data for final timeline
        sentence_raw: List[Dict] = []

        for sentence in sentences[:3]:
            tokens_bytes = list(sentence.encode("utf-8"))
            tokens_tensor = torch.tensor(
                [tokens_bytes], dtype=torch.long, device=device
            )
            word_boundaries = _split_sentence_to_words(sentence)

            extraction_config = ExtractionConfig(
                capture_sparse_activations=True,
                capture_attention_patterns=False,
            )

            with torch.no_grad():
                with model.extraction_mode(extraction_config) as buffer:
                    model(tokens_tensor)

                    raw_per_layer: Dict[int, Dict] = {}

                    for layer_idx in sorted(buffer.x_sparse.keys()):
                        x_all = buffer.x_sparse[layer_idx]
                        y_all = buffer.y_sparse.get(layer_idx)
                        if y_all is None:
                            continue

                        x_sp = x_all[0] if x_all.dim() == 4 else x_all  # (nh, T, N)
                        y_sp = y_all[0] if y_all.dim() == 4 else y_all
                        T = x_sp.shape[1]

                        if layer_idx not in per_layer_deltas:
                            per_layer_deltas[layer_idx] = {h: {} for h in range(n_heads)}

                        layer_xy = {}
                        for h in range(n_heads):
                            # Diagonal outer product: y[t,n] * x[t,n] for all n
                            xy = (y_sp[h] * x_sp[h]).cpu().numpy()  # (T, N)
                            cumsum = np.cumsum(xy, axis=0)  # (T, N)
                            layer_xy[h] = cumsum

                            # Compute Δσ per word for top-active neurons only
                            # (skip neurons with zero activity to save time)
                            max_act = xy.max(axis=0)  # (N,)
                            active_neurons = np.where(max_act > 1e-6)[0]

                            for word, byte_start, byte_end in word_boundaries:
                                last_byte = min(byte_end - 1, T - 1)
                                first_byte = max(byte_start - 1, 0)
                                is_concept = word.lower().strip(".,;:!?'\"") in concept_words

                                for nidx in active_neurons:
                                    curr = cumsum[last_byte, nidx]
                                    prev = cumsum[first_byte, nidx] if byte_start > 0 else 0.0
                                    delta = curr - prev

                                    if nidx not in per_layer_deltas[layer_idx][h]:
                                        per_layer_deltas[layer_idx][h][nidx] = []
                                    per_layer_deltas[layer_idx][h][nidx].append(
                                        (float(delta), is_concept)
                                    )

                        raw_per_layer[layer_idx] = layer_xy

            sentence_raw.append({
                "sentence": sentence,
                "tokens_bytes": tokens_bytes,
                "word_boundaries": word_boundaries,
                "raw_per_layer": raw_per_layer,
            })

        # ── Phase B: Find best (layer, head, neuron) with highest concept selectivity ──
        best_synapses: List[Dict] = []
        for layer_idx, head_data in per_layer_deltas.items():
            for h, neuron_data in head_data.items():
                for nidx, deltas in neuron_data.items():
                    concept_deltas = [d for d, ic in deltas if ic]
                    nonconcept_deltas = [d for d, ic in deltas if not ic]

                    if not concept_deltas or not nonconcept_deltas:
                        continue

                    mean_concept = np.mean(concept_deltas)
                    mean_nonconcept = np.mean(nonconcept_deltas)

                    # Selectivity of Δσ: how much more does σ jump at concept words?
                    denom = abs(mean_concept) + abs(mean_nonconcept) + 1e-10
                    selectivity = mean_concept / denom

                    if selectivity > 0.55 and mean_concept > 1e-4:
                        best_synapses.append({
                            "layer": layer_idx,
                            "head": h,
                            "neuron": nidx,
                            "selectivity": float(selectivity),
                            "mean_concept_delta": float(mean_concept),
                            "mean_nonconcept_delta": float(mean_nonconcept),
                        })

        # Sort by selectivity * magnitude and take top 5
        best_synapses.sort(
            key=lambda s: s["selectivity"] * s["mean_concept_delta"],
            reverse=True,
        )
        top_synapses = best_synapses[:5]

        if not top_synapses:
            # Fallback: pick neurons with largest overall Δσ at concept words
            all_candidates = []
            for layer_idx, head_data in per_layer_deltas.items():
                for h, neuron_data in head_data.items():
                    for nidx, deltas in neuron_data.items():
                        concept_deltas = [d for d, ic in deltas if ic]
                        if concept_deltas:
                            all_candidates.append({
                                "layer": layer_idx,
                                "head": h,
                                "neuron": nidx,
                                "selectivity": 0.5,
                                "mean_concept_delta": float(np.mean(concept_deltas)),
                                "mean_nonconcept_delta": 0.0,
                            })
            all_candidates.sort(key=lambda s: s["mean_concept_delta"], reverse=True)
            top_synapses = all_candidates[:5]

        tracking_layer = top_synapses[0]["layer"] if top_synapses else best_layer
        print(f"     {cid}: tracking layer = L{tracking_layer}, "
              f"top synapse selectivity = {top_synapses[0]['selectivity']:.3f}" if top_synapses else "")

        tracked_synapses = [
            {
                "id": f"σ({s['neuron']},{s['neuron']})",
                "label": f"L{s['layer']}_H{s['head']}_N{s['neuron']}",
                "layer": s["layer"],
                "head": s["head"],
                "i": s["neuron"],
                "j": s["neuron"],
                "selectivity": round(s["selectivity"], 3),
            }
            for s in top_synapses
        ]

        # ── Phase C: Build word-level timeline for the discovered synapses ──
        sentence_tracks: List[Dict] = []
        for sraw in sentence_raw:
            word_timeline: List[Dict] = []
            T = len(sraw["tokens_bytes"])

            for word, byte_start, byte_end in sraw["word_boundaries"]:
                last_byte = min(byte_end - 1, T - 1)
                first_byte = max(byte_start - 1, 0)
                is_concept = word.lower().strip(".,;:!?'\"") in concept_words

                sigma_at_word = {}
                delta_sigma = {}
                for syn in tracked_synapses:
                    layer_data = sraw["raw_per_layer"].get(syn["layer"], {})
                    cumsum = layer_data.get(syn["head"])
                    if cumsum is None:
                        sigma_at_word[syn["id"]] = 0.0
                        delta_sigma[syn["id"]] = 0.0
                        continue

                    nidx = syn["i"]
                    curr = float(cumsum[last_byte, nidx]) if last_byte < cumsum.shape[0] else 0.0
                    prev = float(cumsum[first_byte, nidx]) if byte_start > 0 and first_byte < cumsum.shape[0] else 0.0

                    sigma_at_word[syn["id"]] = round(curr, 6)
                    delta_sigma[syn["id"]] = round(curr - prev, 6)

                word_timeline.append({
                    "word": word,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "is_concept": is_concept,
                    "sigma": sigma_at_word,
                    "delta_sigma": delta_sigma,
                })

            sentence_tracks.append({
                "sentence": sraw["sentence"],
                "n_bytes": len(sraw["tokens_bytes"]),
                "words": word_timeline,
            })

        tracking_data[cid] = {
            "synapses": tracked_synapses,
            "sentences": sentence_tracks,
        }

    return tracking_data


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute monosemanticity data for BDH frontend"
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "checkpoints" / "french" / "french_best.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "frontend" / "public" / "monosemanticity" / "precomputed.json"
        ),
        help="Output JSON path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 BDH Monosemanticity Pre-computation (Enhanced)")
    print("=" * 60)

    # Load model
    print(f"\n📂 Loading model from {args.model}")
    model = load_model(args.model, args.device)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    n_neurons = model.config.n_neurons
    print(f"   Config: {n_layers}L × {n_heads}H × {n_neurons}N")

    # ── Phase 1: Extract fingerprints for every category ──
    print("\n" + "=" * 60)
    print("Phase 1: Extracting concept fingerprints")
    print("=" * 60)
    concepts: Dict[str, Any] = {}
    for cat_id, cat in CATEGORIES.items():
        print(f"\n   ▶ {cat['name']}: {cat['words']}")
        result = extract_fingerprint(
            model,
            cat["words"],
            cat["name"],
            args.device,
            n_layers,
            n_heads,
            n_neurons,
        )
        concepts[cat_id] = result
        for layer_str, matrix in result["similarity"].items():
            n = len(matrix)
            avg = sum(
                matrix[i][j] for i in range(n) for j in range(n) if i != j
            ) / max(n * (n - 1), 1)
            print(f"     L{layer_str} avg cosine: {avg:.4f}")

    # ── Phase 2: Find best layer ──
    best_layer = compute_best_layer(concepts, n_layers)
    print(f"\n🏆 Best layer (highest avg within-concept similarity): L{best_layer}")

    # ── Phase 3: Cross-concept distinctness ──
    print("\n" + "=" * 60)
    print("Phase 3: Cross-concept distinctness")
    print("=" * 60)
    cross_concept = compute_cross_concept(concepts, n_layers, n_heads)
    for cc in cross_concept:
        avg_d = np.mean(cc["distinctness_per_layer"])
        print(
            f"   {cc['primary']} vs {cc['secondary']}: avg distinctness = {avg_d:.4f}"
        )

    # ── Phase 4: Selectivity scores & Mann-Whitney U ──
    print("\n" + "=" * 60)
    print("Phase 4: Neuron selectivity & Mann-Whitney U test")
    print("=" * 60)
    selectivity_data = compute_selectivity(
        concepts, best_layer, n_heads, n_neurons
    )

    # Attach monosemantic_neurons to each concept
    for cid, neurons in selectivity_data["per_concept"].items():
        concepts[cid]["monosemantic_neurons"] = neurons
        sig_count = sum(1 for n in neurons if n["p_value"] < 0.05)
        print(
            f"   {cid}: {len(neurons)} selective neurons, "
            f"{sig_count} significant (p < 0.05)"
        )

    print(
        f"\n   📊 Global: {selectivity_data['total_neurons']} neurons total, "
        f"{selectivity_data['total_selective']} selective (>0.6), "
        f"mean = {selectivity_data['mean_selectivity']:.4f}"
    )

    # ── Phase 5: Synapse tracking timeseries ──
    print("\n" + "=" * 60)
    print("Phase 5: Synapse tracking timeseries")
    print("=" * 60)
    synapse_tracking = compute_synapse_tracking(
        model, args.device, best_layer, n_heads, n_neurons, concepts
    )
    for cid, track in synapse_tracking.items():
        n_syn = len(track["synapses"])
        n_sent = len(track["sentences"])
        print(f"   {cid}: {n_syn} tracked synapses × {n_sent} sentences")

    # ── Phase 6: Write JSON ──
    # Remove raw_x before serialization
    for cid in concepts:
        concepts[cid].pop("raw_x", None)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_info": {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_neurons": n_neurons,
        },
        "best_layer": best_layer,
        "concepts": concepts,
        "cross_concept": cross_concept,
        "selectivity": {
            "histogram": selectivity_data["histogram"],
            "total_neurons": selectivity_data["total_neurons"],
            "total_selective": selectivity_data["total_selective"],
            "mean_selectivity": selectivity_data["mean_selectivity"],
        },
        "synapse_tracking": synapse_tracking,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n💾 Wrote {output_path} ({size_mb:.2f} MB)")
    print("✅ Done! Frontend will load this statically.")


if __name__ == "__main__":
    main()
