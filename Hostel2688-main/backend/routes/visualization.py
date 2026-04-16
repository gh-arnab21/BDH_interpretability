"""Visualization API routes."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

router = APIRouter()
class PlaybackRequest(BaseModel):
    """Request for playback data generation."""
    text: str
    model_name: str = Field(default="french")
    include_attention: bool = Field(default=True)


class HebbianTrackRequest(BaseModel):
    """Request for Hebbian state tracking."""
    text: str
    model_name: str = Field(default="french")
    layer: Optional[int] = Field(default=None, description="Specific layer to track, None=all")
    top_k_synapses: int = Field(default=5, description="Number of top synapses to track")
@router.post("/playback")
def generate_playback(request: PlaybackRequest, req: Request):
    """
    Generate playback data for frontend animation with interpretability data.
    """
    import time
    t0 = time.perf_counter()
    
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    
    tokens = torch.tensor(
        [list(request.text.encode('utf-8'))],
        dtype=torch.long,
        device=model_service.device
    )
    T = tokens.shape[1]
    
    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=request.include_attention,
        capture_pre_relu=True,
        capture_residuals=True,
    )
    
    t1 = time.perf_counter()
    
    # Run model and extract activations
    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            logits, _ = model(tokens)
            
            # Get embedding
            embed_output = model.embed(tokens)
            
            # Transfer to CPU
            layer_data_cpu = {}
            for layer_idx in sorted(buffer.x_sparse.keys()):
                x_cpu = buffer.x_sparse[layer_idx][0].cpu()
                y_cpu = buffer.y_sparse[layer_idx][0].cpu()
                
                x_pre = None
                y_pre = None
                attn_cpu = None
                attn_out_cpu = None
                
                if layer_idx in buffer.x_pre_relu:
                    x_pre = buffer.x_pre_relu[layer_idx][0].cpu()
                if layer_idx in buffer.y_pre_relu:
                    y_pre = buffer.y_pre_relu[layer_idx][0].cpu()
                if request.include_attention and layer_idx in buffer.attention_scores:
                    attn_cpu = buffer.attention_scores[layer_idx][0].cpu()
                if layer_idx in buffer.attention_output:
                    attn_out_cpu = buffer.attention_output[layer_idx][0].cpu()
                
                layer_data_cpu[layer_idx] = {
                    'x_sparse': x_cpu,
                    'y_sparse': y_cpu,
                    'x_pre_relu': x_pre,
                    'y_pre_relu': y_pre,
                    'attention': attn_cpu,
                    'attention_output': attn_out_cpu,
                    'residual': buffer.residuals[layer_idx][0, 0].cpu() if layer_idx in buffer.residuals else None,
                }
            
            logits_cpu = logits[0].cpu()
            embed_cpu = embed_output[0].cpu()
            
            # Pre-LayerNorm embedding (raw, before normalization)
            pre_ln_cpu = None
            if buffer.pre_layernorm is not None:
                pre_ln_cpu = buffer.pre_layernorm[0, 0].cpu()  # (T, D)
    
    t2 = time.perf_counter()
    
    # Build token metadata
    token_bytes = tokens[0].cpu().tolist()
    input_chars = []
    for b in token_bytes:
        try:
            input_chars.append(chr(b) if 32 <= b < 127 else f"\\x{b:02x}")
        except:
            input_chars.append(f"\\x{b:02x}")
    
    # Build frames
    frames = []
    n_head = config.n_head
    n_neurons = config.n_neurons
    TOP_K = 15
    GRID_BINS = 64
    
    # Build ρ matrices (head-averaged, full T×T per layer) for global response
    rho_matrices = {}
    for layer_idx, layer_data in layer_data_cpu.items():
        attn_raw = layer_data['attention']
        if attn_raw is not None:
            # attn shape: (nh, T, T) — head-average to (T, T)
            rho_avg = attn_raw.numpy().mean(axis=0)
            rho_matrices[layer_idx] = np.round(rho_avg, 4).tolist()
    
    for layer_idx, layer_data in layer_data_cpu.items():
        x_sparse = layer_data['x_sparse'].numpy()
        y_sparse = layer_data['y_sparse'].numpy()
        x_pre_relu = layer_data['x_pre_relu'].numpy() if layer_data['x_pre_relu'] is not None else None
        y_pre_relu = layer_data['y_pre_relu'].numpy() if layer_data['y_pre_relu'] is not None else None
        attn = layer_data['attention'].numpy() if layer_data['attention'] is not None else None
        attn_out = layer_data['attention_output'].numpy() if layer_data['attention_output'] is not None else None
        
        x_total = n_head * n_neurons
        y_total = n_head * n_neurons
        
        for t in range(T):
            x_active = []
            y_active = []
            x_top_neurons = []
            y_top_neurons = []
            all_x_active = set()
            all_y_active = set()
            
            for h in range(n_head):
                x_h = x_sparse[h, t]
                y_h = y_sparse[h, t]
                
                x_nz = np.flatnonzero(x_h > 0)
                y_nz = np.flatnonzero(y_h > 0)
                
                for idx in x_nz:
                    all_x_active.add((h, int(idx)))
                for idx in y_nz:
                    all_y_active.add((h, int(idx)))
                
                # Top neurons per head
                if len(x_nz) > 0:
                    top_k = min(5, len(x_nz))
                    top_idx = x_nz[np.argsort(x_h[x_nz])[-top_k:]][::-1]
                    for idx in top_idx:
                        x_top_neurons.append({
                            "head": int(h),
                            "neuron": int(idx),
                            "value": round(float(x_h[idx]), 3)
                        })
                
                if len(y_nz) > 0:
                    top_k = min(5, len(y_nz))
                    top_idx = y_nz[np.argsort(y_h[y_nz])[-top_k:]][::-1]
                    for idx in top_idx:
                        y_top_neurons.append({
                            "head": int(h),
                            "neuron": int(idx),
                            "value": round(float(y_h[idx]), 3)
                        })
                
                # Limit for basic active list
                if len(x_nz) > TOP_K:
                    top = np.argpartition(x_h[x_nz], -TOP_K)[-TOP_K:]
                    x_nz = x_nz[top]
                if len(y_nz) > TOP_K:
                    top = np.argpartition(y_h[y_nz], -TOP_K)[-TOP_K:]
                    y_nz = y_nz[top]
                
                x_active.append({
                    "indices": x_nz.tolist(),
                    "values": np.round(x_h[x_nz], 3).tolist(),
                })
                y_active.append({
                    "indices": y_nz.tolist(),
                    "values": np.round(y_h[y_nz], 3).tolist(),
                })
            
            # Activation grids (downsampled per-head neuron activity)
            bin_size = n_neurons // GRID_BINS
            x_grid = []
            y_grid = []
            hadamard_grid_data = []
            for h in range(n_head):
                x_h_full = x_sparse[h, t]
                y_h_full = y_sparse[h, t]
                x_row = []
                y_row = []
                h_row = []
                for b in range(GRID_BINS):
                    s, e = b * bin_size, (b + 1) * bin_size
                    x_bin = x_h_full[s:e]
                    y_bin = y_h_full[s:e]
                    x_mx = float(x_bin.max())
                    y_mx = float(y_bin.max())
                    x_row.append(round(x_mx, 3) if x_mx > 0 else 0)
                    y_row.append(round(y_mx, 3) if y_mx > 0 else 0)
                    h_row.append(int(((x_bin > 0) & (y_bin > 0)).sum()))
                x_grid.append(x_row)
                y_grid.append(y_row)
                hadamard_grid_data.append(h_row)
            
            # Gating
            gated = all_x_active & all_y_active
            x_only = len(all_x_active - all_y_active)
            y_only = len(all_y_active - all_x_active)
            both = len(gated)
            
            # Sparsity
            x_active_count = len(all_x_active)
            y_active_count = len(all_y_active)
            x_sparsity = round(1 - (x_active_count / x_total), 4)
            y_sparsity = round(1 - (y_active_count / y_total), 4)
            
            # Pre-relu stats (with histogram for visualization)
            x_pre_stats = None
            if x_pre_relu is not None:
                x_pre_t = x_pre_relu[:, t, :]
                x_flat = x_pre_t.flatten()
                counts, edges = np.histogram(x_flat, bins=20)
                x_pre_stats = {
                    "mean": round(float(x_pre_t.mean()), 3),
                    "std": round(float(x_pre_t.std()), 3),
                    "max": round(float(x_pre_t.max()), 3),
                    "min": round(float(x_pre_t.min()), 3),
                    "positive_count": int((x_pre_t > 0).sum()),
                    "total": int(x_pre_t.size),
                    "histogram": [
                        {"start": round(float(edges[i]), 3), "end": round(float(edges[i+1]), 3), "count": int(counts[i])}
                        for i in range(len(counts))
                    ],
                }
            
            y_pre_stats = None
            if y_pre_relu is not None:
                y_pre_t = y_pre_relu[:, t, :]
                y_flat = y_pre_t.flatten()
                y_counts, y_edges = np.histogram(y_flat, bins=20)
                y_pre_stats = {
                    "mean": round(float(y_pre_t.mean()), 3),
                    "std": round(float(y_pre_t.std()), 3),
                    "max": round(float(y_pre_t.max()), 3),
                    "min": round(float(y_pre_t.min()), 3),
                    "positive_count": int((y_pre_t > 0).sum()),
                    "total": int(y_pre_t.size),
                    "histogram": [
                        {"start": round(float(y_edges[i]), 3), "end": round(float(y_edges[i+1]), 3), "count": int(y_counts[i])}
                        for i in range(len(y_counts))
                    ],
                }
            
            # Attention stats (with full weights for bar chart)
            attention_stats = None
            attention_weights_full = None
            if attn is not None and t > 0:
                attn_t = attn[:, t, :t+1]
                attn_avg = attn_t.mean(axis=0)
                top_attn_idx = np.argsort(attn_avg)[-3:][::-1]
                attention_stats = {
                    "top_attended": [
                        {"token_idx": int(idx), "char": input_chars[idx], "weight": round(float(attn_avg[idx]), 3)}
                        for idx in top_attn_idx if idx < len(input_chars)
                    ]
                }
                attention_weights_full = [
                    {"token_idx": int(idx), "char": input_chars[idx], "weight": round(float(attn_avg[idx]), 4)}
                    for idx in range(len(attn_avg)) if idx < len(input_chars)
                ]
            
            # Embedding info (layer 0, with downsampled vector for heatmap)
            embed_info = None
            if layer_idx == 0:
                embed_vec = embed_cpu[t].numpy()
                ds_size = 64
                group = max(1, len(embed_vec) // ds_size)
                vector_ds = [round(float(embed_vec[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                embed_info = {
                    "byte_value": token_bytes[t],
                    "norm": round(float(np.linalg.norm(embed_vec)), 3),
                    "mean": round(float(embed_vec.mean()), 3),
                    "std": round(float(embed_vec.std()), 3),
                    "vector_ds": vector_ds,
                }
                # LayerNorm before/after comparison
                if pre_ln_cpu is not None:
                    pre_ln_vec = pre_ln_cpu[t].numpy()
                    pre_ln_ds = [round(float(pre_ln_vec[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                    embed_info["pre_ln_ds"] = pre_ln_ds
                    embed_info["pre_ln_norm"] = round(float(np.linalg.norm(pre_ln_vec)), 3)
                    embed_info["pre_ln_mean"] = round(float(pre_ln_vec.mean()), 5)
                    embed_info["pre_ln_std"] = round(float(pre_ln_vec.std()), 5)
            
            frame = {
                "token_idx": t,
                "token_byte": token_bytes[t],
                "token_char": input_chars[t],
                "layer": layer_idx,
                "x_active": x_active,
                "y_active": y_active,
                "x_sparsity": x_sparsity,
                "y_sparsity": y_sparsity,
                "x_active_count": x_active_count,
                "y_active_count": y_active_count,
                "x_top_neurons": sorted(x_top_neurons, key=lambda x: -x["value"])[:10],
                "y_top_neurons": sorted(y_top_neurons, key=lambda x: -x["value"])[:10],
                "gating": {
                    "x_only": x_only,
                    "y_only": y_only,
                    "both": both,
                    "survival_rate": round(both / max(x_active_count, 1), 3),
                },
                "x_activation_grid": x_grid,
                "y_activation_grid": y_grid,
                "hadamard_grid": hadamard_grid_data,
            }
            
            if x_pre_stats:
                frame["x_pre_relu"] = x_pre_stats
            if y_pre_stats:
                frame["y_pre_relu"] = y_pre_stats
            if attention_stats:
                frame["attention_stats"] = attention_stats
            if attention_weights_full:
                frame["attention_weights"] = attention_weights_full
            if embed_info:
                frame["embedding"] = embed_info
            
            # a* vector (attention output for current token, downsampled)
            if attn_out is not None:
                # attn_out shape: (nh, T, D) — head-average at position t
                a_star_vec = attn_out[:, t, :].mean(axis=0)  # (D,)
                ds_size = 64
                D = len(a_star_vec)
                group = max(1, D // ds_size)
                a_star_ds = [round(float(a_star_vec[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                a_star_norm = round(float(np.linalg.norm(a_star_vec)), 3)
                frame["a_star_ds"] = a_star_ds
                frame["a_star_norm"] = a_star_norm
            
            # Decoder output Δv* (residual vector for current token, downsampled)
            residual_vec = layer_data.get('residual')
            if residual_vec is not None:
                res_np = residual_vec[t].numpy()  # (D,)
                ds_size = 64
                D = len(res_np)
                group = max(1, D // ds_size)
                decoder_ds = [round(float(res_np[i*group:(i+1)*group].mean()), 4) for i in range(ds_size)]
                decoder_norm = round(float(np.linalg.norm(res_np)), 3)
                decoder_mean = round(float(res_np.mean()), 4)
                decoder_std = round(float(res_np.std()), 4)
                frame["decoder_ds"] = decoder_ds
                frame["decoder_norm"] = decoder_norm
                frame["decoder_mean"] = decoder_mean
                frame["decoder_std"] = decoder_std
            
            frames.append(frame)
    
    # Predictions
    probs = torch.softmax(logits_cpu, dim=-1).numpy()
    predictions = []
    for t in range(T):
        top_idx = np.argsort(probs[t])[-5:][::-1]
        token_preds = []
        for idx in top_idx:
            try:
                char = chr(idx) if 32 <= idx < 127 else f"\\x{idx:02x}"
            except:
                char = f"\\x{idx:02x}"
            token_preds.append({
                "byte": int(idx),
                "char": char,
                "prob": round(float(probs[t, idx]), 4),
            })
        predictions.append(token_preds)
    
    t3 = time.perf_counter()
    print(f"[PERF] setup={t1-t0:.3f}s inference={t2-t1:.3f}s frames={t3-t2:.3f}s total={t3-t0:.3f}s ({T}tok × {len(layer_data_cpu)}layers)")
    
    return JSONResponse({
        "input_text": request.text,
        "input_tokens": token_bytes,
        "input_chars": input_chars,
        "num_layers": config.n_layer,
        "num_heads": config.n_head,
        "neurons_per_head": config.n_neurons,
        "embedding_dim": config.n_embd,
        "total_neurons": config.n_head * config.n_neurons,
        "frames": frames,
        "predictions": predictions,
        "rho_matrices": rho_matrices,
    })


@router.post("/hebbian-track")
def track_hebbian(request: HebbianTrackRequest, req: Request):
    """
    Track Hebbian σ dynamics word-by-word through BDH layers.
    
    Computes the real Hebbian outer product σ(i,j) = Σ y_sparse[τ,i] · x_sparse[τ,j]
    accumulated token-by-token, then aggregated to word boundaries.
    Also provides before/after prediction comparison demonstrating
    how context shifts the model's output distribution.
    """
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
    from bdh import ExtractionConfig
    import torch.nn.functional as F
    
    text = request.text.strip()
    tokens_bytes = list(text.encode("utf-8"))
    T = len(tokens_bytes)
    tokens_tensor = torch.tensor([tokens_bytes], dtype=torch.long, device=model_service.device)
    
    nh = config.n_head
    N = config.n_neurons
    n_layers = config.n_layer
    
    def split_words(text: str):
        words = []
        encoded = text.encode("utf-8")
        i = 0
        word_start = None
        current_word = b""
        for pos, byte in enumerate(encoded):
            ch = chr(byte) if byte < 128 else None
            is_space = ch is not None and ch in " \t\n\r"
            if is_space:
                if current_word:
                    words.append((current_word.decode("utf-8", errors="replace"), word_start, pos))
                    current_word = b""
                    word_start = None
            else:
                if word_start is None:
                    word_start = pos
                current_word += bytes([byte])
        if current_word:
            words.append((current_word.decode("utf-8", errors="replace"), word_start, len(encoded)))
        return words
    
    word_boundaries = split_words(text)
    
    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=False,
    )
    
    # Per-layer, per-head σ tracking word-by-word
    layer_word_data = {}  # {layer: {head: [{word, sigma_cumulative, delta_sigma, ...}]}}
    
    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            logits, _ = model(tokens_tensor)
            
            # Get top predictions at final position (after seeing full text)
            final_logits = logits[0, -1, :]  # (vocab_size,)
            final_probs = F.softmax(final_logits, dim=-1)
            top_vals, top_ids = torch.topk(final_probs, 10)
            after_predictions = []
            for v, idx in zip(top_vals.tolist(), top_ids.tolist()):
                try:
                    ch = chr(idx) if 32 <= idx < 127 else f"0x{idx:02x}"
                except:
                    ch = f"0x{idx:02x}"
                after_predictions.append({"byte": idx, "char": ch, "prob": round(v, 6)})
            
            # Process each layer
            layers_to_process = sorted(buffer.x_sparse.keys())
            if request.layer is not None:
                layers_to_process = [l for l in layers_to_process if l == request.layer]
            
            for layer_idx in layers_to_process:
                x_sparse = buffer.x_sparse[layer_idx][0]  # (nh, T, N)
                y_sparse_data = buffer.y_sparse.get(layer_idx)
                if y_sparse_data is None:
                    continue
                y_sparse = y_sparse_data[0]  # (nh, T, N)
                
                head_data = {}
                for h in range(nh):
                    # Compute σ(i,j) = y_sparse[t,i] * x_sparse[t,j] token by token
                    # For visualization we track diagonal σ(i,i) = y_sparse[t,i] * x_sparse[t,i]
                    # This is the "gate" signal — the Hebbian co-activation
                    x_h = x_sparse[h].cpu().numpy()  # (T, N) 
                    y_h = y_sparse[h].cpu().numpy()  # (T, N)
                    
                    # Token-level gate: element-wise product
                    gate = x_h * y_h  # (T, N) — this IS the Hebbian signal
                    
                    # Cumulative σ along token axis
                    sigma_cumsum = np.cumsum(gate, axis=0)  # (T, N)
                    
                    # Find top synapses: neurons with highest final cumulative σ
                    final_sigma = sigma_cumsum[-1]  # (N,)
                    top_k = min(request.top_k_synapses, N)
                    top_neuron_ids = np.argsort(final_sigma)[::-1][:top_k]
                    
                    # Build word-level timeline for tracked synapses
                    word_timeline = []
                    for wi, (word, byte_start, byte_end) in enumerate(word_boundaries):
                        last_byte = min(byte_end - 1, T - 1)
                        first_byte = max(byte_start - 1, 0) if byte_start > 0 else 0
                        
                        word_entry = {
                            "word": word,
                            "byte_range": [byte_start, byte_end],
                        }
                        
                        # Per-synapse sigma and delta at this word
                        synapses = {}
                        for nidx in top_neuron_ids:
                            nidx_int = int(nidx)
                            curr = float(sigma_cumsum[last_byte, nidx_int])
                            prev = float(sigma_cumsum[first_byte, nidx_int]) if byte_start > 0 else 0.0
                            delta = curr - prev
                            synapses[f"N{nidx_int}"] = {
                                "sigma": round(curr, 4),
                                "delta": round(delta, 4),
                            }
                        
                        # Aggregate gate activity at this word (sum across all neurons)
                        word_gate_sum = float(gate[max(byte_start, 0):byte_end].sum())
                        word_entry["synapses"] = synapses
                        word_entry["gate_activity"] = round(word_gate_sum, 4)
                        word_timeline.append(word_entry)
                    
                    # Tracked synapse metadata
                    tracked = []
                    for nidx in top_neuron_ids:
                        nidx_int = int(nidx)
                        tracked.append({
                            "id": f"N{nidx_int}",
                            "neuron": nidx_int,
                            "final_sigma": round(float(final_sigma[nidx_int]), 4),
                        })
                    
                    head_data[h] = {
                        "tracked_synapses": tracked,
                        "words": word_timeline,
                    }
                
                layer_word_data[layer_idx] = head_data
    
    # Feed just the first few bytes (before the main content)
    # to show what the model predicts without context
    prefix_len = min(3, T)  # first 3 bytes as minimal context
    if prefix_len > 0:
        prefix_tensor = tokens_tensor[:, :prefix_len]
        with torch.no_grad():
            prefix_logits, _ = model(prefix_tensor)
            prefix_final = prefix_logits[0, -1, :]
            prefix_probs = F.softmax(prefix_final, dim=-1)
            top_v, top_i = torch.topk(prefix_probs, 10)
            before_predictions = []
            for v, idx in zip(top_v.tolist(), top_i.tolist()):
                try:
                    ch = chr(idx) if 32 <= idx < 127 else f"0x{idx:02x}"
                except:
                    ch = f"0x{idx:02x}"
                before_predictions.append({"byte": idx, "char": ch, "prob": round(v, 6)})
    else:
        before_predictions = []
    
    layer_summary = []
    for layer_idx in sorted(layer_word_data.keys()):
        head_summaries = []
        for h in range(nh):
            if h not in layer_word_data[layer_idx]:
                continue
            hdata = layer_word_data[layer_idx][h]
            top_syn = hdata["tracked_synapses"][0] if hdata["tracked_synapses"] else None
            total_gate = sum(w["gate_activity"] for w in hdata["words"])
            head_summaries.append({
                "head": h,
                "total_gate_activity": round(total_gate, 4),
                "top_synapse": top_syn,
            })
        layer_summary.append({
            "layer": layer_idx,
            "heads": head_summaries,
        })
    
    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            model(tokens_tensor)
            sparsity_stats = buffer.get_sparsity_stats()
    
    return {
        "input_text": text,
        "num_bytes": T,
        "num_words": len(word_boundaries),
        "words": [w[0] for w in word_boundaries],
        "model_config": {
            "n_layer": n_layers,
            "n_head": nh,
            "n_neurons": N,
        },
        "predictions": {
            "before": before_predictions,
            "after": after_predictions,
            "prefix_text": text[:prefix_len],
        },
        "layer_summary": layer_summary,
        "layer_data": {
            int(k): {
                int(h): v for h, v in heads.items()
            } for k, heads in layer_word_data.items()
        },
        "sparsity": {
            k: round(v, 4) for k, v in sparsity_stats.items()
        },
    }


@router.get("/architecture-spec")
async def get_architecture_spec():
    """Get architecture specification for the interactive diagram."""
    return {
        "name": "BDH (Baby Dragon Hatchling)",
        "components": [
            {"id": "input", "type": "input", "label": "x_{l-1}"},
            {"id": "encoder_e", "type": "linear", "label": "Linear E", "formula": "x_latent = x @ E"},
            {"id": "relu_x", "type": "activation", "label": "ReLU", "formula": "x_sparse = ReLU(x_latent)"},
            {"id": "attention", "type": "attention", "label": "Linear Attention", "formula": "ρ += x^T v"},
            {"id": "encoder_v", "type": "linear", "label": "Linear E_v"},
            {"id": "relu_y", "type": "activation", "label": "ReLU"},
            {"id": "decoder", "type": "linear", "label": "Linear D"},
            {"id": "residual", "type": "operation", "label": "+"},
            {"id": "output", "type": "output", "label": "x_l"},
        ],
    }


@router.get("/color-scheme")
async def get_color_scheme():
    """Get color scheme for visualizations."""
    return {
        "encoder_path": {"fill": "#FFF3CD", "stroke": "#E6AC00"},
        "attention": {"fill": "#D4EDDA", "stroke": "#28A745"},
        "attention_state": {"fill": "#CCE5FF", "stroke": "#0066CC"},
        "activation_relu": {"fill": "#F8D7DA", "stroke": "#DC3545"},
    }
