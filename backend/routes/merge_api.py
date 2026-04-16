"""Merge API routes (heritage probe, side-by-side generation)."""

import torch
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request
from typing import Optional, Dict, Any
from bdh import ExtractionConfig
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))


router = APIRouter()
class HeritageProbeRequest(BaseModel):
    text: str = Field(..., description="Text to probe")
    model_name: str = Field(
        default="merged", description="Merged model to probe")


class SideBySideRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text")
    max_tokens: int = Field(default=60, ge=1, le=300)
@router.post("/heritage-probe")
def heritage_probe(request: HeritageProbeRequest, req: Request):
    """
    Run a heritage probe on merged model: feed text, measure which neuron
    bank (model1-origin vs model2-origin) activates.
    """
    model_service = req.app.state.model_service

    model_name = request.model_name
    try:
        model = model_service.get_or_load(model_name)
    except (FileNotFoundError, KeyError):
        # Fallback chain
        for fallback in ["merged", "merged_finetuned", "merged_polyglot"]:
            if fallback != model_name:
                try:
                    model = model_service.get_or_load(fallback)
                    model_name = fallback
                    break
                except (FileNotFoundError, KeyError):
                    continue
        else:
            raise HTTPException(404, "No merged model available")

    heritage = model_service.get_heritage(model_name)
    if not heritage:
        raise HTTPException(
            400, f"Model '{model_name}' has no heritage info — is it a merged model?")

    config = model_service.get_config(model_name)
    # neurons_per_head_original is total neurons across all heads from the
    # source specialist.  Convert to per-head for the split index.
    N_orig_total = heritage["neurons_per_head_original"]
    N_orig = N_orig_total // config.n_head  # per-head split index
    device = model_service.device

    # Tokenize
    text_bytes = list(request.text.encode("utf-8")[:512])
    tokens = torch.tensor([text_bytes], dtype=torch.long, device=device)

    # Extract sparse activations
    ext_cfg = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=False,
        capture_pre_relu=False,
        capture_layer_outputs=False,
    )

    with torch.no_grad():
        with model.extraction_mode(ext_cfg) as buf:
            model(tokens)

    layers_result: Dict[str, Any] = {}
    for l in range(config.n_layer):
        if l not in buf.x_sparse:
            continue
        xs = buf.x_sparse[l][0]  # (nh, T, N_merged)
        act = xs.mean(dim=(0, 1))  # (N_merged,)
        fr_act = act[:N_orig]
        pt_act = act[N_orig:]

        fr_active = int((fr_act > 0).sum().item())
        pt_active = int((pt_act > 0).sum().item())
        fr_energy = float(fr_act.sum().item())
        pt_energy = float(pt_act.sum().item())
        total = fr_energy + pt_energy

        layers_result[str(l)] = {
            "french": {
                "origin": heritage["model1_name"],
                "active_count": fr_active,
                "total_count": int(N_orig),
                "activation_ratio": round(fr_energy / max(total, 1e-8), 4),
            },
            "portuguese": {
                "origin": heritage["model2_name"],
                "active_count": pt_active,
                "total_count": int(N_orig),
                "activation_ratio": round(pt_energy / max(total, 1e-8), 4),
            },
        }

    # Summary
    all_fr = sum(d["french"]["activation_ratio"]
                 for d in layers_result.values())
    all_pt = sum(d["portuguese"]["activation_ratio"]
                 for d in layers_result.values())
    total = all_fr + all_pt
    fr_pct = round(100 * all_fr / max(total, 1e-8), 2)
    pt_pct = round(100 * all_pt / max(total, 1e-8), 2)

    return {
        "text": request.text,
        "model_name": model_name,
        "heritage_split": N_orig,
        "layers": layers_result,
        "summary": {
            "french_percentage": fr_pct,
            "portuguese_percentage": pt_pct,
            "dominant_heritage": heritage["model1_name"] if fr_pct > pt_pct else heritage["model2_name"],
        },
    }
@router.post("/side-by-side")
def side_by_side(request: SideBySideRequest, req: Request):
    """
    Generate text from all available models for comparison.
    """
    model_service = req.app.state.model_service
    device = model_service.device

    # Model names to try, in order
    model_names = [
        "french",
        "portuguese",
        "merged",
    ]

    prompt_bytes = list(request.prompt.encode("utf-8"))
    tokens = torch.tensor([prompt_bytes], dtype=torch.long, device=device)

    generations: Dict[str, str] = {}

    for name in model_names:
        try:
            model = model_service.get_or_load(name)
            with torch.no_grad():
                output = model.generate(
                    tokens.clone(),
                    max_new_tokens=request.max_tokens,
                    top_k=5,
                    temperature=0.8,
                )
            text = bytes(output[0].cpu().tolist()).decode(
                "utf-8", errors="replace")
            # Return only the generated part (strip prompt)
            generations[name] = text[len(request.prompt):]
        except (FileNotFoundError, KeyError):
            continue

    if not generations:
        raise HTTPException(404, "No models available for generation")

    return {
        "prompt": request.prompt,
        "generations": generations,
    }
