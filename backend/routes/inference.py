"""Inference API routes."""

import torch
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))
from bdh import ExtractionConfig


router = APIRouter()
class InferenceRequest(BaseModel):
    """Request for model inference."""
    text: str = Field(..., description="Input text to process")
    model_name: str = Field(default="french", description="Model to use")
    extract_sparse: bool = Field(
        default=True, description="Extract sparse activations")
    extract_attention: bool = Field(
        default=False, description="Extract attention patterns")
    layers: Optional[List[int]] = Field(
        default=None, description="Specific layers to extract")


class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Prompt text")
    model_name: str = Field(default="french")
    max_tokens: int = Field(default=50, ge=1, le=500)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_k: Optional[int] = Field(default=5, ge=1, le=100)


class TokenActivation(BaseModel):
    """Activation data for a single token."""
    token_idx: int
    token_byte: int
    token_char: str
    x_sparsity: float
    y_sparsity: float
    x_active_count: int
    y_active_count: int


class InferenceResponse(BaseModel):
    """Response from inference."""
    input_text: str
    input_tokens: List[int]
    input_chars: List[str]
    num_layers: int
    num_heads: int
    neurons_per_head: int
    overall_sparsity: float
    sparsity_by_layer: List[float]
    token_activations: Optional[List[TokenActivation]] = None


class GenerateResponse(BaseModel):
    """Response from generation."""
    prompt: str
    generated: str
    full_text: str
    num_tokens_generated: int
@router.post("/run", response_model=InferenceResponse)
def run_inference(request: InferenceRequest, req: Request):
    """
    Run inference with activation extraction.

    This is the core endpoint for visualization - it processes text through
    the BDH model and returns activation data for visualization.
    """
    model_service = req.app.state.model_service

    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")

    # Tokenize
    try:
        tokens = torch.tensor(
            [list(request.text.encode('utf-8'))],
            dtype=torch.long,
            device=model_service.device
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tokenization error: {e}")

    T = tokens.shape[1]

    # Configure extraction
    extraction_config = ExtractionConfig(
        capture_sparse_activations=request.extract_sparse,
        capture_attention_patterns=request.extract_attention,
        capture_pre_relu=False,
        capture_layer_outputs=False,
        layers_to_capture=request.layers,
    )

    # Run inference
    sparsity_by_layer = []
    token_activations = []

    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            logits, _ = model(tokens)

            # Compute sparsity stats
            for layer_idx in sorted(buffer.x_sparse.keys()):
                x = buffer.x_sparse[layer_idx]
                y = buffer.y_sparse[layer_idx]

                x_sparsity = 1 - ((x > 0).sum().item() / x.numel())
                y_sparsity = 1 - ((y > 0).sum().item() / y.numel())
                sparsity_by_layer.append((x_sparsity + y_sparsity) / 2)

                # Token-level activations (first layer only for response size)
                if layer_idx == 0:
                    for t in range(T):
                        x_t = x[0, :, t, :]  # (nh, N)
                        y_t = y[0, :, t, :]

                        token_byte = tokens[0, t].item()
                        try:
                            token_char = chr(
                                token_byte) if 32 <= token_byte < 127 else f"\\x{token_byte:02x}"
                        except:
                            token_char = f"\\x{token_byte:02x}"

                        token_activations.append(TokenActivation(
                            token_idx=t,
                            token_byte=token_byte,
                            token_char=token_char,
                            x_sparsity=1 -
                            ((x_t > 0).sum().item() / x_t.numel()),
                            y_sparsity=1 -
                            ((y_t > 0).sum().item() / y_t.numel()),
                            x_active_count=(x_t > 0).sum().item(),
                            y_active_count=(y_t > 0).sum().item(),
                        ))

    # Build character list
    input_chars = []
    for byte in tokens[0].cpu().tolist():
        try:
            char = chr(byte) if 32 <= byte < 127 else f"\\x{byte:02x}"
        except:
            char = f"\\x{byte:02x}"
        input_chars.append(char)

    return InferenceResponse(
        input_text=request.text,
        input_tokens=tokens[0].cpu().tolist(),
        input_chars=input_chars,
        num_layers=config.n_layer,
        num_heads=config.n_head,
        neurons_per_head=config.n_neurons,
        overall_sparsity=sum(sparsity_by_layer) /
        len(sparsity_by_layer) if sparsity_by_layer else 0,
        sparsity_by_layer=sparsity_by_layer,
        token_activations=token_activations,
    )


@router.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest, req: Request):
    """
    Generate text from a prompt.

    Uses the BDH model's autoregressive generation capability.
    """
    model_service = req.app.state.model_service

    try:
        model = model_service.get_or_load(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")

    # Tokenize prompt
    prompt_tokens = torch.tensor(
        [list(request.prompt.encode('utf-8'))],
        dtype=torch.long,
        device=model_service.device
    )

    # Generate
    with torch.no_grad():
        output_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

    # Decode
    full_bytes = bytes(output_tokens[0].cpu().tolist())
    full_text = full_bytes.decode('utf-8', errors='backslashreplace')
    generated_text = full_text[len(request.prompt):]

    return GenerateResponse(
        prompt=request.prompt,
        generated=generated_text,
        full_text=full_text,
        num_tokens_generated=output_tokens.shape[1] - prompt_tokens.shape[1],
    )


@router.post("/extract-detailed")
def extract_detailed(request: InferenceRequest, req: Request):
    """
    Extract detailed activation data for a single input.

    Returns full activation arrays (large response, use for specific analysis).
    """
    model_service = req.app.state.model_service

    try:
        model = model_service.get_or_load(request.model_name)
        config = model_service.get_config(request.model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")

    # Tokenize
    tokens = torch.tensor(
        [list(request.text.encode('utf-8'))],
        dtype=torch.long,
        device=model_service.device
    )

    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=True,
        capture_pre_relu=True,
        layers_to_capture=request.layers,
    )

    with torch.no_grad():
        with model.extraction_mode(extraction_config) as buffer:
            _, _ = model(tokens)

            # Convert to serializable format
            result = {
                "input_tokens": tokens[0].cpu().tolist(),
                "model_config": {
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "n_neurons": config.n_neurons,
                },
                "layers": {}
            }

            for layer_idx in sorted(buffer.x_sparse.keys()):
                x = buffer.x_sparse[layer_idx][0].cpu()  # (nh, T, N)
                y = buffer.y_sparse[layer_idx][0].cpu()

                # Only return non-zero indices and values (sparse format)
                layer_data = {
                    "x_sparse": {},
                    "y_sparse": {},
                }

                for head in range(config.n_head):
                    for t in range(tokens.shape[1]):
                        x_ht = x[head, t]
                        y_ht = y[head, t]

                        x_nz = (x_ht > 0).nonzero().squeeze(-1)
                        y_nz = (y_ht > 0).nonzero().squeeze(-1)

                        layer_data["x_sparse"][f"h{head}_t{t}"] = {
                            "indices": x_nz.tolist() if x_nz.numel() > 0 else [],
                            "values": x_ht[x_nz].tolist() if x_nz.numel() > 0 else [],
                        }
                        layer_data["y_sparse"][f"h{head}_t{t}"] = {
                            "indices": y_nz.tolist() if y_nz.numel() > 0 else [],
                            "values": y_ht[y_nz].tolist() if y_nz.numel() > 0 else [],
                        }

                # Attention patterns
                if layer_idx in buffer.attention_scores:
                    # (nh, T, T)
                    attn = buffer.attention_scores[layer_idx][0].cpu()
                    layer_data["attention"] = attn.tolist()

                result["layers"][layer_idx] = layer_data

    return result
