"""Model management routes."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()
class ModelInfo(BaseModel):
    """Information about a model."""
    name: str
    loaded: bool
    config: Optional[Dict[str, Any]] = None
    heritage: Optional[Dict[str, Any]] = None  # For merged models


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_name: str
    checkpoint_path: Optional[str] = None
@router.get("/list")
async def list_models(req: Request):
    """
    List all available and loaded models.
    """
    model_service = req.app.state.model_service
    
    available = model_service.list_available_models()
    loaded = model_service.list_loaded_models()
    
    return {
        "available": available,
        "loaded": loaded,
    }


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: str, req: Request):
    """
    Get information about a specific model.
    """
    model_service = req.app.state.model_service
    
    is_loaded = model_service.is_loaded(model_name)
    
    config = None
    heritage = None
    
    if is_loaded:
        cfg = model_service.get_config(model_name)
        config = {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "n_neurons": cfg.n_neurons,
            "total_neurons": cfg.total_neurons,
            "vocab_size": cfg.vocab_size,
        }
        heritage = model_service.get_heritage(model_name)
    
    return ModelInfo(
        name=model_name,
        loaded=is_loaded,
        config=config,
        heritage=heritage,
    )


@router.post("/load")
def load_model(request: LoadModelRequest, req: Request):
    """
    Load a model checkpoint.
    """
    model_service = req.app.state.model_service
    
    try:
        model = model_service.load_model(
            request.model_name,
            request.checkpoint_path
        )
        config = model_service.get_config(request.model_name)
        
        return {
            "status": "loaded",
            "model_name": request.model_name,
            "config": {
                "n_layer": config.n_layer,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_neurons": config.n_neurons,
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{model_name}/unload")
async def unload_model(model_name: str, req: Request):
    """
    Unload a model to free memory.
    """
    model_service = req.app.state.model_service
    
    if not model_service.is_loaded(model_name):
        raise HTTPException(status_code=404, detail=f"Model not loaded: {model_name}")
    
    model_service.unload_model(model_name)
    
    return {
        "status": "unloaded",
        "model_name": model_name,
    }


@router.get("/{model_name}/graph")
def get_model_graph(
    model_name: str,
    req: Request,
    threshold: float = 0.01
):
    """
    Get graph topology from model weights.
    
    Returns the structure that emerges from encoder/decoder weight matrices.
    """
    model_service = req.app.state.model_service
    
    try:
        model = model_service.get_or_load(model_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model error: {e}")
    
    topology = model.get_graph_topology(threshold=threshold)
    
    # Convert numpy arrays to lists
    return {
        "n_heads": topology["n_heads"],
        "n_neurons_per_head": topology["n_neurons_per_head"],
        "density": topology["density"],
        "edges_per_head": topology["edges_per_head"],
        "total_edges": topology["total_edges"],
        "out_degree": topology["out_degree"].tolist(),
        "in_degree": topology["in_degree"].tolist(),
        # Note: Full adjacency matrix is too large to return
        # Use /visualization/graph for pre-computed layout
    }
