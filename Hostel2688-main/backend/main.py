"""BDH Interpretability Suite - FastAPI Backend"""

import torch
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pathlib import Path
import os
import sys

# Fix OpenBLAS memory allocation errors on Windows
# Must be set BEFORE importing numpy/torch
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Add training/ to sys.path so `from bdh import ...` works everywhere
# MUST happen BEFORE importing routes, since they do `from bdh import ...`
_training_dir = str(Path(__file__).parent.parent / "training")
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)


# Import routes (after sys.path is set up)
from backend.routes import inference, analysis, models, visualization, graph as graph_routes
from backend.routes import sparsity as sparsity_routes
try:
    from backend.routes import merge_api as merge_routes
except ImportError:
    merge_routes = None
# Get the project root directory (parent of backend/)
_PROJECT_ROOT = Path(__file__).parent.parent


class Settings:
    """Application settings."""
    PROJECT_NAME: str = "BDH Interpretability Suite"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Model paths - resolve relative to project root
    CHECKPOINT_DIR: Path = Path(
        os.getenv("CHECKPOINT_DIR", str(_PROJECT_ROOT / "checkpoints")))
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "french")

    # Device
    DEVICE: str = os.getenv(
        "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # CORS
    CORS_ORIGINS: List[str] = [
        "https://bdh-pathway.vercel.app",  # Vercel production
        "https://*.vercel.app",            # Vercel preview deploys
        "https://kumarutkarsh9263-backend.hf.space",  # HF Space itself
        "http://localhost:5173",           # Vite dev server
        "http://localhost:5174",           # Alternative Vite port
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "*",                               # Allow all origins (for debugging)
    ]


settings = Settings()
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    print("=" * 60)
    print(f"[BDH] {settings.PROJECT_NAME} v{settings.VERSION}")
    print("=" * 60)
    print(f"Device: {settings.DEVICE}")
    print(f"Checkpoint dir: {settings.CHECKPOINT_DIR}")

    # Initialize model registry
    from backend.services.model_service import ModelService
    app.state.model_service = ModelService(
        checkpoint_dir=settings.CHECKPOINT_DIR,
        device=settings.DEVICE
    )

    # Try to load default model
    try:
        app.state.model_service.load_model(settings.DEFAULT_MODEL)
        print(f"Loaded default model: {settings.DEFAULT_MODEL}")
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")

    print("=" * 60)
    print("[READY] Server ready!")
    print("=" * 60)

    yield

    # Shutdown
    print("\n[SHUTDOWN] Shutting down...")
    app.state.model_service.unload_all()
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend API for the BDH Interpretability Suite.",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(
    inference.router,
    prefix=f"{settings.API_PREFIX}/inference",
    tags=["inference"]
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_PREFIX}/analysis",
    tags=["analysis"]
)

app.include_router(
    models.router,
    prefix=f"{settings.API_PREFIX}/models",
    tags=["models"]
)

app.include_router(
    visualization.router,
    prefix=f"{settings.API_PREFIX}/visualization",
    tags=["visualization"]
)
app.include_router(
    graph_routes.router,
    prefix=f"{settings.API_PREFIX}/graph",
    tags=["graph"])

app.include_router(
    sparsity_routes.router,
    prefix=f"{settings.API_PREFIX}/sparsity",
    tags=["sparsity"])

if merge_routes:
    app.include_router(
        merge_routes.router,
        prefix=f"{settings.API_PREFIX}/merge",
        tags=["merge"]
    )
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "device": settings.DEVICE,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get(f"{settings.API_PREFIX}/status")
async def api_status():
    """API status with loaded models."""
    model_service = app.state.model_service

    return {
        "status": "running",
        "loaded_models": model_service.list_loaded_models(),
        "available_models": model_service.list_available_models(),
        "device": settings.DEVICE,
    }
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    import traceback
    tb = traceback.format_exc()
    print(f"[ERROR] {type(exc).__name__}: {exc}\n{tb}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "detail": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
