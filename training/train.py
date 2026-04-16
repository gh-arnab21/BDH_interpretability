#!/usr/bin/env python3
"""BDH training script. Usage: python train.py --config configs/french.yaml"""

import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))
from bdh import BDH, BDHConfig, ExtractionConfig, load_model


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data
    train_data: str = "data/en-fr/train.bin"
    val_data: str = "data/en-fr/val.bin"
    
    # Model
    n_layer: int = 8
    n_embd: int = 256
    n_head: int = 4
    mlp_multiplier: int = 128
    dropout: float = 0.1
    vocab_size: int = 256
    
    # Training
    batch_size: int = 32
    block_size: int = 512  # Context length
    max_iters: int = 10000
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Gradient accumulation (effective_batch = batch_size * grad_accum)
    gradient_accumulation_steps: int = 4
    
    # Logging & Checkpointing
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    eval_iters: int = 100
    
    # Output
    output_dir: str = "checkpoints"
    run_name: str = "bdh_french"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"  # bfloat16, float16, or float32
    compile_model: bool = True  # Use torch.compile
    
    # Wandb (optional)
    wandb_project: str = ""
    wandb_run_name: str = ""


def get_lr(it: int, config: TrainingConfig) -> float:
    """Learning rate schedule with warmup and cosine decay."""
    # Linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    
    # Cosine decay
    if it > config.max_iters:
        return config.min_lr
    
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


class ByteDataset:
    """
    Memory-mapped dataset for byte-level language modeling.
    
    Reads raw bytes from file and creates (input, target) pairs
    where target is input shifted by 1 position.
    """
    
    def __init__(self, data_path: str, block_size: int):
        self.data_path = data_path
        self.block_size = block_size
        
        # Memory-map the file for efficient random access
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.length = len(self.data)
        
        print(f"Loaded {data_path}: {self.length:,} bytes")
    
    def __len__(self) -> int:
        return self.length - self.block_size - 1
    
    def get_batch(self, batch_size: int, device: str) -> tuple:
        """Get a random batch of sequences."""
        ix = torch.randint(len(self), (batch_size,))
        
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        
        return x, y


@torch.no_grad()
def estimate_loss(
    model: BDH,
    train_data: ByteDataset,
    val_data: ByteDataset,
    config: TrainingConfig,
    ctx
) -> Dict[str, float]:
    """Estimate loss on train and val sets."""
    model.eval()
    losses = {}
    
    for split, dataset in [("train", train_data), ("val", val_data)]:
        total_loss = 0.0
        for _ in range(config.eval_iters):
            x, y = dataset.get_batch(config.batch_size, config.device)
            with ctx:
                _, loss = model(x, y)
            total_loss += loss.item()
        losses[split] = total_loss / config.eval_iters
    
    model.train()
    return losses


@torch.no_grad()
def extract_snapshot(
    model: BDH,
    dataset: ByteDataset,
    config: TrainingConfig,
    ctx
) -> Dict[str, Any]:
    """Extract activation snapshot for visualization."""
    model.eval()
    
    # Get a single batch
    x, y = dataset.get_batch(1, config.device)  # Single example
    
    # Run with extraction
    extraction_config = ExtractionConfig(
        capture_sparse_activations=True,
        capture_attention_patterns=True,
        capture_pre_relu=True,
        capture_layer_outputs=True,
    )
    
    with model.extraction_mode(extraction_config) as buffer:
        with ctx:
            _, _ = model(x, y)
        
        # Compute sparsity stats
        sparsity_stats = buffer.get_sparsity_stats()
        
        # Get graph topology
        graph = model.get_graph_topology()
    
    model.train()
    
    return {
        "input_tokens": x[0].cpu().tolist(),
        "sparsity": sparsity_stats,
        "graph_density": graph["density"],
        "edges_per_head": graph["edges_per_head"],
    }


def save_checkpoint(
    model: BDH,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    iteration: int,
    losses: Dict[str, float],
    output_dir: Path,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(BDHConfig(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            n_head=config.n_head,
            mlp_internal_dim_multiplier=config.mlp_multiplier,
            dropout=config.dropout,
            vocab_size=config.vocab_size,
        )),
        "training_config": asdict(config),
        "losses": losses,
    }
    
    # Save regular checkpoint
    checkpoint_path = output_dir / f"checkpoint_{iteration:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[SAVE] Saved checkpoint: {checkpoint_path}")
    
    # Save as latest
    latest_path = output_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save as best if applicable
    if is_best:
        best_path = output_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"[BEST] New best model! Val loss: {losses['val']:.4f}")


def train(config: TrainingConfig):
    """Main training loop."""
    
    print("=" * 60)
    print("BDH Training")
    print("=" * 60)
    print(f"Run name: {config.run_name}")
    print(f"Device: {config.device}")
    print(f"Data: {config.train_data}")
    print(f"Model: {config.n_layer}L, {config.n_embd}D, {config.n_head}H")
    print(f"Neurons per head: {config.n_embd * config.mlp_multiplier // config.n_head}")
    print(f"Total neurons: {config.n_embd * config.mlp_multiplier}")
    print("=" * 60)
    
    # Setup device and precision
    device = config.device
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    ptdtype = dtype_map[config.dtype]
    
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if "cuda" in device
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(
        device=device,
        enabled=(config.dtype == "float16")
    )
    
    # Setup output directory
    output_dir = Path(config.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Load data
    print("\n[DATA] Loading data...")
    train_data = ByteDataset(config.train_data, config.block_size)
    val_data = ByteDataset(config.val_data, config.block_size)
    
    # Create model
    print("\n[MODEL] Creating model...")
    model_config = BDHConfig(
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        n_head=config.n_head,
        mlp_internal_dim_multiplier=config.mlp_multiplier,
        dropout=config.dropout,
        vocab_size=config.vocab_size,
    )
    model = BDH(model_config).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Compile model for speed (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        print("[COMPILE] Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Optional wandb logging
    if config.wandb_project:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or config.run_name,
                config=asdict(config),
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            config.wandb_project = ""
    
    # Training loop
    print("\n[START] Starting training...")
    model.train()
    
    best_val_loss = float("inf")
    running_loss = 0.0
    start_time = time.time()
    
    # Get first batch
    x, y = train_data.get_batch(config.batch_size, device)
    
    for iteration in range(config.max_iters):
        # Update learning rate
        lr = get_lr(iteration, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps
            
            # Get next batch while computing gradients
            x, y = train_data.get_batch(config.batch_size, device)
            
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * config.gradient_accumulation_steps
        
        # Logging
        if iteration % config.log_interval == 0:
            avg_loss = running_loss / config.log_interval if iteration > 0 else running_loss
            elapsed = time.time() - start_time
            tokens_per_sec = (
                config.batch_size * config.block_size *
                config.gradient_accumulation_steps * config.log_interval
            ) / elapsed if iteration > 0 else 0
            
            print(
                f"iter {iteration:6d} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | {tokens_per_sec:.0f} tok/s"
            )
            
            if config.wandb_project:
                import wandb
                wandb.log({
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }, step=iteration)
            
            running_loss = 0.0
            start_time = time.time()
        
        # Evaluation
        if iteration > 0 and iteration % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, ctx)
            print(
                f"📊 Eval @ {iteration}: "
                f"train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}"
            )
            
            if config.wandb_project:
                import wandb
                wandb.log({
                    "eval/train_loss": losses["train"],
                    "eval/val_loss": losses["val"],
                }, step=iteration)
            
            # Extract snapshot for visualization
            snapshot = extract_snapshot(model, val_data, config, ctx)
            print(
                f"   Sparsity: {snapshot['sparsity']['overall_sparsity']:.1%} | "
                f"Graph density: {snapshot['graph_density']:.4f}"
            )
            
            # Check if best model
            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]
            
            # Save checkpoint
            if iteration % config.save_interval == 0 or is_best:
                save_checkpoint(
                    model, optimizer, config, iteration,
                    losses, output_dir, is_best
                )
    
    # Final save
    print("\n✅ Training complete!")
    losses = estimate_loss(model, train_data, val_data, config, ctx)
    save_checkpoint(
        model, optimizer, config, config.max_iters,
        losses, output_dir, losses["val"] < best_val_loss
    )
    
    # Generate sample
    print("\n📝 Generating sample...")
    model.eval()
    prompt = torch.tensor(
        bytearray("<F:en>The European Union", "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=100, top_k=3)
    
    text = bytes(generated[0].cpu().tolist()).decode(errors="backslashreplace")
    print(f"Generated: {text}")
    
    if config.wandb_project:
        import wandb
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train BDH model")
    
    # Config file (overrides defaults)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Data
    parser.add_argument("--train-data", type=str, help="Training data path")
    parser.add_argument("--val-data", type=str, help="Validation data path")
    
    # Model
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mlp-multiplier", type=int, default=128)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--name", type=str, default="bdh_run")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default="")
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    config = TrainingConfig()
    
    if args.config:
        import yaml
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f)
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Override with command line args (only if explicitly provided)
    if args.train_data:
        config.train_data = args.train_data
    if args.val_data:
        config.val_data = args.val_data
    if args.name != "bdh_run":  # Only override if not default
        config.run_name = args.name
    if args.wandb_project:
        config.wandb_project = args.wandb_project
    
    # Only override model/training params if no config file provided
    if not args.config:
        config.n_layer = args.n_layer
        config.n_embd = args.n_embd
        config.n_head = args.n_head
        config.mlp_multiplier = args.mlp_multiplier
        config.batch_size = args.batch_size
        config.block_size = args.block_size
        config.max_iters = args.max_iters
        config.learning_rate = args.lr
        config.output_dir = args.output_dir
    
    train(config)


if __name__ == "__main__":
    main()
