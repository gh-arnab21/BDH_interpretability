#!/usr/bin/env python3
"""
BDH Data Optimizer for Frontend

This script:
1. Organizes your downloaded JSON files
2. Creates optimized summary files for smooth animations
3. Reduces file sizes by removing redundant data
4. Creates a unified manifest for the frontend

Run after downloading your trained model JSONs:
    python optimize_viz_data.py --input viz_data_french/ --output frontend/public/playback/french/
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import shutil
from tqdm import tqdm


def load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, path: Path, indent: int = None):
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def extract_iteration(filename: str) -> int:
    """Extract iteration number from checkpoint filename."""
    # checkpoint_000500.json -> 500
    try:
        name = Path(filename).stem
        if "checkpoint_" in name:
            return int(name.split("_")[1])
        elif "best" in name.lower():
            return 999999  # Put best at end
        else:
            return 0
    except:
        return 0


def create_evolution_timeline(checkpoint_files: List[Path]) -> Dict:
    """
    Create a lightweight timeline of how metrics evolve.
    This is what powers the "evolution animation" slider.
    """
    timeline = []
    
    for cp_file in tqdm(checkpoint_files, desc="Building timeline"):
        try:
            data = load_json(cp_file)
            iteration = extract_iteration(cp_file.name)
            
            entry = {
                "iteration": iteration,
                "checkpoint": cp_file.stem,
                "file": cp_file.name,
            }
            
            # Extract key metrics
            if "sparsity" in data:
                entry["sparsity"] = round(data["sparsity"].get("overall_sparsity", 0), 4)
                entry["x_sparsity"] = round(data["sparsity"].get("overall_x_sparsity", 0), 4)
                entry["y_sparsity"] = round(data["sparsity"].get("overall_y_sparsity", 0), 4)
                entry["sparsity_by_layer"] = [round(s, 4) for s in data["sparsity"].get("sparsity_by_layer", [])]
            
            if "graph" in data:
                entry["graph_density"] = round(data["graph"].get("density", 0), 4)
                entry["total_edges"] = data["graph"].get("total_edges", 0)
            
            # Count active concepts
            if "concepts" in data:
                entry["concepts_detected"] = len(data["concepts"])
            
            timeline.append(entry)
        except Exception as e:
            print(f"  Warning: Could not process {cp_file.name}: {e}")
            continue
    
    # Sort by iteration
    timeline.sort(key=lambda x: x["iteration"])
    
    return {
        "timeline": timeline,
        "num_checkpoints": len(timeline),
        "iterations": [t["iteration"] for t in timeline],
        "metrics": ["sparsity", "x_sparsity", "y_sparsity", "graph_density"],
    }


def create_keyframe_summary(checkpoint_files: List[Path], num_keyframes: int = 10) -> Dict:
    """
    Create a summary with just the key checkpoints for quick loading.
    Useful for the evolution animation without loading ALL files.
    """
    # Select evenly spaced keyframes
    total = len(checkpoint_files)
    if total <= num_keyframes:
        keyframe_files = checkpoint_files
    else:
        step = total // num_keyframes
        keyframe_files = [checkpoint_files[i * step] for i in range(num_keyframes)]
        # Always include last
        if checkpoint_files[-1] not in keyframe_files:
            keyframe_files.append(checkpoint_files[-1])
    
    keyframes = []
    for cp_file in tqdm(keyframe_files, desc="Creating keyframes"):
        try:
            data = load_json(cp_file)
            iteration = extract_iteration(cp_file.name)
            
            keyframe = {
                "iteration": iteration,
                "checkpoint": cp_file.stem,
                "sparsity": data.get("sparsity", {}),
                "graph": data.get("graph", {}),
            }
            
            # Include first playback example only (reduced data)
            if "playback" in data and len(data["playback"]) > 0:
                pb = data["playback"][0]["data"]
                # Only keep first 20 frames for preview
                keyframe["playback_preview"] = {
                    "input_text": pb.get("input_text", "")[:50],
                    "overall_sparsity": pb.get("overall_sparsity", 0),
                    "num_frames": len(pb.get("frames", [])),
                    "frames": pb.get("frames", [])[:20],  # Just first 20
                }
            
            keyframes.append(keyframe)
        except Exception as e:
            print(f"  Warning: Could not process {cp_file.name}: {e}")
    
    keyframes.sort(key=lambda x: x["iteration"])
    
    return {
        "keyframes": keyframes,
        "num_keyframes": len(keyframes),
        "iterations": [k["iteration"] for k in keyframes],
    }


def create_concept_evolution(checkpoint_files: List[Path]) -> Dict:
    """
    Track how concept detection evolves during training.
    Shows when the model starts recognizing currencies, countries, etc.
    """
    concept_timeline = {concept: [] for concept in ["currencies", "countries", "languages", "numbers", "months"]}
    
    for cp_file in tqdm(checkpoint_files, desc="Tracking concepts"):
        try:
            data = load_json(cp_file)
            iteration = extract_iteration(cp_file.name)
            
            if "concepts" not in data:
                continue
            
            for concept_name, concept_data in data["concepts"].items():
                if concept_name in concept_timeline:
                    concept_timeline[concept_name].append({
                        "iteration": iteration,
                        "avg_activation": round(concept_data.get("avg_max_activation", 0), 4),
                        "consistent_neurons": len(concept_data.get("consistent_neurons", [])),
                    })
        except:
            continue
    
    # Sort each concept's timeline
    for concept in concept_timeline:
        concept_timeline[concept].sort(key=lambda x: x["iteration"])
    
    return {
        "concepts": concept_timeline,
        "available_concepts": list(concept_timeline.keys()),
    }


def create_manifest(
    output_dir: Path,
    model_name: str,
    checkpoint_files: List[Path],
    config: Dict = None
) -> Dict:
    """Create the master manifest file."""
    
    manifest = {
        "model_name": model_name,
        "num_checkpoints": len(checkpoint_files),
        "checkpoint_files": sorted([f.name for f in checkpoint_files]),
        "config": config or {},
        "files": {
            "evolution_timeline": "evolution_timeline.json",
            "keyframe_summary": "keyframe_summary.json",
            "concept_evolution": "concept_evolution.json",
        },
        "data_format_version": "1.0",
    }
    
    return manifest


def optimize_single_checkpoint(input_path: Path, output_path: Path):
    """
    Optimize a single checkpoint JSON by removing redundant data.
    Keeps file sizes manageable.
    """
    data = load_json(input_path)
    
    # Optimize playback data - reduce precision
    if "playback" in data:
        for pb_item in data["playback"]:
            if "data" in pb_item and "frames" in pb_item["data"]:
                for frame in pb_item["data"]["frames"]:
                    # Round sparsity values
                    frame["x_sparsity"] = round(frame.get("x_sparsity", 0), 4)
                    frame["y_sparsity"] = round(frame.get("y_sparsity", 0), 4)
                    
                    # Limit active neuron lists (keep top 100 per head)
                    for key in ["x_active", "y_active"]:
                        if key in frame:
                            for head_data in frame[key]:
                                if len(head_data.get("indices", [])) > 100:
                                    # Keep only top 100 by value
                                    indices = head_data["indices"][:100]
                                    values = [round(v, 4) for v in head_data["values"][:100]]
                                    head_data["indices"] = indices
                                    head_data["values"] = values
    
    # Save optimized version (no indent for smaller size)
    save_json(data, output_path, indent=None)


def main():
    parser = argparse.ArgumentParser(description="Optimize BDH visualization data for frontend")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory with checkpoint JSONs")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory (e.g., frontend/public/playback/french/)")
    parser.add_argument("--model-name", type=str, default="french_specialist",
                       help="Model name for manifest")
    parser.add_argument("--num-keyframes", type=int, default=10,
                       help="Number of keyframes for summary")
    parser.add_argument("--copy-full", action="store_true",
                       help="Copy full checkpoint JSONs (not just summaries)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🎬 BDH Data Optimizer")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model_name}")
    print("=" * 60)
    
    # Find all checkpoint JSONs
    checkpoint_files = sorted(
        [f for f in input_dir.glob("checkpoint_*.json")],
        key=lambda x: extract_iteration(x.name)
    )
    
    # Also check for evolution_summary if it exists
    if (input_dir / "evolution_summary.json").exists():
        checkpoint_files = [f for f in checkpoint_files if f.name != "evolution_summary.json"]
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    
    if len(checkpoint_files) == 0:
        print("❌ No checkpoint files found!")
        print(f"   Looking in: {input_dir}")
        print(f"   Expected pattern: checkpoint_*.json")
        return
    
    # Extract config from first checkpoint
    first_data = load_json(checkpoint_files[0])
    config = first_data.get("config", {})
    
    # 1. Create evolution timeline (lightweight)
    print("\n📊 Creating evolution timeline...")
    timeline = create_evolution_timeline(checkpoint_files)
    save_json(timeline, output_dir / "evolution_timeline.json", indent=2)
    print(f"   Saved: evolution_timeline.json ({len(timeline['timeline'])} points)")
    
    # 2. Create keyframe summary (medium weight)
    print("\n🎯 Creating keyframe summary...")
    keyframes = create_keyframe_summary(checkpoint_files, args.num_keyframes)
    save_json(keyframes, output_dir / "keyframe_summary.json", indent=2)
    print(f"   Saved: keyframe_summary.json ({len(keyframes['keyframes'])} keyframes)")
    
    # 3. Create concept evolution
    print("\n🧠 Creating concept evolution...")
    concepts = create_concept_evolution(checkpoint_files)
    save_json(concepts, output_dir / "concept_evolution.json", indent=2)
    print(f"   Saved: concept_evolution.json")
    
    # 4. Copy/optimize full checkpoint files if requested
    if args.copy_full:
        print("\n📁 Copying optimized checkpoint files...")
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        for cp_file in tqdm(checkpoint_files, desc="Optimizing"):
            optimize_single_checkpoint(cp_file, checkpoints_dir / cp_file.name)
        
        print(f"   Saved {len(checkpoint_files)} optimized checkpoints")
    
    # 5. Create manifest
    print("\n📋 Creating manifest...")
    manifest = create_manifest(output_dir, args.model_name, checkpoint_files, config)
    manifest["has_full_checkpoints"] = args.copy_full
    save_json(manifest, output_dir / "manifest.json", indent=2)
    print(f"   Saved: manifest.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Optimization complete!")
    print("=" * 60)
    
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.json")) / 1024
    print(f"\nOutput files in {output_dir}:")
    for f in sorted(output_dir.glob("*.json")):
        size = f.stat().st_size / 1024
        print(f"  {f.name}: {size:.1f} KB")
    print(f"\nTotal size: {total_size:.1f} KB")
    
    print("\n📌 Next steps:")
    print(f"  1. Copy {output_dir} to frontend/public/playback/{args.model_name}/")
    print(f"  2. Run: cd frontend && npm install && npm run dev")
    print(f"  3. Open http://localhost:5173")


if __name__ == "__main__":
    main()
