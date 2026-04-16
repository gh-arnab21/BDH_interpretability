"""Quick inspection of hero_token per-head activation patterns."""
import json, os
from collections import defaultdict

HERO_DIR = "changed_checkpoints_data/viz_data_complete/hero_tokens"

# Load all hero files grouped by (layer, tok)
all_files = sorted(os.listdir(HERO_DIR))
by_layer_tok = {}
for f in all_files:
    d = json.load(open(os.path.join(HERO_DIR, f)))
    by_layer_tok[(d["layer"], d["tok"])] = d

# Check all layers for "euros" bytes (26-30)
euros_toks = [26, 27, 28, 29, 30]

print("=== REAL PER-HEAD ACTIVATIONS FOR 'euros' bytes ===\n")
for layer in range(6):
    print(f"--- Layer {layer} ---")
    for tok in euros_toks:
        key = (layer, tok)
        if key not in by_layer_tok:
            print(f"  tok={tok}: NO DATA")
            continue
        d = by_layer_tok[key]
        x = d["x"]  # [4][3072]
        for h_idx, hv in enumerate(x):
            nz = sum(1 for v in hv if abs(v) > 1e-6)
            top5_idx = sorted(range(len(hv)), key=lambda i: -abs(hv[i]))[:5]
            top5 = [(i, round(hv[i], 4)) for i in top5_idx]
            print(f"  tok={tok} char={repr(d['char'])} H{h_idx}: {nz:3d} active | top5={top5}")
    print()

# Aggregate: mean activation per head across euros bytes at layer 3
print("=== LAYER 3 AGGREGATE: mean x per head across 'euros' ===")
for h_idx in range(4):
    vals = []
    for tok in euros_toks:
        key = (3, tok)
        if key in by_layer_tok:
            hv = by_layer_tok[key]["x"][h_idx]
            vals.extend(v for v in hv if abs(v) > 1e-6)
    mean_v = sum(vals) / len(vals) if vals else 0
    print(f"  H{h_idx}: {len(vals)} nonzero values, mean={mean_v:.6f}")
