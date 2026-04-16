"""Quick check of precomputed.json state."""
import json, os

f = "frontend/public/monosemanticity/precomputed.json"
print("Size:", round(os.path.getsize(f)/1024, 1), "KB")

d = json.load(open(f))
print("Keys:", list(d.keys()))
print("best_layer:", d.get("best_layer"))
print("model_info:", d.get("model_info"))
print()

for cname, concept in d["concepts"].items():
    print(f"=== {cname} ({len(concept['words'])} words) ===")
    for w in concept["words"]:
        print(f"  {w['word']}:")
        for la in w["layers"]:
            if la["layer"] == 3:  # best layer
                for h in la["heads"]:
                    nz = sum(1 for v in h["x_ds"] if abs(v) > 1e-6)
                    tn = len(h["top_neurons"])
                    print(f"    H{h['head']}: {nz} active in x_ds, {tn} top_neurons")
    # Similarity check
    sim = concept.get("similarity", {})
    if "3" in sim:
        m = sim["3"]
        n = len(m)
        print(f"  similarity['3']: {n}x{len(m[0]) if m else 0}")
        if n >= 2:
            print(f"    [0][1]={m[0][1]:.3f}, [0][-1]={m[0][-1]:.3f}")
    print()

# Cross-concept
print(f"cross_concept: {len(d.get('cross_concept', []))} pairs")
if d.get("cross_concept"):
    cc = d["cross_concept"][0]
    print(f"  first: {cc['primary']} vs {cc['secondary']}, distinctness={[round(x,3) for x in cc['distinctness_per_layer']]}")

# Selectivity
sel = d.get("selectivity")
if sel:
    print(f"\nselectivity histogram: {[b['count'] for b in sel['histogram']]}")
    print(f"  total_neurons={sel['total_neurons']}, total_selective={sel['total_selective']}")

# Synapse tracking
st = d.get("synapse_tracking", {})
print(f"\nsynapse_tracking: {list(st.keys())}")
for cname, track in st.items():
    print(f"  {cname}: {len(track['synapses'])} synapses, {len(track['sentences'])} sentences")
