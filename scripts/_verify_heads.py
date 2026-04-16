"""Quick check: verify all 4 heads populated and show head-level stats."""
import json

d = json.load(open("frontend/public/monosemanticity/precomputed.json"))

for cname, c in d["concepts"].items():
    print(f"\n=== {cname} ===")
    for w in c["words"]:
        word = w["word"]
        for l_data in w["layers"]:
            if l_data["layer"] != 3:
                continue
            line_parts = [f"  {word:12s} L{l_data['layer']}:"]
            for h in l_data["heads"]:
                nt = len(h["top_neurons"])
                na = h["x_active"]
                nz_xds = sum(1 for v in h["x_ds"] if abs(v) > 1e-6)
                line_parts.append(f"H{h['head']}={na}active({nz_xds}xds,{nt}top)")
            print("  ".join(line_parts))

# Check similarity at layer 3 for currency
print("\n=== Similarity matrix (currency, L3) ===")
sim = d["concepts"]["currency"]["similarity"]["3"]
words = [w["word"] for w in d["concepts"]["currency"]["words"]]
print(f"{'':10}", "  ".join(f"{w:>8}" for w in words))
for i, row in enumerate(sim):
    print(f"{words[i]:10}", "  ".join(f"{v:8.4f}" for v in row))
