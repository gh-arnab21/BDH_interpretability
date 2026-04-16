import json
from collections import defaultdict

tl = json.load(open(r"changed_checkpoints_data\viz_data_complete\synapses\timeline.json"))

# Calculate total non-zero activity per synapse across all sentences
synapse_activity = defaultdict(float)
for s in tl["sentences"]:
    for e in s["timeline"]:
        for vi, v in enumerate(e["vals"]):
            synapse_activity[vi] += abs(v)

print("Synapse activity ranking (total |vals| across all sentences):")
ranked = sorted(synapse_activity.items(), key=lambda x: -x[1])
for idx, total in ranked:
    syn = tl["tracked"][idx]
    print(f"  idx={idx:2d} id={syn['id']:2d} i={syn['i']:4d} j={syn['j']:4d} w={syn['weight']:+.3f}  total_activity={total:.4f}")

# Show which synapses are active at concept words
concept_positions = {
    "euros": (26, 31),   # sentence 0
    "pounds": (43, 49),  # sentence 0
    "dollar": None,      # sentence 1 - need to find
    "livres": None,      # sentence 0 - need to find
}
print("\nSynapse activity at 'euros' (t=26..31) in sentence 0:")
for vi in range(20):
    total = sum(abs(tl["sentences"][0]["timeline"][t]["vals"][vi]) for t in range(26, 31) if t < len(tl["sentences"][0]["timeline"]))     
    if total > 0:
        print(f"  synapse {vi}: {total:.6f}")

print("\nSynapse activity at 'pounds' (t=43..49) in sentence 0:")
for vi in range(20):
    total = sum(abs(tl["sentences"][0]["timeline"][t]["vals"][vi]) for t in range(43, 49) if t < len(tl["sentences"][0]["timeline"]))     
    if total > 0:
        print(f"  synapse {vi}: {total:.6f}")
