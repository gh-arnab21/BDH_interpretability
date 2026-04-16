import json, os

path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'public', 'monosemanticity', 'precomputed.json')
with open(path, 'r') as f:
    data = json.load(f)

best = data['best_layer']
print(f"Best layer: {best}\n")

# For each concept, for each word, print top-5 neurons per head
for cid, cr in data['concepts'].items():
    print(f"=== {cid.upper()} ===")
    for w in cr['words']:
        layer = [l for l in w['layers'] if l['layer'] == best][0]
        for h in layer['heads']:
            top5 = sorted(h['top_neurons'], key=lambda n: -n['val'])[:5]
            idxs = [n['idx'] for n in top5]
            vals = [round(n['val'], 4) for n in top5]
            print(f"  {w['word']:12s} H{h['head']}: idx={idxs}  val={vals}")
    print()

# Cross-concept comparison: which neuron indices appear in ALL 4 concepts?
print("=" * 60)
print("CROSS-CONCEPT NEURON OVERLAP")
print("=" * 60)
for head in range(4):
    concept_neuron_sets = {}
    for cid, cr in data['concepts'].items():
        all_idxs = set()
        for w in cr['words']:
            layer = [l for l in w['layers'] if l['layer'] == best][0]
            h = [hd for hd in layer['heads'] if hd['head'] == head][0]
            for n in h['top_neurons']:
                all_idxs.add(n['idx'])
        concept_neuron_sets[cid] = all_idxs

    # Find neurons shared across ALL concepts
    shared_all = set.intersection(*concept_neuron_sets.values())
    print(f"\nH{head}: Neurons in ALL 4 concepts: {sorted(shared_all)[:20]} (total: {len(shared_all)})")

    for c1 in concept_neuron_sets:
        for c2 in concept_neuron_sets:
            if c1 >= c2:
                continue
            overlap = concept_neuron_sets[c1] & concept_neuron_sets[c2]
            total = len(concept_neuron_sets[c1] | concept_neuron_sets[c2])
            print(f"  {c1:12s} vs {c2:12s}: {len(overlap)}/{total} shared ({100*len(overlap)/max(total,1):.0f}%)")
