import json
from collections import Counter
d = json.load(open("changed_checkpoints_data/viz_data_complete/monosemanticity/precomputed.json"))
heads = Counter(n["head"] for n in d["top_200"])
print("top_200 head distribution:", dict(heads))
fp = d["fingerprints"]
for c in fp:
    heads_c = Counter(n["head"] for n in fp[c]["top"])
    print(f"  {c}: heads={dict(heads_c)}")
