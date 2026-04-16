import json

tl = json.load(open(r"changed_checkpoints_data\viz_data_complete\synapses\timeline.json"))
s = tl["sentences"][0]
timeline = s["timeline"]

# Show timeline entries for "euros" at positions 26-30
print("Timeline entries at t=26..31 (euros):")
for e in timeline[26:31]:
    print(f"  t={e['t']} char='{e['char']}' vals[0:5]={e['vals'][:5]}")

# The script computes: for entry in timeline: if entry["t"] >= w_start and entry["t"] < w_end
# For "euros" w_start=26, w_end=31
# So it checks entry["t"] >= 26 and entry["t"] < 31
# entry["t"] IS the position in the array

# Let me manually compute ds for synapse 0 (id=0)
ds = 0.0
for entry in timeline:
    if entry["t"] >= 26 and entry["t"] < 31:
        ds += entry["vals"][0]
        print(f"  t={entry['t']}: vals[0]={entry['vals'][0]}")
print(f"Total ds for synapse 0 at 'euros': {ds}")

# Now check ALL vals entries for any non-zero
print("\nFirst 10 non-zero vals in timeline:")
count = 0
for e in timeline:
    for vi, v in enumerate(e["vals"]):
        if abs(v) > 0.0001:
            print(f"  t={e['t']} char='{e['char']}' vals[{vi}]={v}")
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break
