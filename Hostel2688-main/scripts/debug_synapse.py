import json

tl = json.load(open(r"changed_checkpoints_data\viz_data_complete\synapses\timeline.json"))
s = tl["sentences"][0]
print("Sentence text:", s["text"][:120])

# Show chars from timeline
chars = [e["char"] for e in s["timeline"]]
print("First 50 chars:", "".join(chars[:50]))

# Word splitting (same logic as script)
text = s["text"]
words = []
current_word = ""
current_start = 0
for i, ch in enumerate(text):
    if ch.isalpha() or ch in "àâäéèêëïîôùûüçœæ'":
        if not current_word:
            current_start = i
        current_word += ch
    else:
        if current_word:
            words.append((current_word, current_start, i))
            current_word = ""
if current_word:
    words.append((current_word, current_start, len(text)))

print("Words:", [(w, st, en) for w, st, en in words[:20]])
print("Total words:", len(words))

# Find 'euro' in the words
concept_words = {"euro", "dollar", "franc", "pound", "yen", "livre"}
for w, ws, we in words:
    if w.lower() in concept_words:
        print(f"  CONCEPT WORD: '{w}' at [{ws}:{we}]")

# Check timeline vals at word positions
euro_pos = text.lower().find("euro")
print(f"\n'euro' at text position: {euro_pos}")
if euro_pos >= 0:
    for e in s["timeline"][max(0, euro_pos-2):euro_pos+6]:
        nz = sum(1 for v in e["vals"] if abs(v) > 0.001)
        print(f"  t={e['t']} char='{e['char']}' byte={e['byte']} nonzero={nz}")

# The issue: timeline 't' maps to byte position, not char position in text
# The text has multi-byte UTF-8 chars (é, è, etc.)
print("\nTimeline t vs text chars:")
for i in range(min(10, len(s["timeline"]))):
    e = s["timeline"][i]
    text_char = text[i] if i < len(text) else "?"
    print(f"  t={e['t']}: timeline_char='{e['char']}', text_char='{text_char}'")
