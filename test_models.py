from bdh import load_model
import sys
import torch
sys.path.insert(0, 'training')

device = 'cpu'

for name, path in [('French', 'checkpoints/french/french_best.pt'),
                   ('Portuguese', 'checkpoints/portuguese/portuguese_best.pt')]:
    model = load_model(path, device)

    # Check what loss looks like on a simple string
    text = "The European Parliament adopted the resolution on trade policy."
    tokens = torch.tensor([list(text.encode('utf-8'))], dtype=torch.long)
    with torch.no_grad():
        logits, loss = model(tokens[:, :-1], tokens[:, 1:])
    print(f'{name} model loss on English text: {loss.item():.4f}')

    # Generate
    prompt = "Le parlement " if 'French' in name else "O parlamento "
    ptokens = torch.tensor([list(prompt.encode('utf-8'))], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(ptokens, max_new_tokens=50,
                             top_k=5, temperature=0.8)
    gen = bytes(out[0].tolist()).decode('utf-8', errors='replace')
    print(f'{name} generation: {gen}\n')
