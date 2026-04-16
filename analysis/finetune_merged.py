#!/usr/bin/env python3
"""
BDH Merged Model Fine-Tuning

After merging two specialists by concatenating neurons and averaging
embed/lm_head, the shared parameters need a small amount of adaptation
to learn proper routing to both neuron banks.

This script:
1. Loads the merged checkpoint
2. Creates mixed French+Portuguese training data (built-in, no downloads)
3. Fine-tunes for ~500 iterations (5-15 min on CPU)
4. Saves the fine-tuned checkpoint

Usage:
    python analysis/finetune_merged.py \
        --checkpoint checkpoints/merged_polyglot/checkpoint_best.pt \
        --output checkpoints/merged_finetuned/checkpoint_best.pt \
        --iters 500

If you have the Europarl .bin files locally, use them for better results:
    python analysis/finetune_merged.py \
        --checkpoint checkpoints/merged_polyglot/checkpoint_best.pt \
        --output checkpoints/merged_finetuned/checkpoint_best.pt \
        --french-data data/en-fr/train.bin \
        --portuguese-data data/en-pt/train.bin \
        --iters 500
"""

import argparse
import sys
import time
import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from bdh import BDH, BDHConfig

# ═══════════════════════════════════════════════════════════════════════
#  Built-in training data (Europarl-style mixed French + Portuguese)
# ═══════════════════════════════════════════════════════════════════════

MIXED_DATA = """<F:en>The European Parliament has adopted the resolution on trade policy.<T:fr>Le Parlement européen a adopté la résolution sur la politique commerciale.
<F:en>We must ensure that all citizens have access to healthcare.<T:fr>Nous devons veiller à ce que tous les citoyens aient accès aux soins de santé.
<F:en>The Commission presented its annual report on economic growth.<T:fr>La Commission a présenté son rapport annuel sur la croissance économique.
<F:en>Mr President, I would like to raise a point of order.<T:fr>Monsieur le Président, je voudrais soulever une motion de procédure.
<F:en>This directive aims to protect the environment and public health.<T:fr>Cette directive vise à protéger l'environnement et la santé publique.
<F:en>The Council has reached an agreement on the new regulation.<T:fr>Le Conseil est parvenu à un accord sur le nouveau règlement.
<F:en>I believe this proposal will benefit all member states.<T:fr>Je crois que cette proposition profitera à tous les États membres.
<F:en>The rapporteur has done excellent work on this report.<T:fr>Le rapporteur a fait un excellent travail sur ce rapport.
<F:en>We need to strengthen cooperation between our institutions.<T:fr>Nous devons renforcer la coopération entre nos institutions.
<F:en>The budget for next year must be approved by December.<T:fr>Le budget pour l'année prochaine doit être approuvé avant décembre.
<F:en>Freedom of expression is a fundamental right in Europe.<T:fr>La liberté d'expression est un droit fondamental en Europe.
<F:en>The single market has been a great success for our continent.<T:fr>Le marché unique a été un grand succès pour notre continent.
<F:en>Climate change is one of the greatest challenges of our time.<T:fr>Le changement climatique est l'un des plus grands défis de notre temps.
<F:en>We must protect the rights of workers across the European Union.<T:fr>Nous devons protéger les droits des travailleurs dans toute l'Union européenne.
<F:en>The digital economy offers new opportunities for growth and employment.<T:fr>L'économie numérique offre de nouvelles possibilités de croissance et d'emploi.
<F:en>Education and training are essential for the future of Europe.<T:fr>L'éducation et la formation sont essentielles pour l'avenir de l'Europe.
<F:en>The European Parliament has adopted the resolution on trade policy.<T:pt>O Parlamento Europeu adoptou a resolução sobre a política comercial.
<F:en>We must ensure that all citizens have access to healthcare.<T:pt>Devemos assegurar que todos os cidadãos tenham acesso aos cuidados de saúde.
<F:en>The Commission presented its annual report on economic growth.<T:pt>A Comissão apresentou o seu relatório anual sobre o crescimento económico.
<F:en>Mr President, I would like to raise a point of order.<T:pt>Senhor Presidente, gostaria de levantar uma questão de ordem.
<F:en>This directive aims to protect the environment and public health.<T:pt>Esta directiva tem por objectivo proteger o ambiente e a saúde pública.
<F:en>The Council has reached an agreement on the new regulation.<T:pt>O Conselho chegou a acordo sobre o novo regulamento.
<F:en>I believe this proposal will benefit all member states.<T:pt>Creio que esta proposta beneficiará todos os Estados-Membros.
<F:en>The rapporteur has done excellent work on this report.<T:pt>O relator realizou um excelente trabalho sobre este relatório.
<F:en>We need to strengthen cooperation between our institutions.<T:pt>Precisamos de reforçar a cooperação entre as nossas instituições.
<F:en>The budget for next year must be approved by December.<T:pt>O orçamento para o próximo ano deve ser aprovado até Dezembro.
<F:en>Freedom of expression is a fundamental right in Europe.<T:pt>A liberdade de expressão é um direito fundamental na Europa.
<F:en>The single market has been a great success for our continent.<T:pt>O mercado único tem sido um grande êxito para o nosso continente.
<F:en>Climate change is one of the greatest challenges of our time.<T:pt>As alterações climáticas são um dos maiores desafios do nosso tempo.
<F:en>We must protect the rights of workers across the European Union.<T:pt>Devemos proteger os direitos dos trabalhadores em toda a União Europeia.
<F:en>The digital economy offers new opportunities for growth and employment.<T:pt>A economia digital oferece novas oportunidades de crescimento e emprego.
<F:en>Education and training are essential for the future of Europe.<T:pt>A educação e a formação são essenciais para o futuro da Europa.
"""


class MixedByteDataset:
    """Byte-level dataset from either built-in text or .bin files."""

    def __init__(self, french_bin=None, portuguese_bin=None, block_size=256):
        self.block_size = block_size

        if french_bin and Path(french_bin).exists() and portuguese_bin and Path(portuguese_bin).exists():
            # Use real .bin files — interleave chunks
            fr = np.memmap(french_bin, dtype=np.uint8, mode="r")
            pt = np.memmap(portuguese_bin, dtype=np.uint8, mode="r")
            # Take equal amounts from each
            n = min(len(fr), len(pt))
            chunk = 4096
            parts = []
            for i in range(0, n, chunk * 2):
                parts.append(np.array(fr[i:i+chunk]))
                parts.append(np.array(pt[i:i+chunk]))
            self.data = np.concatenate(parts)
            print(f"  Loaded bin files: {len(self.data):,} bytes")
        else:
            # Use built-in sentences — repeat to get enough data
            base = MIXED_DATA.encode("utf-8")
            repeats = max(1, (50000 // len(base)) + 1)
            raw = base * repeats
            self.data = np.frombuffer(raw, dtype=np.uint8).copy()
            print(f"  Using built-in data: {len(self.data):,} bytes ({repeats} repeats)")

    def get_batch(self, batch_size, device):
        ix = np.random.randint(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)


def load_merged_checkpoint(path, device="cpu"):
    """Load merged model checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        state_dict = ckpt["model_state_dict"]
        # Filter to valid BDHConfig fields
        import dataclasses
        valid = {f.name for f in dataclasses.fields(BDHConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        config = BDHConfig(**cfg_dict)
    else:
        raise ValueError("Checkpoint has no 'config' key — is this a merged checkpoint?")

    # Strip _orig_mod prefix
    clean = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = BDH(config)
    model.load_state_dict(clean, strict=False)
    model.to(device)

    heritage = ckpt.get("heritage", None)
    return model, config, heritage


@torch.no_grad()
def evaluate(model, dataset, device, n_batches=20, batch_size=8):
    model.eval()
    total, count = 0.0, 0
    for _ in range(n_batches):
        x, y = dataset.get_batch(batch_size, device)
        _, loss = model(x, y)
        if loss is not None:
            total += loss.item()
            count += 1
    model.train()
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune merged BDH model")
    parser.add_argument("--checkpoint", required=True, help="Merged model checkpoint")
    parser.add_argument("--output", required=True, help="Output path for fine-tuned model")
    parser.add_argument("--french-data", default="", help="French .bin file (optional)")
    parser.add_argument("--portuguese-data", default="", help="Portuguese .bin file (optional)")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (lower than initial training)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--freeze-sparse", action="store_true",
                        help="Freeze encoder/decoder/freqs (specialist neurons) and only train embed+lm_head. "
                             "Preserves heritage routing at the cost of slightly higher final loss.")
    args = parser.parse_args()

    print("=" * 60)
    print("BDH Merged Model Fine-Tuning")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Iterations: {args.iters}")
    print(f"  LR: {args.lr}")
    print(f"  Freeze sparse: {args.freeze_sparse}")
    print(f"  Device: {args.device}")

    # Load model
    print("\n  Loading merged model...")
    model, config, heritage = load_merged_checkpoint(args.checkpoint, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {config.n_layer}L {config.n_embd}D {config.n_head}H N={config.n_neurons}")
    print(f"  Params: {n_params:,}")

    # Load data
    print("\n  Preparing data...")
    dataset = MixedByteDataset(args.french_data, args.portuguese_data, args.block_size)

    # Pre-finetune evaluation
    print("\n  Pre-finetune loss:")
    pre_loss = evaluate(model, dataset, args.device)
    print(f"  Mixed loss: {pre_loss:.4f}")

    # Freeze sparse layers if requested — preserves specialist neuron identity
    # so the heritage probe shows clean routing
    model.train()
    if args.freeze_sparse:
        # These are the concatenated specialist weights — freezing them preserves routing
        sparse_keywords = ['encoder', 'encoder_v', 'decoder', 'attn.freqs',
                           'encoders.', 'encoders_v.', 'decoders.', 'rho']
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            if any(kw in name for kw in sparse_keywords):
                param.requires_grad = False
                frozen += param.numel()
            else:
                trainable += param.numel()
        print(f"  Frozen: {frozen:,} params (specialist neurons)")
        print(f"  Trainable: {trainable:,} params (embed + lm_head + layernorms)")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Cosine LR schedule
    def get_lr(step):
        if step < 50:
            return args.lr * step / 50  # warmup
        progress = (step - 50) / max(args.iters - 50, 1)
        return args.lr * 0.5 * (1 + np.cos(np.pi * progress))

    print(f"\n  Fine-tuning for {args.iters} iterations...")
    t0 = time.time()
    running_loss = 0

    for step in range(args.iters):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = dataset.get_batch(args.batch_size, args.device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % args.log_interval == 0:
            avg = running_loss / args.log_interval
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta = (args.iters - step - 1) / rate
            print(f"  step {step+1:4d}/{args.iters} | loss {avg:.4f} | lr {lr:.2e} | {rate:.1f} it/s | eta {eta:.0f}s")
            running_loss = 0

    elapsed = time.time() - t0
    print(f"\n  Training done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Post-finetune evaluation
    print("\n  Post-finetune loss:")
    model.eval()
    post_loss = evaluate(model, dataset, args.device)
    print(f"  Mixed loss: {post_loss:.4f}")
    print(f"  Improvement: {pre_loss:.4f} -> {post_loss:.4f} ({(1-post_loss/pre_loss)*100:.1f}% reduction)")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "heritage": heritage,
        "finetune_info": {
            "source_checkpoint": args.checkpoint,
            "iters": args.iters,
            "lr": args.lr,
            "pre_loss": round(pre_loss, 4),
            "post_loss": round(post_loss, 4),
        },
    }
    torch.save(save_dict, out)
    print(f"\n  Saved: {out}")
    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
