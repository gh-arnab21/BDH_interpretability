#!/usr/bin/env python3
"""
BDH Model Merging + Evaluation

Merges two BDH specialists, evaluates all models (including fine-tuned if available),
and produces frontend JSON for the merge page.

Usage (basic merge):
    python analysis/merge.py \
        --model1 checkpoints/french/french_best.pt \
        --model2 checkpoints/portuguese/portuguese_best.pt \
        --output checkpoints/merged/merged_merged.pt

Usage (with fine-tuned comparison):
    python analysis/merge.py \
        --model1 checkpoints/french/french_best.pt \
        --model2 checkpoints/portuguese/portuguese_best.pt \
        --output checkpoints/merged/merged_merged.pt \
        --finetuned checkpoints/merged/merged_finetuned.pt
"""

from bdh import BDH, BDHConfig, load_model
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))


@dataclass
class MergeConfig:
    model1_path: str = ""
    model2_path: str = ""
    output_path: str = ""
    model1_name: str = "french"
    model2_name: str = "portuguese"
    merge_embeddings: str = "average"
    merge_lm_head: str = "average"


# ═══════════════════════════════════════════════════════════════════════
#  Built-in test data
# ═══════════════════════════════════════════════════════════════════════
FRENCH_TEST = [
    "<F:en>The European Parliament has adopted the resolution on trade policy.<T:fr>Le Parlement européen a adopté la résolution sur la politique commerciale.",
    "<F:en>We must ensure that all citizens have access to healthcare.<T:fr>Nous devons veiller à ce que tous les citoyens aient accès aux soins de santé.",
    "<F:en>The Commission presented its annual report on economic growth.<T:fr>La Commission a présenté son rapport annuel sur la croissance économique.",
    "<F:en>Mr President, I would like to raise a point of order.<T:fr>Monsieur le Président, je voudrais soulever une motion de procédure.",
    "<F:en>This directive aims to protect the environment and public health.<T:fr>Cette directive vise à protéger l'environnement et la santé publique.",
    "<F:en>The Council has reached an agreement on the new regulation.<T:fr>Le Conseil est parvenu à un accord sur le nouveau règlement.",
    "<F:en>I believe this proposal will benefit all member states.<T:fr>Je crois que cette proposition profitera à tous les États membres.",
    "<F:en>The rapporteur has done excellent work on this report.<T:fr>Le rapporteur a fait un excellent travail sur ce rapport.",
    "<F:en>We need to strengthen cooperation between our institutions.<T:fr>Nous devons renforcer la coopération entre nos institutions.",
    "<F:en>The budget for next year must be approved by December.<T:fr>Le budget pour l'année prochaine doit être approuvé avant décembre.",
    "<F:en>Freedom of expression is a fundamental right in Europe.<T:fr>La liberté d'expression est un droit fondamental en Europe.",
    "<F:en>The single market has been a great success for our continent.<T:fr>Le marché unique a été un grand succès pour notre continent.",
]

PORTUGUESE_TEST = [
    "<F:en>The European Parliament has adopted the resolution on trade policy.<T:pt>O Parlamento Europeu adoptou a resolução sobre a política comercial.",
    "<F:en>We must ensure that all citizens have access to healthcare.<T:pt>Devemos assegurar que todos os cidadãos tenham acesso aos cuidados de saúde.",
    "<F:en>The Commission presented its annual report on economic growth.<T:pt>A Comissão apresentou o seu relatório anual sobre o crescimento económico.",
    "<F:en>Mr President, I would like to raise a point of order.<T:pt>Senhor Presidente, gostaria de levantar uma questão de ordem.",
    "<F:en>This directive aims to protect the environment and public health.<T:pt>Esta directiva tem por objectivo proteger o ambiente e a saúde pública.",
    "<F:en>The Council has reached an agreement on the new regulation.<T:pt>O Conselho chegou a acordo sobre o novo regulamento.",
    "<F:en>I believe this proposal will benefit all member states.<T:pt>Creio que esta proposta beneficiará todos os Estados-Membros.",
    "<F:en>The rapporteur has done excellent work on this report.<T:pt>O relator realizou um excelente trabalho sobre este relatório.",
    "<F:en>We need to strengthen cooperation between our institutions.<T:pt>Precisamos de reforçar a cooperação entre as nossas instituições.",
    "<F:en>The budget for next year must be approved by December.<T:pt>O orçamento para o próximo ano deve ser aprovado até Dezembro.",
    "<F:en>Freedom of expression is a fundamental right in Europe.<T:pt>A liberdade de expressão é um direito fundamental na Europa.",
    "<F:en>The single market has been a great success for our continent.<T:pt>O mercado único tem sido um grande êxito para o nosso continente.",
]

# ═══════════════════════════════════════════════════════════════════════
#  Load / Detect / Verify
# ═══════════════════════════════════════════════════════════════════════


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in ckpt:
        import dataclasses
        valid = {f.name for f in dataclasses.fields(BDHConfig)}
        cfg = {k: v for k, v in ckpt["config"].items() if k in valid}
        config = BDHConfig(**cfg)
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
        if "encoder" in sd:
            e = sd["encoder"]
            config = BDHConfig(n_layer=6, n_embd=e.shape[1], n_head=e.shape[0],
                               mlp_internal_dim_multiplier=(e.shape[2]*e.shape[0]//e.shape[1]))
        else:
            raise ValueError("Cannot infer config")
    clean = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return clean, config


def verify_compatible(c1, c2):
    ok = True
    for n, a, b in [("n_layer", c1.n_layer, c2.n_layer), ("n_embd", c1.n_embd, c2.n_embd),
                    ("n_head", c1.n_head, c2.n_head), ("mlp", c1.mlp_internal_dim_multiplier, c2.mlp_internal_dim_multiplier)]:
        if a != b:
            print(f"  MISMATCH {n}: {a} vs {b}")
            ok = False
        else:
            print(f"  ok {n}={a}")
    return ok

# ═══════════════════════════════════════════════════════════════════════
#  Merge
# ═══════════════════════════════════════════════════════════════════════


def merge_models(s1, s2, c1, mc):
    m = {}
    nh, D, N = c1.n_head, c1.n_embd, c1.n_neurons
    if "encoder" in s1:  # V1
        m["encoder"] = torch.cat([s1["encoder"], s2["encoder"]], dim=2)
        m["encoder_v"] = torch.cat([s1["encoder_v"], s2["encoder_v"]], dim=2)
        m["decoder"] = torch.cat([s1["decoder"], s2["decoder"]], dim=0)
        m["attn.freqs"] = torch.cat(
            [s1["attn.freqs"], s2["attn.freqs"]], dim=3)
    else:  # V2
        for l in range(c1.n_layer):
            m[f"encoders.{l}"] = torch.cat(
                [s1[f"encoders.{l}"], s2[f"encoders.{l}"]], dim=2)
            if f"encoders_v.{l}" in s1:
                m[f"encoders_v.{l}"] = torch.cat(
                    [s1[f"encoders_v.{l}"], s2[f"encoders_v.{l}"]], dim=2)
            m[f"decoders.{l}"] = torch.cat(
                [s1[f"decoders.{l}"], s2[f"decoders.{l}"]], dim=0)
        if "rho" in s1:
            m["rho"] = torch.cat([s1["rho"], s2["rho"]], dim=2)
        if "attn.freqs" in s1:
            m["attn.freqs"] = torch.cat(
                [s1["attn.freqs"], s2["attn.freqs"]], dim=3)

    if "embed.weight" in s1:
        m["embed.weight"] = (s1["embed.weight"]+s2["embed.weight"])/2
    if "lm_head" in s1:
        m["lm_head"] = (s1["lm_head"]+s2["lm_head"])/2
    for k in s1:
        if k not in m:
            m[k] = ((s1[k]+s2[k])/2 if k in s2 and s1[k].shape ==
                    s2[k].shape else s1[k].clone())

    mc2 = BDHConfig(n_layer=c1.n_layer, n_embd=c1.n_embd, n_head=c1.n_head,
                    mlp_internal_dim_multiplier=c1.mlp_internal_dim_multiplier*2,
                    dropout=c1.dropout, vocab_size=c1.vocab_size)
    print(f"  Merged: {N}->{2*N} N/head, {nh*N}->{2*nh*N} total")
    return m, mc2


def create_heritage_map(c1, mc):
    N, nh = c1.n_neurons, c1.n_head
    return {"model1_name": mc.model1_name, "model2_name": mc.model2_name,
            "neurons_per_head_original": N, "neurons_per_head_merged": 2*N,
            "total_neurons_per_model": N*nh, "total_neurons_merged": 2*N*nh,
            "ranges": {mc.model1_name: {"start": 0, "end": N-1}, mc.model2_name: {"start": N, "end": 2*N-1}}}


def validate(ms, mc, dev):
    try:
        model = BDH(mc)
        model.load_state_dict(ms, strict=False)
        model.to(dev).eval()
        with torch.no_grad():
            lo, _ = model(torch.randint(0, 256, (1, 32), device=dev))
        assert lo.shape == (1, 32, 256)
        del model
        print("  Validation OK")
        return True
    except Exception as e:
        print(f"  Validation FAILED: {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def eval_on_bytes(model, data_bytes, dev, bs=256):
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    if len(data) < bs+1:
        return -1.0
    total, count = 0.0, 0
    n = min(40, len(data)//bs)
    for i in range(n):
        s = i*(len(data)//n)
        if s+bs+1 > len(data):
            break
        x = torch.from_numpy(
            data[s:s+bs].astype(np.int64)).unsqueeze(0).to(dev)
        y = torch.from_numpy(
            data[s+1:s+1+bs].astype(np.int64)).unsqueeze(0).to(dev)
        _, loss = model(x, y)
        if loss is not None:
            total += loss.item()
            count += 1
    return total/max(count, 1)


@torch.no_grad()
def eval_on_file(model, path, dev, bs=256, nb=50, bsz=8):
    if not Path(path).exists():
        return -1.0
    data = np.memmap(path, dtype=np.uint8, mode="r")
    total, count = 0.0, 0
    for _ in range(nb):
        ix = np.random.randint(0, len(data)-bs-1, size=bsz)
        x = torch.stack([torch.from_numpy(data[i:i+bs].astype(np.int64))
                        for i in ix]).to(dev)
        y = torch.stack(
            [torch.from_numpy(data[i+1:i+1+bs].astype(np.int64)) for i in ix]).to(dev)
        _, loss = model(x, y)
        if loss is not None:
            total += loss.item()
            count += 1
    return total/max(count, 1)


def eval_model(model, fv, pv, dev):
    fb = "\n".join(FRENCH_TEST).encode("utf-8")
    pb = "\n".join(PORTUGUESE_TEST).encode("utf-8")
    use_f = Path(fv).exists() if fv else False
    use_p = Path(pv).exists() if pv else False
    fr = eval_on_file(model, fv, dev) if use_f else eval_on_bytes(
        model, fb, dev)
    pt = eval_on_file(model, pv, dev) if use_p else eval_on_bytes(
        model, pb, dev)
    return {"french_loss": round(fr, 4) if fr >= 0 else None, "portuguese_loss": round(pt, 4) if pt >= 0 else None}


@torch.no_grad()
def gen(model, prompt, dev, n=80):
    t = torch.tensor([list(prompt.encode("utf-8"))],
                     dtype=torch.long, device=dev)
    o = model.generate(t, max_new_tokens=n, top_k=5, temperature=0.8)
    return bytes(o[0].cpu().tolist()).decode("utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--name1", default="french")
    p.add_argument("--name2", default="portuguese")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--french-val", default="data/en-fr/val.bin")
    p.add_argument("--portuguese-val", default="data/en-pt/val.bin")
    p.add_argument("--finetuned", default="",
                   help="Path to fine-tuned merged checkpoint")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-merge", action="store_true",
                   help="Skip merge, just evaluate existing models")
    p.add_argument("--frontend-json", default="")
    a = p.parse_args()

    mc = MergeConfig(model1_path=a.model1, model2_path=a.model2, output_path=a.output,
                     model1_name=a.name1, model2_name=a.name2)

    print("="*60)
    print("BDH Model Merger")
    print("="*60)

    # Load source models
    s1, c1 = load_checkpoint(a.model1)
    s2, c2 = load_checkpoint(a.model2)
    print(f"  M1: {c1.n_layer}L {c1.n_embd}D {c1.n_head}H N={c1.n_neurons}")
    print(f"  M2: {c2.n_layer}L {c2.n_embd}D {c2.n_head}H N={c2.n_neurons}")
    if not verify_compatible(c1, c2):
        return 1

    # Merge (or load existing)
    if not a.skip_merge:
        ms, mc2 = merge_models(s1, s2, c1, mc)
        heritage = create_heritage_map(c1, mc)
        if not validate(ms, mc2, a.device):
            return 1
        # Save
        op = Path(a.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": ms, "config": asdict(mc2), "heritage": heritage,
                    "merge_config": asdict(mc)}, op)
        print(f"  Saved: {op}")
    else:
        ckpt = torch.load(a.output, map_location="cpu", weights_only=False)
        import dataclasses
        valid = {f.name for f in dataclasses.fields(BDHConfig)}
        cfg = {k: v for k, v in ckpt["config"].items() if k in valid}
        mc2 = BDHConfig(**cfg)
        ms = ckpt["model_state_dict"]
        heritage = ckpt.get("heritage", create_heritage_map(c1, mc))

    # Evaluate
    ev = {}
    if not a.skip_eval:
        print("\n  Evaluating...")
        fv, pv = a.french_val, a.portuguese_val

        # Specialist 1
        m = load_model(a.model1, a.device)
        ev[a.name1] = eval_model(m, fv, pv, a.device)
        print(f"  {a.name1}: {ev[a.name1]}")
        del m

        # Specialist 2
        m = load_model(a.model2, a.device)
        ev[a.name2] = eval_model(m, fv, pv, a.device)
        print(f"  {a.name2}: {ev[a.name2]}")
        del m

        # Merged (zero-shot)
        mm = BDH(mc2)
        mm.load_state_dict(ms, strict=False)
        mm.to(a.device).eval()
        ev["merged"] = eval_model(mm, fv, pv, a.device)
        print(f"  merged: {ev['merged']}")
        del mm

        # Fine-tuned (if available)
        if a.finetuned and Path(a.finetuned).exists():
            ft = load_model(a.finetuned, a.device)
            ev["finetuned"] = eval_model(ft, fv, pv, a.device)
            print(f"  finetuned: {ev['finetuned']}")
            del ft

    # Generate samples from all models
    print("\n  Generating samples...")
    m1 = load_model(a.model1, a.device)
    m2 = load_model(a.model2, a.device)
    mm = BDH(mc2)
    mm.load_state_dict(ms, strict=False)
    mm.to(a.device).eval()
    ft_model = None
    if a.finetuned and Path(a.finetuned).exists():
        ft_model = load_model(a.finetuned, a.device)

    prompts = [("French prompt", "Le parlement européen"), ("Portuguese prompt", "O parlamento europeu"),
               ("English prompt", "The European Parliament"), ("Mixed context", "Bonjour, como está")]
    samples = []
    for label, pr in prompts:
        s = {"label": label, "prompt": pr,
             "french_generated": gen(m1, pr, a.device), "portuguese_generated": gen(m2, pr, a.device),
             "merged_generated": gen(mm, pr, a.device)}
        if ft_model:
            s["finetuned_generated"] = gen(ft_model, pr, a.device)
        s["generated"] = s.get("finetuned_generated", s["merged_generated"])
        samples.append(s)
        print(f"  {label}:")
        print(f"    {a.name1}: {s['french_generated'][:60]}...")
        print(f"    {a.name2}: {s['portuguese_generated'][:60]}...")
        print(f"    merged: {s['merged_generated'][:60]}...")
        if "finetuned_generated" in s:
            print(f"    finetuned: {s['finetuned_generated'][:60]}...")
    del m1, m2, mm
    if ft_model:
        del ft_model

    # Frontend JSON
    fjp = a.frontend_json or str(
        ROOT/"frontend"/"public"/"merge"/"merge_data.json")
    Path(fjp).parent.mkdir(parents=True, exist_ok=True)
    p1 = sum(v.numel() for v in s1.values())
    p2 = sum(v.numel() for v in s2.values())
    pm = sum(v.numel() for v in ms.values())

    models_data = {
        a.name1: {"name": a.name1.capitalize(), "flag": "\U0001f1eb\U0001f1f7", "params": p1,
                  "n_neurons": c1.n_neurons, "n_heads": c1.n_head, "n_layers": c1.n_layer, "n_embd": c1.n_embd},
        a.name2: {"name": a.name2.capitalize(), "flag": "\U0001f1f5\U0001f1f9", "params": p2,
                  "n_neurons": c2.n_neurons, "n_heads": c2.n_head, "n_layers": c2.n_layer, "n_embd": c2.n_embd},
        "merged": {"name": "Merged (zero-shot)", "flag": "\U0001f504", "params": pm,
                   "n_neurons": mc2.n_neurons, "n_heads": mc2.n_head, "n_layers": mc2.n_layer, "n_embd": mc2.n_embd},
    }
    if "finetuned" in ev:
        models_data["finetuned"] = {
            "name": "Merged (fine-tuned)", "flag": "\U0001f30d", "params": pm,
            "n_neurons": mc2.n_neurons, "n_heads": mc2.n_head, "n_layers": mc2.n_layer, "n_embd": mc2.n_embd}

    fd = {"heritage": heritage, "models": models_data,
          "evaluation": ev, "samples": samples}
    with open(fjp, "w", encoding="utf-8") as f:
        json.dump(fd, f, indent=2, ensure_ascii=False)
    print(f"\n  Frontend JSON: {fjp}")
    print("="*60+"\n  Done!\n"+"="*60)
    return 0


if __name__ == "__main__":
    exit(main())
