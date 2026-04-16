#!/usr/bin/env python3
"""Monosemanticity analysis — discovers concept-specific synapses in BDH models."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import random

import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from bdh import BDH, BDHConfig, ExtractionConfig, load_model


CONCEPT_CATEGORIES = {
    "currencies": {
        "description": "Monetary units and currency names",
        "examples": [
            "dollar", "euro", "pound", "yen", "franc", "peso", "zloty",
            "crown", "rupee", "yuan", "ruble", "krona", "dinar", "lira",
            "US dollars", "British pounds", "Japanese yen", "Swiss francs",
            "EUR", "USD", "GBP", "JPY", "CHF",
        ],
        "contexts": [
            "The price is 100 {word}.",
            "They paid in {word}.",
            "The {word} exchange rate.",
            "Converting to {word}.",
        ]
    },
    "countries": {
        "description": "Nation names, especially EU members (from Europarl)",
        "examples": [
            "France", "Germany", "Spain", "Italy", "Portugal", "Poland",
            "Netherlands", "Belgium", "Austria", "Greece", "Sweden",
            "Denmark", "Finland", "Ireland", "Luxembourg", "Czech",
            "Hungary", "Romania", "Bulgaria", "Croatia", "Slovakia",
            "Slovenia", "Estonia", "Latvia", "Lithuania", "Cyprus", "Malta",
            "United Kingdom", "Britain", "European Union",
        ],
        "contexts": [
            "The government of {word}.",
            "Citizens of {word}.",
            "In {word}, the law states.",
            "The delegation from {word}.",
        ]
    },
    "languages": {
        "description": "Language names",
        "examples": [
            "English", "French", "German", "Spanish", "Portuguese",
            "Italian", "Polish", "Dutch", "Greek", "Swedish",
            "Danish", "Finnish", "Czech", "Hungarian", "Romanian",
        ],
        "contexts": [
            "Translated into {word}.",
            "The {word} version.",
            "Speaking in {word}.",
            "The {word} language.",
        ]
    },
    "numbers": {
        "description": "Numeric words",
        "examples": [
            "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "hundred", "thousand", "million",
            "billion", "first", "second", "third", "fourth", "fifth",
            "dozen", "half", "quarter", "twice", "triple",
        ],
        "contexts": [
            "There are {word} members.",
            "Article {word} states.",
            "The {word} amendment.",
            "Approximately {word} people.",
        ]
    },
    "legal_terms": {
        "description": "Legal and parliamentary terminology",
        "examples": [
            "amendment", "regulation", "directive", "treaty", "article",
            "paragraph", "clause", "legislation", "statute", "provision",
            "resolution", "motion", "vote", "majority", "unanimous",
            "ratification", "implementation", "enforcement", "compliance",
        ],
        "contexts": [
            "The {word} was adopted.",
            "According to the {word}.",
            "This {word} requires.",
            "The proposed {word}.",
        ]
    },
    "temporal": {
        "description": "Time-related words",
        "examples": [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "yesterday", "today", "tomorrow", "year", "month", "week",
        ],
        "contexts": [
            "On {word}, the meeting.",
            "During {word}, we expect.",
            "Since {word}, the policy.",
            "By {word}, the deadline.",
        ]
    },
}

# Negative examples (should NOT activate concept synapses)
NEGATIVE_EXAMPLES = [
    "apple", "running", "beautiful", "quickly", "mountain",
    "chair", "water", "music", "happy", "green", "walking",
    "book", "computer", "weather", "smile", "dance", "river",
]


@dataclass
class SynapseActivation:
    """Record of a synapse's activation for a specific input."""
    layer: int
    head: int
    neuron_x: int  # Pre-gating neuron
    neuron_y: int  # Post-attention neuron
    activation_x: float
    activation_y: float
    gated_activation: float  # x * y
    input_text: str
    concept_category: Optional[str] = None


@dataclass
class MonosemanticSynapse:
    """A discovered monosemantic synapse."""
    layer: int
    head: int
    neuron_idx: int
    activation_type: str  # "x_sparse" or "y_sparse"
    concept: str
    selectivity_score: float  # Higher = more selective
    mean_activation_in_concept: float
    mean_activation_out_concept: float
    std_in_concept: float
    consistency_score: float  # How consistent across examples
    example_activations: List[Tuple[str, float]] = field(default_factory=list)


@dataclass 
class ProbeResults:
    """Results from concept probing."""
    concept: str
    num_examples: int
    activations: Dict[str, np.ndarray]  # layer_head_type -> (num_examples, N)
    mean_activations: Dict[str, np.ndarray]  # layer_head_type -> (N,)
    std_activations: Dict[str, np.ndarray]  # layer_head_type -> (N,)


class MonosemanticityAnalyzer:
    """
    Analyzer for discovering monosemantic synapses in BDH models.
    """
    
    def __init__(
        self,
        model: BDH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.config = model.config
        self.n_layers = self.config.n_layer
        self.n_heads = self.config.n_head
        self.n_neurons = self.config.n_neurons
        
        # Storage for probe results
        self.probe_results: Dict[str, ProbeResults] = {}
        self.discovered_synapses: List[MonosemanticSynapse] = []
    
    def _text_to_tokens(self, text: str) -> torch.Tensor:
        """Convert text to token tensor."""
        tokens = torch.tensor(
            [list(text.encode('utf-8'))],
            dtype=torch.long,
            device=self.device
        )
        return tokens
    
    def _extract_activations(self, text: str) -> Dict[str, torch.Tensor]:
        """Run text through model and extract sparse activations."""
        tokens = self._text_to_tokens(text)
        
        extraction_config = ExtractionConfig(
            capture_sparse_activations=True,
            capture_attention_patterns=False,
            capture_pre_relu=False,
            capture_layer_outputs=False,
        )
        
        with torch.no_grad():
            with self.model.extraction_mode(extraction_config) as buffer:
                _, _ = self.model(tokens)
                
                # Extract activations for each layer
                activations = {}
                for layer_idx in buffer.x_sparse.keys():
                    # x_sparse shape: (1, nh, T, N)
                    x = buffer.x_sparse[layer_idx][0]  # (nh, T, N)
                    y = buffer.y_sparse[layer_idx][0]  # (nh, T, N)
                    
                    # Average across tokens to get neuron activation profile
                    x_mean = x.mean(dim=1)  # (nh, N)
                    y_mean = y.mean(dim=1)  # (nh, N)
                    
                    for head in range(self.n_heads):
                        activations[f"L{layer_idx}_H{head}_x"] = x_mean[head].cpu().numpy()
                        activations[f"L{layer_idx}_H{head}_y"] = y_mean[head].cpu().numpy()
        
        return activations
    
    def probe_concept(
        self,
        concept_name: str,
        examples: List[str],
        contexts: Optional[List[str]] = None,
        use_contexts: bool = True
    ) -> ProbeResults:
        """
        Probe model with concept examples and record activations.
        
        Args:
            concept_name: Name of the concept (e.g., "currencies")
            examples: List of example words/phrases
            contexts: Optional context templates with {word} placeholder
            use_contexts: Whether to embed examples in contexts
        """
        print(f"\n🔍 Probing concept: {concept_name}")
        
        # Generate probe inputs
        probe_inputs = []
        if use_contexts and contexts:
            for example in examples:
                for ctx in contexts:
                    probe_inputs.append(ctx.format(word=example))
        else:
            probe_inputs = examples
        
        print(f"   {len(probe_inputs)} probe inputs")
        
        # Collect activations
        all_activations = defaultdict(list)
        
        for text in tqdm(probe_inputs, desc=f"   Probing {concept_name}"):
            try:
                acts = self._extract_activations(text)
                for key, arr in acts.items():
                    all_activations[key].append(arr)
            except Exception as e:
                print(f"   Warning: Failed on '{text[:50]}': {e}")
                continue
        
        # Stack into arrays
        activations = {k: np.stack(v) for k, v in all_activations.items()}
        mean_activations = {k: v.mean(axis=0) for k, v in activations.items()}
        std_activations = {k: v.std(axis=0) for k, v in activations.items()}
        
        results = ProbeResults(
            concept=concept_name,
            num_examples=len(probe_inputs),
            activations=activations,
            mean_activations=mean_activations,
            std_activations=std_activations,
        )
        
        self.probe_results[concept_name] = results
        return results
    
    def probe_all_concepts(self) -> Dict[str, ProbeResults]:
        """Probe all predefined concept categories."""
        for concept_name, concept_info in CONCEPT_CATEGORIES.items():
            self.probe_concept(
                concept_name=concept_name,
                examples=concept_info["examples"],
                contexts=concept_info.get("contexts"),
                use_contexts=True
            )
        
        # Also probe negative examples
        self.probe_concept(
            concept_name="_negative",
            examples=NEGATIVE_EXAMPLES,
            contexts=None,
            use_contexts=False
        )
        
        return self.probe_results
    
    def compute_selectivity(
        self,
        target_concept: str,
        other_concepts: Optional[List[str]] = None,
        top_k: int = 50
    ) -> List[MonosemanticSynapse]:
        """
        Compute selectivity scores for each neuron.
        
        A neuron is selective for a concept if:
        1. High mean activation for that concept
        2. Low mean activation for other concepts
        3. Low variance within the concept (consistent)
        
        Selectivity = (mean_in - mean_out) / (std_in + epsilon)
        """
        if target_concept not in self.probe_results:
            raise ValueError(f"Concept '{target_concept}' not probed yet")
        
        if other_concepts is None:
            other_concepts = [c for c in self.probe_results.keys() 
                           if c != target_concept and c != "_negative"]
            other_concepts.append("_negative")
        
        target_results = self.probe_results[target_concept]
        
        discovered = []
        
        for key in target_results.mean_activations.keys():
            # Parse key: "L{layer}_H{head}_{x/y}"
            parts = key.split("_")
            layer = int(parts[0][1:])
            head = int(parts[1][1:])
            act_type = parts[2]
            
            mean_in = target_results.mean_activations[key]
            std_in = target_results.std_activations[key]
            
            # Compute mean activation for other concepts
            other_means = []
            for other_concept in other_concepts:
                if other_concept in self.probe_results:
                    other_results = self.probe_results[other_concept]
                    if key in other_results.mean_activations:
                        other_means.append(other_results.mean_activations[key])
            
            if not other_means:
                continue
            
            mean_out = np.mean(other_means, axis=0)
            
            # Compute selectivity for each neuron
            epsilon = 1e-6
            selectivity = (mean_in - mean_out) / (std_in + epsilon)
            
            # Also compute consistency (inverse of coefficient of variation)
            consistency = mean_in / (std_in + epsilon)
            
            # Find top-k most selective neurons
            top_indices = np.argsort(selectivity)[-top_k:][::-1]
            
            for idx in top_indices:
                if selectivity[idx] > 1.0:  # Threshold for "selective"
                    synapse = MonosemanticSynapse(
                        layer=layer,
                        head=head,
                        neuron_idx=int(idx),
                        activation_type=f"{act_type}_sparse",
                        concept=target_concept,
                        selectivity_score=float(selectivity[idx]),
                        mean_activation_in_concept=float(mean_in[idx]),
                        mean_activation_out_concept=float(mean_out[idx]),
                        std_in_concept=float(std_in[idx]),
                        consistency_score=float(consistency[idx]),
                    )
                    discovered.append(synapse)
        
        # Sort by selectivity
        discovered.sort(key=lambda s: s.selectivity_score, reverse=True)
        
        return discovered[:top_k]
    
    def find_all_monosemantic_synapses(self, top_k_per_concept: int = 20) -> List[MonosemanticSynapse]:
        """Find monosemantic synapses for all concepts."""
        print("\n🧠 Finding monosemantic synapses...")
        
        all_synapses = []
        
        for concept_name in CONCEPT_CATEGORIES.keys():
            print(f"\n   Analyzing: {concept_name}")
            synapses = self.compute_selectivity(concept_name, top_k=top_k_per_concept)
            all_synapses.extend(synapses)
            
            if synapses:
                top = synapses[0]
                print(f"   Top synapse: L{top.layer}_H{top.head}_N{top.neuron_idx}")
                print(f"   Selectivity: {top.selectivity_score:.2f}")
        
        self.discovered_synapses = all_synapses
        return all_synapses
    
    def cross_validate_synapse(
        self,
        synapse: MonosemanticSynapse,
        held_out_examples: List[str]
    ) -> Dict[str, float]:
        """
        Validate a discovered synapse on held-out examples.
        
        Returns:
            Dict with validation metrics
        """
        # Probe held-out examples
        activations = []
        for text in held_out_examples:
            acts = self._extract_activations(text)
            key = f"L{synapse.layer}_H{synapse.head}_{synapse.activation_type[0]}"
            if key in acts:
                activations.append(acts[key][synapse.neuron_idx])
        
        if not activations:
            return {"valid": False, "reason": "No activations extracted"}
        
        activations = np.array(activations)
        
        return {
            "valid": True,
            "mean_activation": float(activations.mean()),
            "std_activation": float(activations.std()),
            "activation_rate": float((activations > 0).mean()),
            "consistent_with_training": activations.mean() > synapse.mean_activation_in_concept * 0.5
        }
    
    def export_for_frontend(self, output_path: Path) -> Dict[str, Any]:
        """
        Export analysis results for frontend visualization.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export discovered synapses
        synapses_data = []
        for s in self.discovered_synapses:
            synapses_data.append({
                "layer": s.layer,
                "head": s.head,
                "neuron_idx": s.neuron_idx,
                "activation_type": s.activation_type,
                "concept": s.concept,
                "selectivity_score": s.selectivity_score,
                "mean_in": s.mean_activation_in_concept,
                "mean_out": s.mean_activation_out_concept,
                "consistency": s.consistency_score,
            })
        
        with open(output_path / "monosemantic_synapses.json", "w") as f:
            json.dump(synapses_data, f, indent=2)
        
        # Export concept activation profiles
        profiles = {}
        for concept_name, results in self.probe_results.items():
            if concept_name.startswith("_"):
                continue
            profiles[concept_name] = {
                "num_examples": results.num_examples,
                "mean_activations": {
                    k: v.tolist() for k, v in results.mean_activations.items()
                },
            }
        
        with open(output_path / "concept_profiles.json", "w") as f:
            json.dump(profiles, f, indent=2)
        
        # Export summary statistics
        summary = {
            "total_synapses_found": len(self.discovered_synapses),
            "concepts_analyzed": list(CONCEPT_CATEGORIES.keys()),
            "synapses_per_concept": {},
            "model_config": {
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "n_neurons": self.n_neurons,
            }
        }
        
        for concept in CONCEPT_CATEGORIES.keys():
            count = sum(1 for s in self.discovered_synapses if s.concept == concept)
            summary["synapses_per_concept"][concept] = count
        
        with open(output_path / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Exported results to {output_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Monosemanticity analysis for BDH")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top K synapses per concept")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 BDH Monosemanticity Analysis")
    print("=" * 60)
    
    # Load model
    print(f"\n📂 Loading model from {args.model}")
    model = load_model(args.model, args.device)
    print(f"   Config: {model.config.n_layer}L, {model.config.n_embd}D, {model.config.n_head}H")
    print(f"   Neurons per head: {model.config.n_neurons}")
    
    # Create analyzer
    analyzer = MonosemanticityAnalyzer(model, args.device)
    
    # Probe all concepts
    print("\n" + "=" * 60)
    print("Phase 1: Concept Probing")
    print("=" * 60)
    analyzer.probe_all_concepts()
    
    # Find monosemantic synapses
    print("\n" + "=" * 60)
    print("Phase 2: Synapse Discovery")
    print("=" * 60)
    synapses = analyzer.find_all_monosemantic_synapses(top_k_per_concept=args.top_k)
    
    # Export results
    print("\n" + "=" * 60)
    print("Phase 3: Export Results")
    print("=" * 60)
    summary = analyzer.export_for_frontend(args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Total monosemantic synapses found: {summary['total_synapses_found']}")
    print("\nPer concept:")
    for concept, count in summary["synapses_per_concept"].items():
        print(f"  {concept}: {count}")
    
    if synapses:
        print("\n🏆 Top 5 most selective synapses:")
        for i, s in enumerate(synapses[:5]):
            print(f"  {i+1}. L{s.layer}_H{s.head}_N{s.neuron_idx} -> {s.concept}")
            print(f"     Selectivity: {s.selectivity_score:.2f}, Consistency: {s.consistency_score:.2f}")


if __name__ == "__main__":
    main()
