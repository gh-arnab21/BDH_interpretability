export interface NewPrecomputedData {
  concepts: string[];
  analysis_layer: number;
  total_active: number;
  total_neurons: number;
  top_200: NewNeuronEntry[];
  fingerprints: Record<string, NewFingerprint>;
}

export interface NewNeuronEntry {
  head: number;
  neuron: number;
  global: number;
  concept: string;
  selectivity: number;
  activations: Record<string, number>;
}

export interface NewFingerprint {
  count: number;
  top: NewNeuronEntry[];
  mean_sel: number;
}

export interface OldPrecomputedData {
  model_info: { n_layers: number; n_heads: number; n_neurons: number };
  best_layer: number;
  concepts: Record<string, OldFingerprintResult>;
  cross_concept: OldCrossConceptEntry[];
  selectivity?: {
    histogram: { bin_start: number; bin_end: number; count: number }[];
    total_neurons: number;
    total_selective: number;
    mean_selectivity: number;
  };
  synapse_tracking?: Record<string, OldConceptTracking>;
}

interface OldTopNeuron {
  idx: number;
  val: number;
  raw?: number;
}

interface OldHeadFingerprint {
  head: number;
  x_ds: number[];
  x_active: number;
  top_neurons: OldTopNeuron[];
}

interface OldLayerFingerprint {
  layer: number;
  heads: OldHeadFingerprint[];
}

interface OldWordFingerprint {
  word: string;
  layers: OldLayerFingerprint[];
}

interface OldSharedNeuron {
  layer: number;
  head: number;
  neuron: number;
  mean_activation: number;
  active_in: number;
  per_word: number[];
}

interface OldMonosemanticNeuron {
  layer: number;
  head: number;
  neuron: number;
  selectivity: number;
  mean_in: number;
  mean_out: number;
  p_value: number;
  per_word: number[];
}

export interface OldFingerprintResult {
  concept: string;
  words: OldWordFingerprint[];
  similarity: Record<string, number[][]>;
  shared_neurons: OldSharedNeuron[];
  monosemantic_neurons?: OldMonosemanticNeuron[];
  model_info: { n_layers: number; n_heads: number; n_neurons: number };
}

interface OldCrossConceptEntry {
  primary: string;
  secondary: string;
  distinctness_per_layer: number[];
  secondary_result: OldFingerprintResult;
}

interface OldConceptTracking {
  synapses: {
    id: string;
    label: string;
    layer: number;
    head: number;
    i: number;
    j: number;
    selectivity: number;
  }[];
  sentences: {
    sentence: string;
    n_bytes: number;
    words: {
      word: string;
      byte_start: number;
      byte_end: number;
      is_concept: boolean;
      sigma: Record<string, number>;
      delta_sigma: Record<string, number>;
    }[];
  }[];
}

/**
 * Detect whether the loaded JSON is in the new or old format
 */
export function isNewMonoFormat(data: unknown): data is NewPrecomputedData {
  if (!data || typeof data !== "object") return false;
  const d = data as Record<string, unknown>;
  return Array.isArray(d.concepts) && "analysis_layer" in d && "top_200" in d;
}

/**
 * Adapt the new precomputed monosemanticity data to the old format
 * expected by MonosemanticityPage and FindingsPage
 */
export function adaptMonoData(raw: unknown): OldPrecomputedData {
  // If already in old format, return as-is
  if (!isNewMonoFormat(raw)) {
    return raw as OldPrecomputedData;
  }

  const data = raw as NewPrecomputedData;
  const N_LAYERS = 6;
  const N_HEADS = 4;
  const N_NEURONS = data.total_neurons / N_HEADS; // 3072

  const modelInfo = {
    n_layers: N_LAYERS,
    n_heads: N_HEADS,
    n_neurons: N_NEURONS,
  };

  // Build concepts map
  const concepts: Record<string, OldFingerprintResult> = {};
  const conceptNames = data.concepts;

  for (const conceptName of conceptNames) {
    const fp = data.fingerprints[conceptName];
    if (!fp) continue;

    // Create synthetic word fingerprints from the top neurons
    // Group neurons by head
    const neuronsByHead: Record<number, NewNeuronEntry[]> = {};
    for (let h = 0; h < N_HEADS; h++) neuronsByHead[h] = [];
    for (const n of fp.top) {
      if (neuronsByHead[n.head]) {
        neuronsByHead[n.head].push(n);
      }
    }

    // Create a single synthetic "word" fingerprint from the concept-level data
    const wordFingerprint: OldWordFingerprint = {
      word: conceptName,
      layers: [
        {
          layer: data.analysis_layer,
          heads: Array.from({ length: N_HEADS }, (_, h) => {
            const headNeurons = neuronsByHead[h] || [];
            // Build a sparse activation vector
            const xDs = new Array(Math.min(N_NEURONS, 100)).fill(0);
            for (const n of headNeurons) {
              if (n.neuron < xDs.length) {
                xDs[n.neuron] =
                  n.activations[conceptName] || n.selectivity || 0;
              }
            }
            return {
              head: h,
              x_ds: xDs,
              x_active: headNeurons.length,
              top_neurons: headNeurons.map((n) => ({
                idx: n.neuron,
                val: n.selectivity,
                raw: n.activations[conceptName] || 0,
              })),
            };
          }),
        },
      ],
    };

    // Shared neurons: neurons that appear in this concept's top list
    const sharedNeurons: OldSharedNeuron[] = fp.top.map((n) => ({
      layer: data.analysis_layer,
      head: n.head,
      neuron: n.neuron,
      mean_activation: n.activations[conceptName] || 0,
      active_in: 1,
      per_word: [n.activations[conceptName] || 0],
    }));

    // Monosemantic neurons from selectivity data
    const monosemantic: OldMonosemanticNeuron[] = fp.top
      .filter((n) => n.selectivity > 0.5)
      .map((n) => ({
        layer: data.analysis_layer,
        head: n.head,
        neuron: n.neuron,
        selectivity: n.selectivity,
        mean_in: n.activations[conceptName] || 0,
        mean_out: 0,
        p_value: 0,
        per_word: [n.activations[conceptName] || 0],
      }));

    // Build similarity matrix (self-similarity = 1)
    const similarity: Record<string, number[][]> = {};
    similarity[String(data.analysis_layer)] = [[1.0]];

    concepts[conceptName] = {
      concept: conceptName,
      words: [wordFingerprint],
      similarity,
      shared_neurons: sharedNeurons,
      monosemantic_neurons: monosemantic,
      model_info: modelInfo,
    };
  }

  // Build cross-concept entries from top_200 data
  const crossConcept: OldCrossConceptEntry[] = [];
  for (let i = 0; i < conceptNames.length; i++) {
    for (let j = i + 1; j < conceptNames.length; j++) {
      const c1 = conceptNames[i];
      const c2 = conceptNames[j];

      // Compute distinctness from neuron overlap
      const neurons1 = new Set(
        data.top_200
          .filter((n) => n.concept === c1)
          .map((n) => `${n.head}_${n.neuron}`),
      );
      const neurons2 = new Set(
        data.top_200
          .filter((n) => n.concept === c2)
          .map((n) => `${n.head}_${n.neuron}`),
      );
      const intersection = [...neurons1].filter((n) => neurons2.has(n)).length;
      const union = new Set([...neurons1, ...neurons2]).size;
      const jaccard = union > 0 ? intersection / union : 0;
      const distinctness = 1 - jaccard;

      // Create per-layer distinctness (same value for the analysis layer, 0 elsewhere)
      const distinctnessPerLayer = new Array(N_LAYERS).fill(0);
      distinctnessPerLayer[data.analysis_layer] = distinctness;

      crossConcept.push({
        primary: c1,
        secondary: c2,
        distinctness_per_layer: distinctnessPerLayer,
        secondary_result: concepts[c2]!,
      });
    }
  }

  // Build selectivity histogram from top_200
  const selectivities = data.top_200.map((n) => n.selectivity);
  const bins = 10;
  const histogram: { bin_start: number; bin_end: number; count: number }[] = [];
  for (let b = 0; b < bins; b++) {
    const binStart = b / bins;
    const binEnd = (b + 1) / bins;
    const count = selectivities.filter(
      (s) => s >= binStart && s < binEnd,
    ).length;
    histogram.push({ bin_start: binStart, bin_end: binEnd, count });
  }

  const totalSelective = selectivities.filter((s) => s > 0.5).length;
  const meanSelectivity =
    selectivities.length > 0
      ? selectivities.reduce((a, b) => a + b, 0) / selectivities.length
      : 0;

  return {
    model_info: modelInfo,
    best_layer: data.analysis_layer,
    concepts,
    cross_concept: crossConcept,
    selectivity: {
      histogram,
      total_neurons: data.total_neurons,
      total_selective: totalSelective,
      mean_selectivity: meanSelectivity,
    },
    synapse_tracking: undefined,
  };
}

interface NewMergeData {
  evaluation: {
    merged: {
      fr_data: number;
      pt_data: number;
    };
  };
  heritage: {
    model_a: string;
    model_b: string;
    neurons_a: number;
    neurons_b: number;
    total: number;
  };
  samples: { prompt: string; output: string }[];
}

export interface OldMergeData {
  heritage: {
    model1_name: string;
    model2_name: string;
    neurons_per_head_original: number;
    neurons_per_head_merged: number;
    total_neurons_per_model: number;
    total_neurons_merged: number;
    ranges: {
      french: { start: number; end: number };
      portuguese: { start: number; end: number };
    };
  };
  models: Record<
    string,
    {
      name: string;
      flag: string;
      params: number;
      n_neurons: number;
      n_heads: number;
      n_layers: number;
      n_embd: number;
    }
  >;
  evaluation: Record<
    string,
    {
      french_loss: number | null;
      portuguese_loss: number | null;
    }
  >;
  samples: {
    label: string;
    prompt: string;
    french_generated?: string;
    portuguese_generated?: string;
    merged_generated?: string;
    finetuned_generated?: string;
  }[];
  heritage_probe?: Record<string, unknown>;
  finetune_info?: Record<string, unknown>;
}

export function isNewMergeFormat(data: unknown): data is NewMergeData {
  if (!data || typeof data !== "object") return false;
  const d = data as Record<string, unknown>;
  return (
    "heritage" in d &&
    typeof d.heritage === "object" &&
    d.heritage !== null &&
    "model_a" in (d.heritage as Record<string, unknown>)
  );
}

export function adaptMergeData(raw: unknown): OldMergeData {
  if (!isNewMergeFormat(raw)) {
    return raw as OldMergeData;
  }

  const data = raw as NewMergeData;
  const neuronsPerHead = data.heritage.neurons_a / 4; // 4 heads
  const mergedNeuronsPerHead = data.heritage.total / 4;

  return {
    heritage: {
      model1_name: "french",
      model2_name: "portuguese",
      neurons_per_head_original: neuronsPerHead,
      neurons_per_head_merged: mergedNeuronsPerHead,
      total_neurons_per_model: data.heritage.neurons_a,
      total_neurons_merged: data.heritage.total,
      ranges: {
        french: { start: 0, end: neuronsPerHead - 1 },
        portuguese: {
          start: neuronsPerHead,
          end: mergedNeuronsPerHead - 1,
        },
      },
    },
    models: {
      french: {
        name: "French",
        flag: "\u{1F1EB}\u{1F1F7}",
        params: 7962624,
        n_neurons: 3072,
        n_heads: 4,
        n_layers: 6,
        n_embd: 192,
      },
      portuguese: {
        name: "Portuguese",
        flag: "\u{1F1F5}\u{1F1F9}",
        params: 7962624,
        n_neurons: 3072,
        n_heads: 4,
        n_layers: 6,
        n_embd: 192,
      },
      merged: {
        name: "Merged (zero-shot)",
        flag: "\u{1F504}",
        params: 7962624 * 2,
        n_neurons: 6144,
        n_heads: 4,
        n_layers: 6,
        n_embd: 192,
      },
    },
    evaluation: {
      french: {
        french_loss: null,
        portuguese_loss: null,
      },
      portuguese: {
        french_loss: null,
        portuguese_loss: null,
      },
      merged: {
        french_loss: data.evaluation.merged.fr_data,
        portuguese_loss: data.evaluation.merged.pt_data,
      },
    },
    samples: data.samples.map((s) => ({
      label: s.prompt.includes("<T:fr>")
        ? "French prompt"
        : "Portuguese prompt",
      prompt: s.prompt,
      merged_generated: s.output,
    })),
    heritage_probe: undefined,
    finetune_info: undefined,
  };
}
