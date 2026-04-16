import { create } from "zustand";

interface TimelinePoint {
  iteration: number;
  checkpoint: string;
  file: string;
  sparsity: number;
  x_sparsity: number;
  y_sparsity: number;
  sparsity_by_layer: number[];
  graph_density: number;
  total_edges?: number;
  concepts_detected?: number;
}

interface EvolutionTimeline {
  timeline: TimelinePoint[];
  num_checkpoints: number;
  iterations: number[];
  metrics: string[];
}

interface KeyframeData {
  iteration: number;
  checkpoint: string;
  sparsity: {
    overall_sparsity: number;
    overall_x_sparsity: number;
    overall_y_sparsity: number;
    sparsity_by_layer: number[];
  };
  graph: {
    density: number;
    total_edges: number;
  };
  playback_preview?: {
    input_text: string;
    overall_sparsity: number;
    num_frames: number;
    frames: PlaybackFrame[];
  };
}

interface PlaybackFrame {
  token_idx: number;
  token_byte: number;
  token_char: string;
  layer: number;
  x_active: Array<{ indices: number[]; values: number[] }>;
  y_active: Array<{ indices: number[]; values: number[] }>;
  x_sparsity: number;
  y_sparsity: number;
  x_active_count?: number;
  y_active_count?: number;
  x_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  y_top_neurons?: Array<{ head: number; neuron: number; value: number }>;
  x_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
  };
  y_pre_relu?: {
    mean: number;
    std: number;
    max: number;
    min: number;
    positive_count: number;
    total: number;
  };
  gating?: {
    x_only: number;
    y_only: number;
    both: number;
    survival_rate: number;
  };
  attention?: number[];
  attention_stats?: {
    top_attended: Array<{ token_idx: number; char: string; weight: number }>;
  };
  embedding?: {
    byte_value: number;
    norm: number;
    mean: number;
    std: number;
  };
}

interface ConceptEvolution {
  concepts: {
    [key: string]: Array<{
      iteration: number;
      avg_activation: number;
      consistent_neurons: number;
    }>;
  };
  available_concepts: string[];
}

interface Manifest {
  model_name: string;
  num_checkpoints: number;
  checkpoint_files: string[];
  config: {
    n_layer?: number;
    n_embd?: number;
    n_head?: number;
    n_neurons?: number;
  };
  has_full_checkpoints?: boolean;
}

interface EvolutionState {
  currentModel: "french" | "portuguese" | "merged";
  manifest: Manifest | null;
  timeline: EvolutionTimeline | null;
  keyframes: KeyframeData[];
  conceptEvolution: ConceptEvolution | null;

  currentIteration: number;
  currentKeyframeIndex: number;
  isPlaying: boolean;
  playbackSpeed: number;

  isLoading: boolean;
  loadingProgress: number;
  error: string | null;

  loadedCheckpoints: { [key: string]: any };

  setModel: (model: "french" | "portuguese" | "merged") => Promise<void>;
  loadModelData: (model: string) => Promise<void>;
  setIteration: (iteration: number) => void;
  setKeyframeIndex: (index: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  nextKeyframe: () => void;
  prevKeyframe: () => void;
  setPlaybackSpeed: (speed: number) => void;
  loadFullCheckpoint: (checkpoint: string) => Promise<any>;

  getCurrentKeyframe: () => KeyframeData | null;
  getIterationData: (iteration: number) => TimelinePoint | null;
  getSparsityAtIteration: (iteration: number) => number;
  getClosestKeyframeIndex: (iteration: number) => number;
}

async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }
  return response.json();
}

export const useEvolutionStore = create<EvolutionState>((set, get) => ({
  // Initial state
  currentModel: "french",
  manifest: null,
  timeline: null,
  keyframes: [],
  conceptEvolution: null,
  currentIteration: 0,
  currentKeyframeIndex: 0,
  isPlaying: false,
  playbackSpeed: 500,
  isLoading: false,
  loadingProgress: 0,
  error: null,
  loadedCheckpoints: {},

  setModel: async (model) => {
    set({ currentModel: model });
    await get().loadModelData(model);
  },

  loadModelData: async (model) => {
    set({ isLoading: true, loadingProgress: 0, error: null });

    const basePath = `/playback/${model}`;

    try {
      // Load manifest first
      set({ loadingProgress: 10 });
      const manifest = await fetchJSON<Manifest>(`${basePath}/manifest.json`);
      set({ manifest, loadingProgress: 30 });

      // Load evolution timeline
      const timeline = await fetchJSON<EvolutionTimeline>(
        `${basePath}/evolution_timeline.json`,
      );
      set({ timeline, loadingProgress: 50 });

      // Load keyframe summary
      const keyframeSummary = await fetchJSON<{ keyframes: KeyframeData[] }>(
        `${basePath}/keyframe_summary.json`,
      );
      set({ keyframes: keyframeSummary.keyframes, loadingProgress: 70 });

      // Load concept evolution
      try {
        const concepts = await fetchJSON<ConceptEvolution>(
          `${basePath}/concept_evolution.json`,
        );
        set({ conceptEvolution: concepts, loadingProgress: 90 });
      } catch {
        // Concept evolution might not exist
        set({ conceptEvolution: null });
      }

      // Set initial iteration to the last one
      const lastIteration =
        timeline.iterations[timeline.iterations.length - 1] || 0;
      set({
        currentIteration: lastIteration,
        currentKeyframeIndex: keyframeSummary.keyframes.length - 1,
        loadingProgress: 100,
        isLoading: false,
      });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Unknown error",
        isLoading: false,
      });
    }
  },

  setIteration: (iteration) => {
    const { keyframes } = get();
    // Find closest keyframe
    let closestIndex = 0;
    let closestDist = Infinity;
    keyframes.forEach((kf, i) => {
      const dist = Math.abs(kf.iteration - iteration);
      if (dist < closestDist) {
        closestDist = dist;
        closestIndex = i;
      }
    });
    set({ currentIteration: iteration, currentKeyframeIndex: closestIndex });
  },

  setKeyframeIndex: (index) => {
    const { keyframes } = get();
    if (index >= 0 && index < keyframes.length) {
      set({
        currentKeyframeIndex: index,
        currentIteration: keyframes[index].iteration,
      });
    }
  },

  play: () => set({ isPlaying: true }),
  pause: () => set({ isPlaying: false }),
  togglePlay: () => set((state) => ({ isPlaying: !state.isPlaying })),

  nextKeyframe: () => {
    const { currentKeyframeIndex, keyframes } = get();
    if (currentKeyframeIndex < keyframes.length - 1) {
      get().setKeyframeIndex(currentKeyframeIndex + 1);
    }
  },

  prevKeyframe: () => {
    const { currentKeyframeIndex } = get();
    if (currentKeyframeIndex > 0) {
      get().setKeyframeIndex(currentKeyframeIndex - 1);
    }
  },

  setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),

  loadFullCheckpoint: async (checkpoint) => {
    const { currentModel, loadedCheckpoints } = get();

    // Return cached if available
    if (loadedCheckpoints[checkpoint]) {
      return loadedCheckpoints[checkpoint];
    }

    // Load from server
    const url = `/playback/${currentModel}/checkpoints/${checkpoint}.json`;
    try {
      const data = await fetchJSON(url);
      set((state) => ({
        loadedCheckpoints: { ...state.loadedCheckpoints, [checkpoint]: data },
      }));
      return data;
    } catch {
      // Try without checkpoints subdirectory
      const altUrl = `/playback/${currentModel}/${checkpoint}.json`;
      const data = await fetchJSON(altUrl);
      set((state) => ({
        loadedCheckpoints: { ...state.loadedCheckpoints, [checkpoint]: data },
      }));
      return data;
    }
  },

  getCurrentKeyframe: () => {
    const { keyframes, currentKeyframeIndex } = get();
    return keyframes[currentKeyframeIndex] || null;
  },

  getIterationData: (iteration) => {
    const { timeline } = get();
    if (!timeline) return null;
    return timeline.timeline.find((t) => t.iteration === iteration) || null;
  },

  getSparsityAtIteration: (iteration) => {
    const data = get().getIterationData(iteration);
    return data?.sparsity || 0;
  },

  getClosestKeyframeIndex: (iteration) => {
    const { keyframes } = get();
    let closest = 0;
    let minDist = Infinity;
    keyframes.forEach((kf, i) => {
      const dist = Math.abs(kf.iteration - iteration);
      if (dist < minDist) {
        minDist = dist;
        closest = i;
      }
    });
    return closest;
  },
}));

interface PlaybackData {
  input_text: string;
  input_tokens: number[];
  input_chars: string[];
  num_layers: number;
  num_heads: number;
  neurons_per_head: number;
  frames: PlaybackFrame[];
  overall_sparsity?: number;
  sparsity_by_layer?: number[];
  embedding_dim?: number;
  total_neurons?: number;
  predictions?: Array<Array<{ byte: number; char: string; prob: number }>>;
}

interface PlaybackState {
  playbackData: PlaybackData | null;
  currentFrame: number;
  isLoading: boolean;
  loadingMessage: string;
  error: string | null;
  mode: "live" | "playback";
  currentModel: string;

  loadPlayback: (text: string, modelName?: string) => Promise<void>;
  loadFromCheckpoint: (
    model: string,
    checkpoint: string,
    exampleIdx?: number,
  ) => Promise<void>;
  setModel: (model: string) => void;
  setFrame: (frame: number) => void;
  nextFrame: () => void;
  prevFrame: () => void;
  reset: () => void;
}

export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  playbackData: null,
  currentFrame: 0,
  isLoading: false,
  loadingMessage: "",
  error: null,
  mode: "playback",
  currentModel: "french",

  loadPlayback: async (text: string, modelName?: string) => {
    const model = modelName || get().currentModel;
    set({
      isLoading: true,
      error: null,
      loadingMessage: "Checking backend...",
    });

    // Resolve API origin (in production the backend lives on a different domain)
    const apiOrigin =
      (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
      "";

    // Quick health check first (2s timeout)
    try {
      const hc = await fetch(`${apiOrigin}/api/status`, {
        signal: AbortSignal.timeout(3000),
      });
      if (!hc.ok) {
        set({
          isLoading: false,
          loadingMessage: "",
          error:
            "Backend responded but API status check failed. Check server logs.",
        });
        return;
      }
      const status = await hc.json();
      if (!status.loaded_models || status.loaded_models.length === 0) {
        set({
          loadingMessage:
            "Backend is loading model... (first run may take longer)",
        });
      }
    } catch {
      set({
        isLoading: false,
        loadingMessage: "",
        error:
          "Cannot reach backend. Start the server with: uvicorn backend.main:app --reload --port 8000  (from the project root)",
      });
      return;
    }

    const MAX_RETRIES = 3;
    const TIMEOUT_MS = 120_000; // 120 seconds — model inference can be slow

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        // Update loading message
        if (attempt === 1) {
          set({
            loadingMessage: "Running model inference... (this may take 10-30s)",
          });
        } else {
          set({
            loadingMessage: `Retry ${attempt}/${MAX_RETRIES} — running inference...`,
          });
        }

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

        const response = await fetch(
          `${apiOrigin}/api/visualization/playback`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text,
              model_name: model,
              include_attention: true,
            }),
            signal: controller.signal,
          },
        );

        clearTimeout(timeoutId);

        if (response.ok) {
          set({ loadingMessage: "Processing response..." });
          const data = await response.json();
          set({
            playbackData: data,
            currentFrame: 0,
            isLoading: false,
            loadingMessage: "",
            mode: "live",
            error: null,
          });
          return; // Success!
        } else {
          const errData = await response
            .json()
            .catch(() => ({ detail: "Unknown error" }));
          console.warn(`API error (attempt ${attempt}):`, errData);

          // Don't retry on 4xx client errors
          if (response.status >= 400 && response.status < 500) {
            set({
              isLoading: false,
              loadingMessage: "",
              error: `API Error: ${errData.detail || response.statusText}`,
            });
            return;
          }
          // Server errors — retry
          if (attempt < MAX_RETRIES) {
            const backoff = attempt * 2000;
            set({
              loadingMessage: `Server error — retrying in ${backoff / 1000}s...`,
            });
            await new Promise((r) => setTimeout(r, backoff));
            continue;
          }
          set({
            isLoading: false,
            loadingMessage: "",
            error: `Server error after ${MAX_RETRIES} attempts: ${errData.detail || response.statusText}`,
          });
          return;
        }
      } catch (err: any) {
        if (err?.name === "AbortError") {
          console.warn(`Request timed out (attempt ${attempt})`);
          if (attempt < MAX_RETRIES) {
            set({
              loadingMessage: `Timed out — retrying (${attempt + 1}/${MAX_RETRIES})...`,
            });
            continue;
          }
          set({
            isLoading: false,
            loadingMessage: "",
            error:
              "Request timed out after multiple attempts. Ensure the backend is running (uvicorn on port 8000) and the model checkpoint is loaded.",
          });
          return;
        } else {
          console.warn(`API not available (attempt ${attempt}):`, err);
          if (attempt < MAX_RETRIES) {
            const backoff = attempt * 2000;
            set({
              loadingMessage: `Backend unreachable — retrying in ${backoff / 1000}s...`,
            });
            await new Promise((r) => setTimeout(r, backoff));
            continue;
          }
          set({
            isLoading: false,
            loadingMessage: "",
            error:
              "Backend is offline. Start the server with: uvicorn backend.main:app --reload --port 8000  (from the project root)",
          });
          return;
        }
      }
    }

    set({ isLoading: false, loadingMessage: "" });
  },

  setModel: (model: string) => {
    set({ currentModel: model });
  },

  loadFromCheckpoint: async (
    model: string,
    checkpoint: string,
    exampleIdx: number = 0,
  ) => {
    set({ isLoading: true, error: null });

    try {
      const url = `/playback/${model}/${checkpoint}.json`;
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to load ${url}`);
      }

      const data = await response.json();

      // Extract playback data from checkpoint
      if (data.playback && data.playback[exampleIdx]) {
        set({
          playbackData: data.playback[exampleIdx].data,
          currentFrame: 0,
          isLoading: false,
          mode: "playback",
        });
      } else {
        throw new Error("No playback data in checkpoint");
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Unknown error",
        isLoading: false,
      });
    }
  },

  setFrame: (frame: number) => {
    const { playbackData } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: Math.max(0, Math.min(frame, maxFrame)) });
  },

  nextFrame: () => {
    const { playbackData, currentFrame } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: currentFrame < maxFrame ? currentFrame + 1 : 0 });
  },

  prevFrame: () => {
    const { playbackData, currentFrame } = get();
    if (!playbackData) return;
    const maxFrame = playbackData.frames.length - 1;
    set({ currentFrame: currentFrame > 0 ? currentFrame - 1 : maxFrame });
  },

  reset: () => set({ currentFrame: 0 }),
}));
