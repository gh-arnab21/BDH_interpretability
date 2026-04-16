# BDH Interpretability Suite

DEMO Link - https://youtu.be/Mlw_ITRiuyc?si=lQbud51_9Ua69HBG

**KRITI 2026 -- AI Interpretability Challenge |**

An interactive visualization and interpretability platform for the BDH (Baby Dragon Hatchling) post-transformer architecture. This project delivers a complete pipeline: training BDH models from scratch on multilingual data, extracting every internal signal the architecture exposes, and rendering those signals in a 9-page interactive frontend with live backend inference.

We primarily targeted **Path A (Visualization)**, building rich interactive explorations of every BDH component. In doing so we delivered significantly in **Path B (Interpretability)** -- monosemanticity probing, synapse tracking, selectivity analysis, and cross-concept correlation -- and contributed to **Path C (Frontier)** by training separate French and Portuguese specialist models and merging them into a single polyglot via neuron concatenation.

---

## Table of Contents

1. [Track Coverage](#track-coverage)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Data Extraction Pipeline](#data-extraction-pipeline)
5. [Frontend Pages](#frontend-pages)
6. [Tech Stack](#tech-stack)
7. [Repository Structure](#repository-structure)
8. [Setup and Installation](#setup-and-installation)
9. [Deployment](#deployment)
10. [Key Findings](#key-findings)
11. [Links](#links)

---

## Track Coverage

| Track                          | Coverage    | What We Delivered                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Path A -- Visualization**    | Primary     | 9 interactive pages covering every BDH component: architecture walkthrough, sparsity comparison, 3D graph topology, Hebbian playback, monosemanticity dashboard (6 tabs), merge visualization, findings dashboard, and a full educational tutorial                                                                                                             |
| **Path B -- Interpretability** | Significant | Monosemantic neuron discovery across 4 concept categories (currencies, countries, languages, politics), selectivity distributions, sparse fingerprinting, cross-concept correlation matrices, shared neuron analysis, per-neuron graph visualization, synapse tracking with Hebbian sigma accumulation, category affinity probing via sparse cosine similarity |
| **Path C -- Frontier**         | Partial     | Trained French and Portuguese BDH specialists on Europarl, merged via neuron concatenation (doubling MLP width), evaluated merged model on both language pairs, heritage routing analysis showing which specialist neurons activate for which language                                                                                                         |

---

## Model Architecture

The BDH architecture is a biologically-inspired language model that replaces softmax attention and layer normalization with sparse ReLU activations and Hebbian synaptic updates.

| Parameter                | Value                                      |
| ------------------------ | ------------------------------------------ |
| Layers                   | 6                                          |
| Attention Heads          | 4                                          |
| Embedding Dimension (D)  | 192                                        |
| MLP Expansion Multiplier | 64 (128 after merge)                       |
| Neurons per Head (N)     | 3,072 (6,144 after merge)                  |
| Total MLP Neurons        | 12,288 (24,576 after merge)                |
| Vocabulary               | 256 (raw UTF-8 bytes)                      |
| Positional Encoding      | RoPE (Rotary Position Embeddings)          |
| Attention                | Linear causal -- O(T) complexity           |
| Activation               | ReLU sparsification (~95% inactive)        |
| Synaptic Update          | Hebbian -- no backpropagation at inference |
| Context Length           | 256 tokens (training block size)           |

**Core Layer Equations:**

```
x_sparse = ReLU(x @ W_up)                          -- Sparse Encoding
attn = (Q @ K^T) * causal_mask  (no softmax)        -- Linear Attention
y = attn @ (x_sparse @ W_val)                       -- Value Encoding
sigma += rho * (x_pre * y_post)                      -- Hebbian Update
```

---

## Training Pipeline

All models were trained using the notebook at `changed_checkpoints_data/bdh_best_model.ipynb` on Google Colab with an A100 GPU.

### Dataset

- **Source**: Europarl parallel corpus (European Parliament proceedings)
- **Language Pairs**: English-French (en-fr) and English-Portuguese (en-pt)
- **Encoding**: Byte-level (UTF-8), no subword tokenizer
- **Format**: `<F:en>...english text...<T:fr>...french text...` per sample
- **Preprocessing**: Raw bytes converted to `.bin` files via memory-mapped arrays

### Training Configuration

| Setting               | French                        | Portuguese |
| --------------------- | ----------------------------- | ---------- |
| Architecture          | 6L / 192D / 4H / mlp_mult=64  | Same       |
| Iterations            | 50,000                        | 40,000     |
| Batch Size            | 16                            | 16         |
| Block Size            | 256 tokens                    | 256        |
| Learning Rate         | 3e-4 (cosine decay to 3e-5)   | Same       |
| Warmup                | 200 iterations                | 200        |
| Optimizer             | AdamW (beta1=0.9, beta2=0.95) | Same       |
| Weight Decay          | 0.1                           | 0.1        |
| Gradient Accumulation | 4 steps                       | 4          |
| Precision             | bfloat16                      | bfloat16   |
| Compilation           | torch.compile enabled         | Same       |
| Eval Interval         | Every 500 iterations          | 500        |
| GPU                   | A100 (Colab)                  | A100       |

### Model Merging

French and Portuguese specialists are merged via **neuron concatenation**:

1. Both models share identical architecture (6L/192D/4H)
2. For each layer, MLP weight matrices (`W_up`, `W_gate`, `W_down`) are concatenated along the neuron dimension
3. The merged model's `mlp_dim_mult` doubles from 64 to 128, giving 6,144 neurons per head (24,576 total)
4. Attention weights, embeddings, and output head are averaged
5. RoPE frequency buffers are rebuilt from scratch (not copied) to avoid shape mismatches
6. No fine-tuning is performed after the merge

### Checkpoints

| Model                 | File                 | Location                                                                  |
| --------------------- | -------------------- | ------------------------------------------------------------------------- |
| French Specialist     | `french_best.pt`     | `checkpoints/french/` and `changed_checkpoints_data/new_best_models/`     |
| Portuguese Specialist | `portuguese_best.pt` | `checkpoints/portuguese/` and `changed_checkpoints_data/new_best_models/` |
| Merged Polyglot       | `merged_merged.pt`   | `checkpoints/merged/` and `changed_checkpoints_data/new_best_models/`     |

---

## Data Extraction Pipeline

The notebook (`changed_checkpoints_data/bdh_best_model.ipynb`) runs a 6-phase extraction pipeline after training, generating all JSON data consumed by the frontend. This runs on the merged model using the BDH extraction context manager.

### Phase 1: Per-Token Telemetry

For each sentence in the evaluation corpus (15 multilingual sentences), forward-pass through the merged model with extraction enabled. For every token at every layer and head, record:

- Pre-ReLU activation histograms (50 bins)
- Post-ReLU (gated) activation histograms
- Sparsity fraction, mean, standard deviation
- Count of active neurons
- Top-30 most active neuron indices and values
- Hero neuron dumps (top-5 neurons per layer/head)

Output: `telemetry/sentence_*.json`, `hero_tokens/hero_*.json`

### Phase 2: Graph Topology

Extracts the internal wiring of each attention head:

- Computes the graph matrix `G* = D_x @ E` where `D_x` is the down-projection and `E` is the embedding matrix
- Applies a beta threshold to retain only strong connections
- Runs Louvain community detection to identify neuron clusters
- Records node positions, cluster assignments, edge weights

Output: `graph/graph_head_*.json`

### Phase 3: Synapse Tracking

Simulates Hebbian learning across a 10-sentence corpus:

- Manually accumulates `sigma += rho * (x_pre * y_post)` at each token step
- Records sigma snapshots at configurable intervals
- Tracks delta-sigma (change per step) and cumulative sigma
- Per-head, per-layer tracking of gate activity

Output: `synapses/synapses_head_*.json`

### Phase 4: Monosemanticity Analysis

Probes individual neurons for concept specificity:

- Defines a concept bank with 4 categories: currencies, countries, institutions, action verbs
- For each category, runs ~52 curated French sentences through the model
- Extracts activations at concept-word byte positions
- Computes per-neuron selectivity scores across categories
- Identifies monosemantic neurons (those that fire strongly for one category and weakly for others)
- Records selectivity distributions, cross-concept correlations, shared neuron sets

Output: `monosemanticity/*.json`

### Phase 5: Global Sparsity Statistics

Aggregates sparsity metrics across the full evaluation corpus:

- Per-layer, per-head sparsity fractions
- Active neuron counts at each layer
- Distribution statistics (mean, std, percentiles)
- Comparison data formatted for BDH-vs-Transformer visualization

Output: `sparsity/*.json`

### Phase 6: Evolution and Merge Data

Copies training evolution logs and merge metadata:

- Loss curves over training iterations
- Sparsity evolution over training
- Merge heritage mapping (which neurons came from which specialist)
- Per-language evaluation metrics before and after merge

Output: `evolution/*.json`, `merge/*.json`

---

## Frontend Pages

The frontend consists of 9 interactive pages, each targeting specific aspects of BDH interpretability.

### 1. Home Page

The landing page. Features a wireframe terrain with animated liquid blobs, a live hero activation heatmap rendered from real neuron data (`hero_tokens/*.json`), a stats bar (5% active neurons, O(T) attention, infinite context, 1:1 synapse-to-concept), the four core BDH layer equations rendered in KaTeX, an 8-card feature grid linking to every other page, and a differentiators section contrasting BDH with standard transformers.

### 2. Architecture Page

A 13-step animated architecture walkthrough with playback controls. Each step highlights a different part of the forward pass: byte embedding, sparse encoding, RoPE injection, linear attention, value encoding, gated sparsification, Hebbian update, residual connection, and output decoding. Includes a math detail panel showing the relevant equation for each step, a layer progress indicator, and customizable input text.

### 3. Sparsity Page

Side-by-side comparison of BDH versus a standard transformer. Displays two 400-cell neuron grids: BDH shows ~5% active (sparse), transformer shows ~95% active (dense). Supports model selection (French, Portuguese, Merged) and shows per-layer sparsity breakdowns. Loads data from precomputed JSON with a live/precomputed badge indicator. Token-level hover reveals which specific neurons fire for each input byte.

### 4. Graph Brain (3D Topology)

A WebGL-rendered 3D force-directed graph built with Three.js and react-force-graph-3d. Visualizes the internal wiring of each attention head using the graph matrix `G* = D_x @ E`. Features a head selector (4 heads), a beta threshold slider to filter weak connections, cluster expansion on click (Louvain communities), and falls back to static JSON when the backend is unavailable. Handles WebGL context cleanup on unmount to prevent memory leaks.

### 5. Monosemanticity Page

The most feature-dense page. Contains 6 tabs:

- **Synapse Tracking**: Animated Hebbian sigma accumulation showing how synaptic weights evolve token-by-token during inference
- **Selectivity**: Per-neuron selectivity distributions showing which neurons respond to which concept categories, with layer and head selectors
- **Sparse Fingerprinting**: Activation pattern visualization showing the sparse firing signature for specific concepts
- **Cross-Concept**: Correlation matrices between concept categories revealing how much neuron overlap exists between, e.g., currencies and countries
- **Shared Neurons**: Identifies neurons that respond to multiple concepts and quantifies the degree of sharing
- **Neuron Graph**: Network visualization of neuron relationships based on co-activation patterns

Also includes a "Try It Yourself" category affinity probe: enter a sentence, the backend runs it through the model, and returns a sparse cosine similarity score against each concept category.

### 6. Hebbian Learning Page

Word-by-word sigma playback that visualizes Hebbian memory formation during inference. Select a layer and head, choose between delta-sigma (instantaneous change) and cumulative sigma views, adjust playback speed, and watch gate activity bars respond in real time. Shows how synaptic weights strengthen or weaken as each token is processed, without any backpropagation.

### 7. Merge Page

Visualizes the neuron concatenation merge process. Includes an animated merge diagram, training evolution charts (loss and sparsity curves for both specialists), model cards summarizing each specialist's configuration, a loss comparison table, sample text generations from each model, a neuron heritage map showing which neurons in the merged model originated from which specialist, and a heritage probe that shows language-dependent routing.

### 8. Findings Page

A summary dashboard aggregating key results across all analyses. Displays hero statistics, a loss landscape chart, selectivity histogram with radial gauge, heritage routing donut chart, sigma replay animation, distinctness chart comparing specialist versus merged neurons, and a neuron activation heatmap. All visualization data is precomputed.

### 9. Learn BDH Page

An 8-step educational tutorial walking through the full BDH layer computation:

1. Byte Embedding
2. Sparse Encoding (ReLU expansion)
3. RoPE (Rotary Position Embeddings)
4. Linear Attention (no softmax)
5. Value Encoding
6. Sparse Gating
7. Decode + Residual
8. Full Layer Assembly

Each step includes a description tab and a theory tab with mathematical details, code snippets from the actual architecture, difficulty badges, and animated visualizations. Designed to teach the BDH architecture from first principles.

---

## Tech Stack

### Frontend

| Technology           | Version | Purpose                            |
| -------------------- | ------- | ---------------------------------- |
| React                | 18      | UI framework                       |
| TypeScript           | 5.2     | Type safety                        |
| Vite                 | 5       | Build tool and dev server          |
| Tailwind CSS         | 3.3     | Utility-first styling              |
| Framer Motion        | 10      | Page transitions and animations    |
| Three.js             | 0.183   | WebGL 3D rendering                 |
| react-force-graph-3d | 1.29    | Force-directed graph visualization |
| D3                   | 7       | Data-driven charts and histograms  |
| KaTeX                | 0.16    | LaTeX math rendering               |
| Zustand              | 4       | State management                   |
| Axios                | 1.7     | HTTP client for backend API        |
| Lucide React         | 0.344   | Icon library                       |

### Backend

| Technology     | Version | Purpose                         |
| -------------- | ------- | ------------------------------- |
| FastAPI        | 0.100+  | API framework                   |
| PyTorch        | 2.0+    | Model loading and inference     |
| uvicorn        | 0.23+   | ASGI server                     |
| NetworkX       | 3.1+    | Graph algorithms                |
| python-louvain | 0.16+   | Community detection             |
| scikit-learn   | 1.3+    | Distance metrics and clustering |
| NumPy          | 1.24+   | Numerical computation           |
| SciPy          | 1.11+   | Statistical functions           |

### Training

| Technology               | Purpose                                 |
| ------------------------ | --------------------------------------- |
| PyTorch + torch.compile  | Model definition and optimized training |
| Google Colab (A100)      | GPU compute                             |
| Europarl Corpus          | Multilingual training data              |
| AdamW + Cosine LR        | Optimization schedule                   |
| bfloat16 mixed precision | Memory-efficient training               |

---

## Repository Structure

```
.
|-- README.md                          # This file
|-- requirements.txt                   # Python dependencies
|-- Dockerfile                         # Backend Docker image (HF Spaces)
|-- hf_space_README.md                 # HuggingFace Spaces metadata
|-- upload_to_hf.py                    # Script to push backend to HF
|-- test_models.py                     # Model loading tests
|-- script.md                          # Demo video script
|
|-- training/
|   |-- bdh.py                         # Official BDH architecture with extraction hooks
|   |-- train.py                       # Training script
|   |-- download_europarl.py           # Europarl dataset downloader
|   |-- configs/
|       |-- french.yaml                # French specialist config
|       |-- portuguese.yaml            # Portuguese specialist config
|
|-- changed_checkpoints_data/
|   |-- bdh_best_model.ipynb           # MAIN TRAINING NOTEBOOK (Colab)
|   |-- new_best_models/               # Final trained checkpoints
|   |   |-- french_best.pt
|   |   |-- portuguese_best.pt
|   |   |-- merged_merged.pt
|   |-- viz_data_complete/             # Complete extracted visualization data
|       |-- corpus.json
|       |-- meta.json
|       |-- evolution/
|       |-- graph/
|       |-- hero_tokens/
|       |-- merge/
|       |-- monosemanticity/
|       |-- sparsity/
|       |-- synapses/
|       |-- telemetry/
|
|-- checkpoints/
|   |-- french/                        # French specialist + evolution logs
|   |-- portuguese/                    # Portuguese specialist + evolution logs
|   |-- merged/                        # Merged polyglot model
|
|-- analysis/
|   |-- merge.py                       # Model merging logic
|   |-- monosemanticity.py             # Interpretability analysis
|   |-- finetune_merged.py             # Post-merge fine-tuning (experimental)
|
|-- scripts/
|   |-- precompute_monosemanticity.py  # V2 monosemanticity extraction
|   |-- assemble_mono_data.py          # Assemble monosemanticity JSON
|   |-- optimize_viz_data.py           # Optimize JSON sizes for frontend
|   |-- generate_playback.py           # Generate synapse playback data
|   |-- check_neurons.py               # Neuron inspection utilities
|   |-- inspect_hero.py                # Hero neuron inspector
|   |-- debug_synapse*.py              # Synapse debugging scripts
|
|-- backend/
|   |-- main.py                        # FastAPI application entry point
|   |-- routes/
|   |   |-- inference.py               # Live text generation endpoint
|   |   |-- sparsity.py                # Sparsity analysis endpoint
|   |   |-- graph.py                   # Graph topology endpoint
|   |   |-- analysis.py                # General analysis endpoint
|   |   |-- merge_api.py               # Merge operations endpoint
|   |   |-- models.py                  # Model management endpoint
|   |   |-- visualization.py           # Visualization data endpoint
|   |-- services/
|       |-- model_service.py           # Model loading and caching
|
|-- frontend/
|   |-- index.html                     # HTML entry point
|   |-- package.json                   # Node.js dependencies
|   |-- vite.config.ts                 # Vite build configuration
|   |-- tailwind.config.js             # Tailwind CSS configuration
|   |-- Dockerfile                     # Frontend Docker image (nginx)
|   |-- nginx.conf                     # nginx reverse proxy config
|   |-- src/
|   |   |-- App.tsx                    # Root component and routing
|   |   |-- main.tsx                   # React entry point
|   |   |-- index.css                  # Global styles
|   |   |-- pages/                     # 9 page components (see above)
|   |   |-- components/                # Shared UI components
|   |   |-- features/architecture/     # Architecture diagram logic
|   |   |-- stores/                    # Zustand state stores
|   |   |-- utils/                     # API client, helpers
|   |-- public/                        # Precomputed JSON data
|       |-- telemetry/                 # Per-token activation data
|       |-- graph/                     # Graph topology JSON
|       |-- hero_tokens/               # Hero neuron dumps
|       |-- monosemanticity/           # Concept probing results
|       |-- sparsity/                  # Sparsity statistics
|       |-- synapses/                  # Hebbian tracking data
|       |-- evolution/                 # Training evolution logs
|       |-- merge/                     # Merge metadata
|       |-- viz_data/                  # Additional visualization data
|
|-- notebooks/
    |-- BDH_Finetune_Kaggle.ipynb
    |-- BDH_Kaggle_Pipeline.ipynb
    |-- BDH_Training_Colab.ipynb
```

---

## Setup and Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ (20 recommended)
- A CUDA-capable GPU for training (or use Google Colab with A100)
- ~10GB disk space for Europarl datasets

### Backend Setup

```bash
# Clone the repository
git clone <repo-url>
cd BDH_Pathway-main

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server from the project root
uvicorn backend.main:app --reload --port 8000
```

The backend serves the API at `http://localhost:8000/api/`. It loads model checkpoints from `checkpoints/` and serves precomputed visualization data.

### Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

The frontend runs at `http://localhost:5173`. In development mode, API requests to `/api` are proxied to the backend at `localhost:8000`.

### Training (Colab)

Open `changed_checkpoints_data/bdh_best_model.ipynb` in Google Colab with a GPU runtime (A100 recommended). The notebook contains the complete pipeline:

1. Installs dependencies
2. Downloads Europarl en-fr and en-pt datasets
3. Trains the French specialist (50k iterations)
4. Trains the Portuguese specialist (40k iterations)
5. Merges models via neuron concatenation
6. Evaluates on both language pairs
7. Runs the full 6-phase data extraction pipeline
8. Packages outputs for the frontend

### Production Build

```bash
cd frontend
npm run build    # Outputs to frontend/dist/
```

The production build bundles all assets and can be served by any static file server or the included nginx Docker image.

---

## Deployment

### Frontend -- Coolify (Docker + nginx)

The frontend is deployed as a Docker container using `frontend/Dockerfile`:

```
Stage 1: node:20-alpine   -- npm install, npm run build
Stage 2: nginx:stable-alpine -- serves dist/ on port 80
```

The VITE_API_URL environment variable is set at build time to point to the HuggingFace Spaces backend URL.

### Backend -- HuggingFace Spaces (Docker)

The backend is deployed as a Docker container on HuggingFace Spaces using the root `Dockerfile`:

```
Base: python:3.11-slim
Port: 7860 (HF Spaces requirement)
Entrypoint: uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

Model checkpoints are included in the Docker image. The `hf_space_README.md` file provides HuggingFace Spaces metadata (SDK: docker, port: 7860).

---

## Key Findings

1. **Sparsity is structural, not regularized.** BDH achieves ~95% sparsity through ReLU after a high-ratio expansion (192 -> 3,072 per head). No dropout or L1 penalty is needed. The architecture forces the model to compress information into a small fraction of active neurons.

2. **Individual neurons encode specific concepts.** Probing across 4 semantic categories (currencies, countries, languages, politics) reveals neurons with high selectivity -- they fire strongly for one category and remain near-silent for others. This monosemanticity is a direct consequence of the extreme sparsity.

3. **Merging works without fine-tuning.** Concatenating the MLP neurons of separately trained specialists produces a working polyglot model. The merged model routes inputs to the appropriate specialist neurons: French neurons activate for French text, Portuguese neurons for Portuguese text, with minimal interference.

4. **Hebbian updates create context-dependent memory.** During inference, synaptic weights accumulate via `sigma += rho * (x_pre * y_post)`. This means the model's internal state adapts to the sequence it is processing, forming transient "memories" without any gradient computation.

5. **Linear attention scales linearly.** By removing softmax normalization, BDH's attention mechanism operates in O(T) time instead of O(T^2). Combined with the sparse MLP, this makes the compute profile qualitatively different from standard transformers.

6. **Graph topology reveals modular structure.** The graph matrix `G* = D_x @ E` exposes how neurons wire together within each attention head. Louvain community detection finds clear clusters, suggesting the network self-organizes into functional modules during training.

---

## Links

- [BDH Paper (arXiv:2509.26507)](https://arxiv.org/abs/2509.26507)
- [Official BDH Repository](https://github.com/pathwaycom/bdh)
- [KRITI 2026 Challenge](https://kriti.org)

---

**KRITI 2026 AI Interpretability Challenge**
