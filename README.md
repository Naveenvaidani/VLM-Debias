# VLM-SHIELD: Architectural Mitigation of Shortcut Learning via Local Feature Bottlenecks

### Technical Specification & Engineering Protocol

The persistence of shortcut learning in Large Vision-Language Models (LVLMs) stems from a fundamental architectural flaw: standard Multi-Layer Perceptron (MLP) projectors act as "shortcut funnels." When optimized strictly via auto-regressive next-token prediction, the language model naturally learns to exploit spurious global correlations (e.g., backgrounds, watermarks, texture statistics) because they offer a mathematically cheaper path to minimize the loss function than parsing dense, local visual features.

**VLM-SHIELD** replaces the standard linear projector with a custom **Information-Preserving Attention Bottleneck**. This modified cross-attention architecture forces the LLM to explicitly query spatial vision patches, extinguishing global shortcut collapse.

---

## Key Components

1. **Vision Encoder**: `google/siglip-base-patch16-224`
   - *Rationale*: SigLIP uses a pairwise sigmoid loss, retaining significantly denser local patch features than standard contrastive models (196 spatial patches of size 16x16, dimension 768).
2. **Language Backbone**: `Qwen/Qwen2.5-1.5B-Instruct`
   - *Rationale*: At 1.5 billion parameters (1536-dim), Qwen2.5 is highly articulate while allowing efficient gradient checkpointing and fine-tuning on resource-constrained GPUs.
3. **Local Feature Bottleneck (`LocalFeatureBottleneck`)**:
   - Cross-attention bottleneck where LLM token embeddings act as queries ($Q$), attending to SigLIP vision patch representations ($K, V$).
   - If a background patch contains a spurious shortcut, the cross-attention mechanism allows the network to dynamically drop its attention weight to zero.
4. **Information-Sparsity Loss ($\mathcal{L}_{total} = \mathcal{L}_{NTP} + \lambda \mathcal{L}_{sparsity}$)**:
   - Penalizes spatial attention collapse onto single shortcut patches via entropy regularization across visual patches.

---

## Directory Structure

```
VLM Debias/
├── src/
│   ├── models/
│   │   └── debiased_vlm.py   # LocalFeatureBottleneck & DebiasedVLM architecture
│   └── training/
│       └── loss.py           # DebiasedVLMLoss (L_NTP + lambda * L_sparsity)
├── scripts/
│   └── download_models.py    # Local foundation model downloader script
├── tests/
│   └── test_debiased_vlm.py  # Unit tests for bottleneck, model, and loss function
├── train.py                  # Training pipeline with synthetic shortcut dataset
├── evaluate.py               # Quantitative debiasing benchmark & spatial entropy evaluation
├── app.py                    # Validation UI interface (Gradio & OpenCV jet heatmap overlay)
├── requirements.txt          # Dependencies list
└── README.md                 # Technical documentation
```

---

## Quick Start & Usage

### 1. Download Pre-Trained Foundation Models Locally
Download `google/siglip-base-patch16-224` and `Qwen/Qwen2.5-1.5B-Instruct` to local storage:
```bash
python scripts/download_models.py --output_dir ./local_models
```

### 2. Run Unit Tests
Verify model architecture, dimensions, cross-attention mechanics, and loss calculations:
```bash
python -m unittest discover tests
```

### 3. Train the LocalFeatureBottleneck
Execute the custom debiasing training pipeline:
```bash
# Fast mock training check (CPU friendly)
python train.py --epochs 3 --batch_size 4 --mock

# Full pre-trained training
python train.py --epochs 5 --batch_size 4 --lr 1e-4 --lambda_sparsity 0.1
```

### 4. Evaluate Shortcut Mitigation
Quantify spatial attention entropy and shortcut reliance:
```bash
python evaluate.py
```

### 5. Launch the Validation UI Interface
Visualize internal cross-attention heatmaps to visually verify local grounding:
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser. Upload an image to generate grounded captions and inspect proof of grounding heatmaps.

---

## Mathematical Formulation: Dual Grounding Loss

The optimization objective is formulated as:

$$\mathcal{L}_{total} = \mathcal{L}_{NTP} + \lambda_{ent} \mathcal{L}_{entropy} + \lambda_{supp} \mathcal{L}_{shortcut\_supp}$$

Where:
- $\mathcal{L}_{NTP} = -\sum_{t} \log P(w_t \mid w_{<t}, \mathbf{z}_{\text{fused}})$ (Causal Next-Token Prediction Loss)
- $\mathcal{L}_{entropy} = \mathbb{E} \left[ \frac{1}{H(\mathbf{A}) + \epsilon} \right]$ (Inverse Entropy preventing single-patch collapse)
- $H(\mathbf{A}) = -\sum_{p=1}^{196} A_p \log(A_p + \epsilon)$ (Spatial Patch Attention Entropy)
- $\mathcal{L}_{shortcut\_supp} = \mathbb{E} \left[ \sum_{p \in \mathcal{S}} A_p \right]$ (Explicit penalty extinguishing attention leakage onto spurious shortcut regions $\mathcal{S}$)

---

## Evaluation Against Bleeker et al. Benchmark

We evaluate shortcut learning mitigation using:
- **Relative Shortcut Reliance (RSR)**:
  $$\text{RSR} = \frac{\text{Acc}_{ID} - \text{Acc}_{OOD}}{\text{Acc}_{ID}}$$
- **Out-of-Distribution Performance Drop**: $\Delta_{OOD} = \text{Acc}_{ID} - \text{Acc}_{OOD}$
- **Shortcut Patch Leakage**: $\text{Attn}_{\mathcal{S}} = \frac{1}{T} \sum_{t=1}^T \sum_{p \in \mathcal{S}} A_{t, p}$
- **Spatial Grounding Entropy**: $H_{attn}$ across object extent.
