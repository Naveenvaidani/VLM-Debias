# VLM-Debias: Does the Vision-Language Connector Cause Shortcut Learning?

**A controlled study of shortcut learning (spurious correlations) in LLM-based vision-language models (VLMs).**
We hold the LLM, the vision encoder, the data and the training budget fixed and swap only the
vision-language connector: **MLP projector (LLaVA)**, **Q-Former (BLIP-2)**, and a **Flamingo-style gated
cross-attention** block with a learnable or a fixed injection gate. We measure group robustness
(worst-group accuracy) on **Waterbirds**, a synthetic **Watermark** task and **CelebA**, with Qwen3.5-4B and
Phi-4-mini (4-bit QLoRA) on frozen SigLIP features.

**Keywords:** shortcut learning · spurious correlations · group robustness · worst-group accuracy · vision-language
models · multimodal LLM · connector / projector design · Q-Former · gated cross-attention · Flamingo gate ·
Deep Feature Reweighting (DFR) · Waterbirds · CelebA · QLoRA · Qwen3.5 · Phi-4-mini · SigLIP

![Architecture: frozen SigLIP features, four swappable connectors, LLM with LoRA](docs/figures/architecture.svg)

## Key findings

1. **A learnable injection gate causes shortcut reliance.** The Flamingo-style `tanh(α)` gate stays shut after
   alignment (|tanh α| ≈ 0.0005) and the connector then passes almost no visual signal. Fixing the gate open
   raises worst-group accuracy by **+10.6 points on Waterbirds** (15 seeds, p = 4e-4) and **+23.8 on Watermark**
   (13 seeds, p = 3e-4), and removes the collapsed seeds (2/15 and 8/13 → 0). Pre-declared tests H1 and H2 are
   both confirmed on Qwen3.5-4B.
2. **It replicates on a second LLM.** Phi-4-mini, 5 seeds: fixed > learned on both tasks (Waterbirds +9.7,
   p = 0.005; Watermark +31.6, p = 0.044, which misses the pre-declared one-sided α = 0.025).
3. **Once the gate is fixed, connector type barely matters.** MLP, Q-Former and fixed-gate cross-attention are
   within seed noise of each other (MLP vs Q-Former, 5 seeds: p = 0.23 and 0.40). Matching image-token counts
   (MLP with 36 instead of 49 tokens) changes nothing (74.7 vs 74.9).
4. **A stronger encoder hides the problem, but the gate still shuts.** With SigLIP-large features the learned
   gate again ends alignment at 0.0006, yet its Waterbirds WGA no longer collapses (79.3 vs 80.1 fixed,
   2 seeds). All four connectors gain 6–16 points from the bigger encoder.
5. **A cheap fix: retrain only the connector on a small balanced set (connector-DFR).** On Watermark it lifts the
   fixed-gate connector from 72.5 to **89.6 ± 0.5** WGA (5 seeds) and beats MLP + DFR (81.2) and a linear DFR
   probe on the same features (85.1). On Waterbirds it gives no gain (73.9 → 74.1).
6. **Off-the-shelf VLMs are no better.** Qwen3.5-4B with its own vision tower, zero-shot and calibrated on
   validation, reaches only **35.5%** WGA on Waterbirds. In released IDEFICS-80B checkpoints, **9 of 20** gated
   cross-attention layers have |tanh α| < 0.05, so near-shut gates also occur at scale.

![Main result: worst-group accuracy per connector, every seed shown](docs/figures/final/main_connectors.png)

## Results

All numbers: test **worst-group accuracy (WGA, %)**, mean ± s.d. over seeds, checkpoint chosen by validation WGA,
averaged over three question phrasings (one held out from training).

### 1. Connector comparison, Qwen3.5-4B, SigLIP-base

| Task | MLP projector | Q-Former | Cross-attn, learned gate | Cross-attn, fixed gate |
| :--- | :---: | :---: | :---: | :---: |
| Waterbirds | 74.5 ± 1.5 (5 seeds: 74.4) | 71.7 ± 2.5 (5 seeds: 72.8) | 63.6 ± 10.4 (15) | **74.2 ± 3.4** (15) |
| Watermark | 63.6 ± 4.4 (5 seeds: 66.2) | 68.6 ± 4.1 (5 seeds: 69.2) | 41.4 ± 19.0 (13) | **65.2 ± 10.4** (13) |

### 2. Pre-declared gate hypotheses (paired by seed)

| LLM | Task | Seeds | Learned gate | Fixed gate | Gain (95% CI) | H1 paired t | H2 variance F | Collapsed (< 50) learned / fixed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3.5-4B | Waterbirds | 15 | 63.6 ± 10.4 | 74.2 ± 3.4 | +10.6 [+5.3, +16.0] | **4e-4** | **9e-5** | 2 / 0 |
| Qwen3.5-4B | Watermark | 13 | 41.4 ± 19.0 | 65.2 ± 10.4 | +23.8 [+12.5, +35.1] | **3e-4** | **0.02** | 8 / 0 |
| Phi-4-mini | Waterbirds | 5 | 60.4 ± 5.7 | 70.1 ± 3.6 | +9.7 [+3.8, +15.7] | **0.005** | 0.20 | 0 / 0 |
| Phi-4-mini | Watermark | 5 | 28.5 ± 21.3 | 60.1 ± 15.1 | +31.6 [−7.4, +70.6] | 0.044 | 0.26 | 4 / 2 |

H1: fixed > learned (one-sided, α = 0.025). H2: learned gate varies more across seeds. Bold = significant.
Tables: [`results_study/hypotheses.md`](results_study/hypotheses.md), [`results_study/phi/hypotheses.md`](results_study/phi/hypotheses.md).

### 3. Second LLM: Phi-4-mini, 5 seeds

| Task | MLP | Q-Former | Learned gate | Fixed gate |
| :--- | :---: | :---: | :---: | :---: |
| Waterbirds | **76.6 ± 3.6** | 71.9 ± 3.1 | 60.4 ± 5.7 | 70.1 ± 3.6 |
| Watermark | 62.2 ± 10.0 | **66.1 ± 10.7** | 28.5 ± 21.3 | 60.1 ± 15.1 |

![Phi-4-mini](docs/figures/final/phi_connectors.png)

### 4. Vision encoder scale: SigLIP-base vs SigLIP-large (Waterbirds)

| Encoder | MLP | Q-Former | Learned gate | Fixed gate |
| :--- | :---: | :---: | :---: | :---: |
| SigLIP-base (196 × 768) | 74.5 | 71.7 | 63.6 | 74.2 |
| SigLIP-large (256 × 1024), 2 seeds | **82.9 ± 3.4** | 81.7 ± 0.4 | 79.3 ± 1.9 | 80.1 ± 3.1 |

![Encoder scale](docs/figures/final/encoder_scale.png)

### 5. Fixes: connector-DFR, balanced sampling, linear DFR

| Method | Uses group labels | Waterbirds | Watermark |
| :--- | :--- | :---: | :---: |
| MLP (ERM) | selection only | 74.9 ± 1.4 | 64.9 ± 8.9 |
| MLP + connector-DFR | small balanced val half | 75.0 ± 3.6 | 81.2 ± 2.2 |
| Fixed gate (ERM) | selection only | 73.9 ± 2.5 | 72.5 ± 9.5 |
| **Fixed gate + connector-DFR** | small balanced val half | 74.1 ± 5.6 | **89.6 ± 0.5** |
| MLP + group-balanced sampling, 3 seeds | yes (training) | – | 89.1 ± 0.9 |
| Fixed gate + group-balanced sampling, 3 seeds | yes (training) | – | 87.1 ± 3.5 |
| Linear DFR on frozen SigLIP features (no LLM), 3 seeds | small balanced val half | **84.2 ± 1.5** | 85.1 ± 1.3 |
| Connector-JTT (1 seed) | selection only | 68.4 / 80.2 (fixed / MLP) | 49.9 / 58.6 |

Rows 1–4: 5 seeds each. Fixed gate + DFR beats MLP + DFR on Watermark by +8.4 [+6.0, +9.8] (p < 0.001, Holm-corrected).

![Fixes](docs/figures/final/fixes.png)

### 6. CelebA (Blond_Hair vs Male, official splits)

| Connector | Seeds | WGA | Avg acc |
| :--- | :---: | :---: | :---: |
| MLP | 4 | **54.3 ± 13.7** | 94.1 ± 1.4 |
| Cross-attn, learned gate | 4 | 36.3 ± 18.7 | 87.4 ± 9.8 |
| Cross-attn, fixed gate | 2 | 49.3 ± 4.5 | 94.8 ± 0.3 |

Same direction as the other tasks (learned gate worst and least stable); not significant with these seeds.

### 7. Ablations (Waterbirds, MLP unless noted)

| Variant | Seeds | WGA |
| :--- | :---: | :---: |
| Default (49 image tokens, 4 epochs) | 5 | 74.9 ± 1.4 |
| 36 image tokens | 2 | 74.7 ± 1.1 |
| + latent-target-decoding caption loss (LTD) | 3 | 75.9 ± 3.3 |
| 8 epochs | 1 | 79.1 |
| Fixed gate, 8 epochs | 1 | 44.5 |
| Frozen LLM (no LoRA): learned gate / MLP / Q-Former | 1–3 | 24.2 / 73–75 |
| Attention-entropy regulariser | 3 | no gain |

![Ablations](docs/figures/final/ablations.png)

### 8. Reference points

| Model | Waterbirds WGA |
| :--- | :---: |
| Qwen3.5-4B native vision tower, zero-shot | 0.0 (always one class) |
| same, answer threshold calibrated on validation | 35.5 (avg 50.8) |
| Linear probe on SigLIP-base, ERM | 62.0 |
| Best connector in this study (MLP, SigLIP-large) | 82.9 |
| Linear DFR on SigLIP-base | 84.2 |

## Observations

- **The gate, not the connector family, decides robustness.** Every failure we see traces to the learnable
  gate: it is near zero after alignment, its gradient changes sign every step, and with it shut the LLM
  falls back on whatever cue is easiest. A fixed gate with a zero-initialised output layer starts as the
  identity, so the optimiser cannot close it.
- **The failure is a seed lottery.** The learned gate is not uniformly worse: some seeds match the fixed gate,
  others collapse below 50%. Variance, not only the mean, is the effect (H2).
- **Stronger features mask it.** With SigLIP-large the gate still shuts, but the learned-gate model no longer
  collapses on Waterbirds. Two seeds only; the safe reading is "a better encoder hides the problem", not
  "fixes it".
- **The information is in the features; the VLM does not use it.** A linear DFR probe on the same frozen
  features beats every ERM-trained VLM by about 10 points. Retraining only the connector on a few hundred
  balanced examples closes that gap on Watermark but not on Waterbirds.
- **Attention on the shortcut does not mean reliance on it.** Both cross-attention variants attend to the
  watermark corner about 2.5x more when it is present, yet only the learned-gate model fails.
- **Waterbirds on land is the hardest group** for every connector.
- **Longer training is not a free win.** At 8 epochs the MLP gains and the fixed gate collapses on one seed;
  with one seed each this is a warning, not a result.

---

## Earlier experiment: VLM-SHIELD local feature bottleneck (Waterbirds, 1 seed)

The repository started as **VLM-SHIELD**: the prompt tokens cross-attend to individual SigLIP patches and the
attended context enters the LLM through a zero-initialised tanh gate, so the answer token's attention map is
inspectable. The connector study above grew out of it.

### VLM-SHIELD results (Waterbirds test set, n = 5,794)

| Model | Avg acc | Worst-group acc | Acc_ID | Acc_OOD | Δ_OOD | RSR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MLP projector (baseline) | 90.4% | 80.4% | 98.3% | 82.5% | 15.8 pp | 0.161 |
| VLM-SHIELD bottleneck | 89.2% | 77.3% | 97.1% | 81.3% | 15.8 pp | 0.163 |
| MLP projector + group-balanced sampling | **90.8%** | **82.8%** | 96.9% | **84.6%** | **12.3 pp** | **0.127** |
| VLM-SHIELD + group-balanced sampling | 89.4% | 80.9% | 96.5% | 82.3% | 14.2 pp | 0.147 |

Qwen3.5-4B (4-bit QLoRA) for every row; checkpoint selected by validation worst-group accuracy.
Per-group numbers are in [`metrics/research_paper_metrics.md`](metrics/research_paper_metrics.md),
and everything is in [`metrics/benchmark_results.json`](metrics/benchmark_results.json).

![Waterbirds test accuracy](metrics/accuracy_summary.png)

![Per-group test accuracy](metrics/per_group_accuracy.png)

#### What the numbers say

- **The shortcut is real.** Every model is 12–16 points less accurate on birds shown on the
  "wrong" background, and `waterbird/land` or `landbird/water` is always the worst group.
- **The cross-attention bottleneck does not beat the MLP projector here.** It is 2–3 points lower
  on worst-group accuracy in both the ERM and balanced settings. With one seed per configuration
  that gap is within noise, so the honest reading is "no measurable benefit", not "worse".
- **Group-balanced sampling helps both architectures** (+2.5 pp worst-group for the MLP, +3.6 pp
  for the bottleneck) and gives the best model overall: 82.8% worst-group, RSR 0.127.
- **Training is unstable under ERM.** Validation worst-group accuracy swings by 30–40 points
  between epochs (below), so early stopping on worst-group accuracy matters more than the number
  of epochs. Balanced sampling makes the curves much smoother.

![Training curves](metrics/training_curves.png)

#### Where the bottleneck looks

The answer token's cross-attention over the 14×14 SigLIP patches, on test images of waterbirds
photographed on land backgrounds (the hardest group). Attention is sparse (normalised entropy
0.50 for the balanced model) and lands on both bird and background patches. The bottleneck makes
this inspectable, but it does not force the model onto the bird.

![Attention maps](metrics/attention_bottleneck_balanced.png)

## Where each result lives

| Experiment | Results | Kaggle script |
| :--- | :--- | :--- |
| Four connectors, Qwen3.5-4B (3–5 seeds) | [`results_study/summary.md`](results_study/summary.md), [`five_seed_mlp_qformer.md`](results_study/five_seed_mlp_qformer.md) | `study/kaggle_kernels/study*` |
| Gate hypotheses H1/H2 (13–15 seeds) | [`results_study/hypotheses.md`](results_study/hypotheses.md), [`h1/`](results_study/h1) | `h1a`, `h1b`, `seeds34` |
| Phi-4-mini (5 seeds) | [`results_study/phi/`](results_study/phi) | `phi-*` |
| SigLIP-large | [`results_study/siglip_large/`](results_study/siglip_large) | `siglip-large` |
| Connector-DFR / JTT | [`results_study/stage3/`](results_study/stage3) | `stage3*` |
| Group-balanced sampling | [`results_study/balanced/`](results_study/balanced) | `extras` |
| CelebA | [`results_study/celeba/`](results_study/celeba) | `celeba*` |
| Ablations (36 tokens, LTD, 8 epochs, frozen LLM) | [`mlp36/`](results_study/mlp36), [`ltd/`](results_study/ltd), [`long8/`](results_study/long8), [`frozen/`](results_study/frozen) | `extras`, `ltd-mlp`, `study-sweep` |
| Linear probes and DFR | [`results_study/linear_baselines.md`](results_study/linear_baselines.md) | `study/linear_baselines.py` |
| Native zero-shot VLM | [`results_study/native_waterbirds.json`](results_study/native_waterbirds.json) | `study/native_vlm.py` |
| Released IDEFICS gate values | [`results_study/pretrained_gates.md`](results_study/pretrained_gates.md) | `study/pretrained_gates.py` |
| Figures | [`docs/figures/final/`](docs/figures/final) (PNG + PDF) | `study/final_figures.py` |

## Related work and where this study stands

Literature check (October 2026): Google Scholar, OpenAlex (covers Springer, Elsevier, IEEE, ACM, ACL,
AAAI and arXiv), forward citations of the closest papers, and full reads of the five nearest. No
paper we found compares connector designs for shortcut reliance in LLM-based VLMs. Numbers below are
as reported by each paper, on their own setups, so they are context rather than a like-for-like ranking.

### Shortcuts in vision-language models

| Work | What they did | Models | Data | Reported result |
| :--- | :--- | :--- | :--- | :--- |
| Bleeker et al., 2024 (arXiv 2402.17510) | Inject synthetic per-pair shortcuts (MNIST digit in image + number in caption) into contrastive training; mitigate with latent target decoding (LTD) or implicit feature modification (IFM) | CLIP (fine-tuned), VSE++ | Flickr30k, MS-COCO retrieval | Flickr30k recall sum: VSE++ 409.0 without shortcuts, 3.4 with them, 274.6 with LTD; CLIP 543.5 without, 436 with, 464 with LTD, 495 with IFM |
| SpurLens (arXiv 2503.08884) | Automatic detection of spurious cues in multimodal LLMs | LLaVA and other MLLMs, zero-shot | Waterbirds and others | LLaVA per-group accuracy: waterbird on land 13.2%, waterbird on water 59.3%, landbird groups 77.4% and 99.8% |
| SpuriVerse (arXiv 2506.18322) | Benchmark of 124 spurious correlations mined from real VQA data | Large VLMs | SpuriVerse | Fine-tuning on spurious cases trades off accuracy on ordinary samples |
| PURGE (arXiv 2608.29996) | Unlearning to remove object-background correlations | LLaVA-1.6-7B, Qwen3-VL-8B, Qwen3.5-9B | Object-background benchmarks | Connector treated as fixed |
| MD3 / MAD (ICLR 2026 submission, OpenReview NTbAH4UD6K) | GPT-4o "critic agents" rewrite biased answers into ~50k debiased reasoning chains, then fine-tune MLLMs | LLaVA-v1.5-7B, LLaVA-Llama-3-8B, Llama-3.2-11B-Vision | VQA-CP, VQA-CE, GQA-OOD; Waterbirds on ResNet-50 | VQA-CP (LLaVA-v1.5-7B): 63.98 → 68.83. Waterbirds WGA (ResNet-50): ERM 63.7, GroupDRO 89.9, DFR 90.2, MOO 91.8, MAD 93.6 |
| Scaling VLMs Is Not Enough (arXiv 2607.28211) | 194 contrastive models, 16 families; which design factors predict robustness | OpenCLIP models | ImageNet, CelebA, UrbanCars | Data size and curation matter more than scale; connector type not studied |

### Connector design

| Work | What they did | Reported result | Shortcuts measured? |
| :--- | :--- | :--- | :--- |
| Modality Integration Rate (arXiv 2410.07167) | Alignment quality of LLaVA-1.5 with an MLP vs a Q-Former connector | MLP clearly outperforms Q-Former after instruction tuning | No |
| Rethinking Projector Training (arXiv 2610.18598) | What projector alignment learns; training-free PORTAL projector | Alignment is coarse; 10-20% of the 558k pairs recover most gains; PORTAL matches a trained MLP with ~135x fewer alignment FLOPs | No |
| The Gate Always Closes (Farazi et al., arXiv 2607.23335) | Learnable gates for injecting signals into frozen VLMs, across five designs | Every gated pathway ends up behaviourally closed; recommends fixed injection | No |
| Honeybee, TokenPacker (IJCV 2025), Cambrian-1 | Locality-aware and efficient projectors | Benchmark accuracy and token efficiency | No |

### Group-robustness baselines (Waterbirds worst-group accuracy, ResNet-50)

| Method | Uses group labels in training | WGA | Source |
| :--- | :--- | :---: | :--- |
| ERM | No | 63.7 | as reported in MD3 |
| JTT | Only for model selection | 86.7 | Liu et al., 2021 |
| GroupDRO | Yes | 89.9 | as reported in MD3 |
| DFR | Small group-balanced set | 90.2 | as reported in MD3 |
| **This repo, MLP projector (Qwen3.5-4B, connector study)** | Only for model selection | **74.5 ± 1.5** | 3 seeds |
| **This repo, MLP + group-balanced sampling (PR #1 run)** | Yes | **82.8** | 1 seed |
| **This repo, DFR on frozen SigLIP-base features** | Small group-balanced set | **84.2 ± 1.5** | 3 seeds |

### Where this study stands out

- **Question nobody isolated:** with the LLM, vision features, data and budget held fixed, does the
  connector (MLP, Q-Former, cross-attention) change how much a generative VLM relies on a shortcut?
  Prior shortcut work studies contrastive models (Bleeker; scaling study) or whole MLLMs (SpurLens,
  SpuriVerse, PURGE, MD3); prior connector work measures benchmark accuracy, not robustness.
- **Group-level evaluation of generative VLMs:** worst-group accuracy, ID-to-OOD gap, a held-out question
  phrasing, and two checkpoint-selection rules, over seeds.
- **A mechanism with a fix:** a learnable injection gate stays shut (0.0005), and that alone turns a
  cross-attention connector from on-par to the most shortcut-reliant and unstable; fixing the gate open
  restores it (+10.6 and +23.8 WGA points over 15 and 13 seeds). This links the gate-suppression effect of Farazi et al. to
  shortcut reliance, which they did not measure.
- **Reference points on the same features:** a linear ERM probe and DFR bracket every VLM, showing how
  much of the gap is the model's, not the features'.
- **Once that pitfall is avoided, connector type is a near-null factor**, consistent with MD3's and the
  scaling study's view that data and training, not architecture, are the main lever.

## Setup

| Component | Choice |
| :--- | :--- |
| Vision encoder | `google/siglip-base-patch16-224`, frozen; 196 patch features (768-d) cached once |
| Language model | `Qwen/Qwen3.5-4B` text decoder, 4-bit NF4 (QLoRA), LoRA r=16 on all attention/MLP projections |
| VLM-SHIELD connector | `LocalFeatureBottleneck`: prompt tokens attend over the 196 patches; output added to the token embeddings through `tanh(gate)`, gate initialised at 0 |
| Baseline connector | LLaVA-style 2-layer MLP over 2×2-pooled patches → 49 prefix tokens |
| Task | Prompt *"Is the bird in this image a landbird or a waterbird?"*; prediction = LLM next-token logits restricted to the two answer tokens |
| Loss | Cross-entropy + 0.1 × normalised attention-entropy penalty `1 − H(A)/log 196` (bottleneck only) |
| Training | Paged 8-bit AdamW, cosine schedule, batch 16, gradient checkpointing, bf16; up to 8 epochs with early stopping on validation worst-group accuracy |
| Variants | ERM (natural sampling) and group-balanced sampling (each bird/background group equally likely) |
| Hardware | One RTX 3050 Laptop GPU (4 GB) |

The 248k-row token embedding matrix (tied with the LM head) is never put on the GPU: the prompt is
fixed, so its embeddings are computed once, and only the two answer-token rows of the LM head are
kept. This, plus 4-bit weights, is what lets a 4B model train in 4 GB.

### Metrics

Groups are (bird, background): `landbird/land`, `landbird/water`, `waterbird/land`, `waterbird/water`.

- **Average accuracy** over the test set.
- **Worst-group accuracy**: the lowest of the four group accuracies (the standard Waterbirds metric).
- **Acc_ID / Acc_OOD**: accuracy on majority groups (bird on its usual background) vs minority groups.
- **Δ_OOD = Acc_ID − Acc_OOD** and **RSR = Δ_OOD / Acc_ID** (relative shortcut reliance).

All numbers are computed by `evaluate.py` from model predictions on the held-out test split.

---

## Experimental protocol (connector study)

| Item | Setting |
| :--- | :--- |
| LLM | Qwen/Qwen3.5-4B, NF4 4-bit (double quantisation), fp16 compute; native vision tower removed |
| Vision features | google/siglip-base-patch16-224, frozen, last layer, 196 patches x 768, cached once |
| Connectors | MLP (LayerNorm-Linear-GELU-Linear, 2x2 patch pooling, 49 tokens, 8.5M); Q-Former (32 queries, 8 heads, 1 block, 11.4M); cross-attention (attn dim 1024, 8 heads, 8.9M) with a learned zero-initialised tanh gate or a fixed gate with zero-initialised output |
| Stage 1 (alignment) | 10k COCO caption pairs (no task val/test images), 300 steps, batch 16, AdamW lr 1e-3, 3% warmup + cosine, LLM frozen |
| Stage 2 (task) | LoRA r 16, alpha 32, dropout 0.05 on attention, linear-attention and MLP projections; PagedAdamW8bit, lr 2e-4 (LoRA and connector), weight decay 0.01, batch 16, 4 epochs, 5% warmup + cosine, grad clip 1.0, early stop after 2 epochs without val WGA gain |
| Prompting | Chat template; answer = first token of the class word; 2 training question phrasings, 1 held out for test |
| Checkpoint selection | Best val worst-group accuracy (uses group labels); best val average accuracy reported as the label-free alternative |
| Stage 3 (connector-only) | From the selected stage-2 model, LLM and LoRA frozen. DFR-style: group-balanced half of val, other half selects. JTT-style: training set with the stage-2 model's errors repeated 20x. 1 epoch |
| Seeds | 3 per configuration (5 planned); seeds vary data order, LoRA init and stage-2 sampling; per-seed alignment in a separate run |
| Hardware | Kaggle 2x NVIDIA T4 (16 GB); one job per GPU |

## Statistics

- Point estimates are means over seeds of test worst-group accuracy, averaged over the three question phrasings.
- Pairwise comparisons ([`results_study/significance_wga.md`](results_study/significance_wga.md)): 95% intervals
  from 2,000 stratified bootstrap resamples of the test set, and a **hierarchical bootstrap** that also resamples
  seeds. Significance uses the hierarchical p-value with Holm-Bonferroni correction within each task.
- The gate hypotheses were declared before the seed-extension runs and tested with one-sided paired t-tests,
  Wilcoxon signed-rank tests and variance F-tests ([`study/hypotheses.py`](study/hypotheses.py)).
- Apart from the gate (and DFR on Watermark), connector differences stay inside training variance.

## Reproducibility

- Every Kaggle run writes `environment.txt` (exact `pip freeze`, GPU driver, git commit) next to its results.
- Data splits are deterministic (`study/build_tasks.py`, seed 0); Waterbirds uses the official splits.
- `study/kaggle_kernels/` holds the exact scripts for every run; runs resume from exported results.
- Unit tests: `python -m pytest tests` (18).

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python scripts/prepare_waterbirds.py          # Waterbirds -> data/waterbirds/{images,metadata.csv}
python scripts/download_models.py             # SigLIP + Qwen3.5-4B into the HF cache
python scripts/cache_siglip_features.py       # frozen SigLIP features -> data/siglip_patch_features.npy

# One run
python train.py --mode bottleneck --balanced --gradient_checkpointing

# All four runs (MLP / bottleneck x ERM / balanced), 110 min budget each, then evaluation
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/run_all.sh Qwen/Qwen3.5-4B 110

python evaluate.py                            # metrics/*.csv|json|md + figures
python -m pytest tests                        # unit tests
python app.py --run bottleneck_balanced       # Gradio demo with attention heatmap
```

## Repository layout

```
src/data/waterbirds.py          dataset, cached-feature loader, group metrics
src/models/debiased_vlm.py      LocalFeatureBottleneck, MLPPrefixProjector, DebiasedVLM (QLoRA Qwen3.5)
src/models/*_scratch.py         small pure-PyTorch SigLIP/Qwen re-implementations (tests / teaching)
src/models/lora.py              minimal LoRA layer used by the scratch models
src/training/loss.py            task loss + attention entropy / shortcut-suppression regularisers
src/training/engine.py          train / eval loops, group-balanced sampler
scripts/                        data preparation, feature caching, model download, run_all.sh
train.py / evaluate.py / app.py training, evaluation + figures, demo
metrics/                        exported results and figures
src/study/, study/              connector study: connectors, generic VLM wrapper, tasks, training, plots
results_study/, docs/figures/        connector-study results; docs/figures/final has the summary figures
```

## Limitations

- Waterbirds study at the top: one seed per configuration; differences of a few points are within run-to-run noise.
- Connector study:
  - 13–15 seeds for the gate comparison, 3–5 for most other cells; SigLIP-large and the 36-token MLP have 2
    seeds, CelebA's fixed gate 2, the 8-epoch runs 1. The learning-rate sweep did not finish within Kaggle's
    session limit, so the learning rate (2e-4) is not tuned.
  - Two 4-bit LLMs of 4B parameters or fewer; results may differ for larger or full-precision models.
  - The Watermark shortcut is partly textual (readable watermark), and Qwen3.5 was pretrained with its own
    vision merger, which our connectors replace.
  - Seeds vary only the task stage; stage-1 alignment ran once per connector (`--align_per_seed` now exists).
  - LoRA also adapts the LLM, which can learn the shortcut whatever the connector; a frozen-LLM run
    (`--freeze_llm`) separates the two.
  - Alignment is short (300 steps on 10k captions; LLaVA uses 558k pairs), and the MLP uses 2x2-pooled
    patches (49 tokens) instead of all 196 for speed.
  - Average accuracy is the plain test mean, not reweighted by training group sizes as in some papers.
  - Checkpoints are selected with group labels (val WGA); the label-free rule is in the summary tables.
- Waterbirds has no shortcut-region masks in this setup, so the shortcut-suppression term of the
  loss is implemented and tested but not used in these runs.
- The vision encoder is frozen; only LoRA adapters and the connector are trained.

## Citation

If you use this code or the findings, please cite the repository:

```bibtex
@misc{vlmdebias2026,
  title  = {Does the Vision-Language Connector Cause Shortcut Learning? A Controlled Study of Injection Gates in LLM-based VLMs},
  author = {{VLM-Debias contributors}},
  year   = {2026},
  howpublished = {\url{https://github.com/Naveenvaidani/VLM-Debias}}
}
```
