# Paper outline: learnable injection gates induce shortcut reliance in LLM-based VLMs

Working title: *A learnable injection gate causes shortcut reliance in vision-language models; fixing the gate restores group robustness*

## Claims (each tied to a pre-declared test)

1. **H1:** a fixed (open) injection gate has higher worst-group accuracy than a learnable tanh gate
   (Waterbirds +10.6 [5.3, 16.0], Watermark +23.8 [12.5, 35.1]; paired t p < 0.001; 13–15 seeds per arm).
2. **H2:** the learnable gate is less stable across seeds (variance F-test p = 9e-5 Waterbirds, 0.02 Watermark;
   8 of 13 Watermark seeds collapse below 50%, none for the fixed gate).
3. **Mechanism:** the learnable gate stays near zero after alignment (0.0005) and its gradient flips sign every step.
4. **Generality:** the effect holds for a second LLM (Phi-4-mini, 5 seeds: Waterbirds +9.7, p = 0.005; Watermark +31.6, p = 0.044, misses α = 0.025).
7. **Encoder scale:** with SigLIP-large the learned gate still shuts (0.0006) but no longer collapses on Waterbirds (79.3 vs 80.1, 2 seeds).
5. **Connector type does not matter once the gate is fixed:** MLP vs Q-Former, five seeds, no significant difference.
6. **A label-free fix:** connector-only retraining on a balanced validation half (DFR-style) raises Watermark
   worst-group accuracy to 89.6 ± 0.5 (fixed gate, 5 seeds); no gain on Waterbirds (73.9 → 74.1); the caption loss rises little.
   Group-balanced sampling is the strongest Waterbirds method (84.7) but uses group labels; report both, clearly labelled.

## Negative results (report)
- Attention-entropy regularizer: no gain.
- Caption-embedding (latent target decoding) loss: no gain for the fixed gate.
- Frozen LLM: learned gate collapses (24.2), MLP and Q-Former do not (73–75).

## Structure
1. **Introduction:** shortcuts in VLMs; connector design is an unexamined factor; contributions (1–6 above).
2. **Related work:** shortcut studies (Bleeker et al. 2024, SpurLens, SpuriVerse, PURGE, MD3); connectors (LLaVA, BLIP-2,
   Flamingo, Honeybee); gate suppression (Farazi et al. 2026); last-layer retraining (DFR, Yang et al. 2023,
   class-balanced LLR); seed variance in group robustness.
3. **Setup:** Qwen3.5-4B (NF4 + LoRA) and Phi-4-mini; frozen SigLIP-base features; three connectors plus the gated variant;
   two-stage training; tasks: Waterbirds (official), Watermark (ours), CelebA (Blond_Hair vs Male, official partitions).
4. **Pre-registration:** H1, H2 and the equivalence margin (±3 points) written before the seed-extension runs.
5. **Results:** H1/H2 table; mechanism; second LLM; connector comparison; fix (connector-DFR) with cost.
6. **Analysis:** attention (both gate variants attend to the watermark; attention alone does not show reliance);
   failure analysis (waterbirds on land is hardest); compute table.
7. **Limitations:**
   - 3–5 seeds for most cells; CelebA has 1–3 seeds.
   - SigLIP-base only; Watermark text is readable (a partly textual shortcut); Qwen was pretrained with its own vision merger.
   - Unequal image-token counts (MLP 49, Q-Former 32, cross-attention 0).
   - Single alignment per seed in most runs; learning-rate sweep incomplete.
   - Balanced sampling needs group labels.
8. **Conclusion.**

## Paper kit
Title/abstract, introduction, methodology, results and discussion drafts, verified references and the literature
matrix are in this folder; the literature review is `docs/LITERATURE.md`.

## Still needed before submission
- Learning-rate sweep: did not finish (Kaggle session time); 8-epoch run has 1 seed. SigLIP-large has 2 of 3 seeds.
- CelebA fixed gate has 2 seeds (4 for the others).
- Calibrated zero-shot native baseline: done (35.5 WGA).
- Figures as vector PDF; final statistics table.
- Code release: license, citation file, pinned requirements from `environment.txt`.
