# Literature review: sources of this project

This review lists the work the connector study is built on, grouped by theme. For each paper: **what they did**,
**what we learned** from it, and **what we used** in this repository. The last section states the gap this
study fills and how its claims relate to prior results.

Identifiers are arXiv numbers unless stated. Papers marked † were checked in full during the October 2026
literature search (Google Scholar, OpenAlex, forward citations); numbers quoted from them are as reported by
the authors, on their own setups.

---

## 1. Shortcut learning and spurious correlations

| Paper | What they did | What we learned | What we used |
| :--- | :--- | :--- | :--- |
| Geirhos et al., 2020, *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence (2004.07780) | Framed shortcut learning: models use decision rules that work on the training distribution but fail under shift. | A model can be accurate on average while solving the task the wrong way; only tests that break the shortcut reveal it. | The study's framing, and the choice of tasks where the shortcut is known and can be broken at test time. |
| Sagawa et al., 2020, *Distributionally Robust Neural Networks for Group Shifts* (GroupDRO), ICLR (1911.08731) | Introduced the Waterbirds benchmark (CUB birds pasted on Places backgrounds) and GroupDRO, which minimises the worst group's loss; showed strong regularisation is needed. | Worst-group accuracy (WGA) is the right metric; average accuracy hides minority-group failure; group labels in training help a lot. | Waterbirds with the official splits; WGA as the primary metric; the (label, background) group definition; group-balanced sampling as a group-label upper reference. |
| Liu et al., 2021, *Just Train Twice* (JTT), ICML (2107.09044) | Train once with ERM, upweight the examples it got wrong, train again; group labels needed only for model selection. | Errors of an ERM model are a usable proxy for minority groups. | Connector-JTT in stage 3 (error set repeated 20x, connector only). It did not help here (1 seed). |
| Kirichenko, Izmailov & Wilson, 2023, *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations* (DFR), ICLR (2204.02937) | Showed that ERM features already contain the core signal; retraining only the last layer on a small group-balanced set recovers most of the robustness. | Robustness failures are often in how features are read out, not in the features. | (1) Linear DFR on our frozen SigLIP features as a reference point. (2) **Connector-DFR**: we apply the DFR idea to the connector, retraining only it on a balanced half of validation; best Watermark result (89.6). |
| Izmailov et al., 2022, *On Feature Learning in the Presence of Spurious Correlations*, NeurIPS (2210.11369) | Measured how much core information ERM features keep, using DFR as a probe; studied architecture and pretraining effects. | Feature quality and the readout must be separated when judging a method. | Motivated running linear probes and DFR on exactly the features the VLM sees, so the VLM's gap can be attributed to the model rather than the features. |
| Idrissi et al., 2022, *Simple Data Balancing Achieves Competitive Worst-Group-Accuracy*, CLeaR (2110.14503) | Showed that subsampling or reweighting to balance groups rivals complex robust-training methods. | Simple group balancing is a strong baseline that any new method must be compared against. | Group-balanced sampling in stage 2 and a group-balanced linear probe, both labelled as using group labels. |
| Koh et al., 2021, *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML (2012.07421) | A suite of real distribution-shift benchmarks with standard evaluation protocols. | Report ID and OOD accuracy separately, plus the gap between them. | The ID / OOD accuracy split and the ID-to-OOD gap reported for every run. |
| Liu et al., 2015, *Deep Learning Face Attributes in the Wild* (CelebA), ICCV (1411.7766) | Large face-attribute dataset; Blond_Hair vs Male later became a standard spurious-correlation benchmark. | A natural, not synthetic, spurious correlation (blond hair correlates with gender). | Third task, official partitions. |

## 2. Shortcuts and robustness in vision-language models

| Paper | What they did | What we learned | What we used |
| :--- | :--- | :--- | :--- |
| Bleeker et al., 2024, *Demonstrating and Reducing Shortcuts in Vision-Language Representation Learning* † (2402.17510) | Injected synthetic shortcuts into image-caption training of CLIP and VSE++; proposed latent target decoding (LTD) and implicit feature modification to reduce them. Flickr30k recall sum for VSE++ falls from 409.0 to 3.4 with shortcuts; LTD recovers 274.6. | Synthetic, controllable shortcuts make causal claims possible; a caption-reconstruction objective can push models to keep more of the image. | The controlled-shortcut design of our Watermark task; an LTD-style caption loss as an ablation (no gain: 75.9 vs 74.9 WGA). |
| SpurLens, 2025 † (2503.08884) | Automatic discovery of spurious cues in multimodal LLMs, zero-shot. | Off-the-shelf MLLMs such as LLaVA fail badly on Waterbirds minority groups (waterbird on land 13.2%). | Motivated our native zero-shot baseline (Qwen3.5-4B with its own vision tower: 35.5 WGA after calibration). |
| SpuriVerse, 2025 † (2506.18322) | Benchmark of 124 spurious correlations mined from real VQA data for large VLMs. | Fine-tuning on spurious cases trades off ordinary accuracy. | Context for reporting average accuracy next to WGA. |
| PURGE, 2026 † (2608.29996) | Unlearning to remove object-background correlations in LLaVA-1.6, Qwen3-VL and Qwen3.5-9B. | Prior VLM debiasing treats the connector as fixed. | Part of the evidence that connector design is unexamined. |
| MD3 / MAD, ICLR 2026 submission † (OpenReview NTbAH4UD6K) | GPT-4o critic agents rewrite biased answers into about 50k debiased reasoning chains for fine-tuning MLLMs; also reports ResNet-50 Waterbirds baselines (ERM 63.7, GroupDRO 89.9, DFR 90.2). | Data, not architecture, is the usual lever; standard group-robustness baselines for comparison. | Baseline numbers for the related-work table. |
| *Scaling Vision-Language Models Is Not Enough*, 2026 † (2607.28211) | 194 contrastive models from 16 families; which design factors predict robustness. | Data size and curation matter more than scale; connector type was not studied. | Supports our finding that connector type is a near-null factor once the gate pitfall is avoided. |

## 3. Vision-language connectors

| Paper | What they did | What we learned | What we used |
| :--- | :--- | :--- | :--- |
| Alayrac et al., 2022, *Flamingo*, NeurIPS (2204.14198) | Frozen LM with interleaved gated cross-attention layers; each block adds `tanh(α)·xattn` with `α` initialised at 0 so the LM starts unchanged. | Zero-initialised tanh gates keep the pretrained LM intact at the start of training. | The **learned-gate cross-attention connector** is this design; it is the object of the main hypothesis. |
| Laurençon et al., 2023, *OBELICS* / IDEFICS (2306.16527) | Open web-scale interleaved image-text dataset and the open Flamingo reproduction IDEFICS (9B, 80B). | Released checkpoints let us inspect trained gate values directly. | We read the gate tensors of IDEFICS 9B/80B from their checkpoints: 9 of 20 cross-attention gates in 80B have \|tanh α\| < 0.05. |
| Li et al., 2023, *BLIP-2*, ICML (2301.12597) | Q-Former: a small set of learned queries cross-attends to frozen image features and feeds the LLM. | A fixed number of learned queries compresses the image into few tokens. | The **Q-Former connector** (32 queries). |
| Liu et al., 2023, *Visual Instruction Tuning* (LLaVA), NeurIPS (2304.08485); Liu et al., 2024, *Improved Baselines* (LLaVA-1.5), CVPR (2310.03744) | Linear (then two-layer MLP) projection of vision patches into the LLM's token space; two-stage training (caption alignment, then instruction tuning). | A simple MLP projector is a strong default; two-stage training is standard. | The **MLP projector connector** and the two-stage protocol (caption alignment with the LLM frozen, then task training). |
| Cha et al., 2024, *Honeybee: Locality-enhanced Projector*, CVPR (2312.06742) | Locality-aware projectors that keep spatial structure while reducing tokens. | Token count and locality are design axes of connectors. | Motivated pooling the MLP's patches (2x2, 49 tokens) and the 36-token ablation (no change: 74.7 vs 74.9). |
| Li et al., 2024, *TokenPacker* (2407.02392) | Efficient visual projector compressing image tokens. | Same as above, measured on benchmark accuracy and efficiency only. | Context: connector work measures accuracy, not robustness. |
| Tong et al., 2024, *Cambrian-1* (2406.16860) | Vision-centric study of encoders and connectors for MLLMs (Spatial Vision Aggregator). | Encoder choice matters a lot for MLLM quality. | Motivated the SigLIP-base vs SigLIP-large comparison. |
| *Modality Integration Rate* †, 2024 (2410.07167) | Measured alignment quality of LLaVA-1.5 with MLP vs Q-Former connectors. | MLP clearly beats Q-Former after instruction tuning on benchmarks; robustness was not measured. | A prediction to test: on shortcut tasks we find no significant MLP vs Q-Former difference. |
| *Rethinking Projector Training* †, 2026 (2610.18598) | What projector alignment learns; training-free PORTAL projector. | Alignment is coarse; 10–20% of LLaVA's 558k pairs recover most gains. | Justified our short alignment (300 steps on 10k COCO captions) as a reasonable approximation. |
| Farazi et al., 2026, *The Gate Always Closes* † (2607.23335) | Studied learnable gates for injecting signals into frozen VLMs across five designs. | Every learnable gated pathway ends up behaviourally closed; they recommend fixed injection. Shortcut reliance was not measured. | **Direct basis of hypothesis H1.** We link gate closure to shortcut reliance and test the recommended fix (fixed gate with zero-initialised output). |
| Bachlechner et al., 2021, *ReZero is All You Need*, UAI (2003.04887) | Residual branches scaled by a learnable scalar initialised at zero. | Zero-initialised gates start as the identity but rely on the gradient to open them. | Contrast for our fixed-gate design, which gets the identity start from a zero-initialised output layer instead of a closed gate. |

## 4. Models, encoders and efficient training

| Paper | What they did | What we learned | What we used |
| :--- | :--- | :--- | :--- |
| Zhai et al., 2023, *Sigmoid Loss for Language Image Pre-Training* (SigLIP), ICCV (2303.15343) | Contrastive image-text pretraining with a pairwise sigmoid loss. | Strong frozen patch features at small scale. | Frozen `siglip-base-patch16-224` (196 × 768) and `siglip-large-patch16-256` (256 × 1024) features, cached once. |
| Qwen team, 2026, *Qwen3.5* (model card, `Qwen/Qwen3.5-4B`) | Open LLM family with native vision support. | A 4B model fits a T4 in 4-bit. | Main LLM; its native vision tower is removed for the study and used as-is for the zero-shot baseline. |
| Microsoft, 2025, *Phi-4-Mini Technical Report* (2503.01743) | 3.8B open language model. | A different architecture and tokenizer for replication. | Second LLM, to check that the gate effect is not Qwen-specific. |
| Hu et al., 2022, *LoRA*, ICLR (2106.09685) | Low-rank adapters for fine-tuning large models. | Fine-tune a large LLM cheaply while the base stays frozen. | LoRA r = 16, alpha 32 on attention and MLP projections. |
| Dettmers et al., 2023, *QLoRA*, NeurIPS (2305.14314) | 4-bit NF4 quantisation with double quantisation, paged optimisers, LoRA on top. | 4B-scale fine-tuning fits on 16 GB GPUs. | NF4 + double quantisation, paged 8-bit AdamW. |
| Chen et al., 2016, *Training Deep Nets with Sublinear Memory Cost* (1604.06174) | Gradient checkpointing. | Trade compute for memory. | Enabled for the SigLIP-large runs after an out-of-memory failure. |
| Lin et al., 2014, *Microsoft COCO* (1405.0312) | Image dataset with captions and object labels. | Captions for alignment; object labels for a controllable task. | Stage-1 alignment captions and the Watermark task images ("is there a dog?"). |

## 5. Statistics and reporting

| Source | What it gives | What we used |
| :--- | :--- | :--- |
| Dror et al., 2018, *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL | When to use paired tests, bootstrap and non-parametric tests. | Paired t-test and Wilcoxon signed-rank test over seeds for H1. |
| Efron & Tibshirani, 1993, *An Introduction to the Bootstrap* | Bootstrap confidence intervals. | Stratified bootstrap over test images and a hierarchical bootstrap over seeds and images. |
| Holm, 1979, *A Simple Sequentially Rejective Multiple Test Procedure*, Scand. J. Statistics | Family-wise error control. | Holm–Bonferroni correction within each task for all connector pairs. |
| Schuirmann, 1987, two one-sided tests (TOST), J. Pharmacokinetics and Biopharmaceutics | Equivalence testing. | ±3-point equivalence margin for the "connector type does not matter" claim. |
| Variance F-test (classical) | Compare variances. | H2: the learned gate varies more across seeds. |

---

## 6. The gap and how this study fills it

**What prior work established**
- Shortcut learning is measured with worst-group accuracy (Sagawa et al.).
- The features usually contain the core signal, and readout fixes such as DFR recover it (Kirichenko et al.; Izmailov et al.).
- Learnable injection gates in VLMs tend to close (Farazi et al.).
- MLLMs fail on spurious-correlation benchmarks (SpurLens; SpuriVerse).

**What was missing**
- No study holds the LLM, features, data and budget fixed and varies **only the connector**, measuring shortcut reliance with group metrics over many seeds.
- Gate closure had not been linked to shortcut reliance.
- Connector studies (Modality Integration Rate; Honeybee; TokenPacker; Cambrian-1) report benchmark accuracy, not robustness.

**What this study adds**

1. **A controlled connector comparison for shortcut reliance in generative VLMs:**
   - three connector families;
   - two LLMs;
   - two encoder sizes;
   - three tasks;
   - up to 15 seeds per arm.
2. **A causal link from gate closure (Farazi et al.) to shortcut reliance.** Fixing the gate raises WGA by +10.6 and +23.8 points, with pre-declared tests.
3. **Connector-DFR:** the DFR readout idea moved from the classifier head to the connector.
   - It gives 89.6 WGA on Watermark, above linear DFR on the same features (85.1).
   - It gives no gain on Waterbirds.
4. **Evidence at scale:** 9 of 20 IDEFICS-80B cross-attention gates are near closed.

**What a reviewer could ask, and where it is answered**

| Question | Where answered |
| :--- | :--- |
| Seed variance | 13–15 seeds per arm; H2 tested; per-seed figures |
| Token-count confound | 36-token MLP ablation |
| Encoder confound | SigLIP-large runs |
| LLM confound | Phi-4-mini replication |
| Group-label use | Every method labelled by whether it uses group labels |
| Ceiling from the features | Linear ERM and DFR probes on the same features |

**Limitations relative to the literature**
- Models are 4B scale or smaller.
- SigLIP-large has 2 seeds.
- The learning rate was not tuned.
- The ResNet-50 group-robustness numbers (GroupDRO 89.9, DFR 90.2) come from a different setting and are context, not a head-to-head comparison.
