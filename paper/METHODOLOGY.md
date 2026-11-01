# Methodology

*Structured to the Q1 blueprint: design → data → variables and instruments → procedure → analysis →
validity and ethics. Each step follows the "action → justification → citation" pattern. Citation keys refer to
[`references.bib`](references.bib).*

---

## 1. Research design and paradigm

**Design.**
- The study is a **controlled, between-architecture experiment with repeated randomised runs**: a full
  factorial over connector × task with seeds as replicates, plus targeted follow-ups.
- Only one factor varies at a time: the vision-language connector, or one specified follow-up factor.
- Everything else is held fixed: the LLM, the frozen vision features, the data, the prompt, the optimiser and
  the training budget.

**Why this design.**
- The research question is causal: *does the connector change shortcut reliance?*
- Observational comparisons of released VLMs cannot answer it, because released models differ in data, scale,
  encoder and training recipe at the same time.
- Design studies of VLMs [mckinzie2024mm1; karamcheti2024prismatic] show that only controlled ablations can
  attribute effects to one component.
- Seed-level replication is required because a single seed can reverse conclusions in deep learning
  [henderson2018matters; picard2021seed; bouthillier2021variance].

**Paradigm.**
- The study is positivist and hypothesis-testing.
- The primary hypotheses (H1, H2), their tests, the significance level (one-sided α = 0.025) and the
  equivalence margin (±3 WGA points) were written down before the seed-extension runs (see
  [`OUTLINE.md`](OUTLINE.md) and [`study/hypotheses.py`](../study/hypotheses.py)).
- Analyses beyond these are labelled exploratory.

## 2. Data sources, sampling and inclusion criteria

| Task | Source | Shortcut (spurious attribute *a*) | Label *y* | Train | Val | Test |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| Waterbirds | CUB-200 birds on Places backgrounds [wah2011cub; zhou2018places], official splits [sagawa2020groupdro] | background (land / water) | landbird / waterbird | 4,795 | 1,199 | 5,794 |
| Watermark (ours) | MS-COCO images [lin2014coco] | stock-photo text watermark in the corner | dog present | 4,800 (95% shortcut-aligned) | 1,200 (group-balanced) | 2,400 (group-balanced) |
| CelebA | CelebA, official partitions [liu2015celeba] | gender (Male) | Blond_Hair | 8,000 (stratified subsample) | 5,000 (≥ 100 per group) | full official test |
| Alignment captions | MS-COCO captions | – | – | 10,000 pairs | – | – |

**Sampling.**
- *Waterbirds:* official splits, kept unchanged for comparability with the group-robustness literature
  [sagawa2020groupdro; kirichenko2023dfr].
- *Watermark:*
  - Training uses stratified sampling with 95% of examples shortcut-aligned (watermark ↔ dog), the same
    strength as Waterbirds.
  - Validation and test are **group-balanced**, so every group is measured with equal precision.
  - All splits are deterministic (`study/build_tasks.py`, seed 0).
- *CelebA:* train and validation are stratified subsamples that preserve group proportions, to fit the
  compute budget; the official test set is used in full.

**Inclusion and exclusion.**
- Alignment captions exclude every image in any task's validation or test split, to prevent leakage.
- The Watermark label comes from the COCO object annotations: an image is positive if any annotation is
  "dog".
- No run was excluded after the fact. Runs that crashed (out-of-memory, session limits) were re-run with the
  same seed, and every finished run is reported (`results_study/*/all_runs.csv`).

**Power analysis (sample size = seeds).**
- The unit of replication is the training seed. Each seed varies data order, LoRA initialisation and
  stage-2 sampling [bouthillier2021variance].
- Power for the one-sided paired t-test at α = 0.025 was computed from the non-central t distribution
  [cohen1988power].

| Paired effect size d_z | Power with 3 seeds | 5 seeds | 10 seeds | 15 seeds | Seeds for 80% power |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.8 (large) | 0.13 | 0.28 | 0.62 | 0.82 | 15 |
| 1.0 | 0.18 | 0.40 | 0.80 | 0.95 | 10 |
| 1.5 | 0.32 | 0.71 | 0.99 | 1.00 | 6 |
| 2.0 | 0.47 | 0.91 | 1.00 | 1.00 | 5 |

The initial 3-seed design had power below 0.5 even for large effects. The gate comparison was therefore
extended to **15 (Waterbirds) and 13 (Watermark) seeds**, enough for 80% power at d_z ≥ 0.8. Observed effects
(Section 4 of the results) are d_z = 1.10 and 1.27, giving achieved power of 0.98 and 0.99.

## 3. Variables and instrumentation

**Independent variable: connector**, four levels:

| Level | Design | Image tokens | Trainable params |
| :--- | :--- | :---: | :---: |
| MLP projector [liu2024llava15] | LayerNorm–Linear–GELU–Linear over 2×2-pooled patches | 49 | 8.5M |
| Q-Former [li2023blip2] | 32 learned queries, 8 heads, 1 block | 32 | 11.4M |
| Cross-attention, learned gate [alayrac2022flamingo] | prompt tokens attend to patches; `e ← e + tanh(α)·attn`, α₀ = 0 | 0 | 8.9M |
| Cross-attention, fixed gate | identical block; gate fixed open, output layer zero-initialised | 0 | 8.9M |

The two cross-attention levels have the **same parameters and FLOPs**; only whether `α` is trainable
differs. Both start as the identity map. This isolates the gate as the manipulated mechanism and removes
capacity as a rival explanation.

**Dependent variables.**
- **Primary: test worst-group accuracy (WGA)** [sagawa2020groupdro].
  - Mean over three question phrasings, one of them never seen in training.
  - Checkpoint chosen by validation WGA.
- **Secondary:**
  - average accuracy;
  - ID and OOD accuracy and their gap [koh2021wilds];
  - balanced accuracy;
  - seed variance of WGA (H2);
  - WGA with the label-free checkpoint rule (validation average accuracy).
- **Mechanism measures:** the learned gate value `tanh(α)` and its gradient, logged every 10 alignment steps.
  Also the share of attention on the watermark corner.

**Controlled variables, and why each is controlled.**
- **LLM:** Qwen3.5-4B [qwen2026qwen35], NF4 4-bit with LoRA r = 16 [dettmers2023qlora; hu2022lora].
  Controlled because LLM capacity and pretraining change shortcut reliance [sagawa2020overparam]. Varied once
  (Phi-4-mini [abouelenin2025phi4mini]) to test generality.
- **Vision features:** frozen SigLIP [zhai2023siglip], cached once, so every connector receives
  bit-identical inputs. Encoder choice strongly affects MLLMs [tong2024cambrian; tong2024eyes]. Varied once
  (SigLIP-large).
- **Data, prompts and answer read-out:**
  - The same splits, the same chat template and the same 2 training phrasings for every connector.
  - The prediction is the more likely of the two answer tokens, which removes free-text parsing as a
    confounder.
- **Optimisation and budget:**
  - *Stage 1:* 300 alignment steps on 10k COCO captions with the LLM frozen.
  - *Stage 2:* 4 epochs, batch 16, learning rate 2e-4, cosine schedule.
  - The same for every connector, with early stopping after 2 epochs without validation-WGA gain.
  - The budget is equalised in steps, not wall-clock time, so faster connectors gain no extra training.
- **Image-token count:** differs by design (49 / 32 / 0). It is checked with a 36-token MLP ablation,
  because token count is a known design factor [cha2024honeybee; mckinzie2024mm1].

## 4. Procedure

1. **Data preparation** (`study/build_tasks.py`, `study/cache_features.py`): build splits and cache frozen
   patch features (one Kaggle CPU/GPU session per encoder).
2. **Stage 1, alignment** (`study/align.py`): train only the connector on COCO captions, with the LLM frozen.
   This follows the two-stage practice of [liu2023llava; liu2024llava15]. 300 steps suffice because alignment
   is coarse and data-efficient [groger2026rethinking].
3. **Stage 2, task training** (`study/train_task.py`): train LoRA and the connector on each task under ERM
   with natural (shortcut-biased) sampling.
4. **Stage 3, optional connector-only fixes:** freeze the LLM and LoRA, then retrain only the connector:
   - **Connector-DFR:** on a group-balanced half of validation, the other half selecting the model
     [kirichenko2023dfr].
   - **Connector-JTT:** stage-2 errors upweighted ×20 [liu2021jtt].
5. **Evaluation:**
   - Test metrics for each phrasing.
   - Per-image predictions saved (`arrays.npz`) for the bootstrap.
   - `environment.txt` (exact package versions and GPU) written next to every result.
6. **Execution:**
   - Runs on Kaggle (2× NVIDIA T4, 16 GB), one job per GPU.
   - Jobs are resumable; results are exported after every run.
   - Seeds were assigned before the runs; no run was selected after seeing its result.

**Bias-minimisation steps.**
- Identical code paths for all connectors: one `train_task.py` with a `--connector` flag.
- A held-out question phrasing is used to detect prompt over-fitting.
- Two checkpoint-selection rules are reported: one uses group labels, one does not.
- Group-label use is stated for every method.
- The full per-seed results are released.

## 5. Analysis strategy

| Question | Test | Why this test |
| :--- | :--- | :--- |
| H1: fixed > learned gate | One-sided **paired t-test** over seeds, with a **Wilcoxon signed-rank** test as the assumption-free check [dror2018hitchhiker] | Seeds pair the two arms (same data order and initialisation), removing between-seed variance |
| H2: learned gate more variable | One-sided **variance-ratio F-test**, with the **Brown–Forsythe** test as a robust check | The F-test is sensitive to non-normality; Brown–Forsythe is not |
| Pairwise connector differences | **Stratified bootstrap** of test images (2,000 resamples) and a **hierarchical bootstrap** over seeds and images [efron1993bootstrap]; **Holm–Bonferroni** correction within each task [holm1979simple] | The hierarchical bootstrap covers training variance, not just test-sample noise |
| "Connector type does not matter" | **Two one-sided tests (TOST)**, margin ±3 points [schuirmann1987tost] | A non-significant difference is not evidence of equivalence |
| Effect size | Paired **Cohen's d_z** (mean difference / s.d. of differences), plus d_av for comparison across designs [cohen1988power] | p-values alone do not show practical importance |

**Assumption checks (reported in full).**
- *Normality of the paired differences (Shapiro–Wilk):*
  - Violated for Qwen Waterbirds (p = 0.009), which is why the Wilcoxon test is reported alongside;
    it agrees (p = 3e-5).
  - Not violated elsewhere (p ≥ 0.34).
- *Normality of the learned-gate WGA:* violated on Qwen Waterbirds (p = 0.003), driven by the collapsed
  seeds. The robust Brown–Forsythe test is therefore reported for H2.
- *Independence:* each seed is a separate training run with its own random state. The two arms share only
  the seed value, which is the intended pairing.

## 6. Validity, reliability and ethics

**Internal validity.**
- One factor varies at a time, and the two gate arms are identical except for the trainability of `α`.
- Pairing by seed controls initialisation and data order.
- Rival explanations are tested directly: capacity (identical parameters), token count (36-token
  ablation), encoder (SigLIP-large), LLM (Phi-4-mini) and training length (8 epochs).

**Construct validity.**
- Shortcut reliance is measured behaviourally, as WGA and the ID–OOD gap on test sets where the shortcut is
  broken, not inferred from attention. Attention is not explanation [jain2019attention].
- The mechanism is measured directly: the logged gate value and its gradient.

**External validity.**
- Coverage:
  - three tasks: a natural background shortcut, a synthetic text shortcut and a demographic attribute;
  - two LLM families;
  - two encoder sizes;
  - released 9B and 80B IDEFICS checkpoints, inspected for gate closure.
- Bounds:
  - models are 4B scale or smaller and 4-bit;
  - the vision encoder is frozen;
  - the alignment budget is short.

**Reliability and replicability.**
- Deterministic splits.
- Exact scripts for every run in [`study/kaggle_kernels/`](../study/kaggle_kernels/).
- Pinned requirements ([`requirements-study.txt`](../requirements-study.txt)) and an `environment.txt` per run.
- 18 unit tests.
- All per-seed results and figure scripts.

**Ethics.**
- All data are public research datasets used under their terms: Waterbirds/CUB, Places, MS-COCO, CelebA.
- No new human-subject data were collected, so no IRB approval was required.
- CelebA contains face images and a binary gender attribute. It is used only as the standard
  group-robustness benchmark [sagawa2020groupdro]; no identity inference is performed.
- Compute: about 150 T4 GPU-hours of task training across 222 logged runs (from `train_minutes`), plus data preparation, alignment and evaluation, on Kaggle free-tier accounts.
