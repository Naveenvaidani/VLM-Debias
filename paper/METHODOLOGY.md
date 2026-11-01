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
extended to **15 seeds per arm on both tasks**, enough for 80% power at d_z ≥ 0.8. Observed effects
(Section 4 of the results) are d_z = 1.10 and 1.34, giving achieved power of 0.98 and > 0.99. Phi-4-mini
Watermark (d_z ≈ 1.0, power 0.40 at 5 seeds) was extended to 10 seeds *after* its 5-seed test missed
α = 0.025 (p = 0.044). This data-dependent extension is disclosed, and the new seeds 5–9 are also tested on
their own as fresh data (p = 0.023).

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
  - average accuracy: the plain mean over test images. It is *not* reweighted to the training-group
    frequencies, as some Waterbirds papers do [sagawa2020groupdro], so it is not directly comparable to their
    "average accuracy";
  - ID and OOD accuracy and their gap [koh2021wilds];
  - balanced accuracy;
  - seed variance of WGA (H2);
  - WGA with the label-free checkpoint rule (validation average accuracy).
- **Mechanism measures:** the learned gate value `tanh(α)` and its gradient, logged every 10 alignment steps.
  Up to round 4 the logged gradient is the value after global-norm clipping at 1 (as in Flamingo); from round 5
  the raw pre-clip gradient and the pre-clip global norm are logged as well (`grad_raw`, `grad_norm_total`).
- **Behavioural pathway test:** from round 5, every selected checkpoint is re-tested with the connector's visual
  contribution removed (`ablated_mean_over_templates`). If WGA does not change, the pathway is behaviourally
  closed, the criterion of [farazi2026gate].
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
| H1: fixed > learned gate | One-sided **paired t-test** over seeds, with a **Wilcoxon signed-rank** test as the assumption-free check [dror2018hitchhiker]; **Holm** correction across the four H1 cells (2 LLMs × 2 tasks); unpaired Welch and Mann–Whitney tests as checks | Seeds pair the two arms (same data order and initialisation), removing between-seed variance |
| H2: learned gate more variable | One-sided **variance-ratio F-test**, with the **Brown–Forsythe** test as a robust check, and **Fisher's exact test** on the number of collapsed runs (WGA < 50) | The F-test is sensitive to non-normality; Brown–Forsythe is not; the collapse count tests the practical claim directly |
| Pairwise connector differences | **Stratified bootstrap** of test images (2,000 resamples) and a **hierarchical bootstrap** over seeds and images [efron1993bootstrap]; **Holm–Bonferroni** correction within each task [holm1979simple] | The hierarchical bootstrap covers training variance, not just test-sample noise |
| "Connector type does not matter" | **Two one-sided tests (TOST)**, margin ±3 points [schuirmann1987tost] | A non-significant difference is not evidence of equivalence |
| Effect size | Paired **Cohen's d_z** (mean difference / s.d. of differences), plus d_av for comparison across designs [cohen1988power] | p-values alone do not show practical importance |

**Confirmatory vs exploratory.** Only the four H1 tests (2 LLMs × 2 tasks, Holm-corrected) and H2 on
Qwen3.5-4B are confirmatory. Every control (open and half-open gate, learning rate, alignment length, Group DRO,
injection ratio, SigLIP-large, CelebA, p-sweep, per-layer, fp16) is exploratory: its p-values are reported
without correction and are not used as evidence for the hypotheses.

**Hypothesis timeline (stated openly, since there is no public pre-registration).**
- 28 Oct 2026: 3-seed pilot of all connectors, including the fixed gate (commit `2c948b4`).
- 28 Oct 2026, after the pilot: seed-extension runs launched (Kaggle kernels `h1a`, `h1b`, `seeds34`).
- 29 Oct 2026: H1/H2 and their tests committed with the extended results (`7ab0563`, `study/hypotheses.py`).
- Because H1 was formed after seeing the pilot, it is also tested on the confirmation seeds only (seeds ≥ 3):
  Waterbirds +10.5, t(11) = 4.06, p = 9.5 × 10⁻⁴, d_z = 1.17; Watermark +26.7, t(11) = 4.86, p = 2.5 × 10⁻⁴,
  d_z = 1.40. The pilot alone was not significant (p = 0.16 and 0.12). (`results_study/robustness_checks.md`)

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
- One factor varies at a time. The two gate arms share every parameter, but differ in two ways: whether `α`
  trains, and where the zero initialisation sits (the learned gate starts with α = 0 and a default-initialised
  `out_proj`; the fixed gate starts with tanh α = 1 and `out_proj` = 0).
- The second difference is forced, not chosen. If both α and `out_proj` start at zero, the gradient of the
  output with respect to α is proportional to `out_proj` and that with respect to `out_proj` to tanh α, so
  both stay at zero and the visual pathway never trains. A learnable zero-initialised gate therefore needs a
  non-zero `out_proj`, as in Flamingo and LLaMA-Adapter.
- Two further arms separate trainability from the starting value. The design follows the residual-scaling
  literature, where the start value of a learnable scale is the variable: ReZero and SkipInit start it at 0
  [bachlechner2021rezero; de2020skipinit], and LayerScale at a small ε, reporting that ε trains better than 0
  [touvron2021cait]. Those scales are linear; Flamingo's gate passes through tanh, so its gradient is scaled
  by 1 − tanh² α and vanishes as the gate saturates. Both arms are learnable, with `out_proj` = 0 like the
  fixed gate: `bottleneck_open` starts at tanh α = 0.995 and `bottleneck_half` at tanh α = 0.5. The open
  gate's gradient is scaled by 1 − tanh² ≈ 0.01, so it is nearly fixed in practice; the half-open gate's is
  scaled by 0.75, so it can move toward 0 or 1. The value 0.5 is our choice, not taken from prior work: it
  is the midpoint of the gate's range at which the gradient is still within 25% of its maximum (at tanh α = 0). If the half-open gate also matches the fixed gate, the harm is
  tied to starting at zero; if it drifts shut and loses WGA, trainability itself is the problem.
- Pairing by seed shares the data order and the LoRA initialisation between the two arms of a seed (the
  connector initialisations differ by design, see above). A seed does not replay a run exactly: GPU kernels
  (4-bit matrix multiplies, linear-attention kernels, atomic additions) are non-deterministic, and library
  versions drifted between jobs (transformers 5.21.0 to 5.22.1, recorded per job in `environment.txt`).
  Rerunning seeds 0–1 changed WGA by 12.7 points on average for the learned gate and 5.2 for the fixed gate
  (`results_study/extra_analyses.md` §9). Seeds are therefore treated as random replicates. Pairing is a
  design choice, so unpaired Welch and Mann–Whitney tests are reported as well
  (`results_study/robustness_checks.md`); they agree.
- The headline WGA averages three question phrasings, two of which were used in training. On the held-out
  phrasing alone the gate effect is the same (+10.8, +25.5, +10.2 and +25.8; all p ≤ 0.006;
  `extra_analyses.md` §8).
- Model selection uses validation WGA (group labels), as is standard [sagawa2020groupdro; kirichenko2023dfr];
  the label-free rule (best validation average accuracy) is reported for every run and gives a larger gate
  effect.
- Missing runs: Watermark seeds 8 and 14 were never launched because of the session budget; no finished run
  was excluded. Both are being added so that each arm has 15 seeds.
- Rival explanations are tested directly: capacity (identical parameters), token count (36-token
  ablation), encoder (SigLIP-large), LLM (Phi-4-mini) and training length (8 epochs).

**Construct validity.**
- Shortcut reliance is measured behaviourally, as WGA and the ID–OOD gap on test sets where the shortcut is
  broken, not inferred from attention. Attention is not explanation [jain2019attention].
- The mechanism is measured directly: the logged gate value and its gradient and, from round 4, the injected
  signal `‖tanh(α)·out_proj(attn)‖ / ‖e‖`, since a small gate can be offset by large output weights.
- Worst-group accuracy is reported with average and in-distribution accuracy, so a robustness gain cannot hide
  an accuracy loss.

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
- Requirements ([`requirements-study.txt`](../requirements-study.txt), Kaggle pins in
  [`study/requirements-kaggle.txt`](../study/requirements-kaggle.txt)) and an `environment.txt` per run.
- Not bit-reproducible: GPU kernels are non-deterministic and Kaggle jobs installed the latest
  `transformers`; same-seed reruns differ (§6, internal validity), so seeds are random replicates.
- 23 unit tests.
- All per-seed results and figure scripts.

**Ethics.**
- All data are public research datasets used under their terms:
  - Waterbirds [sagawa2020groupdro]: CUB-200-2011 birds [wah2011cub] on Places backgrounds [zhou2017places];
    both are for non-commercial research.
  - MS-COCO 2017 [lin2014coco]: annotations CC BY 4.0; images under Flickr terms of use.
  - CelebA [liu2015celeba]: non-commercial research purposes only.
  - Models: Qwen3.5-4B (Apache-2.0), Phi-4-mini-instruct (MIT), SigLIP (Apache-2.0). Check each model card
    for the exact version used (`environment.txt`).
- SigLIP was pre-trained on web data (WebLI) that may contain CUB, Places or COCO images; this affects every
  connector equally, because the encoder is frozen and shared.
- No new human-subject data were collected, so no IRB approval was required.
- CelebA contains face images and a binary gender attribute. It is used only as the standard
  group-robustness benchmark [sagawa2020groupdro]; no identity inference is performed.
- Compute: about 150 T4 GPU-hours of task training across 222 logged runs (from `train_minutes`), plus data
  preparation, alignment and evaluation, on Kaggle free-tier accounts. Rounds 3–5 (controls and extensions)
  add roughly 120–150 GPU-hours. Preliminary work (the VLM-SHIELD pilot, failed or cancelled sessions) is not
  in these totals and should be estimated by the authors for the NeurIPS-style compute statement.

## 7. Statements required at submission

**Data availability.** Waterbirds, COCO 2017 and CelebA are public; the task-building scripts
(`study/build_tasks.py`, `scripts/prepare_waterbirds.py`) regenerate every split deterministically. Per-run
results (`results.json`, per-sample predictions in `arrays.npz`) are in `results_study/`.

**Code availability.** All code, Kaggle kernels and exact package versions (`requirements-study.txt`, plus the
`environment.txt` exported by every run) are at https://github.com/Naveenvaidani/VLM-Debias. A licence must
be added before submission (author decision).

**Use of generative AI.** Springer Nature, Elsevier and IEEE require authors to disclose AI assistance. A
statement of the following kind is needed, adjusted to what the authors actually did: *"During this work the
authors used an AI coding and writing assistant (Claude, Anthropic) to help write experiment code, run
analyses and draft text. The authors reviewed and edited all content and take full responsibility for it."*
AI tools cannot be listed as authors.

**Competing interests and funding.** To be completed by the authors.

**Author contributions (CRediT).** To be completed by the authors.

