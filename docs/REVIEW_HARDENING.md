# Preparing the study for peer review: what was changed, why, and where

This page explains, in plain language, every change made to get the study ready for a Q1 journal review
(1 November 2026). Each entry gives the concern a reviewer is likely to raise, what we did about it, and where
to find the evidence. The per-item status table is in
[`paper/SUBMISSION_CHECKLIST.md`](../paper/SUBMISSION_CHECKLIST.md).

**How the concerns were found.**
- We re-read our own paper drafts against the closest prior work:
  - **Flamingo** (Alayrac et al., 2022): where its gates sit, how they are initialised, how long it trains, and
    its optimiser settings;
  - **Farazi et al., 2026**, *The Gate Always Closes* (arXiv 2607.23335): how it defines a closed gate;
  - **SpuriVerse** and **SpurLens**: what they attribute spurious bias to, and their benchmark formats.
- We checked the study against the **NeurIPS paper checklist**, whose items journals in this area also ask for:
  claims, limitations, reproducibility, significance, compute, licences and LLM use.
- We checked the git history to see what was decided when.

Nothing was changed without first reading the code involved. Every new option was tested with unit tests and a
CPU end-to-end run (tiny Llama) before any Kaggle job used it. Defaults are unchanged, so new runs stay
comparable with the earlier ones.

---

## 1. Wording that a reviewer would correct

| Before | After | Why |
| :--- | :--- | :--- |
| "the learned gate **closes** during training" | "initialised at zero, the gate **never opens**" | `tanh(0) = 0` at the start. The logged trace peaks at ≈ 0.006 and decays to ≈ 0.001, so nothing "closes". |
| "learnable gates **cause** shortcut learning" (title) | "**induce**" | With SigLIP-large the gate is still ≈ 0.0006, yet WGA no longer collapses (79.3 vs 80.1). A shut gate is a risk factor, not a sufficient cause. |
| "connector type makes **no difference**" | "**no significant difference detected; equivalence not established**" | The ±3-point TOST does not pass, and on CelebA MLP leads by 5 points (n.s.). |
| "**Flamingo-style** cross-attention" | "**input-level** gated cross-attention with a **Flamingo-type** `tanh` gate" | Flamingo inserts a gated block before every 4th (9B) or 7th (80B) LM layer and trains for 500k steps. Ours injects once, into the input embeddings, after 300 steps. |
| "**pre-declared** hypotheses" | "formulated after a 3-seed pilot, confirmed on new seeds" | Git shows the hypotheses were committed (29 Oct, `7ab0563`) after the pilot (28 Oct, `2c948b4`). There is no public pre-registration. |

## 2. New analyses of existing runs (no new GPU time)

All of these come from [`study/robustness_checks.py`](../study/robustness_checks.py). The output is in
[`results_study/robustness_checks.md`](../results_study/robustness_checks.md) and in README §16.

| Reviewer question | Answer from the data |
| :--- | :--- |
| Does the gate effect depend on choosing checkpoints with group labels? | No. With label-free selection (best validation average accuracy) the gain is larger: +17.7 and +24.1 points. |
| Is pairing seeds what makes the test significant? | No. Unpaired Welch tests give p = 7.6 × 10⁻⁴ and 4.4 × 10⁻⁴; Mann–Whitney gives 1.2 × 10⁻⁵ and 1.2 × 10⁻³. |
| Is worst-group accuracy bought with average accuracy? | No. The fixed gate also raises average accuracy (84.1 → 88.0, 76.9 → 85.8). In-distribution accuracy stays ≈ 98%. |
| The hypothesis was formed on the pilot. Does it hold on new data alone? | Yes. On seeds 3+ only: Waterbirds +10.5 (12 seeds, p = 9.5 × 10⁻⁴, d_z = 1.17) and Watermark +25.1 (10 seeds, p = 0.0012, d_z = 1.31). The 3-seed pilot alone was not significant (p = 0.16, 0.12). |
| Four H1 tests (2 LLMs × 2 tasks): is there a multiple-testing correction? | Holm: three survive at 0.025 (adjusted p = 0.0012, 0.0012, 0.011). Phi-4-mini Watermark (0.044) does not; its seeds 5–9 have finished and are being added. |
| Why are there 13 Watermark seeds but 15 Waterbirds seeds? | Seeds 8 and 14 were never launched, because of the session budget. No finished run was excluded. Both are now running. |

## 3. Code changes (all optional flags; defaults unchanged)

| Change | File | What it does | Why |
| :--- | :--- | :--- | :--- |
| Injection strength | `src/study/vlm.py`, `src/study/engine.py` | Logs `‖tanh(α)·out_proj(attn)‖ / ‖e‖` on every evaluation (`inject_ratio`). | A near-zero gate can be offset by larger output weights. This measures what actually reaches the LLM. |
| Pathway ablation | `src/study/vlm.py`, `study/train_task.py` | Re-tests each selected checkpoint with the connector's visual contribution removed (`ablated_mean_over_templates`). | Farazi et al. call a pathway closed when removing it at inference changes nothing. This tests that criterion directly. |
| Raw gate gradient | `src/study/engine.py` (`clip_and_log`), `study/align.py` | Logs the gate gradient **before** clipping, plus the pre-clip global norm. | The earlier trace logged the clipped value (≈ ±1). That only showed the gate dominated the clipped norm, not its real size. |
| `--clip_exclude_gate` | `study/align.py`, `study/train_task.py`, `study/kaggle_study.py` | Leaves the gate out of global-norm clipping. | With global clipping, a dominant gate gradient shrinks every other connector update. This control removes that coupling. |
| `--gate_no_wd` | `study/train_task.py`, `study/kaggle_study.py` | No weight decay on the gate in stage 2. | Weight decay pulls the gate towards zero. The effect is tiny (relative 2 × 10⁻⁶ per step), but it is controlled anyway. |
| `--loss groupdro` | `src/study/engine.py`, `study/train_task.py` | Group DRO (Sagawa et al., 2020) over the four groups. | Gives a standard group-robust baseline at the full model, not only at the connector. |
| `bottleneck_open` | `src/study/connectors.py` | A learnable gate initialised open (tanh 3 = 0.995) with a zero-initialised output layer. | Learned and fixed gates differ in two ways: trainability, and where the zero initialisation sits. This variant separates them. |

Tests: `tests/test_study.py`, 21 passing. They include Group DRO up-weighting the worst group, and clipping
with and without the gate.

## 4. New Kaggle runs

| Round | Account | Run | Question it answers |
| :---: | :--- | :--- | :--- |
| 3 | account C | Learning rate 5e-5 and 1e-3 | Is the effect an artefact of an untuned learning rate? |
| 3 | account C | 3,000-step alignment | Does the gate open with 10× longer alignment, closer to Flamingo's regime? |
| 3 | account D | SigLIP-large seeds 2–4, plus Watermark | Encoder-scale result at 5 seeds |
| 3 | account E | Phi-4-mini Watermark seeds 5–9 | Power for the second-LLM replication |
| 3 | account E | CelebA fixed gate seeds 3–4 | Equal seeds on CelebA |
| 3 | account E | Open-gate control | Trainability vs initialisation |
| 4 | account A | Injection strength, SigLIP-base | Is the learned gate's injected signal really small? |
| 4 | account A | 8 epochs, seeds 1–4 | Does the fixed gate's 44.5 at 8 epochs replicate? |
| 4 | account B | Injection strength, SigLIP-large | Why the shut gate does not hurt with the larger encoder |
| 4 | account B | Group DRO | Standard group-robust baseline |
| 4 | jeyadevr | Watermark seeds 8 and 14 | 15 seeds per arm |
| 5 | jeyadevr | Controls: pathway ablation, gate excluded from clipping, no gate weight decay | Behavioural closure; clipping and weight-decay confounds |
| 5 | jeyadevr | Watermark at p = 0.9 and 0.99 | Dose–response over shortcut strength |

## 5. Paper documents updated

**[`paper/METHODOLOGY.md`](../paper/METHODOLOGY.md):**
- the two-way difference between the gate arms, and the open-gate control;
- why seeds are paired;
- the checkpoint-selection rule;
- the missing-seed rule;
- the hypothesis timeline, with commit hashes;
- Holm correction across the four H1 tests;
- the raw-gradient and pathway-ablation measures;
- dataset and model licences;
- pre-training overlap;
- the compute estimate;
- data and code availability;
- a generative-AI disclosure template.

**[`paper/RESULTS_DISCUSSION.md`](../paper/RESULTS_DISCUSSION.md):**
- R1: robustness, confirmation sample and multiplicity;
- R3: the gradient-clipping caveat;
- R5: new control rows;
- D2: scope relative to Flamingo, behavioural closure, the encoder-vs-connector account (SpurLens), and why a
  linear probe is not the point;
- D5: a shut gate as a risk factor;
- D9: new limitations.

**Introduction, title and abstract:**
- [`paper/INTRODUCTION.md`](../paper/INTRODUCTION.md) and
  [`paper/TITLE_ABSTRACT.md`](../paper/TITLE_ABSTRACT.md): wording in §1;
- the recommended title is now *"Learnable Injection Gates Induce Shortcut Learning in Vision-Language
  Models"*.

**References** ([`paper/references.bib`](../paper/references.bib)):
- all 94 entries now have full author lists, fetched from arXiv / Crossref by
  [`paper/fetch_authors.py`](../paper/fetch_authors.py);
- the first author of each entry is cross-checked against the verified record;
- the Places database (Zhou et al., TPAMI 2018) was added for the licence statement.

## 6. What only the authors can do

- Author names, affiliations and ORCIDs (the README citation still says "VLM-Debias contributors").
- A code licence for the repository.
- Adjust the generative-AI disclosure in METHODOLOGY §7 to describe what was actually done.
- An estimate of preliminary and failed compute, for the compute statement.
- Set the repository's About text and topics. This needs admin rights on `Naveenvaidani/VLM-Debias`.

## 7. Still open (would strengthen a TPAMI / IJCV submission)

- **Released-model behavioural test:** force IDEFICS-9B's near-shut gates open and compare WGA.
- **Natural benchmark:** SpuriVerse (124 spurious correlations, 1,364 multiple-choice questions) needs a
  generative multiple-choice protocol rather than our two-answer readout.
- **Multi-layer injection:** a per-layer gated block, as in Flamingo.
- **Scale and precision:** a larger or full-precision LLM (T4 memory).
