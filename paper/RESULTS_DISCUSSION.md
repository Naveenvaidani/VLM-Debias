# Results and discussion

*Results report the facts in the order of the hypotheses, using the pattern: target → test → verdict with
statistics → table or figure. The discussion interprets them in a "sandwich": finding → comparison with the
literature → mechanism. Every number below comes from `results_study/` and can be regenerated with
`study/hypotheses.py`, `study/significance.py`, `study/aggregate.py` and `study/final_figures.py`.*

**Notation:**
- WGA: test worst-group accuracy (%); means are ± s.d. over seeds.
- d_z: paired Cohen's d (mean difference ÷ s.d. of the differences).
- All H1/H2 tests are one-sided at α = 0.025, fixed before the confirmation seeds were run.

---

## Part I — Results

### R1. H1: a fixed gate yields higher worst-group accuracy than a learned gate

**Target and test.**
- Target: H1, fixed gate > learned gate.
- Test: one-sided paired t-test over seeds, with the Wilcoxon signed-rank test as an assumption-free check
  (Table 1, Figure 1).

**Verdict: supported for Qwen3.5-4B on both tasks.**
- *Waterbirds:* the fixed gate exceeds the learned gate by 10.6 points.
  - Paired test: 95% CI [5.3, 16.0], t(14) = 4.26, p = 4.0 × 10⁻⁴, d_z = 1.10.
  - Wilcoxon: p = 3.1 × 10⁻⁵.
- *Watermark:* the gain is 25.2 points.
  - Paired test: 95% CI [14.8, 35.6], t(14) = 5.21, p = 6.6 × 10⁻⁵, d_z = 1.34.
  - Wilcoxon: p = 6.1 × 10⁻⁵.
- Both effects are large by conventional thresholds [cohen1988power]. Achieved power is 0.98 and 0.99.
- *Robustness of the verdict* (`results_study/robustness_checks.md`):
  - Unpaired tests, which do not rely on seed pairing, agree: Welch p = 7.6 × 10⁻⁴ and 5.1 × 10⁻⁵;
    Mann–Whitney p = 1.2 × 10⁻⁵ and 2.5 × 10⁻⁴.
  - With label-free model selection (best validation average accuracy) the gain is +17.7 and +25.5
    (paired p = 0.002 and 6.0 × 10⁻⁵), so it does not depend on using group labels for selection.
  - The fixed gate also raises average accuracy (84.1 → 88.0, p = 0.007; 76.9 → 86.1, p = 7.4 × 10⁻⁴), with
    in-distribution accuracy unchanged (≈ 98%). The worst-group gain is not a trade-off against accuracy.
  - *Confirmation sample.* H1 was formed after a 3-seed pilot, so it is also tested on the later seeds only:
    Waterbirds +10.5 (12 seeds, t(11) = 4.06, p = 9.5 × 10⁻⁴, d_z = 1.17); Watermark +26.7 (12 seeds,
    t(11) = 4.86, p = 2.5 × 10⁻⁴, d_z = 1.40). The pilot alone was not significant (p = 0.16 and 0.12).
  - *Multiplicity.* With Holm correction across the four H1 tests (2 LLMs × 2 tasks), all four survive at 0.025
    (adjusted p = 0.0012 and 2.7 × 10⁻⁴ for Qwen3.5-4B, 0.0086 and 0.0086 for Phi-4-mini;
    `results_study/extra_analyses.md` §1).

**Phi-4-mini, second LLM.**
- *Waterbirds:* H1 is supported (+9.7, t(4) = 4.52, p = 0.005, d_z = 2.02).
- *Watermark, 10 seeds:* H1 is supported (+25.5, 95% CI [8.2, 42.7], t(9) = 3.34, p = 0.0043,
  Wilcoxon p = 9.8 × 10⁻⁴, d_z = 1.06).
  - With the first 5 seeds the gain (+31.6) missed α = 0.025 (p = 0.044). Seeds 5–9 were added to reach the
    ≈ 10 seeds a power analysis asked for, before looking at them; fixed > learned in all five new seeds.
  - The fixed gate is not robust on this task either: 6 of 10 fixed-gate seeds fall below 50% WGA (8 of 10
    learned). Phi-4-mini finds the watermark task harder with any gate.

**Table 1.** Gate hypotheses (WGA, %).

| LLM | Task | Seeds | Learned | Fixed | Δ [95% CI] | t (df) | p (t) | p (Wilcoxon) | d_z | Collapsed < 50 (L / F) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3.5-4B | Waterbirds | 15 | 63.6 ± 10.4 | 74.2 ± 3.4 | +10.6 [5.3, 16.0] | 4.26 (14) | 4.0e-4 | 3.1e-5 | 1.10 | 2 / 0 |
| Qwen3.5-4B | Watermark | 15 | 40.5 ± 18.0 | 65.7 ± 10.3 | +25.2 [14.8, 35.6] | 5.21 (14) | 6.6e-5 | 6.1e-5 | 1.34 | 10 / 0 |
| Phi-4-mini | Waterbirds | 5 | 60.4 ± 5.7 | 70.1 ± 3.6 | +9.7 [3.8, 15.7] | 4.52 (4) | 5.3e-3 | 3.1e-2 | 2.02 | 0 / 0 |
| Phi-4-mini | Watermark | 10 | 29.4 ± 18.7 | 54.9 ± 14.5 | +25.5 [8.2, 42.7] | 3.34 (9) | 4.3e-3 | 9.8e-4 | 1.06 | 8 / 6 |

With n = 5 the smallest attainable one-sided Wilcoxon p is 0.031, so the Wilcoxon test cannot reach
α = 0.025 for Phi-4-mini Waterbirds. Figure 1: `results_study/h1_gate_seeds.png`, `results_study/phi/h1_gate_seeds.png`
(one line per seed).

### R2. H2: the learned gate is less stable across seeds

**Target and test.**
- Target: H2, the learned gate is more variable.
- Test: one-sided F-test on seed variances, with the Brown–Forsythe test as a robust check, since learned-gate
  WGA is non-normal on Qwen Waterbirds (Shapiro–Wilk p = 0.003).

**Verdict: supported for Qwen3.5-4B by the F-test; the robust test weakens Watermark.**

| Task | F-test | Brown–Forsythe (two-sided) | Reading |
| :--- | :--- | :--- | :--- |
| Waterbirds | F = 9.2, p = 8.6 × 10⁻⁵ | p = 0.045 | Supported |
| Watermark | F = 3.1, p = 0.023 | not recomputed for 15 seeds | Supported by the F-test |

- The learned gate collapses below 50% WGA in 2 of 15 and 10 of 15 seeds; the fixed gate never does.
- For Phi-4-mini, neither variance test is significant (F p = 0.20 and 0.23).

### R3. Mechanism: the learned gate stays closed

**Target.** Exploratory: whether the gate opens. It is initialised at 0 (tanh 0 = 0), so the question is
whether training opens it, not whether it closes.

**Gate value.**
- After 300 alignment steps, |tanh α| = 0.0005 with SigLIP-base and 0.0006 with SigLIP-large.
- It never opens: in the logged trace it peaks at about 0.006 within 20 steps and decays to ≈ 0.001.
- The logged gradient is near ±1 and changes sign between logged steps (`align.json`, `gate_trace`). It is the
  value *after* global-norm clipping at 1, so it means the gate's gradient made up almost all of the clipped
  norm, not that its raw gradient is 1. Global clipping therefore shrinks every other connector update when the
  gate is learnable. Flamingo uses the same clipping; a control that leaves the gate out of clipping, and raw
  gradient logging, are in round 5.
- A small gate is not by itself a small injection, because `out_proj` is unconstrained and can grow to
  offset it. The injected signal `‖tanh(α)·out_proj(attn)‖ / ‖e‖` is therefore logged on every evaluation
  from round 4 on (`inject_ratio`), for SigLIP-base and SigLIP-large.
- *Result (`results_study/extra_analyses.md` §6):* `out_proj` does partly compensate. With tanh α ≈ 0.003 the
  learned gate still injects 1.2–7.5 times the text-embedding norm (mean 3.5, Qwen, 6 runs), but the fixed
  gate injects 95–129 (mean 112) and the open gate 106–145. The learned gate passes 15–35× less visual
  signal. Within learned-gate runs the ratio does not predict WGA (Spearman ρ = 0.33, p = 0.27, 13 runs).
- *Longer alignment does not open the gate (§4):* after 3,000 alignment steps instead of 300 the gate peaks at
  |tanh α| = 0.003 and ends at −0.0002.

**Attention on the shortcut.** Both cross-attention variants attend to the watermark corner more when it is
present (uniform share 8.2%):

| Variant | Watermark present | Watermark absent |
| :--- | :---: | :---: |
| Learned gate | 16.7% | 6.1% |
| Fixed gate | 18.2% | 7.0% |

Yet only the learned gate fails.

**Released checkpoints.** Gates with |tanh α| < 0.05:
- IDEFICS-80B: 9 of 20 cross-attention layers. IDEFICS-9B: 1 of 8.
- Llama 3.2 Vision: all of them. Mean |tanh α| is 0.003 at 11B (8 of 8 layers) and 0.001 at 90B (20 of 20).
  Values rise with depth (0.001 at layer 3 to 0.009 at layer 38 in 11B).
- These fully trained models clearly use the image, so a small gate there must be offset by large
  cross-attention outputs. The gate parameter alone is therefore not a measure of transmitted signal, which is
  why we add the injection ratio and the ablation test.

### R4. Connector type, once the gate is fixed

**Test.** Paired t-test over 5 seeds, plus TOST with a ±3-point margin.

| Task | Q-Former − MLP | Paired t | TOST 90% CI | Reading |
| :--- | :---: | :---: | :---: | :--- |
| Waterbirds | −1.6 | p = 0.23 | [−4.1, 0.8] | Not significant; equivalence not shown |
| Watermark | +3.0 | p = 0.40 | [−3.9, 9.9] | Not significant; equivalence not shown |

- Neither difference is significant.
- **Equivalence is not established:** TOST p = 0.15 and 0.50. Both 90% intervals extend beyond the
  ±3-point margin.
- Across Holm-corrected hierarchical-bootstrap comparisons, only gate contrasts and connector-DFR contrasts
  survive (`results_study/*/significance_wga.md`).

### R5. Robustness of the effect to rival explanations (exploratory)

| Rival explanation | Check | Result |
| :--- | :--- | :--- |
| Image-token count | MLP with 36 instead of 49 tokens, 2 seeds | 74.7 ± 1.1 vs 74.9 ± 1.4 |
| Encoder capacity | SigLIP-large, 4–5 seeds | All connectors +6 to +11; learned 74.5 ± 7.4 (5) vs fixed 80.9 ± 2.6 (4); paired +8.0, p = 0.088. Watermark, 2 seeds: 68.8 vs 68.2 |
| LLM adaptation | Frozen LLM, 3 seeds | Learned gate 24.2 ± 2.8; MLP 73.4 ± 1.1; Q-Former 74.6 ± 0.9 |
| Single alignment per seed | Alignment repeated per seed, 3 seeds | MLP 76.1 ± 2.1, fixed gate 74.2 ± 0.8 |
| Training length | 8 epochs, 1 seed (seeds 1–4 running) | MLP 79.1; fixed gate 44.5. Unresolved: the fixed gate may also degrade with long training |
| Gate trainable vs initialisation | Learnable gate initialised open (`bottleneck_open`, tanh = 0.995), 5 seeds | Open gate matches the fixed gate (71.8 vs 72.9; 66.8 vs 72.1) and beats the zero-init gate (+9.9, +17.1). The harm comes from the zero initialisation |
| Learning rate | Connector LR 5e-5 / 2e-4 / 1e-3, 2 seeds | Gap +29.6 / +16.7 / +0.7 as the gate moves further from 0 (0.002 / 0.004 / 0.018): dose-response |
| Alignment length | 3,000 instead of 300 steps, 3 seeds | Gate still shut (peak 0.003); learned-gate WGA collapses (11.4, 4.1) while fixed stays (70.8, 59.5): gap widens to +59 and +55 |
| Standard group-robust training | Group DRO at the full model, 2 seeds | Gap shrinks to +3.5 and +9.9; the fixed gate still leads |
| Injection strength | ‖injected‖ / ‖text‖ at test | Learned gate ≈ 3–8, fixed and open ≈ 100–125: 15–35× less visual signal, but not zero |
| Caption-reconstruction loss (LTD) | 3 seeds | MLP 75.9 ± 3.3 (vs 74.9); fixed gate 72.3 ± 3.7 |

### R6. Connector-only fixes

Table 2, Figure 2: `docs/figures/final/fixes.png`. Stage-2 baseline vs connector-DFR, 5 seeds each:

| Task | Fixed gate | Fixed + DFR | MLP | MLP + DFR | Linear DFR (no LLM) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Watermark | 72.5 ± 9.5 | **89.6 ± 0.5** | 64.9 ± 8.9 | 81.2 ± 2.2 | 85.1 ± 1.3 |
| Waterbirds | 73.9 ± 2.5 | 74.1 ± 5.6 | 74.9 ± 1.4 | 75.0 ± 3.6 | 84.2 ± 1.5 |

**Watermark.**
- Connector-DFR raises fixed-gate WGA by 17.1 points.
- It also reduces seed s.d. from 9.5 to 0.5.
- Fixed + DFR exceeds MLP + DFR by 8.4 points (bootstrap 95% CI [6.0, 9.8]; Holm-corrected hierarchical
  p < 0.001).

**Waterbirds.** Connector-DFR gives no gain, and every VLM stays about 10 points below linear DFR on the same
features.

**Other fixes.**
- Group-balanced sampling, which uses group labels in training, reaches 89.1 ± 0.9 (MLP) and 87.1 ± 3.5
  (fixed gate) on Watermark.
- Connector-JTT (1 seed) did not help.

### R7. Generalisation to CelebA and to native VLMs (exploratory)

- **CelebA** (8k-image stratified training subsample, so not comparable to full-data results): the ordering
  matches the other tasks, but no difference is significant:
  - MLP 54.3 ± 13.7 (4 seeds);
  - learned gate 36.3 ± 18.7 (4 seeds);
  - fixed gate 53.1 ± 5.8 (4 seeds); fixed − learned on 3 paired seeds +13.3, p = 0.24.
- **Native zero-shot:** Qwen3.5-4B with its own vision tower reaches 0.0 WGA (it always predicts one class),
  and 35.5 after a validation-calibrated threshold.

---

## Part II — Discussion

### D1. Overview

A learnable injection gate, not the connector family, determines how strongly a generative VLM relies on a
shortcut:
- Fixing the gate open raised worst-group accuracy by 10–24 points with large effect sizes (d_z = 1.1–1.3).
- It removed every collapsed seed.
- The direction replicated on a second LLM.
- Once the gate was fixed, MLP, Q-Former and cross-attention connectors were not significantly different.

### D2. The gate as a transmission bottleneck

**Finding.** The learned gate, initialised at zero, never opens during alignment, and models built on it rely
on the shortcut and vary widely across seeds.

**Comparison with the literature.**
- *Gate closure:* this confirms [farazi2026gate], who found that learnable gates in frozen VLMs end up closed
  across five designs. We extend that observation from *whether* the gate stays closed to *what the model then
  relies on*, which they did not measure.
- *Feature–readout view:* the result also refines the decomposition of [izmailov2022feature] and
  [kirichenko2023dfr]. Between features and readout sits a transmission stage, and its failure produces
  shortcut reliance even when the features are adequate. Linear DFR on the same frozen features reaches about
  85 WGA.
- *Conflicting cross-attention evidence:* it helps explain why reports on cross-attention connectors disagree
  [laurencon2024matters vs farazi2026gate]. Averages over a few seeds hide a seed lottery: some learned-gate
  runs match the fixed gate, others collapse.

**Mechanism.**
- With `tanh(α) ≈ 0`, unless `out_proj` grows enough to offset it, the visual pathway contributes little to
  the language stream, so the LoRA-adapted LLM must fit the training labels from whatever survives. The
  injection ratio confirms a 15–35× weaker visual signal (R3).
- On shortcut-aligned training data, the cue with the largest, most linearly available signal wins. This
  matches the "easiest rule" account of [geirhos2020shortcut].
- Whether a run escapes depends on a near-zero gate with a sign-flipping gradient. That is a fragile,
  seed-dependent event, which is exactly what H2 predicts.
- The frozen-LLM result supports this. Without LoRA to compensate, the learned gate falls to 24.2 WGA while
  MLP and Q-Former stay near 74.

**Scope relative to Flamingo.** Flamingo's gates "quickly grow" from zero over 500k steps with one gated block
every 4th–7th LM layer [alayrac2022flamingo]. Our connector injects once, at the input embeddings, after 300
alignment steps. The finding is therefore scoped to short alignment and input-level injection, the regime of
adapter-style fine-tuning on a budget. Ten times longer alignment (3,000 steps) does not move it toward
Flamingo's behaviour: the gate stays shut and the learned-gate model collapses further (WGA 11.4 and 4.1 vs
70.8 and 59.5 for the fixed gate). In our regime more alignment widens the gap. The IDEFICS gate
readings (9 of 20 layers below 0.05 after full training) show that near-shut gates also survive long training.
Flamingo also reports that removing the gate costs 4.2 points and destabilises training. Our fixed gate is not
"no gate": the output layer is zero-initialised, so the model still starts as the pretrained LLM, and no
instability was observed.

**Relation to modality imbalance.** A gate that stays shut is a concrete, architectural instance of the
greedy under-use of one modality [wu2022greedy] and of modality competition, in which the winner depends on
initialisation [huang2022modality]. That account predicts the seed lottery we observe (H2). Prior work fixes
imbalance by modulating gradients [peng2022balanced; wei2025onthefly]. Here, fixing one component removes the
problem at no cost, and the consequence is measured as worst-group accuracy rather than average accuracy.

**If the gate is shut, how is the visual shortcut learned?** The watermark and the background are themselves
visual, so some image signal must reach the LLM. A gate of ≈ 0.0005 passes little unless the output layer grows
to offset it. Our account is that the narrow residual pathway carries the strongest, most linearly available cue
and loses the finer object evidence. The injection ratio supports the first half: the pathway is open (the
injected signal is several times the text-embedding norm, because `out_proj` grows) but 15–35× weaker than
with a fixed gate. Group DRO (R5) supports the second: with group-balanced weighting of the loss, the same
narrow pathway reaches 72.5 and 69.2 WGA, so it carries enough object evidence; under ERM the model does not
use it. The pathway ablation (round 5) adds the behavioural test. If ablation sends the learned-gate model to chance, the pathway is narrow but open. If it changes
nothing, the pathway is behaviourally closed and the shortcut comes from elsewhere.

**Behavioural closure.** [farazi2026gate] define a pathway as closed when ablating it at inference leaves
accuracy unchanged, even when the gate nominally passes 30–45% of the signal. Our gate value is a parameter
reading; the round-5 runs add their behavioural test (`ablated_mean_over_templates`) and the injected-signal
ratio, so both criteria can be reported.

**Where the bias sits.** SpurLens attributes most of the spurious bias of MLLMs to the vision encoder
[hosseini2025spurlens]. Our design holds the encoder fixed and shared across arms, so the gate effect cannot
come from the encoder; the two findings are complementary: the encoder decides what spurious information is
available, the connector decides how much of the rest reaches the LLM.

**Why not just use a linear probe?** Linear DFR on the same frozen features reaches about 85 WGA, above every
VLM (≈ 74). The study does not propose a better classifier; it explains a failure of VLMs. The gap is itself
evidence for the transmission account: the information is in the features but is lost between features and
answer.

### D3. Attention is not reliance

**Finding.** Both gate variants attend to the watermark about 2.5× more when it is present, yet only one relies
on it.

**Comparison.**
- Attention-based diagnostics of spurious features [chakraborty2024exmap; lapuschkin2019clever] therefore
  need care in VLMs. This aligns with [jain2019attention].
- It also shows the condition under which [wiegreffe2019attention] expect attention to be informative:
  attention indicates reliance only when the pathway carrying it is open.

**Mechanism.** The learned gate multiplies attended features by ~0.0005. Attention can be focused on the
watermark while transmitting almost nothing, so the shortcut reaches the answer through the LLM's own priors
and LoRA instead.

### D4. Connector type is a weak factor for robustness

**Finding.** With the gate pitfall avoided, MLP and Q-Former differ by under 3 points, and no comparison is
significant.

**Comparison.**
- This agrees with [mckinzie2024mm1], who found connector type matters little for benchmark accuracy, and
  with [sarridis2026scaling], who found data rather than architecture drives bias.
- It is less consistent with [huang2024mir], whose MLP-over-Q-Former advantage in alignment quality does not
  appear as a robustness advantage here.

**Caveat.** Our TOST does not establish equivalence with 5 seeds, so the defensible claim is
*no detectable difference*, not *no difference*.

### D5. A stronger encoder masks the problem without removing its cause

**Finding.** With SigLIP-large all connectors improve, and the learned gate no longer collapses on Waterbirds,
although its value is still 0.0006. A shut gate is therefore not sufficient for shortcut reliance: it is a
risk factor whose effect depends on how separable the features are.

**Comparison.**
- This fits the encoder-centric account of MLLM failures [tong2024eyes; tong2024cambrian]: better features
  help.
- It also warns, in line with [sagawa2020overparam], that more capacity does not fix the mechanism.

**Mechanism.** Larger, more separable features let even a nearly closed pathway carry enough signal, or let
LoRA compensate. The fragility remains, and may resurface on harder tasks. Two seeds are not enough to
conclude more.

### D6. Readout fixes transfer to the connector, task-dependently

**Finding.** Connector-DFR is the best label-light fix on Watermark (89.6, s.d. 0.5) but does nothing on
Waterbirds.

**Comparison.**
- This extends DFR [kirichenko2023dfr] and its contrastive-VLM analogue [you2024calibrating] to the connector
  of generative VLMs.
- It also qualifies their premise that readout retraining suffices. It suffices when the shortcut is a
  localised, separable cue (a corner watermark). It does not suffice when the core and spurious features are
  entangled across the image (bird and background).
- Linear DFR's Waterbirds advantage (84.2) shows the core information is present but not recoverable through
  this LLM readout with a few hundred balanced examples.

### D7. Theoretical implications

1. **A transmission stage in the feature–readout account.** Shortcut reliance in modular models can arise
   between features and readout, through what an interface lets through. The gate is a concrete, measurable
   instance.
2. **Variance as an outcome.** Architectural choices can change *how often* a model learns the shortcut, not
   only how much on average. Seed-level reporting is therefore required for claims about robustness
   [bouthillier2021variance].
3. **A boundary condition for readout fixes.** Connector-level DFR works when the spurious cue is spatially
   localised and separable, and fails when it is entangled with the core object.

### D8. Practical implications

- **Fixed injection:** prefer fixed (or zero-initialised-output) injection over learnable tanh gates when
  adding cross-attention adapters to an LLM. Log the gate value if a learnable gate is kept.
- **Evaluation:** evaluate worst-group accuracy over at least 5–10 seeds before choosing a connector.
- **Cheap fix:** when a small group-balanced set exists, retrain only the connector. It costs one epoch, needs
  no LLM retraining, and stabilises seeds.
- **Cost:** the fixed-gate cross-attention connector matches MLP robustness at about half the step time and
  memory (1.7 vs 4.0 s per step; 6.2 vs 12.1 GB on a T4).

### D9. Limitations and future research

**Limitations.**
- Models are 4-bit and 4B parameters or smaller.
- The encoder is frozen.
- SigLIP-large Watermark has 2 seeds, CelebA 3–4, and the 8-epoch run 1 (extension running).
- The learning-rate sweep and Group DRO have 2 seeds per setting and are descriptive.
- The open-gate control separates initialisation from learnability only partly: at α = 3 the gate gradient
  is scaled by 1 − tanh² ≈ 0.01, so the open gate barely moves.
- The fixed gate fell to 44.5 in the single 8-epoch run; long-training behaviour is unresolved.
- CelebA trains on an 8k subsample.
- Group DRO is the only full-model group-robust baseline; JTT is tested only at the connector.
- Phi-4-mini Watermark is underpowered (0.40).
- The Watermark shortcut is partly textual [materzynska2022disentangling].
- Equivalence of connector types is not established.
- Injection is input-level (one block), not per-layer as in Flamingo, and alignment is 300 steps.
- In stage 2 the learned gate has weight decay 0.01 (none in alignment); it shrinks α by a relative
  2 × 10⁻⁶ per step (lr × decay), negligible, and a no-weight-decay control is in round 5.
- Watermark uses one shortcut strength (p = 0.95); a dose–response at p = 0.9 and 0.99 is in round 5.
- SigLIP's web pre-training may include CUB, Places or COCO images (shared by every arm).
- CelebA and Waterbirds are licensed for non-commercial research only.

**Three directions for future work.**
1. Test gate dynamics in full-precision, larger models with trainable encoders: IDEFICS-scale models and
   Llama 3 vision adapters [grattafiori2024llama3].
2. Design injection schemes that start as the identity but cannot close, and test them on natural-image
   benchmarks of spurious cues [yang2025spuriverse; hosseini2025spurlens].
3. Explain why connector-DFR fails on entangled shortcuts, for example with subspace explanations
   [chormai2024disentangled], and combine it with data-level fixes [noohdani2024decompose].
