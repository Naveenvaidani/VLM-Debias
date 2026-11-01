# Results and discussion

*Results report the facts in the order of the hypotheses, using the pattern: target → test → verdict with
statistics → table or figure. The discussion interprets them in a "sandwich": finding → comparison with the
literature → mechanism. Every number below comes from `results_study/` and can be regenerated with
`study/hypotheses.py`, `study/significance.py`, `study/aggregate.py` and `study/final_figures.py`.*

**Notation:**
- WGA: test worst-group accuracy (%); means are ± s.d. over seeds.
- d_z: paired Cohen's d (mean difference ÷ s.d. of the differences).
- All H1/H2 tests are one-sided at the pre-declared α = 0.025.

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
- *Watermark:* the gain is 23.8 points.
  - Paired test: 95% CI [12.5, 35.1], t(12) = 4.59, p = 3.1 × 10⁻⁴, d_z = 1.27.
  - Wilcoxon: p = 2.4 × 10⁻⁴.
- Both effects are large by conventional thresholds [cohen1988power]. Achieved power is 0.98 and 0.99.

**Phi-4-mini, second LLM.**
- *Waterbirds:* H1 is supported (+9.7, t(4) = 4.52, p = 0.005, d_z = 2.02).
- *Watermark:* the gain is larger (+31.6) but **not significant** at α = 0.025 (t(4) = 2.25, p = 0.044,
  d_z = 1.01).
  - Achieved power is 0.40; about 10 seeds would be needed for 80% power.
  - Two of the five fixed-gate seeds also fall below 50% WGA on this task.

**Table 1.** Gate hypotheses (WGA, %).

| LLM | Task | Seeds | Learned | Fixed | Δ [95% CI] | t (df) | p (t) | p (Wilcoxon) | d_z | Collapsed < 50 (L / F) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3.5-4B | Waterbirds | 15 | 63.6 ± 10.4 | 74.2 ± 3.4 | +10.6 [5.3, 16.0] | 4.26 (14) | 4.0e-4 | 3.1e-5 | 1.10 | 2 / 0 |
| Qwen3.5-4B | Watermark | 13 | 41.4 ± 19.0 | 65.2 ± 10.4 | +23.8 [12.5, 35.1] | 4.59 (12) | 3.1e-4 | 2.4e-4 | 1.27 | 8 / 0 |
| Phi-4-mini | Waterbirds | 5 | 60.4 ± 5.7 | 70.1 ± 3.6 | +9.7 [3.8, 15.7] | 4.52 (4) | 5.3e-3 | 3.1e-2 | 2.02 | 0 / 0 |
| Phi-4-mini | Watermark | 5 | 28.5 ± 21.3 | 60.1 ± 15.1 | +31.6 [−7.4, 70.6] | 2.25 (4) | 4.4e-2 | 3.1e-2 | 1.01 | 4 / 2 |

With n = 5 the smallest attainable one-sided Wilcoxon p is 0.031, so the Wilcoxon test cannot reach
α = 0.025 for Phi-4-mini. Figure 1: `results_study/h1_gate_seeds.png`, `results_study/phi/h1_gate_seeds.png`
(one line per seed).

### R2. H2: the learned gate is less stable across seeds

**Target and test.**
- Target: H2, the learned gate is more variable.
- Test: one-sided F-test on seed variances, with the Brown–Forsythe test as a robust check, since learned-gate
  WGA is non-normal on Qwen Waterbirds (Shapiro–Wilk p = 0.003).

**Verdict: supported for Qwen3.5-4B by the pre-declared F-test; the robust test weakens Watermark.**

| Task | F-test | Brown–Forsythe (two-sided) | Reading |
| :--- | :--- | :--- | :--- |
| Waterbirds | F = 9.2, p = 8.6 × 10⁻⁵ | p = 0.045 | Supported |
| Watermark | F = 3.4, p = 0.023 | p = 0.070 | Supported only by the F-test |

- The learned gate collapses below 50% WGA in 2 of 15 and 8 of 13 seeds; the fixed gate never does.
- For Phi-4-mini, neither variance test is significant (F p = 0.20 and 0.26).

### R3. Mechanism: the learned gate stays closed

**Target.** Exploratory: whether the gate closes.

**Gate value.**
- After 300 alignment steps, |tanh α| = 0.0005 with SigLIP-base and 0.0006 with SigLIP-large.
- Its gradient is near ±1 and changes sign between logged steps (`align.json`, `gate_trace`).

**Attention on the shortcut.** Both cross-attention variants attend to the watermark corner more when it is
present (uniform share 8.2%):

| Variant | Watermark present | Watermark absent |
| :--- | :---: | :---: |
| Learned gate | 16.7% | 6.1% |
| Fixed gate | 18.2% | 7.0% |

Yet only the learned gate fails.

**Released checkpoints.** In IDEFICS-80B, 9 of 20 gated cross-attention layers have |tanh α| < 0.05; in
IDEFICS-9B, 1 of 8.

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
| Encoder capacity | SigLIP-large, 2 seeds | All connectors +6 to +16; learned 79.3 ± 1.9 vs fixed 80.1 ± 3.1 |
| LLM adaptation | Frozen LLM, 3 seeds | Learned gate 24.2 ± 2.8; MLP 73.4 ± 1.1; Q-Former 74.6 ± 0.9 |
| Single alignment per seed | Alignment repeated per seed, 3 seeds | MLP 76.1 ± 2.1, fixed gate 74.2 ± 0.8 |
| Training length | 8 epochs, 1 seed | MLP 79.1; fixed gate 44.5 |
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

- **CelebA:** the ordering matches the other tasks, but no difference is significant:
  - MLP 54.3 ± 13.7 (4 seeds);
  - learned gate 36.3 ± 18.7 (4 seeds);
  - fixed gate 49.3 ± 4.5 (2 seeds).
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

**Finding.** The learned gate shuts during alignment, and models built on it rely on the shortcut and vary
widely across seeds.

**Comparison with the literature.**
- *Gate closure:* this confirms [farazi2026gate], who found that learnable gates in frozen VLMs end up closed
  across five designs. We extend that observation from *whether* the gate closes to *what the model then
  relies on*, which they did not measure.
- *Feature–readout view:* the result also refines the decomposition of [izmailov2022feature] and
  [kirichenko2023dfr]. Between features and readout sits a transmission stage, and its failure produces
  shortcut reliance even when the features are adequate. Linear DFR on the same frozen features reaches about
  85 WGA.
- *Conflicting cross-attention evidence:* it helps explain why reports on cross-attention connectors disagree
  [laurencon2024matters vs farazi2026gate]. Averages over a few seeds hide a seed lottery: some learned-gate
  runs match the fixed gate, others collapse.

**Mechanism.**
- With `tanh(α) ≈ 0` the visual pathway contributes almost nothing to the language stream, so the
  LoRA-adapted LLM must fit the training labels from whatever survives.
- On shortcut-aligned training data, the cue with the largest, most linearly available signal wins. This
  matches the "easiest rule" account of [geirhos2020shortcut].
- Whether a run escapes depends on a near-zero gate with a sign-flipping gradient. That is a fragile,
  seed-dependent event, which is exactly what H2 predicts.
- The frozen-LLM result supports this. Without LoRA to compensate, the learned gate falls to 24.2 WGA while
  MLP and Q-Former stay near 74.

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
although it still closes to 0.0006.

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
- SigLIP-large has 2 seeds, CelebA's fixed gate 2, and the 8-epoch run 1.
- The learning rate was not tuned; the sweep did not complete.
- Phi-4-mini Watermark is underpowered (0.40).
- The Watermark shortcut is partly textual [materzynska2022disentangling].
- Equivalence of connector types is not established.

**Three directions for future work.**
1. Test gate dynamics in full-precision, larger models with trainable encoders: IDEFICS-scale models and
   Llama 3 vision adapters [grattafiori2024llama3].
2. Design injection schemes that start as the identity but cannot close, and test them on natural-image
   benchmarks of spurious cues [yang2025spuriverse; hosseini2025spurlens].
3. Explain why connector-DFR fails on entangled shortcuts, for example with subspace explanations
   [chormai2024disentangled], and combine it with data-level fixes [noohdani2024decompose].
