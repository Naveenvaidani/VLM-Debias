# Where our results contradict, extend or agree with prior work

For each closely related paper: what it found, which of its stated limitations our setup can test, what our
data says, and how strong that evidence is. Papers were read in full ([`NOVELTY.md`](NOVELTY.md) §6). Our numbers
come from [`results_study/rounds_4_8.md`](../results_study/rounds_4_8.md), [`extra_analyses.md`](../results_study/extra_analyses.md)
and the README.

**Ground rule.** Our setting differs from each paper's: a 4B LLM in 4-bit, frozen SigLIP features, a few hundred
alignment steps, binary tasks. A disagreement shows that a finding does **not hold in our setting**. It does not
show that the other paper is wrong in its own setting. Each verdict below says which.

Verdicts: **Contradicts** (our data disagrees with their claim or expectation), **Extends** (we test something
they list as a limitation or never test), **Agrees**, **Pending** (round 9 will decide).

---

## 1. Farazi et al., *The Gate Always Closes* (arXiv 2607.23335) [farazi2026gate]

### 1a. The mechanism: their dead gradient vs our live, cancelling gradient — **Contradicts (in our setting)**

- **Their claim.** For an image-derived signal that does not depend on the caption, the first-order gradient on
  the gate is zero at initialisation (Lemma 1, Prop. 1), so the gate stalls (S5) or drifts down (S3). Their
  measured paired gradient is indistinguishable from zero (|t| = 0.19, Table S7).
- **Our data.** Our connector is *text-coupled*: text tokens query the image patches, so the injected signal
  depends on the prompt, and their caption-invariance condition does not hold. The gradient on our gate is
  **large, not zero**. The raw, pre-clipping mean |∂L/∂α| per epoch is 3–1,528 across 13 learned-gate runs
  (round 5, logged by `src/study/engine.py`). In several runs it accounts for almost all of the global gradient
  norm (for example 1,437 of 1,439 in one Waterbirds run).
- **Yet the gate does not move.** Waterbirds has about 300 steps per epoch (4,795 images, batch 16), so 3–4
  epochs is roughly 900–1,200 Adam steps at learning rate 2 × 10⁻⁴. Adam's step is at most about the learning
  rate, so a gradient of consistent sign would move α by up to about 0.2. The observed change is at most 0.003.
  The gradient therefore has to flip sign almost every step: its signs agree on the order of 1–2% of steps,
  net. Leaving the gate out of clipping (3 seeds) or out of weight decay (2 seeds) does not change this
  (final |tanh α| ≤ 0.0028).
- **What this changes.** In a text-coupled connector the gate is not stuck because the gradient is dead; it is
  stuck because a large gradient cancels itself. Their account covers caption-invariant pathways, and they list
  text-coupled injection as untested future work. Our data shows near-zero gates also arise there, through a
  different mechanism.
- **Limits of our evidence.** Rounds 1–8 logged |∂L/∂α| but not its sign, so the cancellation is inferred from
  Adam's bounded step, not observed directly. The 1–2% figure is an order of magnitude, not a measurement.
- **Round 9 (`r9-ablate`, 7 zero-start runs) — measured.** Mean |∂L/∂α| 37–1,282 per epoch; 37–60% of steps push
  the gate up; the sign flips on 45–49% of consecutive steps; final gate 0.001–0.005. The cancellation is now
  observed directly, not inferred.

### 1b. Text-coupled injection does not escape closure — **Contradicts their expectation**

- **Their expectation.** Text-coupled (query-conditioned) injection breaks caption-invariance, so it "would not
  be expected to close"; they cite LLaMA-Adapter as such a case and leave it as future work (Sec. 3.3, Table S11).
- **Our data.** In our text-coupled connector the learned gate ends at |tanh α| ≈ 0.0005–0.004 after alignment
  and stage 2. It stays there at 10× alignment length (peak 0.003) and with the gate protected from clipping and
  weight decay. The per-layer version, which is also text-coupled, ends at a mean of 0.027.
- **Strength.** Seen in every learned-gate run we have (15 + 15 Qwen seeds, 5 + 10 Phi-4-mini seeds, all
  controls). Single block and per-layer, one LLM family per run, short training.

### 1c. Small gate, but the pathway is used — **Contradicts their behavioural finding; agrees with their principle**

- **Their finding.** At the learned gate, ablating the pathway changes GQA-rel by at most 0.5 points (S1–S4):
  "behaviourally closed". Their principle: a gate's value is not evidence of use; only ablation is.
- **Our data.** With the gate at about 0.004, removing the pathway drops the learned-gate model to the
  majority-class share (Waterbirds average accuracy 89.4 → 77.8; Watermark 77.9 → 50.0). The pathway is small
  in parameter terms but **used**, because the output projection grows to compensate (injection ratio 2.6
  against 99 for the fixed gate).
- **Reading.** Their principle holds in the opposite direction too: a near-zero gate can come with an *active*
  pathway. Gate value alone misreads both cases.
- **Round 9 (`r9-ablate`).** Pathway removed or image features zeroed: the learned-gate model gives one answer to
  every image in every run (Waterbirds 83.0 → one-class share, −46.9, 4/4 seeds, p = 0.035; Watermark 81.8 → 50.0,
  3/3 seeds, p = 0.003). **Strength now moderate** (3–4 seeds; the session ran out before seed 4).
- **Shuffled-image control: invalid as run.** Test files are stored group by group and evaluated in order, so
  "batch rolled by one" swapped each image with a same-group neighbour (same label and shortcut). It changed
  accuracy by ≤ 0.5 points for both connectors, which is expected for a same-group swap and says nothing about
  instance-level use. Not reported in the paper. Fixed in code (seeded random test order for the shuffle mode)
  for future runs; round-10 runs cloned the old code, so their shuffle numbers are also invalid.
- **Shuffled-image control, rerun with a random order (round 11, `results_study/r11_shuffle_runs.csv`).** Valid now.
  Another example's image drops average accuracy to chance on Watermark for both gates (learnable 68.6 → 51.1,
  fixed 86.2 → 52.0, 4 seeds) and on Waterbirds 85.8 → 62.3 and 90.1 → 64.9 (3 seeds). The learnable gate's
  Watermark WGA rises with the wrong image (24.6 → 44.2). **Supports** instance-level image use: weak, not closed.
- **W_o-init confound (round 11, `results_study/r11_wo_runs.csv`).** Starts 0.01 / 0.1 with the default W_o (only
  the start value differs from the zero-start arm). 0.01: 66.6 / 46.1 vs zero start 64.4 / 44.2 on the same seeds,
  so the extra collapse of the W_o = 0 version came from W_o. 0.1: gate stays 0.070–0.074; 73.2 / 58.9 vs fixed
  74.6 / 67.4 (−1.4, −8.5, ns; 3 / 4 seeds). **Partly supports** the threshold: 0.1 matches fixed on Waterbirds,
  recovers ~2/3 of the Watermark gap. Paper wording changed from "match" to "approach".

### 1d. Gate start value — **Extends (they never vary it)**

- **Their setup.** Every learnable gate starts at 0, or at softplus(0) for the attention-bias designs. No
  initialisation ablation.
- **Our data.** A learnable gate started at tanh α = 0.5 matches the fixed gate and beats the zero start by
  +11.4 and +20.1 worst-group points (p = 0.014 and 0.006, 5 seeds, exploratory). It moves little (final 0.44).
  The start at 0.995 behaves the same. So the outcome is set by where the gate starts, not by whether it can learn.
- **Round 9, start 0.1 (`r9-eps1`, 5 seeds).** Waterbirds 73.1 and Watermark 65.3, against 61.2 / 42.6 for the
  zero start (+11.9, p = 0.041; +22.7, p = 0.072) and 73.1 / 67.2 for the fixed gate. The gate shrinks to about
  0.06 but stays about 100× above the zero-start gate. So a start of 0.1 is already enough.
- **Round 9, start 0.01 (`r9-eps01`, 5 seeds).** The gate falls back to 0.001–0.008 and behaves like the zero
  start or worse: Waterbirds 69.1, Watermark 22.5 with 5/5 seeds collapsed (−20.1 vs zero start, p = 0.041).
  Threshold lies between 0.01 and 0.1. Signed gradient: large (up to 159), but 30–58% of steps push the gate up,
  which is the self-cancelling gradient measured directly (sign no longer inferred).

### 1e. Their recommendation: matched-norm injection without a gate — **Partly agrees; robustness side Pending**

- **Their recommendation.** If the signal must be injected, inject it at matched norm without an optional gate.
  The unnormalised prefix (36× the text norm) breaks generation.
- **Our data, part that agrees.** Our ungated fixed injection (70–130× the text norm) is the *more* robust option
  on binary tasks but **hurts free-form generation** on VQA-CP (−5.0 in distribution, p = 0.010). Removing its
  image pathway even *raises* shifted-prior accuracy by 4.6 points. This is their norm warning, seen in a
  generative robustness benchmark.
- **Round 9, generation side (`r9-vqacp`, 3 seeds) — Agrees.** The capped connector (per-token cap at the text
  norm) scores 49.6 in distribution and 30.6 under shifted priors: +7.0 over the uncapped fixed gate in distribution
  (p = 0.020) and +3.5 over a text-only model under shifted priors (p = 0.028). It is the only model whose accuracy
  falls when the image is removed or zeroed (−2.8 to −3.5 in distribution; the shuffled control is invalid, see 1b). The uncapped fixed gate is
  *below* the text-only model (−5.7, p = 0.001); the learnable gate equals it in distribution (−0.6, p = 0.23).
- **Round 9, robustness side (`r9-capped`, 5 seeds) — Contradicts.** Matching the norm costs robustness:
  Waterbirds 69.9 vs 73.1 for the uncapped fixed gate (p = 0.13); Watermark 50.6 vs 67.2 (−16.6, p = 0.002,
  d_z = −3.1), 2/5 seeds collapsed. Still above the zero-start gate (+8.7, p = 0.055; +8.0, p = 0.30).
- **Reading.** Their recommendation trades shortcut robustness for generation quality. Their benchmarks have no
  shortcut measure, so the cost was invisible to them. This is the one finding where we contradict their advice. If the cap keeps the binary gain and repairs
  VQA-CP, their recommendation and ours combine into one rule. If it loses the gain, their recommendation trades
  robustness for generation safety, which would be a real contradiction.

### 1f. Limitations of theirs that our design already removes — **Extends**

| Their stated limitation | Ours |
| :--- | :--- |
| Most LLaVA configurations are single runs | 15 paired seeds for the main comparison, 4–5 for most controls |
| No spurious-correlation, shortcut or worst-group measure | Worst-group accuracy on three tasks, plus VQA-CP shifted priors |
| No random or mismatched-input control | Done in round 11 (random order); round-9 version was invalid |
| One backbone for the behavioural ablations | Two LLMs (Qwen3.5-4B, Phi-4-mini) for the main effect; ablation on Qwen only |

---

## 2. Li et al., *The Unseen Bias* (arXiv 2512.08374, ICLR 2026) [li2025unseen]

- **Their claim.** In pre-norm MLLMs, visual tokens carry far larger norms than text tokens (5–66× after the
  projector) and this loses visual information. A LayerNorm after the projector helps (one seed, small gains).
- **Our data — Contradicts on binary tasks, Agrees on generation.**
  - On the binary tasks the **high-norm** injection (fixed gate, about 100× the text norm) is the one that uses
    the image well and resists the shortcut; the **low-norm** learned gate (2.6×) is the one that falls back on it.
    A larger visual norm does not mean less visual information here.
  - On VQA-CP the high-norm injection hurts free-form answers, as their account predicts.
- **Reading.** Norm discrepancy is not uniformly harmful: for a single-token class readout a dominant visual
  signal helps; for free-form generation it hurts. Their benchmarks mix both, so they cannot separate the two.
- **Limits.** Different connector (additive cross-attention vs prefix tokens), different LLMs. Their mechanism
  (slower angular updates of high-norm tokens) concerns prefix tokens in the residual stream; ours adds the
  visual signal into the text tokens. `r9-capped` tests the norm effect directly in our connector.
- **Their limitations we remove:** one seed (we use 5–15); no robustness tests (ours is a robustness study);
  prefix projectors only (we test cross-attention injection).

---

## 3. Balaji et al., *One Modality to Forget Them All* (arXiv 2607.16442) [balaji2026forget]

- **Their claim.** The fusion design (MLP, Q-Former, gated cross-attention) is the main structural mediator of
  how far unlearning transfers across modalities.
- **Our data — Contradicts the general idea that connector type drives behaviour, for robustness.** Once the gate
  is fixed, MLP, Q-Former and fixed-gate cross-attention show no significant worst-group difference (5 seeds;
  equivalence not established). The one connector property that matters is the zero-initialised gate.
- **Their limitation we remove.** Their three models also differ in vision encoder (CLIP ViT-L, EVA-CLIP ViT-G,
  OpenCLIP ViT-H), LLM and size, so their connector effect is confounded. Ours changes only the connector.
- **Limits.** Different outcome (unlearning, not robustness), so this is a contrast, not a refutation. A fair
  sentence for the paper: "when the connector is the only change, its type has no detectable effect on
  worst-group accuracy; differences attributed to connector type in comparisons of released models may partly
  reflect their other differences".

---

## 4. Flamingo [alayrac2022flamingo] and released gated models

- **Their claim.** Zero-initialised tanh gates "quickly grow" during training.
- **Our data — Contradicts, for released Llama 3.2 Vision.** Its attention gates are 0.001–0.012 (11B) and
  0.000–0.003 (90B) in the released weights; every layer is below 0.05 (README §14). IDEFICS-80B has 9 of 20
  layers below 0.05. Near-zero gates survive full-scale training in shipped models.
- **Limits.** Small gate values do not show the released models misuse the image (Llama 3.2 clearly uses it;
  see 1c). We did not test released models for shortcut reliance.

---

## 5. Sarridis et al., *Scaling VLMs Is Not Enough* (ACM MM 2026) [sarridis2026scaling] — **Agrees**

- **Their claim.** Across 194 CLIP-style models, scale raises ImageNet accuracy but not worst-group accuracy under
  multi-attribute bias; data curation matters more.
- **Our data.** A larger encoder (SigLIP-large) narrows the gate gap on Waterbirds (+8.0, p = 0.09) but not on
  Watermark (+22.4, 5 seeds). Bigger encoders improve every connector on Waterbirds but do not remove the
  architectural failure.
- **Their limitation we remove.** They study contrastive models only; we test an LLM-based VLM.

---

## 6. VideoDistill [zou2024videodistill] — **Agrees in direction, differs in mechanism**

- **Their claim.** Replacing cross-attention with a language-aware gate reduces language-shortcut reliance in a
  CLIP-based VideoQA model.
- **Our data.** The gate design changes shortcut reliance in an LLM-based VLM too, but the risk we find is the
  opposite kind of gate: a learnable zero-start gate that lets too little image signal through. On VQA-CP, the
  learned gate answers with the training prior more often (39.9% vs 33.8%), consistent with their concern, yet it
  is not less accurate on shifted priors.
- **Limits.** Different gate, task and model family.

---

## 7. SCER [park2026scer] — **No contradiction; a calibration point we must state**

SCER reaches 91.2 worst-group accuracy on Waterbirds with attribute labels on a ResNet-50 (Table 1). Our best
VLM numbers are 74–84 (84.2 for a linear DFR probe on our frozen features). We isolate a cause of failure in VLM
connectors; we do not compete with label-using robustness methods. The paper must say this.

---

## What to claim, and how strongly

| Claim against prior work | Evidence now | Strength | Settles it |
| :--- | :--- | :--- | :--- |
| Near-zero gates also arise in text-coupled injection (Farazi expected not) | Every learned-gate run | Strong in our setting | – |
| There the gate is stuck by a large, self-cancelling gradient, not a dead one | Logged \|∂L/∂α\| 3–1,528 with Δα ≤ 0.003 | Moderate (sign inferred) | Signed-gradient logging, now in the code; round 9 |
| A near-zero gate can still carry a used pathway | Pathway removed / zeroed, 3–4 seeds, every run | Moderate | none |
| The gate's start value, not its learnability, decides robustness | Starts 0.1, 0.5, 0.995, 5 seeds each | Moderate (exploratory p) | `r9-eps01` |
| High injection norm helps a class readout but hurts generation; a norm cap repairs generation but costs robustness | Capped connector: 5 seeds binary, 3 seeds VQA-CP; text-only VQA-CP | Moderate to strong (Watermark p = 0.002) | none |
| Connector type has no robustness effect when it is the only change | MLP vs Q-Former vs fixed gate, 5 seeds | Moderate (no equivalence) | More seeds, TOST |
| Released gated models ship near-zero gates | Llama 3.2 Vision, IDEFICS weights | Strong (direct read) | – |

**What we must not claim.** That Farazi et al. or Li et al. are wrong in their settings; that small gates make
released models shortcut-prone (untested); that norm matching keeps robustness (untested until round 9).

## Measurement added for 1a (24 Nov 2026)

`src/study/engine.py` now also keeps the signed gate gradient, and `study/train_task.py` records per epoch the
signed mean, the fraction of steps that push the gate up (`gate_grad_frac_positive`; 0.5 means fully cancelling)
and the step-to-step sign-flip rate (`gate_grad_sign_flip_rate`). This turns the cancelling-gradient argument into a
measurement. It needs no new run design: every round-9 job that trains a learnable gate records it
(`r9-eps01`, `r9-eps1`, `r9-ablate`).

### Round 12: longer, stable alignment (`results_study/r12_long_align_runs.csv`)
- Setting: lr 3e-4, 3,000 steps (Watermark and Waterbirds) and 6,000 steps (Watermark), 5 paired seeds, Qwen3.5-4B.
- Alignment stable for both arms (no divergence). Zero-start gate peak |tanh α| ≤ 0.003; caption loss ends higher
  than the fixed gate's (2.23–2.34 vs 1.84–2.05). **Supports** "the gate does not open with more training".
- Watermark: +14.4 (3k, p = 0.11, 2/5 collapsed vs 0/5) and +24.7 (6k, p = 0.014, d_z 1.50, 3/5 vs 0/5); the
  300-step gap on the same seeds is +24.6. **Supports.**
- Waterbirds: learnable 61.2 → 71.3; gap +2.7 (p = 0.15), no collapses. **Contradicts** a task-general effect:
  on the easier task longer alignment makes the closed gate almost harmless. Reported in Results and Limitations.
- Caveat: learning rate changed with length, so length and lr are not separated.

## Round 13 (26 Nov 2026)

- **9B Waterbirds pairing.** Seed 3 of the zero-start gate exists twice: round 10 (19.3%, its fixed partner never
  ran) and round 13 (60.1%, paired with a fixed run from the same session). A first merge averaged the two and gave
  +13.9 (p = 0.034); the paper uses the clean pairing (round-10 seeds 0-2, round-13 seeds 3-4): +9.9 [-1.8, 21.6],
  one-sided p = 0.040, d_z = 1.05. The round-10 seed-3 run is reported as unpaired.
- **Connector-DFR on Waterbirds.** The first runs (stage3, stage3b, stage3c kernels) used `--stage3_epochs 1`
  (+0.2, 5 seeds); round 13 used the default of 3 epochs (+8.9, 3 seeds). Both are reported, labelled by epochs.
  README previously said "1 epoch" for all stage-3 runs; corrected.
- **Waterbirds 6,000-step alignment.** With 4 of 5 seeds the gap was +5.2; with all 5 it is +8.5 (p = 0.062). The
  earlier text "the Waterbirds gap almost closes" (from 3,000 steps, +2.7) is replaced: a lower alignment learning
  rate shrinks the gap (+4.4 at 300 steps, 3e-4), longer alignment does not shrink it further.
- **Citations.** All 75 cited entries matched arXiv/Crossref by title and first author. Wording corrected to what
  the sources state for balaji2026forget, rm2026zerogated, farazi2026gate (gate initialisation was not "untested"),
  li2025unseen, sarridis2026scaling, zou2024videodistill, ahn2026evident. The study earlier described as
  "Akbari et al." is Lialin et al. 2023 (arXiv 2304.02080).
