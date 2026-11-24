# Novelty: what is new here, and what is not

A reviewer's first question is "what does this add over prior work?". This page sets out our answer.
- **Prior work searched** (1 Nov 2026; re-checked 24 Nov 2026, see §6): arXiv full-text queries, OpenAlex, Semantic Scholar, and Google Scholar
  via web search.
- **Search terms:** gated cross-attention with spurious / shortcut; zero-initialised gate with robustness;
  connector or projector with worst-group accuracy / spurious; gate collapse; modality gate.
- **Code inspected:** the released Flamingo-type models' code, in `transformers` 5.21
  (`modeling_idefics.py`, `modeling_mllama.py`).

The claims below are worded so that each one is backed by a result in this repository. Each also names the
closest prior work it must be distinguished from.

## 1. The closest prior work

| Work | What it shows | What it does not show (our gap) |
| :--- | :--- | :--- |
| **Flamingo** [alayrac2022flamingo] | Introduces tanh-gated cross-attention, gates initialised at 0, one block every 4th–7th LM layer. Gates grow over 500k steps. Removing the gate costs 4.2 points. | Whether a gate that stays small changes *what the model relies on*. No worst-group or shortcut evaluation. |
| **IDEFICS** (HF M4) and **Llama 3.2 Vision** [grattafiori2024llama3] | The same zero-initialised tanh gates ship in released models: IDEFICS uses one gate per block; Llama 3.2 gates the attention and the MLP separately, in 8 of 40 layers (11B). | The gate values and their consequences are not reported. We read them from the checkpoints (README §14): every Llama 3.2 Vision attention gate is below 0.012 (11B) and 0.003 (90B), the same regime as our learned gate. |
| **LLaMA-Adapter / V2** [zhang2024llamaadapter; gao2023llamaadapterv2] | A zero-initialised learnable gate injects adapter prompts, including visual ones, into a frozen LLM, to keep it intact early in training. | No analysis of gate values or of robustness. |
| **Gated multimodal fusion** (GMU [arevalo2017gmu] and later dynamic gates) | An input-dependent gate learns how much each modality contributes to a fused representation, and improves average accuracy. | Late-fusion classifiers without an LLM; the gate is computed per sample, not a parameter that starts at zero; never evaluated for shortcut reliance, worst-group accuracy or seed variance. |
| **ReZero / SkipInit / LayerScale** [bachlechner2021rezero; de2020skipinit; touvron2021cait] | Learnable residual scales initialised at 0 or a small ε make deep unimodal networks train stably; LayerScale shows the start value matters (ε beats 0). | Unimodal; outcome is convergence and average accuracy. The scaled branch is never a modality's only path, so the robustness cost of a near-zero start cannot appear. |
| **Zero-initialised attention theory** [diep2025zeroinit] | Prompts and gating factors of zero-initialised attention can be estimated optimally (a mixture-of-experts view). | Estimation theory only; nothing on spurious correlations. |
| **The Gate Always Closes** [farazi2026gate] | Across five gated designs in frozen VLMs, the gated pathway ends up *behaviourally* closed: ablating it at inference changes nothing. Recommends fixed injection. | Measures relational accuracy on GQA and VSR, not shortcut reliance, worst-group accuracy or seed variance. No controlled comparison with other connector types. |
| **Modality imbalance** [wu2022greedy; huang2022modality; peng2022balanced; wei2025onthefly] | Multimodal networks greedily learn one modality and under-fit the other. Which modality wins depends on initialisation. Gradient modulation helps. | Late-fusion audio-visual or RGB-depth classifiers. No LLM-based VLMs, no connectors, no spurious-group evaluation. |
| **Shortcuts in VLMs and MLLMs** [varma2024ravl; hosseini2025spurlens; yang2025spuriverse; sarridis2026scaling; sarker2026purge] | VLMs and MLLMs inherit spurious cues. RaVL fixes them with a region-aware loss; SpurLens attributes them mostly to the vision encoder; scaling does not fix them; unlearning (PURGE, which also updates the projector) reduces them. | None isolates the connector as the manipulated variable with everything else fixed. |
| **Connector comparisons** [mckinzie2024mm1; cha2024honeybee; karamcheti2024prismatic] | Connector type matters little for benchmark accuracy; locality matters for some tasks. | Average accuracy only; no group robustness. |

## 2. Our claims, worded to survive review

**C1. A zero-initialised learnable gate in a single input-level connector block is a controlled cause of shortcut
reliance in LLM-based VLMs.** (Scope narrowed on 24 Nov 2026: per-layer gates show no gap, and VQA-CP does not
reproduce it.)
- *Mechanism evidence:* a learnable gate that starts half open matches the fixed gate (+11.4 and +20.1 over the
  zero start, p = 0.014 and 0.006), so the zero start, not learnability, drives it. Leaving the gate out of
  clipping or weight decay does not open it.
- *Evidence:* one variable is changed with everything else fixed (LLM, features, data, budget, seeds). Fixing
  the gate raises WGA by +10.6 and +25.2 points (15 seeds each).
- *Robustness:* the gain holds on confirmation seeds alone, with unpaired tests and with label-free selection.
  It replicates on Phi-4-mini for Waterbirds.
- *New relative to Farazi et al.:* the outcome is shortcut reliance and worst-group accuracy, not relational
  accuracy, and the connector is the only variable.
- *New relative to the modality-imbalance work:* the cause is a specific, widely shipped architectural
  component, in LLM-based VLMs, measured on spurious-correlation groups.

**C2. A learnable gate turns robustness into a seed lottery.**
- *Evidence:* the learned gate collapses below 50% WGA in 2 of 15 and 10 of 15 seeds; the fixed gate never
  does (Watermark Fisher exact p = 1.0 × 10⁻⁴; Waterbirds variance F-test p = 8.6 × 10⁻⁵, Brown–Forsythe 0.045).
- *Link to theory:* this is the empirical counterpart of modality competition, where the winner depends on
  initialisation [huang2022modality].
- *Gap filled:* no prior gate or connector study reports seed-level robustness.

**C3. Once the gate is fixed, no significant connector-type difference is detected.**
- *Evidence:* MLP vs Q-Former, p = 0.23 and 0.40.
- *Caveat, stated as such:* equivalence is not established.
- *Gap filled:* this extends the "connector type matters little" finding [mckinzie2024mm1] from average
  accuracy to worst-group accuracy.

**C4. A connector-only fix works for localised shortcuts, not entangled ones.**
- *Evidence:* connector-DFR reaches 89.6 ± 0.5 on Watermark, with no gain on Waterbirds.
- *Gap filled:* readout-level fixes [kirichenko2023dfr] had not been tested at the VLM connector.

**Claims we deliberately do not make:**
- We do not claim to be the first to observe that gates stay closed. Farazi et al. did.
- We do not claim that zero-initialised gating is new. ReZero and SkipInit introduced zero-initialised learnable
  residual scales; Flamingo and LLaMA-Adapter carried them into vision-language and adapter models.
- We do not claim that the initial value of a learnable scale matters for *optimisation*; LayerScale showed
  that. We claim it matters for *robustness* when the scaled branch is a modality's only path.
- We do not claim that the gate is *sufficient* for shortcut reliance. With SigLIP-large on Waterbirds the gap
  shrinks to +8.0 (p = 0.09).
- We do not claim the effect for per-layer injection: per-layer gated cross-attention (Flamingo / Llama 3.2
  layout) shows no gap (README §17). The claim covers one input-level gated block.
- We do not claim the effect for open-ended VQA: on a VQA-CP v2 subset the fixed gate is worse (README §18).
- We do not claim the gate pathway is closed. The pathway ablation shows the learned-gate model still uses the
  image (without it, accuracy falls to the majority-class share); its injected signal is 15–35× weaker than the
  fixed gate's. This differs from Farazi et al., whose learned-gate pathways (designs S1–S4) were behaviourally closed.

## 3. Questions a reviewer is likely to ask, and the answers

**"If the gate is shut, how does the learned-gate model see the watermark at all?"**
The shortcut is itself visual, so some visual signal must pass. A gate of ≈ 0.0005 passes very little unless
the output layer grows to offset it.

Our account: the small residual injection carries only the strongest, most linearly available cue (the
watermark, or the background), and the finer object evidence is lost. Two round-4/5 measurements test this
directly:
- **Injection ratio** (`inject_ratio`): how large the injected signal is relative to the text embedding.
- **Pathway ablation** (`ablated_mean_over_templates`): how much accuracy depends on it.

Measured: the injection ratio is 3.5 for the learned gate vs 112 for the fixed gate, and with the image removed
the learned-gate model falls to the majority-class share (Waterbirds average accuracy 89.4 → 77.8; Watermark
77.9 → 50.0, 1–2 seeds). The pathway is open but narrow, not behaviourally closed. Round 9 repeats this on
5 seeds with the pathway absent, with zero image features, and with another example's image (shuffled).

**"Gating and shortcut learning are both well studied. What is new?"**
- *Both halves are old, and we say so.* Gated fusion goes back to the GMU [arevalo2017gmu], zero-initialised
  gates to ReZero, SkipInit and Flamingo; shortcuts in VLMs are documented by RaVL, SpurLens, SpuriVerse, the
  scaling study and PURGE.
- *The two literatures have not met.* Gating papers report convergence and average accuracy; shortcut papers
  fix data, losses or unlearning and keep the architecture fixed. None asks whether a standard gate design
  *causes* shortcut reliance.
- *What we add:* a controlled manipulation (only the gate changes; LLM, features, data, budget and seeds are
  fixed), a confirmatory test at 15 paired seeds with Holm correction, a second LLM, a seed-level outcome
  (collapse rate), controls that isolate the zero start (open, half-open, learning rate, alignment length),
  and a label-free, zero-cost design rule.
- *What we do not claim:* a new architecture, a new fusion mechanism, or a new debiasing method that beats
  label-using ones.

**"Is this not just the modality-imbalance problem?"**
- *What it shares:* it is a case of it, and we say so.
- *What is new:*
  1. The setting: LLM-based VLMs with a connector.
  2. The cause: a single component that ships in released models (Flamingo, IDEFICS, Llama 3.2 Vision,
     LLaMA-Adapter).
  3. The consequence: worst-group failure and seed variance, not only lower average accuracy.
  4. The fix: fixing the gate costs nothing, unlike gradient modulation.
- Gradient modulation (OGM-GE) is a natural extra baseline, listed in §5.

**"Your connector is not Flamingo."**
- *What we say:* we call it input-level gated cross-attention with a Flamingo-type gate.
- *The test:* a per-layer version (`xattn_deep`, gated blocks before layers 0, 8, 16 and 24, parameter-matched).
- *The result:* **no gap** (Waterbirds 70.8 vs 70.4, 5 seeds; Watermark 68.3 vs 70.6, 4 seeds), although its learned
  gates also stay small (mean |tanh α| 0.027). The claim is therefore limited to one input-level block. A likely
  reason: four gates at different depths give the image several routes, and each is ~7× larger than the single
  input-level gate (0.027 vs 0.0037).

**"Flamingo trains 500k steps and its gates open."**
- *What we say:* our regime is short, adapter-style alignment, and the paper says so.
- *The test:* 3,000 alignment steps (10×) leave the gate shut (peak 0.003). The learned-gate alignment then
  diverged (caption loss 2.0 → 7.9), so the wider gap (+59 / +55) shows fragility, not a clean effect
  (README §17).
- *Supporting evidence:* the released IDEFICS-80B gates show that near-shut gates survive full training in
  9 of 20 layers.

**"4-bit models only."**
- *The test:* a full-precision (fp16 weights) replication, 3 seeds: +10.0 and +21.1, same direction as 4-bit
  (not significant with 3 seeds). Quantisation does not explain the gap.

## 4. Why we did not pursue the other two suggestions

**Forcing a released IDEFICS model's gates open.** Not used as evidence:
- A trained model has adapted to its own gate values, so forcing them open is a distribution shift. Any drop
  in accuracy would be uninterpretable.
- IDEFICS-9B has only 1 of 8 near-shut gates, and the 80B model does not fit a T4.
- What we use instead is cheap and unambiguous: we read the gates of released models, now including
  Llama 3.2 Vision, in README §14. In both Llama 3.2 Vision sizes every cross-attention gate is below 0.05
  (11B: 8 of 8; 90B: 20 of 20), so the regime we study is the one a flagship model ships in.

**SpuriVerse.** Out of scope for this paper:
- It is a 1,364-question multiple-choice benchmark of open-ended VQA. Our models are trained per task with a
  two-answer readout, so they cannot be scored on it without a different training protocol.
- It is listed as future work.

## 5. Strengthening in progress (round 9) and remaining

Chosen for how much each adds against reviewer objections and the closest prior work (§6):

| Experiment (round 9) | Why it matters | Prior work it answers |
| :--- | :--- | :--- |
| **Norm-matched injection** (`bottleneck_capped`: fixed gate, injected norm ≤ text norm per token), binary tasks 5 seeds + VQA-CP 3 seeds | Turns the VQA-CP failure into a testable design rule. If capping keeps the binary-task gain and repairs generation, the recommendation becomes "no learnable zero gate, and matched norm". | Farazi et al. recommend matched-norm injection without a gate (batch-mean rescale); Li et al. (2026) tie norm discrepancy to visual information loss. Neither measures shortcut reliance |
| **Gate start-value sweep**: tanh α = 0.01 and 0.1 (`bottleneck_eps01`, `bottleneck_eps1`), 5 seeds | With 0, 0.5 and 0.995 already run, this gives a dose-response in the start value and a practical threshold. | LayerScale shows a small non-zero start beats zero for optimisation; Farazi et al. never vary the gate's start value |
| **Pathway tests on 5 seeds**: pathway absent, zero image features, another example's image (shuffled) | Our main point of difference from Farazi (weak, not closed) currently rests on 1–2 seeds. Shuffled images separate "uses this image" from "uses image statistics". | Farazi et al.'s on / zeroed / absent protocol; the shuffled-image control is one they lack |
| **Text-only baseline on VQA-CP** (`--text_only`, same LLM, LoRA and budget) | Bounds how much image information either VQA-CP model uses. | Language-prior baselines in VQA-CP [agrawal2018vqacp] |

Remaining, not queued:
- Gradient modulation (OGM-GE [peng2022balanced]) as a baseline next to the fixed gate.
- MM-SpuBench, SpuriVerse and SpurLens: open-ended MLLM benchmarks that need a different training protocol.
- Larger LLMs (7–8B) at 4-bit, if a larger GPU becomes available.

## 6. Novelty re-check (24 Nov 2026): papers read in full

Every paper below was read from its arXiv HTML text (not search snippets); numbers are from their tables. Searches:
arXiv full-text (connector/projector, gated cross-attention, Flamingo gate, modality imbalance, worst-group MLLM,
language prior) and OpenAlex. Semantic Scholar refused automated access, so citation-graph coverage is missing.

**Verdict.** No paper studies a learnable zero-initialised injection gate, or the connector type, as a cause of
shortcut reliance or worst-group failure in LLM-based VLMs. The claim "no work compares connector types" is **false
as a general statement**: Balaji et al. (2026) compare the same three connector families, on unlearning. Our claim
is therefore worded as *no controlled comparison on worst-group accuracy or spurious-correlation reliance*.

| Work (read in full) | What it does | Where we differ, and what we take from it |
| :--- | :--- | :--- |
| **Farazi et al., The Gate Always Closes** (arXiv 2607.23335 v2, Nov 2026; preprint) [farazi2026gate] | Five gated injection designs on frozen LLaVA-1.5-7B with LoRA (3 epochs on GQA), plus Qwen2-VL-7B and InternVL3-8B. Gates start at 0 and settle near zero; at the learned gate, ablation moves GQA-rel by ≤ 0.5 points (S1–S4), so the pathway is behaviourally closed. A dead-gradient regime (Lemma 1, Prop. 1) explains it for *caption-invariant* signals. Inference modes: on, zeroed (slots kept, content × 0), absent. Recommends injecting "at matched norm without an optional gate"; their matched norm rescales each prefix token to the norm of the batch-mean text embedding (Eq. 11). | (1) They measure relational accuracy, SugarCrepe, VSR and POPE; **no spurious-correlation, shortcut or worst-group test**. (2) **No gate-initialisation ablation**: our start sweep (0, 0.01, 0.1, 0.5, 0.995) is new relative to them. (3) Their signals are caption-invariant; they expect text-coupled injection *not* to close and list it as future work. **Our connector is text-coupled** (text tokens query the patches) and its gate still stays near zero: new evidence on their open question. (4) Their gated pathways are closed; ours is weak but used (pathway ablation). (5) Their norm fix is a batch-mean rescale; our `bottleneck_capped` is a per-token cap. |
| **Li et al., The Unseen Bias** (arXiv 2512.08374; ICLR 2026 proceedings listing) [li2025unseen] | In pre-norm MLLMs, visual tokens carry 5–66× the text-embedding norm after the projector (Table 2: LLaVA-1.5 ≈ 37×, Qwen2.5-VL ≈ 66×) and update more slowly; one LayerNorm after the projector, initialised to the text norm, gives small benchmark gains (one seed). | Prefix projectors only, no robustness tests. Our fixed gate injects 70–130× the text norm, the same regime, which explains its VQA-CP generation harm. |
| **Balaji et al., One Modality to Forget Them All** (arXiv 2607.16442, Jul 2026) [balaji2026forget] | Compares LLaVA-1.5 (MLP), InstructBLIP (Q-Former) and IDEFICS (gated cross-attention) on cross-modal unlearning; fusion design mediates transfer. | Connector is confounded with encoder, LLM and scale; no shortcut or worst-group measures. Ours changes only the connector. Must be cited for the connector comparison. |
| **Sarridis et al., Scaling VLMs Is Not Enough** (arXiv 2607.28211, ACM MM 2026) [sarridis2026scaling] | 194 CLIP-style models, zero-shot worst-group accuracy on CelebA and UrbanCars; scale helps ImageNet but not worst-group accuracy under multi-attribute bias; data curation matters more. | Contrastive models, no connector. Consistent with our encoder result: SigLIP-large narrows the gap on Waterbirds but not on Watermark. |
| **Zou et al., VideoDistill** (CVPR 2024) [zou2024videodistill] | A language-aware gating unit replaces cross-attention in a CLIP-based VideoQA model and reduces language-shortcut reliance. | Closest precedent that fusion/gating design affects shortcuts; not an LLM-based VLM, no zero-initialised tanh gate, indirect evidence. |
| **Ahn et al., EVIDENT** (NeurIPS 2026) [ahn2026evident] | Entity-grounded adapter with a parameter-free gate on Qwen2.5-VL improves cross-domain temporal grounding against dataset shortcuts. | No worst-group metric; the gate is a fixed similarity function, not a learned zero-start gate. |
| **Ye et al., MM-SpuBench** (KDD 2026 D&B) [ye2026mmspubench] | 2,400 multiple-choice items over nine spurious-attribute types; MLLMs keep relying on spurious cues. | Evaluation only, no architecture analysis. Candidate external benchmark (needs an open-ended protocol). |
| **Park et al., SCER** (ICLR 2026) [park2026scer] | Embedding regulariser on ResNet-50/BERT with attribute labels: Waterbirds worst-group 91.2 vs ERM 69.1 (Table 1). | A label-using fix on classifiers. Our best VLM numbers (74–84) are far below it; we isolate a cause, we do not compete on absolute WGA, and the paper must say so. |

Also seen, cited as context only: ProjLens (2604.19083, projector-level backdoors), "The Expense of Seeing" (2604.20665,
language-prior exploitation in the encoder-projector-LLM design, conceptual), and modality-bias work in MLLMs
(2505.18657, 2508.10552, 2510.22836). These were classified from abstracts, not read in full.

**Consequences for round 9 (applied):**
- The pathway tests are named to match Farazi et al.: `absent` equals their "absent" for our fused connector (and
  their "zeroed" for prefix connectors); `zero_features` and `shuffle` are additional controls. The shuffled-image
  control is one they list as missing.
- `bottleneck_capped` is described as a per-token cap motivated by, not identical to, their matched-norm rescale.
