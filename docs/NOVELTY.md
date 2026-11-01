# Novelty: what is new here, and what is not

A reviewer's first question is "what does this add over prior work?". This page sets out our answer.
- **Prior work searched** (1 Nov 2026): arXiv full-text queries, OpenAlex, Semantic Scholar, and Google Scholar
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
| **ReZero / SkipInit / LayerScale** [bachlechner2021rezero; de2020skipinit; touvron2021cait] | Learnable residual scales initialised at 0 or a small ε make deep unimodal networks train stably; LayerScale shows the start value matters (ε beats 0). | Unimodal; outcome is convergence and average accuracy. The scaled branch is never a modality's only path, so the robustness cost of a near-zero start cannot appear. |
| **Zero-initialised attention theory** [diep2025zeroinit] | Prompts and gating factors of zero-initialised attention can be estimated optimally (a mixture-of-experts view). | Estimation theory only; nothing on spurious correlations. |
| **The Gate Always Closes** [farazi2026gate] | Across five gated designs in frozen VLMs, the gated pathway ends up *behaviourally* closed: ablating it at inference changes nothing. Recommends fixed injection. | Measures relational accuracy on GQA and VSR, not shortcut reliance, worst-group accuracy or seed variance. No controlled comparison with other connector types. |
| **Modality imbalance** [wu2022greedy; huang2022modality; peng2022balanced; wei2025onthefly] | Multimodal networks greedily learn one modality and under-fit the other. Which modality wins depends on initialisation. Gradient modulation helps. | Late-fusion audio-visual or RGB-depth classifiers. No LLM-based VLMs, no connectors, no spurious-group evaluation. |
| **Shortcuts in MLLMs** [hosseini2025spurlens; yang2025spuriverse; sarridis2026scaling; sarker2026purge] | MLLMs inherit spurious cues. SpurLens attributes them mostly to the vision encoder; scaling does not fix them; unlearning (PURGE, which also updates the projector) reduces them. | None isolates the connector as the manipulated variable with everything else fixed. |
| **Connector comparisons** [mckinzie2024mm1; cha2024honeybee; karamcheti2024prismatic] | Connector type matters little for benchmark accuracy; locality matters for some tasks. | Average accuracy only; no group robustness. |

## 2. Our claims, worded to survive review

**C1. The connector's gate is a controlled cause of shortcut reliance in LLM-based VLMs.**
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
  does (H2, F-test p = 8.6 × 10⁻⁵).
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
- We do not claim that the gate is *sufficient* for shortcut reliance. With SigLIP-large it is not.
- We do not claim results for per-layer injection or fp16 weights until the round-6 runs are in.
- We do not claim the gate pathway is closed: the learned gate still injects visual signal, 15–35× less
  than the fixed gate (README §17).

## 3. Questions a reviewer is likely to ask, and the answers

**"If the gate is shut, how does the learned-gate model see the watermark at all?"**
The shortcut is itself visual, so some visual signal must pass. A gate of ≈ 0.0005 passes very little unless
the output layer grows to offset it.

Our account: the small residual injection carries only the strongest, most linearly available cue (the
watermark, or the background), and the finer object evidence is lost. Two round-4/5 measurements test this
directly:
- **Injection ratio** (`inject_ratio`): how large the injected signal is relative to the text embedding.
- **Pathway ablation** (`ablated_mean_over_templates`): how much accuracy depends on it.

Measured so far: the injection ratio is 3.5 for the learned gate vs 112 for the fixed gate, so the pathway
is open but 15–35× weaker, and under Group DRO the same narrow pathway reaches ~70 WGA. If ablation sends the learned-gate model to chance, the pathway is open but narrow. If nothing changes, it is
behaviourally closed and the shortcut is learned from elsewhere. Either way, the answer is reported, not
assumed.

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
- *The test:* a per-layer version (`xattn_deep`, gated blocks before layers 0, 8, 16 and 24, parameter-matched)
  tests whether the result carries over to the Flamingo / Llama 3.2 layout (round 6).

**"Flamingo trains 500k steps and its gates open."**
- *What we say:* our regime is short, adapter-style alignment, and the paper says so.
- *The test:* 3,000 alignment steps (10×) leave the gate shut (peak 0.003) and widen the gap to +59 / +55
  (README §17).
- *Supporting evidence:* the released IDEFICS-80B gates show that near-shut gates survive full training in
  9 of 20 layers.

**"4-bit models only."**
- *The test:* a full-precision (fp16 weights) replication of the gate comparison is in round 6.

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

## 5. Remaining optional strengthening

- Gradient modulation (OGM-GE [peng2022balanced]) as a baseline next to the fixed gate.
- A natural-image MLLM benchmark with a generative protocol (SpuriVerse, SpurLens).
- Larger LLMs (7–8B) at 4-bit, if a larger GPU becomes available.
