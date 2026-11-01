# Introduction (draft, CARS five-paragraph funnel)

*About 1–1.5 journal pages. Citation keys refer to [`references.bib`](references.bib); replace them with the
journal's citation style.*

---

**¶1 — Hook and global context.**

Vision-language models (VLMs) that connect a pretrained vision encoder to a large language model (LLM) have
become the default way to build general-purpose multimodal systems [yin2024mllmsurvey; zhang2024vlmsurvey;
awais2025foundation]. As these systems move into screening, moderation and medical triage, a failure mode
inherited from earlier deep networks becomes critical: **shortcut learning**, the use of a cue that predicts the
label in training but not in deployment [geirhos2020shortcut]. Shortcuts have already produced high-accuracy
models that read hospital markers instead of lung pathology [degrave2021covid] and watermarks instead of
objects [lapuschkin2019clever]. They fail silently on exactly the under-represented groups where errors are
most costly [brown2023shortcut; ongly2024shortcut].

**¶2 — What is known.**

A substantial body of work on spurious correlations establishes how to measure and mitigate such failures in
classifiers:
- worst-group accuracy over groups defined by label and spurious attribute [sagawa2020groupdro];
- group-aware or label-free reweighting [liu2021jtt; idrissi2022balancing];
- the finding that features usually retain the core signal, so retraining only the readout recovers most
  robustness [izmailov2022feature; kirichenko2023dfr].

For VLMs, recent studies show that multimodal LLMs inherit spurious cues and fail on minority groups
[hosseini2025spurlens; yang2025spuriverse; zhang2024oodmllm]. They attribute this to the vision encoder
[tong2024eyes], to data rather than scale [sarridis2026scaling], or remove it post hoc by unlearning
[sarker2026purge]. In parallel, a separate line of work compares the connectors that map visual features into
the LLM, such as MLP projectors, Q-Formers and gated cross-attention [liu2024llava15; li2023blip2;
alayrac2022flamingo], and ranks them by benchmark accuracy and efficiency [mckinzie2024mm1; cha2024honeybee;
karamcheti2024prismatic].

**¶3 — The problem pivot.**

However, these two lines of work have not met, which leaves three limitations.
1. **No controlled test of the connector.** Every robustness study of VLMs treats the connector as fixed, and
   every connector study measures average accuracy rather than worst-group behaviour. Whether the component
   that decides *what visual information reaches the LLM* also decides *whether the model relies on a
   shortcut* is untested.
2. **Conflicting evidence on gated cross-attention.** Zero-initialised, learnable tanh gates are the standard
   way to inject a new modality into a pretrained LLM: Flamingo, IDEFICS, LLaMA-Adapter and Llama 3.2 Vision
   all use them [alayrac2022flamingo; zhang2024llamaadapter; grattafiori2024llama3]. Cross-attention
   injection performs well with frozen backbones [laurencon2024matters]. Yet learnable injection gates can end
   up behaviourally closed [farazi2026gate], and every attention gate in the released Llama 3.2 Vision
   checkpoints is below 0.012. The consequence of a near-shut gate for *what the model relies on* has never
   been measured. Work on modality imbalance shows that multimodal networks can greedily ignore a modality
   [wu2022greedy; huang2022modality], but not which component of an LLM-based VLM causes it, nor what it does
   to spurious-group robustness.
3. **Results from too few seeds.** Comparisons rest on one or a few training runs, although seed variance
   alone can reverse conclusions in deep learning [henderson2018matters; bouthillier2021variance].

The field therefore cannot say whether a widely used architectural choice silently causes shortcut learning,
or how to avoid it.

**¶4 — The intervention.**

To address these limitations, this study tests whether the vision-language connector, and specifically its
injection gate, determines shortcut reliance in LLM-based VLMs.
- **What is held fixed:** the LLM (Qwen3.5-4B, replicated with Phi-4-mini), frozen SigLIP features, data,
  prompts and training budget.
- **What varies:** only the connector, at four levels: an MLP projector, a Q-Former, and gated
  cross-attention with a learnable or a fixed gate. The two gate variants have identical parameters; a third,
  learnable gate initialised open separates trainability from initialisation.
- **Where it is measured:** worst-group accuracy on Waterbirds, a controlled watermark task and CelebA.
- **How it is tested:** hypotheses formed on a 3-seed pilot and confirmed on up to 12 new paired
  seeds (15 in total), with effect sizes,
  robust and non-parametric tests, and an equivalence test.
- **Fixes and rival explanations:** we log the gate during training, read the gates of released IDEFICS
  checkpoints, test a connector-only analogue of deep feature reweighting, and probe rival explanations
  (token count, encoder scale, frozen LLM, training length).

**¶5 — Contributions and roadmap.**

This paper makes three contributions.
1. **A transmission mechanism.** We identify the learnable injection gate as a *transmission bottleneck*
   between features and readout. Initialised at zero, the gate never opens, and the model then relies on the shortcut; fixing it raises
   worst-group accuracy by 10.6 and 23.8 points (d_z = 1.10 and 1.27). This extends the feature–readout account
   of shortcut learning and links gate closure to robustness.
2. **Variance as an outcome.** The learnable gate turns robustness into a seed lottery, with up to 8 of 13
   runs collapsing. Once the gate is fixed, we detect no significant difference between connector types, although equivalence
   is not established. Together these establish
   that architectural robustness claims require seed-level evaluation.
3. **A practical fix and its boundary.** Retraining only the connector on a small group-balanced set raises
   watermark worst-group accuracy to 89.6% with near-zero seed variance, but not when the shortcut is entangled
   with the object.

Section 2 reviews related work. Section 3 describes the design and statistical protocol. Section 4 reports the
results, Section 5 discusses mechanisms and implications, and Section 6 concludes. Code, per-seed results and
figures are public.
