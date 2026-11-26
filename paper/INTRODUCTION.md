# Introduction (draft, CARS five-paragraph funnel)

*About 1.5 journal pages. Framed for the IJCV special issue on multimodal large language models (topics I:
encoder–LLM pipelines and cross-modal alignment; IV: robustness, bias and reproducibility). Citation keys refer to
[`references.bib`](references.bib). Paragraph labels (¶) are writing aids and are removed by `springer/convert.py`.*

---

Multimodal large language models (MLLMs), also called vision-language models (VLMs), are now built by joining a
pretrained vision encoder to a large language model (LLM) through a small trainable connector [yin2024mllmsurvey; huang2025vitsurvey; awais2025foundation].
The connector decides what visual information reaches the LLM, and in the most widely shipped design it does so
through a learnable injection gate. They inherit a failure mode of earlier deep networks: shortcut learning, the use of a cue that predicts
the label in training but not in deployment [geirhos2020shortcut]. Shortcuts have produced accurate models that
read hospital markers instead of lung pathology [degrave2021covid] and watermarks instead of objects
[lapuschkin2019clever], and they fail silently on exactly the under-represented groups where errors cost most
[brown2023shortcut; ongly2024shortcut].

Work on spurious correlations shows how to measure and reduce such failures in classifiers: worst-group accuracy
over groups defined by label and spurious attribute [sagawa2020groupdro], group-aware and label-free reweighting
[liu2021jtt; idrissi2022balancing], invariance-seeking training [huang2024losingtickets; mitchell2026wule], and the
finding that features usually keep the core signal, so retraining only the readout recovers most robustness
[izmailov2022feature; kirichenko2023dfr]. For vision-language models, recent studies show that MLLMs inherit spurious
cues and language priors and fail under shift [hosseini2025spurlens; yang2025spuriverse; ye2026mmspubench;
zhu2024vlalignood]. They attribute this to the vision encoder [tong2024eyes], to data rather than scale
[sarridis2026scaling], or remove it after training [sarker2026purge]. A separate line of work compares the
connectors themselves (MLP projectors, Q-Formers and gated cross-attention [liu2024llava15; li2023blip2;
alayrac2022flamingo]) by benchmark accuracy and efficiency [mckinzie2024mm1; cha2024honeybee; karamcheti2024prismatic].

These lines of work have not met, which leaves three gaps.
1. No controlled test of the connector. Robustness studies keep the connector fixed. Connector studies measure
   average accuracy, or compare released models that also differ in encoder, LLM and scale [balaji2026forget].
   Whether the component that decides *what visual information reaches the LLM* also decides *whether the model
   relies on a shortcut* is untested.
2. The injection gate is common but poorly understood. Zero-initialised learnable tanh gates are the standard
   way to inject a modality into a pretrained LLM: Flamingo, IDEFICS, LLaMA-Adapter and Llama 3.2 Vision all use
   them [alayrac2022flamingo; zhang2024llamaadapter; grattafiori2024llama3]. Such gates are known to move little
   during training, and a larger start can help accuracy [lialin2023scalable; yan2026dgp]. Farazi et al. show that such gates can
   end up near zero and their pathways behaviourally closed, through a vanishing first-order gradient when the
   injected signal does not depend on the text [farazi2026gate]; Li et al. show that visual tokens far larger than
   text tokens lose information [li2025unseen]. Neither measures what a near-shut gate does to *what the model
   relies on*, and Farazi et al. leave text-coupled injection, where the gradient need not vanish, as an open case.   Every attention gate in the released Llama 3.2 Vision weights is at most 0.012.
3. Results from too few seeds. Comparisons rest on one or a few training runs, although seed variance alone
   can reverse conclusions [henderson2018matters; bouthillier2021variance].

We test whether the vision-language connector, and specifically its injection gate, determines shortcut reliance
in LLM-based VLMs.
- Held fixed: the LLM (Qwen3.5-4B, replicated with Phi-4-mini and, exploratorily, Qwen3.5-9B), frozen SigLIP features, data, prompts and
  training budget.
- Varied: only the connector: an MLP projector, a Q-Former, and text-coupled gated cross-attention whose tanh
  gate is learnable from zero, learnable from a non-zero start, or fixed. The gate variants have the same parameter count.
- Measured: worst-group accuracy on Waterbirds, a controlled watermark task and CelebA, and free-form answers
  under shifted answer priors on VQA-CP v2 [agrawal2018vqacp].
- Tested: hypotheses formed on a 3-seed pilot and confirmed on new paired seeds (15 in total), with effect
  sizes, robust and non-parametric tests, Holm correction and an equivalence test; mechanism controls on gate start,
  learning rate, clipping, weight decay, alignment length and alignment learning rate, injection depth and weight precision;
  two standard fixes (DFR and JTT) applied to both gates; and a behavioural
  test that removes the visual pathway at inference.

We do not propose a robustness method that competes with label-using approaches, which reach higher absolute
worst-group accuracy [park2026scer]. We isolate a cause.

1. **A controlled architectural cause.** In a single input-level connector, a learnable gate that starts at zero
   makes shortcut reliance more likely. Fixing the gate raises worst-group accuracy by 10.6 and 25.2 points
   (d_z = 1.10 and 1.34, 15 paired seeds; all four LLM × task cells survive Holm correction). Gates started at
   0.1 or higher come close to the fixed one while a start of 0.01 falls back near zero, so a start at or near
   zero, not learnability, is the cause.
2. **A different mechanism from the one known.** That zero-start gates stay small is known
   [lialin2023scalable; farazi2026gate]; why is not, in a text-coupled connector, where the dead-gradient account does
   not apply. We measure it directly: the gate's gradient is large, yet the gate barely moves, because the gradient
   changes sign on almost half of the steps. The
   pathway is weak, not closed: removing it makes the model give one answer to every image. Gate value is therefore not
   evidence of use in either direction.
3. **Robustness as a seed-level outcome.** The zero-start gate collapses below 50% worst-group accuracy in up to 10 of
   15 seeds on Qwen3.5-4B and in 5 of 5 on Qwen3.5-9B Watermark, the fixed gate in none (on Phi-4-mini Watermark the fixed
   gate still collapses in 6 of 10). With the gate fixed, connector type has no detectable effect (equivalence not
   established).
4. **Boundaries, reported as found.** Per-layer gated injection shows no gap; a stronger encoder narrows the gap on
   one task but not the other; and on open-ended VQA-CP the strong fixed injection harms free-form generation. Capping
   the injection at the text norm repairs generation but, at 4B, gives back most of the robustness gain, a trade-off that
   prior norm-matching advice could not see because it did not measure shortcuts.
5. **Standard fixes and reproducible evidence.** Retraining only the connector on a small group-balanced set reaches
   89.6% worst-group accuracy on the watermark task and rescues the zero-start gate there, but not on Waterbirds,
   where the fixed gate keeps a 14-point lead; error upweighting (JTT) rescues neither. All experiments ran on free
   T4 GPUs; code, job scripts, pinned versions and per-seed results are public.

Section 2 reviews related work. Section 3 describes the design and the statistical protocol. Section 4 reports and
discusses the results, including their limits.
