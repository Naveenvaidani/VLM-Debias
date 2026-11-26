# Title, abstract and keywords

## Target journals (computer vision / machine learning, Q1)

| Journal | Publisher | Why it fits | Abstract limit |
| :--- | :--- | :--- | :--- |
| IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) | IEEE | Publishes VLM surveys and robustness studies we cite; values controlled analysis | ~250 words |
| International Journal of Computer Vision (IJCV) | Springer | Published TokenPacker, VQA v2 and the CIB robust-VQA paper we build on | ~250 words |
| Pattern Recognition | Elsevier | Strong venue for robustness / bias analyses of deep models | ~250 words |
| Neural Networks | Elsevier | Fits the mechanism (gating, optimisation dynamics) angle | ~250 words |
| Information Fusion | Elsevier | Multimodal fusion focus; published the Clever-Hans removal work we cite | ~250 words |

TPAMI or IJCV first. Pattern Recognition is a strong alternative with a faster turnaround.

## Title options

Each option is 10–15 words, with the keywords in the first 65 characters and no fluff.

1. **Declarative (recommended):** *Learnable Injection Gates Induce Shortcut Learning in Vision-Language Models*
   (10 words). "Induce" rather than "Cause": a stronger encoder narrows the gap on Waterbirds, and per-layer gates
   and VQA-CP show no gap, so the gate is a risk factor, not a sufficient cause. After round 5-8 the title should
   keep a scope word, for example *A Zero-Initialised Injection Gate Makes Shortcut Learning More Likely in
   Vision-Language Models*, or add "in Input-Level Connectors".
2. **Structured:** *Shortcut Learning in Vision-Language Models: The Role of the Vision-Language Connector
   and Its Injection Gate* (16 words; can drop "Its Injection Gate")
3. **Hook and subtitle:** *When the Gate Stays Shut: Connector Design and Spurious Correlations in Multimodal
   Large Language Models* (15 words)

## Abstract (as in paper/springer/main.tex, 250 words; IJCV limit 150–250)

> **Purpose.** Vision-language models that connect a vision encoder to a language model can rely on spurious cues such as backgrounds or watermarks. Connectors are compared on average accuracy alone.
>
> **Methods.** We hold the language model, frozen vision features, data and training budget fixed and vary only the connector: a multilayer-perceptron projector, a Q-Former, and gated cross-attention whose tanh injection gate is learnable from different starts or fixed. We measure worst-group accuracy on three shortcut tasks with three language models of 3.8B to 9B parameters, on up to fifteen paired seeds.
>
> **Results.** The zero-initialised gate never opens, although its large gradient changes sign on almost half the steps, and the model relies on the shortcut while still using the image. Fixing the gate raises worst-group accuracy by 10.6 and 25.2 points (d_z = 1.10 and 1.34, p < 0.001) and removes runs below 50%; at 9B the watermark gain doubles to 50.3 points. Gates started at 0.1 or higher approach the fixed gate; a start of 0.01 falls back near zero. Connector-level reweighting removes the gap on only one task. With the gate fixed, connector type has no detectable effect, and per-layer gates show no gap. On open-ended question answering, strong injection harms generation; capping it at the text norm repairs generation but, at 4B, gives back most of the robustness gain.
>
> **Conclusion.** A gate started at or near zero is a controllable architectural cause of shortcut reliance, injection strength trades robustness against generation quality, and robustness claims need seed-level evaluation.

## Keywords (5–7)

Shortcut learning; spurious correlations; vision-language models; multimodal large language models; group
robustness; worst-group accuracy; gated cross-attention

## Highlights (Elsevier journals ask for 3–5, ≤ 85 characters each)

- Zero-initialised learnable gates stay shut and make VLMs rely on shortcuts
- Fixing the gate raises worst-group accuracy by 10.6–25.2 points over 15 seeds
- The learnable gate turns robustness into a seed lottery: up to 10 of 15 runs collapse
- With the gate fixed, no significant difference between connector types
- Retraining only the connector on a small balanced set reaches 89.6% worst-group accuracy
