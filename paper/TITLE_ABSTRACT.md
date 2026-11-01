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
   (10 words). "Induce" rather than "Cause": with a stronger encoder the shut gate no longer hurts, so the gate
   is a risk factor, not a sufficient cause.
2. **Structured:** *Shortcut Learning in Vision-Language Models: The Role of the Vision-Language Connector
   and Its Injection Gate* (16 words; can drop "Its Injection Gate")
3. **Hook and subtitle:** *When the Gate Stays Shut: Connector Design and Spurious Correlations in Multimodal
   Large Language Models* (15 words)

## Abstract (≈ 230 words, five-part formula, no citations, acronyms spelled out)

> **[Context]** Vision-language models built by connecting a vision encoder to a large language model are
> increasingly deployed in settings where reliance on spurious cues, such as image backgrounds or watermarks,
> causes silent failures on under-represented groups.
>
> **[Gap]** Prior work attributes these failures to data, scale or the vision encoder, and compares connector
> designs only on average benchmark accuracy. Whether the connector itself determines shortcut reliance has not
> been tested.
>
> **[Method]** We hold the language model, frozen vision features, data and training budget fixed and vary only
> the connector: a multilayer-perceptron projector, a query transformer, and gated cross-attention with either
> a learnable or a fixed injection gate. We measure worst-group accuracy on three shortcut tasks, two language
> models and two encoder sizes, with hypotheses formed on a pilot and confirmed on new paired
> training seeds (up to fifteen in total).
>
> **[Findings]** The learnable gate, initialised at zero, never opens during alignment, and with it the model
> relies on the shortcut.
> Fixing the gate open raises worst-group accuracy by 10.6 and 25.2 points, large effects (paired Cohen's
> d = 1.10 and 1.34, p < 0.001). It also eliminates the training runs that collapse below 50%, which occur in
> up to 10 of 15 seeds with the learnable gate. A learnable gate initialised open behaves like the fixed
> one, and ten times longer alignment leaves the zero-initialised gate shut. Once the gate is fixed, no significant difference between
> connector types is detected. Retraining only the connector on a small group-balanced set reaches 89.6% worst-group accuracy
> on the watermark task.
>
> **[Contribution]** These results identify the injection gate as a transmission bottleneck between features
> and readout. They show that robustness claims require seed-level evaluation, and they give practitioners a
> zero-cost, label-free design rule that removes one source of shortcut reliance, and a cheap connector-level
> fix when group labels are available.

Word count: about 230 (check against the journal's limit).

## Keywords (5–7)

Shortcut learning; spurious correlations; vision-language models; multimodal large language models; group
robustness; worst-group accuracy; gated cross-attention

## Highlights (Elsevier journals ask for 3–5, ≤ 85 characters each)

- Zero-initialised learnable gates stay shut and make VLMs rely on shortcuts
- Fixing the gate raises worst-group accuracy by 10.6–25.2 points over 15 seeds
- The learnable gate turns robustness into a seed lottery: up to 10 of 15 runs collapse
- With the gate fixed, no significant difference between connector types
- Retraining only the connector on a small balanced set reaches 89.6% worst-group accuracy
