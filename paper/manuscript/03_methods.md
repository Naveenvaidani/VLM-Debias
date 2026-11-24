# Methods (manuscript section)

*Compact version for the journal. The full protocol, power analysis, timeline and validity notes are in
`paper/METHODOLOGY.md` and go to the supplementary material.*

---

## Design

The study is a controlled between-architecture experiment. One factor changes, the connector that maps vision
features into the LLM, while the LLM, the frozen vision features, the data, the prompts, the optimiser and the
step budget stay the same. The unit of replication is the training seed, which sets data order, LoRA
initialisation and sampling. Seeds are paired across connectors.

## Model

A frozen SigLIP-base encoder [zhai2023siglip] gives 196 patch features of dimension 768 per image, computed once
and cached, so every connector receives identical inputs. The LLM is Qwen3.5-4B [qwen2026qwen35] in 4-bit NF4 with
LoRA of rank 16 [dettmers2023qlora; hu2022lora]; Phi-4-mini [abouelenin2025phi4mini] replicates the main
comparison. The answer is read from the next-token distribution restricted to the two class words, which removes
free-text parsing as a confounder on the classification tasks.

## Connectors

Let $P \in \mathbb{R}^{196 \times 768}$ be the patch features and $e_t$ the embedding of text token $t$.

*MLP projector* [liu2024llava15]. A LayerNorm–Linear–GELU–Linear map over $2 \times 2$-pooled patches gives 49 image
tokens that are placed before the text (8.5M parameters).

*Q-Former* [li2023blip2]. Thirty-two learned queries attend to the patches in one block with 8 heads and give 32
image tokens (11.4M parameters).

*Gated cross-attention* (8.9M parameters, no image tokens). Each text token queries the patches and the result is
added to its own embedding:

$$
e_t' = e_t + g \, \Delta_t, \qquad \Delta_t = W_o\,\mathrm{Attn}\big(W_q\,\mathrm{LN}(e_t),\ \mathrm{LN}(P),\ \mathrm{LN}(P)\big).
$$

The injected signal depends on the text, which matters for the mechanism in Section 4. The variants differ only in
the gate $g$:

- *learnable gate:* $g = \tanh\alpha$ with $\alpha_0 = 0$ and the default initialisation of $W_o$, as in Flamingo
  [alayrac2022flamingo];
- *fixed gate:* $g = 1$ with $W_o$ initialised to zero, so the connector still starts as the identity but the
  optimiser cannot close it;
- *learnable gate with a non-zero start* (controls): $g = \tanh\alpha$ with $W_o = 0$ and $\tanh\alpha_0 \in
  \{0.5, 0.995\}$.

Both the gate and $W_o$ cannot start at zero together: $\partial e_t'/\partial \alpha \propto W_o$ and
$\partial e_t'/\partial W_o \propto \tanh\alpha$, so neither would receive a gradient. The learnable arm therefore
starts from $\alpha = 0$ and the fixed arm from $W_o = 0$, and the non-zero-start controls separate the two.

A per-layer variant places four such blocks before decoder layers 0, 8, 16 and 24 with their own gates, the layout
of Flamingo and Llama 3.2 Vision, at about the same parameter count (9.3M).

## Tasks and data

Waterbirds [sagawa2020groupdro] places CUB birds [wah2011cub] on Places backgrounds [zhou2018places] (official
splits: 4,795 / 1,199 / 5,794 images). Watermark is a controlled task we built from COCO [lin2014coco]: the question
asks whether a dog is present, and a corner text watermark co-occurs with dogs in 95% of training images; validation
(1,200) and test (2,400) are group-balanced. CelebA [liu2015celeba] predicts blond hair with gender as the spurious
attribute (stratified 8,000-image training subsample, official test set). VQA-CP v2 [agrawal2018vqacp] tests
free-form answers when the answer prior of each question type differs between training and test; we use an
image-disjoint subset of 8,000 training, 1,000 validation and 2,000 in-distribution test questions from its training
split and 3,000 questions from its test split. Images used for alignment are excluded from every evaluation split.

## Training

Stage 1 aligns each connector for 300 steps on 10,000 COCO caption pairs with the LLM frozen (batch 16, learning
rate $10^{-3}$). Stage 2 trains LoRA and the connector on the task for 4 epochs (batch 16, learning rate
$2 \times 10^{-4}$, cosine schedule, gradient-norm clipping at 1) and stops early after two epochs without
validation gain. The budget is equal in steps for every connector. The checkpoint is chosen by validation
worst-group accuracy; selection by validation average accuracy, which needs no group labels, is reported as a check.

## Measures

The primary outcome is test worst-group accuracy, the lowest accuracy over the four (label, spurious attribute)
groups,

$$
\mathrm{WGA} = \min_{(y,a)} \ \frac{1}{|G_{y,a}|} \sum_{i \in G_{y,a}} \mathbb{1}[\hat y_i = y_i],
$$

averaged over three question phrasings, one never seen in training. Secondary outcomes are average accuracy, the
seed variance of WGA, and the number of collapsed runs (WGA below 50%). On VQA-CP the outcome is VQA accuracy
against the ten human answers.

Three measures probe the mechanism. The gate value $\tanh\alpha$ and its gradient are logged during training; in the
control runs the gradient is also logged before clipping. The injection ratio $\rho = \mathbb{E}_t\|g\,\Delta_t\| / \mathbb{E}_t\|e_t\|$ measures how much visual signal
enters relative to the text. The pathway test re-scores each selected model with the visual pathway removed at
inference, following the protocol of [farazi2026gate].

## Statistics

The hypotheses were formed after a 3-seed pilot and confirmed on new seeds. H1 states that the fixed gate gives
higher worst-group accuracy than the learnable gate; it is tested with one-sided paired t-tests at
$\alpha = 0.025$ in four cells (two LLMs, two tasks), with Holm correction [holm1979simple] and Wilcoxon tests as the
assumption-free check. H2 states that the learnable gate is less stable; it is tested with a variance-ratio F-test,
the Brown–Forsythe test and Fisher's exact test on collapsed runs. Effect sizes are paired Cohen's $d_z$
[cohen1988power]. Connector differences use a hierarchical bootstrap over seeds and images [efron1993bootstrap], and
the claim that connector type does not matter is tested for equivalence with two one-sided tests at a margin of
three points [schuirmann1987tost]. Only the four H1 tests and H2 on Qwen3.5-4B are confirmatory; every control is
exploratory and its p-values are reported uncorrected. Because H1 was formed after the pilot, it is also tested on
the confirmation seeds alone.

Runs are not bit-reproducible on GPUs: rerunning a seed changed worst-group accuracy by 12.7 points on average for the
learnable gate and 5.2 for the fixed gate, so seeds are treated as random replicates and unpaired tests are reported
as well. Code, job scripts, pinned package versions and every run's results are public (see Code availability).

## Use of AI tools

An AI assistant (Claude, Anthropic) was used to write and test experiment code, run the statistical analyses and
draft parts of the text. The authors designed the study, checked the code and every reported number against the
saved run outputs, and revised the text, and they take full responsibility for the content.
