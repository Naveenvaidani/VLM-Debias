# Discussion (manuscript section)

---

## A transmission account of shortcut reliance

Work on spurious correlations in classifiers concludes that features usually carry the core signal and that the
readout decides whether it is used [izmailov2022feature; kirichenko2023dfr]. Our results add a stage between the two:
the connector decides how much of the feature signal reaches the readout. With a zero-start learnable gate, the
injected visual signal is an order of magnitude weaker, and the LLM falls back on the cue that is easiest to read
from what remains. A linear probe on the same frozen features reaches 84 to 85 worst-group accuracy with reweighting,
above every vision-language model trained with plain empirical risk minimisation on those features, which confirms that the information is present in the features and is lost on the way to
the readout. This is a concrete case of modality under-use [wu2022greedy; huang2022modality] located in one component
that ships in released models.

## A toy model of weak injection

Why should a weak injection favour the shortcut rather than simply slow learning? A two-feature toy model gives a
simple account (code in `study/toy/`). A logistic readout sees two image features through an injection gain $g$: a
core feature that separates the classes in every group but has a small scale ($\pm 0.5$, noise s.d. 0.1), and a
shortcut feature that agrees with the label in 95% of training examples and has a large scale ($\pm 2$, noise s.d.
1). To classify the minority groups, the core weight must exceed the shortcut weight by more than the ratio of
their scales, so the effective weights $g\,w$ must grow large. The gain slows that growth: each update changes
$g\,w$ by about $g$ times the step for Adam and $g^2$ times for SGD. With $g = 1$, 1,000 Adam steps reach 84.6%
worst-group accuracy on a balanced test set and 3,000 steps 99.8%; with $g \le 0.1$ the readout is still in the
early regime, where both weights are similar and the larger-scale shortcut decides, and worst-group accuracy stays
below 17% (Table \ref{tab:toy}). More steps move the threshold to smaller gains. The toy fixes the gain, so it explains why a weak injection keeps the model on the shortcut, not why the
learnable gate stays weak; its settings were chosen so that the full-gain model can learn the core feature, as the
fixed-gate model does on our tasks.

| Optimiser, steps | $g$ = 0.01 | 0.03 | 0.1 | 0.3 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Adam, 1,000 | 3.7 | 5.8 | 7.5 | 14.8 | 84.6 |
| Adam, 3,000 | 5.6 | 7.1 | 16.1 | 85.5 | 99.8 |
| SGD, 1,000 | 0.5 | 2.3 | 3.7 | 27.1 | 99.5 |
| SGD, 3,000 | 1.6 | 3.0 | 6.4 | 87.0 | 99.9 |

: Toy model: worst-group accuracy (%) of a logistic readout behind an injection gain $g$, mean over 20 seeds
(s.d. at most 5.4). \label{tab:toy}

## Relation to gate closure in frozen VLMs

Farazi et al. find that learnable gates on image-derived pathways in frozen VLMs settle near zero and that the
pathways become behaviourally closed, and they explain it by a first-order gradient that vanishes when the injected
signal does not depend on the caption [farazi2026gate]. Our connector is text-coupled, the case they leave open. Its
gate also stays near zero, but through a large gradient that cancels itself, and its pathway is still used. Both
results agree that a gate's value is not evidence of how much the pathway is used, in either direction. They differ on
what the small gate does: in our setting it does not remove the image, it weakens it enough for a shortcut to win.

## Initialisation and injection strength must be chosen together

A non-zero start, as LayerScale recommends for optimisation [touvron2021cait] and as recent connectors already use
[lialin2023scalable; yan2026dgp], removes the robustness cost of the gate. Standard fixes do not reliably remove it:
connector-level DFR rescued the zero-start gate on the watermark task but not on Waterbirds, and JTT on neither, so
the start value should be set at design time rather than repaired afterwards.
Strong injection has its own cost: the fixed gate injects about 70 to 130 times the text-embedding norm, which helps
the binary readout but disturbs free-form answers on VQA-CP, where it scores below a text-only model. This matches
reports that visual tokens with much larger norms than text tokens lose information in pre-norm MLLMs
[li2025unseen] and the advice to inject at a matched norm [farazi2026gate]. Capping the injection at the text norm
repairs generation in our connector: the capped model is the best on both VQA-CP test splits and the only one whose
answers depend on the test image. The same cap, however, gives back most of the robustness gain on the watermark
task at 4B (50.6 against 67.2), although with the 9B model it kept the gain (63.9 against 63.2), so the cost may
shrink with scale. Neither earlier study measures shortcut reliance, so this cost of norm matching was not
visible to them. In our data a strong visual signal is what lets the class readout override a textual or
background shortcut, and the same strength disturbs free-form decoding. No single injection strength was best on
both kinds of task, so the strength should be chosen for the output the model must produce, and robustness should
be measured whenever it is changed.

## Why per-layer gates behave differently

With four gated blocks at different depths, the learnable and fixed gates do not differ. The per-layer learnable gates
end about seven times larger than the single input-level gate (mean 0.027 against 0.0037), and four routes into the
decoder give the image several chances to contribute. The effect we report therefore belongs to the single
input-level design, as used in adapter-style injection, and not to every gated architecture.

## Attention is not reliance

Both gate variants attend to the watermark corner about twice as much as a uniform map would, yet only the
learnable-gate model fails on images where the watermark and the label disagree. Attention maps alone would not have
separated the two models [jain2019attention; wiegreffe2019attention]; the pathway test and the group-level accuracies
did.
