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

## Relation to gate closure in frozen VLMs

Farazi et al. find that learnable gates on image-derived pathways in frozen VLMs settle near zero and that the
pathways become behaviourally closed, and they explain it by a first-order gradient that vanishes when the injected
signal does not depend on the caption [farazi2026gate]. Our connector is text-coupled, the case they leave open. Its
gate also stays near zero, but through a large gradient that cancels itself, and its pathway is still used. Both
results agree that a gate's value is not evidence of how much the pathway is used, in either direction. They differ on
what the small gate does: in our setting it does not remove the image, it weakens it enough for a shortcut to win.

## Initialisation and injection strength must be chosen together

A non-zero start, as LayerScale recommends for optimisation [touvron2021cait], removes the robustness cost of the gate.
Strong injection has its own cost: the fixed gate injects about 70 to 130 times the text-embedding norm, which helps
the binary readout but disturbs free-form answers on VQA-CP. This matches reports that visual tokens with much larger
norms than text tokens lose information in pre-norm MLLMs [li2025unseen] and the advice to inject at a matched norm
[farazi2026gate]. Neither of those studies measures shortcut reliance, so whether a norm-matched injection keeps the
robustness gain is an open question that our data does not yet answer.

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
