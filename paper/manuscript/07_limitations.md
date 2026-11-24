# Limitations (manuscript section)

---

The models are at most 4B parameters, run in 4-bit with LoRA, with a frozen vision encoder and a short alignment of
300 steps. Released gated models are larger, trained for longer and update their encoders, so our results show a
risk in this regime, not a property of every gated model. The effect is limited to a single input-level gated block:
per-layer gates showed no gap.

Three of the four tasks use a binary readout from the first answer token. On the one open-ended benchmark, a VQA-CP v2
subset with three seeds, the gate effect did not appear, strong injection hurt generation, and only the capped
connector used the test image beyond a text-only baseline; with three seeds these results are exploratory.

Several controls have few seeds: the pathway ablation three to four, the learning-rate sweep two per setting, fp16 and the
shortcut-strength sweep two or three. They are reported as exploratory. Phi-4-mini Watermark was extended from five to
ten seeds after its first test, which we disclose; the added seeds also pass on their own. Runs are not
bit-reproducible on GPUs, so seeds are random replicates rather than exact repeats.

The learnable and fixed gates differ in two ways, whether the gate trains and whether the zero start sits in the gate
or in the output layer. The second difference cannot be removed, because both at zero cannot train; the open and
half-open controls separate the two as far as possible, but the open gate barely moves because its gradient is scaled
by $1 - \tanh^2\alpha \approx 0.01$.

Equivalence between connector types is not established. CelebA uses an 8,000-image training subsample and a binary,
annotator-assigned "Male" attribute, so its results say nothing about sex or gender beyond that label; the watermark
shortcut is partly textual [materzynska2022disentangling], and SigLIP's web pre-training may include images from the
source datasets, a factor shared by every arm. Group DRO is the only full-model group-robust baseline we ran; we did
not run gradient-modulation methods for modality imbalance [peng2022balanced] or label-free reweighting at full
scale, because the study isolates a cause rather than competing with fixes. Two of the tasks are standard but small
group-robustness benchmarks and one is synthetic; open-ended spurious-cue benchmarks for MLLMs
[ye2026mmspubench; yang2025spuriverse] need a different training protocol and are left to future work. A control
that gave each test example another example's image turned out to swap images only within a group, because the test
files are stored group by group; we report no result from it.
