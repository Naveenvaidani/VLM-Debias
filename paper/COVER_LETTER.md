# Cover letter (draft)

*To be sent through Editorial Manager with the submission. Items in [square brackets] must be filled in or deleted by
the authors before sending. The 9B sentence depends on the round-10 results.*

---

[Date]

To the Guest Editors, Special Issue on Multimodal Large Language Models for Unified Comprehension and Generation,
International Journal of Computer Vision

Dear Editors,

We submit the manuscript "A Zero-Initialised Injection Gate Makes Shortcut Learning More Likely in Vision-Language
Models" for consideration in the special issue.

**What the paper shows.** Many multimodal LLMs join a vision encoder to a language model through a connector, and
several inject visual features through a tanh gate that starts at zero (Flamingo, IDEFICS, Llama 3.2 Vision, and new
adapters in 2026). We hold the language model, frozen visual features, data and training budget fixed and vary only
the connector. A learnable gate started at or near zero lowered worst-group accuracy by 10.6 and 25.2 points on
Qwen3.5-4B (paired $d_z$ 1.10 and 1.34, $p < 0.001$, confirmatory tests on fresh seeds) and by 9.7 and 25.5 points on
Phi-4-mini, and made robustness depend on the random seed. Gates started at 0.1 or higher behaved like a fixed gate.
The zero-start gate stays small because its large gradient keeps changing sign, which we measure directly, and its
visual pathway stays in use rather than closing. On open-ended question answering, strong injection harms generation,
and capping it at the text-embedding norm repairs generation but gives back most of the robustness gain.
[If round 10 holds: The main effect repeats with Qwen3.5-9B.]

**Fit with the call.** The paper addresses two of the listed topics:
- *Topic I, Unified Modeling Architectures (encoder–LLM pipelines; representation alignment across modalities).* The
  connector is the component that aligns visual representations with the LLM. We isolate one of its design defaults
  and show that it changes what the model learns.
- *Topic IV, Evaluation and Benchmarks (robustness and reliability analysis; bias and fairness evaluation;
  reproducibility standards).* The study measures worst-group robustness, treats seed-level collapse as an outcome,
  separates confirmatory from exploratory tests, and releases code, pinned versions and every per-run result.

**Relation to recent work.** The closest preprint (Farazi et al., 2026) shows that learnable gates settle near zero
in frozen VLMs. Our study differs in four ways: the injected signal depends on the text, we vary the gate's start
value, we measure shortcut reliance and worst-group accuracy, and we find the pathway is used rather than closed.
Their recommendation to inject at a matched norm repairs generation in our connector but costs robustness, a cost
their benchmarks could not show.

**Declarations.** The manuscript is original, is not under review elsewhere, and [has / has not] been posted as a
preprint [arXiv identifier and licence]. All authors approved the submission. The use of an AI assistant is described
in the Methods section. [Competing interests statement.]

**Suggested reviewers.** [Two to four independent researchers with institutional e-mail addresses and ORCID or Scopus
IDs; not recent collaborators and not from the authors' institution.]

Yours sincerely,

[Corresponding author name]
Faculty of Engineering and Technology, Sri Ramachandra Institute of Higher Education and Research, Porur, Chennai,
India
[Institutional e-mail]
