# Literature review

*Structured for a Q1 journal: conceptual background → thematic synthesis → contradictions → concrete gaps →
bridge to the hypotheses.* Every reference is in [`paper/references.bib`](../paper/references.bib) (BibTeX keys in
brackets) and in the synthesis matrix [`paper/literature_matrix.csv`](../paper/literature_matrix.csv)
(author, year, venue, objective/method, finding, limitation, use in this study).

All 93 references were checked on 2026-11-01:
- **arXiv papers:** checked against the arXiv API (title, first author, year).
- **Journal and conference DOIs:** checked against Crossref (venue, volume, pages).

They include 24 journal articles: Nature Machine Intelligence, Nature Communications, npj Digital Medicine,
IEEE TPAMI, IEEE TKDE, IJCV (Springer), Information Fusion (Elsevier), Machine Intelligence Research (Springer),
Visual Intelligence (Springer), National Science Review and others.

---

## 1. Conceptual and theoretical background

**Shortcut learning.**
- *Definition.* A model solves a task through a decision rule that holds on the training distribution but not
  on the intended one [geirhos2020shortcut]. For example, it predicts "waterbird" from the background, or "dog"
  from a watermark.
- *Spurious correlation.* A feature that predicts the label in training but has no causal link to it.
- *Groups and metric.* Following [sagawa2020groupdro], the data are partitioned into groups by (label, spurious
  attribute). Robustness is measured by **worst-group accuracy (WGA)**, the minimum accuracy over groups.
  Majority (in-distribution, ID) and minority (out-of-distribution, OOD) groups are reported separately
  [koh2021wilds].

**Theoretical anchor: invariance and the readout view.**
- Invariant Risk Minimization [arjovsky2019irm] formalises robustness as reliance on features whose
  relationship with the label is invariant across environments.
- The *feature–readout decomposition* of [izmailov2022feature] and [kirichenko2023dfr] shows when shortcuts
  arise. Often the representation already contains the invariant (core) feature, and the failure lies in how
  it is read out.

This study applies that decomposition to LLM-based vision-language models (VLMs), whose pipeline is
*encoder → connector → LLM*:
- the **frozen encoder** fixes the features;
- the **connector** controls what reaches the LLM;
- the **LLM** is the readout.

A connector that transmits too little visual signal forces the readout onto whatever cue survives. This
framing yields testable predictions.

**Injection gates.**
- Flamingo-style models add `tanh(α)·CrossAttn(text, image)` to the language stream, with `α` initialised at
  zero so that the pretrained LLM is unchanged at the start [alayrac2022flamingo].
- The same zero-initialised, learnable scaling is used in unimodal networks to stabilise training
  [bachlechner2021rezero], in LLaMA-Adapter's zero-gated prompts [zhang2024llamaadapter; gao2023llamaadapterv2],
  and in released VLMs: IDEFICS and Llama 3.2 Vision (separate attention and MLP gates in 8 of 40 layers)
  [grattafiori2024llama3]. Its estimation theory is analysed in [diep2025zeroinit].
- Flamingo reports that its gates grow from zero over 500k steps, and that removing them costs 4.2 points
  [alayrac2022flamingo].
- A gate is *closed* when `|tanh α| ≈ 0`.

## 2. Thematic synthesis

### Theme A — Shortcut learning is pervasive and costly; fixes act on data and readout

**A1. Evidence of harm.**
- Shortcut reliance has been documented in high-stakes settings: COVID-19 radiograph classifiers that used
  text markers and hospital artefacts [degrave2021covid], and medical models whose ID accuracy over-states
  external performance [ongly2024shortcut; brown2023shortcut].
- "Clever Hans" analyses show high-accuracy models relying on watermarks and source tags
  [lapuschkin2019clever; anders2022finding]. This is the real-world precedent for our Watermark task.

**A2. Methods.** Mitigation splits into three families:
- *Group-aware training:* GroupDRO [sagawa2020groupdro] and balanced sampling [idrissi2022balancing].
- *Group-label-free reweighting:* LfF [nam2020lff], JTT [liu2021jtt], Correct-N-Contrast [zhang2022cnc] and
  AFR [qiu2023afr].
- *Readout retraining:* DFR [kirichenko2023dfr; labonte2023llr].

Recent CVPR work continues to act on data or readout [noohdani2024decompose; chakraborty2024exmap]. The
consensus is that **features usually contain the core signal and cheap readout or data fixes recover most
robustness** [izmailov2022feature; kirichenko2023dfr]. Model capacity can make things worse
[sagawa2020overparam].

**A3. Evaluation.**
- Fair model selection changes conclusions [gulrajani2021domainbed]. Domain-generalisation surveys stress
  protocol over method [zhou2022dgsurvey; wang2022dgsurvey].
- Benchmarks: Waterbirds (CUB on Places backgrounds) [wah2011cub; zhou2018places; sagawa2020groupdro] and
  CelebA [liu2015celeba].
- *Synthesis:* this literature studies **classifiers**. The "readout" is a linear head, and the architecture
  between features and head is not a variable.

### Theme B — Vision-language models inherit and amplify shortcuts

**B1. Language priors.**
- Before LLM-based VLMs, VQA models answered from language priors and ignored the image
  [agrawal2018vqacp]. Balanced data was the remedy [goyal2019vqav2].
- Causal adjustments target cross-modal confounders [yang2022deconfounded; liu2023crossmodal], and
  information bottlenecks discard redundant correlations [jiang2023cib].

**B2. Contrastive VLMs.**
- CLIP-style models [radford2021clip] carry spurious correlations, and their two modalities sit in separate
  regions of the embedding space [liang2022modalitygap]. They also read rendered text strongly
  [materzynska2022disentangling].
- Fixes again act on a small module or the readout: contrastive adapters [zhang2022adapters], fine-tuning
  constraints [yang2023mitigating], prompt projection [chuang2023biasedprompts] and representation calibration
  [you2024calibrating], the last a direct descendant of DFR.

**B3. Generative and LLM-based VLMs.**
- *Contrastive training.* Synthetic shortcuts can erase retrieval performance [bleeker2024shortcuts].
- *Spurious-cue benchmarks.* Multimodal LLMs fail on minority groups zero-shot [hosseini2025spurlens], and
  generalise poorly beyond seen spurious correlations [yang2025spuriverse] and beyond their training domains
  [zhang2024oodmllm].
- *Hallucination.* Models hallucinate objects from co-occurrence priors [li2023pope; leng2024vcd], and
  journal-level evaluations confirm broad robustness gaps [xu2025lvlmehub].
- *Proposed causes and remedies:* the visual encoder [tong2024eyes], unlearning [sarker2026purge], or data
  rather than scale [sarridis2026scaling].
- *Synthesis:* every one of these works treats the connector as fixed.

### Theme C — Connector design is studied for accuracy and efficiency, not robustness

**C1. Families.** Three connector families dominate [yin2024mllmsurvey; chen2023vlpsurvey]:
- *Projection:* MLP projectors [liu2023llava; liu2024llava15] and linear projectors [beyer2024paligemma].
- *Query-based:* the Q-Former [li2023blip2; dai2023instructblip] and resamplers [bai2023qwenvl].
- *Gated cross-attention:* Flamingo, IDEFICS [alayrac2022flamingo; laurencon2023obelics] and the Llama 3 vision
  adapters [grattafiori2024llama3].

**C2. Design studies.** Controlled ablations ask which connector maximises benchmark accuracy or efficiency:
- locality and token count [cha2024honeybee; li2025tokenpacker];
- broad design-space sweeps [mckinzie2024mm1; karamcheti2024prismatic; tong2024cambrian];
- alignment quality [huang2024mir; groger2026rethinking].

Surveys list robustness and bias as open challenges [awais2025foundation; zhang2024vlmsurvey;
jin2025efficientmllm], but no design study measures them.

**C3. Gating.**
- Learnable gates for injecting signals into frozen VLMs end up behaviourally closed across five designs, so
  [farazi2026gate] recommend fixed injection.
- Their consequence for **what the model relies on** was not measured.

**C4. Modality imbalance.**
- Multimodal networks greedily learn one modality and under-fit the other [wu2022greedy]. Which modality wins
  depends on initialisation [huang2022modality].
- Gradient or feature modulation reduces the imbalance [peng2022balanced; wei2025onthefly].
- This literature studies late-fusion classifiers. It does not identify an architectural component of
  LLM-based VLMs as the cause, and it does not measure spurious-group robustness. A shut injection gate is a
  concrete instance of a modality being under-used.

### Theme D — Methodological standards for comparing deep models

**Seeds.**
- Seeds alone can reverse conclusions [henderson2018matters; picard2021seed].
- Randomising every source of variation is needed for valid comparisons [bouthillier2021variance].

**Testing.**
- Paired and non-parametric tests suit seed-paired designs [dror2018hitchhiker].
- Equivalence requires its own test [schuirmann1987tost].
- Multiple comparisons need correction [holm1979simple].
- Power should be planned [cohen1988power].

**Attention.**
- Attention weights do not necessarily explain predictions [jain2019attention]; under some conditions they do
  [wiegreffe2019attention].
- Subspace explanations can isolate Clever-Hans features [chormai2024disentangled].

## 3. Critical comparison and contradictions

| Question | Camp 1 | Camp 2 | Why they disagree | What this study does |
| :--- | :--- | :--- | :--- | :--- |
| Does connector type matter? | Yes: MLP aligns better than Q-Former [huang2024mir]; locality matters [cha2024honeybee] | Little: connector type has small effect vs tokens and resolution [mckinzie2024mm1] | Different outcomes (alignment metric vs benchmark), different budgets | Fixes everything else and measures *group robustness* over many seeds: MLP vs Q-Former not significant (5 seeds) |
| Is cross-attention a good injection? | Yes with frozen backbones [laurencon2024matters]; used at frontier scale [grattafiori2024llama3]; gates grow over long training [alayrac2022flamingo] | Learnable gates end up behaviourally closed [farazi2026gate] | Training length differs (500k vs short adaptation); gate dynamics are rarely logged; outcomes are averages over seeds | Logs the gate and tests learned vs fixed gate head-to-head (H1, H2), input-level and per-layer, 300 and 3,000 alignment steps |
| Is under-use of a modality an optimisation or an architecture problem? | Optimisation: greedy learning, modality competition [wu2022greedy; huang2022modality] | Fixable by modulating gradients [peng2022balanced; wei2025onthefly] | Studied in late-fusion classifiers only | Isolates one architectural component (the gate) and shows its effect on worst-group accuracy |
| Where do MLLM failures come from? | Visual encoder [tong2024eyes] | Data, not scale [sarridis2026scaling]; mapping between modalities [zhang2024oodmllm] | Each study varies one component | Varies the connector and the encoder separately: the gate effect persists across LLMs; a larger encoder masks it |
| Does attention reveal shortcut use? | Explanations can find spurious groups [chakraborty2024exmap; lapuschkin2019clever] | Attention is not explanation [jain2019attention] | Attention ≠ causal use | Both gate variants attend to the watermark equally; only one relies on it |
| Is more capacity safer? | Bigger encoders help MLLMs [tong2024cambrian] | Overparameterisation can worsen reliance [sagawa2020overparam] | Different metrics (average vs worst-group) | SigLIP-large raises WGA for all connectors, but the gate still closes |
| Do readout fixes transfer to VLMs? | DFR works on classifiers [kirichenko2023dfr] and CLIP [you2024calibrating] | Untested on generative VLM connectors | No connector-level analogue existed | Connector-DFR: large gain on Watermark, none on Waterbirds |

## 4. Targeted gaps

1. **Theoretical gap (missing mechanism).** The feature–readout view [izmailov2022feature; kirichenko2023dfr]
   has no account of the *transmission* stage between features and readout. Gate closure [farazi2026gate] is a
   candidate mechanism, but it has never been linked to shortcut reliance. Modality-imbalance work
   [wu2022greedy; huang2022modality] describes the under-use of one modality, but not in LLM-based VLMs, not
   through a specific component, and not in terms of spurious groups.
2. **Methodological gap.**
   - Connector studies [mckinzie2024mm1; karamcheti2024prismatic; huang2024mir] use average benchmark accuracy
     and typically one seed.
   - Robustness studies of VLMs [hosseini2025spurlens; yang2025spuriverse; sarker2026purge] do not vary the
     architecture.
   - No work combines group metrics, stated hypotheses tested on new seeds, and seed-level statistics for a
     connector comparison.
3. **Empirical gap.**
   - The evidence on cross-attention injection conflicts [laurencon2024matters; farazi2026gate].
   - The evidence on connector type conflicts [huang2024mir; mckinzie2024mm1].
   - No controlled data on worst-group behaviour resolves either question.
4. **Contextual gap.** DFR-style readout fixes are established for classifiers and contrastive models
   [kirichenko2023dfr; you2024calibrating] but untested for the connector of generative, LLM-based VLMs. The
   same holds for compute-constrained settings: 4-bit LLMs on commodity GPUs.

## 5. Bridge to this study

These gaps lead to one question:

> Holding the LLM, features, data and budget fixed, does the vision-language connector, and specifically a
> learnable injection gate, determine how much a generative VLM relies on a shortcut?

Pre-declared hypotheses:
- **H1:** a fixed injection gate yields higher worst-group accuracy than a learnable gate (paired by seed).
- **H2:** the learnable gate is less stable across seeds (larger variance).
- **Secondary questions:**
  - Does connector type matter once the gate is fixed?
  - Does the effect replicate with another LLM and survive a stronger encoder?
  - Can a connector-only, DFR-style fix close the gap to a linear probe on the same features?

Each prediction follows from the transmission view in Section 1. A closed gate transmits little visual signal,
so the readout falls back on the most available cue, which predicts lower WGA (H1). Closure depends on a
fragile, sign-flipping gradient, which predicts high seed variance (H2).

See [`paper/METHODOLOGY.md`](../paper/METHODOLOGY.md) for how these are tested and
[`paper/RESULTS_DISCUSSION.md`](../paper/RESULTS_DISCUSSION.md) for the outcomes.
