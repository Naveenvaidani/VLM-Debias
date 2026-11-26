# Related work (manuscript section)

*About one journal page. Condensed from [`../docs/LITERATURE.md`](../docs/LITERATURE.md) and the full-text reading in
[`../docs/NOVELTY.md`](../docs/NOVELTY.md) §6. Converted to LaTeX by `springer/convert.py`.*

---

## Shortcut learning and group robustness

Shortcut learning, the reliance on a cue that predicts the label in training but not at deployment, is documented
from medical imaging to watermark detection [geirhos2020shortcut; degrave2021covid; lapuschkin2019clever]. Its
standard measure is worst-group accuracy over groups defined by label and spurious attribute, with Waterbirds and
CelebA as benchmarks [sagawa2020groupdro; liu2015celeba]. Mitigations act on data or training: group-robust
optimisation and balanced sampling [sagawa2020groupdro; idrissi2022balancing], label-free reweighting
[nam2020lff; liu2021jtt; qiu2023afr], invariance-seeking objectives [huang2024losingtickets; mitchell2026wule], and
embedding regularisers that reach high worst-group accuracy from group-wise mean embeddings, which need attribute labels [park2026scer]. Features usually retain the core signal, so retraining only the readout recovers most robustness
[izmailov2022feature; kirichenko2023dfr]. This literature studies classifiers: the architecture between features and
readout is not a variable. Our study makes that architecture the variable, and tests a readout-style fix at the
connector.

## Shortcuts and language priors in vision-language models

Before LLM-based models, VQA systems answered from question priors, which VQA-CP was built to expose
[agrawal2018vqacp; goyal2019vqav2]. Contrastive models such as CLIP carry spurious correlations and read rendered
text [radford2021clip; materzynska2022disentangling], and fixes act on adapters, prompts or calibration
[zhang2022adapters; chuang2023biasedprompts; you2024calibrating]. For MLLMs, benchmarks show reliance on spurious cues
and poor generalisation beyond them [hosseini2025spurlens; yang2025spuriverse; ye2026mmspubench; zhang2024oodmllm],
and robustness under shift remains open for vision-language alignment [zhu2024vlalignood]. The proposed causes are the
vision encoder [tong2024eyes], training data, which tracks worst-group accuracy more consistently than scale [sarridis2026scaling], or are removed after training
[sarker2026purge]. A gating unit that keeps language from fusing directly into visual features reduces language
shortcuts in video question answering [zou2024videodistill], and an adaptation framework with an entity-grounded gate
improves the out-of-domain robustness of an MLLM [ahn2026evident]. None of these varies the connector of an LLM-based VLM as a
controlled factor.

## Connector design

Three connector families dominate [yin2024mllmsurvey; huang2025vitsurvey]: projection (MLP projectors
[liu2024llava15]), query-based (the Q-Former [li2023blip2]) and gated cross-attention (Flamingo, IDEFICS and the
Llama 3.2 vision adapters [alayrac2022flamingo; laurencon2023obelics; grattafiori2024llama3]). Design studies rank
connectors by benchmark accuracy and efficiency [mckinzie2024mm1; cha2024honeybee; karamcheti2024prismatic;
tong2024cambrian] and find that connector type matters less than token count and resolution [mckinzie2024mm1]. The only
three-way comparison we found on another outcome, cross-modal unlearning, compares released models (LLaVA-1.5, InstructBLIP
and IDEFICS) that also differ in vision encoder and, for IDEFICS, in LLM and size [balaji2026forget]. Surveys list robustness and bias as open challenges for MLLMs
[awais2025foundation; zhang2024vlmsurvey], but no design study measures them.

## Gated injection and initialisation

Zero-initialised learnable scales let a new branch start as the identity: ReZero and SkipInit for residual blocks
[bachlechner2021rezero; de2020skipinit], LayerScale with a small non-zero start that trains better than zero
[touvron2021cait], and LLaMA-Adapter and Flamingo for adding a modality to a frozen LLM [zhang2024llamaadapter;
alayrac2022flamingo]. Zero-initialised gated cross-attention is still proposed for injecting non-text modalities into
frozen LLMs [rm2026zerogated]. That such gates barely move is known: in a Flamingo-style video-captioning adapter
on a frozen LLM, Lialin et al. find gate values changing by less than 0.05 during fine-tuning and report that
starting the gates at $\tanh 1$ instead of zero greatly improves models trained without pre-training
[lialin2023scalable]. Recent connector designs also avoid a zero start: Yan et al. start a sigmoid residual gate at
0.3, arguing that too small a gate injects too little visual signal, and compare starts of 0.1 to 0.5 by benchmark
accuracy [yan2026dgp]. Neither study measures what a small gate does to shortcut reliance. Farazi et al. show that learnable gates on image-derived pathways in frozen VLMs settle near
zero and become behaviourally closed; they explain this by a vanishing first-order gradient when the injected signal
does not depend on the caption, recommend injecting at matched norm without a gate, and leave text-coupled injection
to future work [farazi2026gate]. Li et al. show that visual tokens with much larger norms than
text tokens update more slowly in pre-norm MLLMs, which impairs cross-modal fusion [li2025unseen]. Modality-imbalance work finds that multimodal
networks greedily under-use one modality, with the winner depending on initialisation, and corrects it by gradient
modulation [wu2022greedy; huang2022modality; peng2022balanced]. We bring these lines together: we vary the gate's
start value in a text-coupled connector and measure the consequence for shortcut reliance. Table \ref{tab:closest} sets out how our
design differs from the closest studies.

| Study | Injected signal | Gate start varied | Shortcut or worst-group measure | Seeds per comparison |
| :--- | :--- | :---: | :---: | :---: |
| Flamingo [alayrac2022flamingo] | Image, text-coupled (cross-attention) | No | No | 1 |
| Farazi et al. [farazi2026gate] | Auxiliary, independent of the caption | No | No | mostly 1 |
| Li et al. [li2025unseen] | Image prefix tokens | No gate | No | 1 |
| Zero-gated cross-attention [rm2026zerogated] | Audio and video, text-coupled | No | No | 1 |
| Ours | Image, text-coupled (cross-attention) | Yes (0 to 0.995) | Yes (worst-group accuracy, collapse rate) | 3 to 15 |

: The closest studies of gated injection. \label{tab:closest}

## Evaluation practice

Seed variance alone can reverse deep-learning conclusions [henderson2018matters; bouthillier2021variance;
picard2021seed], and model selection changes robustness results [gulrajani2021domainbed]. We therefore state
hypotheses on a pilot, confirm them on new paired seeds, correct for multiple tests [holm1979simple], report effect
sizes and equivalence tests [schuirmann1987tost], and treat seed-level collapse as an outcome.
