# Recent papers in the target journals (checklist item C1)

Found through the Crossref API (journal ISSN filter, published 2024 or later, keyword queries on spurious
correlations, shortcut learning, group robustness, vision-language models, multimodal LLMs, VQA bias, domain
generalisation). Metadata (title, authors, year, volume, DOI) comes from Crossref. **Abstract checked** means the
abstract was read; **title only** means Crossref had no abstract and relevance is judged from the title. Read the
paper before citing it for anything beyond "the journal publishes this topic".

## International Journal of Computer Vision (ISSN 1573-1405)

| Paper | Year, vol. | DOI | Why cite | Checked |
| :--- | :--- | :--- | :--- | :--- |
| Huang, J. et al. Visual Instruction Tuning towards General-Purpose Multimodal Large Language Model: A Survey | 2025, 133 | 10.1007/s11263-025-02572-7 | IJCV's MLLM survey; context for connector designs. In `references.bib` (`huang2025vitsurvey`) | Title only |
| Zhu, L. et al. Vision-Language Alignment Learning Under Affinity and Divergence Principles for Few-Shot OOD Generalization | 2024, 132 | 10.1007/s11263-024-02036-4 | Vision-language robustness under shift. In bib (`zhu2024vlalignood`) | Title only |
| Huang, Z. et al. Winning Prize Comes from Losing Tickets: Improve Invariant Learning by Exploring Variant Parameters for OOD Generalization | 2024, 133 | 10.1007/s11263-024-02075-x | Invariant learning against spurious features. In bib (`huang2024losingtickets`) | Title only |
| Gao, L. et al. Informative Scene Graph Generation via Debiasing | 2025, 133 | 10.1007/s11263-025-02365-y | Debiasing in a vision-language task | Title only |
| Chen, L. et al. A Causal Inspired Early-Branching Structure for Domain Generalization | 2024, 132 | 10.1007/s11263-024-02061-3 | Architecture as a cause of generalisation (close in spirit to our connector argument) | Title only |
| Li, H. et al. Multimodal Domain Generalization via Adaptive Dual-Objective Feature Learning | 2026, 134 | 10.1007/s11263-026-02991-0 | Multimodal robustness | Title only |
| Guo, Y. and Gu, X. MMRL++: Parameter-Efficient and Interaction-Aware Representation Learning for Vision-Language Models | 2025, 134 | 10.1007/s11263-025-02586-1 | Vision-language adapter design | Title only |
| Ying, Z. et al. SafeBench: A Safety Evaluation Framework for Multimodal Large Language Models | 2025 | 10.1007/s11263-025-02613-1 | MLLM evaluation in IJCV | Title only |

**Fit verdict for IJCV:** the journal publishes MLLM surveys and benchmarks, vision-language robustness and
domain generalisation, but no recent paper on spurious correlations in LLM-based VLMs turned up. That is good for
novelty, but the cover letter has to make the fit explicit (robustness of multimodal vision systems).

## Machine Learning (ISSN 1573-0565)

| Paper | Year, vol. | DOI | Why cite | Checked |
| :--- | :--- | :--- | :--- | :--- |
| Mitchell, J., Martinez-del-Rincon, J. and McLaughlin, N. Weighted UnLearning from Experience for Improved Spurious Correlation Robustness | 2026, 115 | 10.1007/s10994-026-07164-1 | **Direct topic match**: spurious-correlation robustness. In bib (`mitchell2026wule`) | Abstract checked |
| Gao, E. et al. Domain generalization via content factors isolation: a two-level latent variable modeling approach | 2025, 114 | 10.1007/s10994-024-06717-6 | Separating label from domain (spurious) factors | Abstract checked |
| Yuan, J. et al. Entangle-then-disentangle: a novel approach for enhancing large vision-language model | 2025, 114 | 10.1007/s10994-025-06811-3 | Large VLMs in this journal. In bib (`yuan2025entangle`) | Title only |
| Raza, S. et al. Developing safe and responsible large language model: can we balance bias reduction and language understanding? | 2025 | 10.1007/s10994-025-06767-4 | Bias in LLMs | Title only |
| Wang, X. et al. Expanding Domain Generalization Theory: Error Bound Beyond Convex Combination Assumption | 2026 | 10.1007/s10994-026-07003-3 | Theory-minded audience | Title only |

**Fit verdict for Machine Learning:** spurious-correlation robustness appears directly (Mitchell et al., 2026),
and the journal publishes large-VLM and bias work. Topic fit is at least as clear as for IJCV.

Not covered: Crossref rate-limited some queries (they were retried); Springer's own journal pages need a sign-in,
so the journals' editorial statements on scope were not read.
