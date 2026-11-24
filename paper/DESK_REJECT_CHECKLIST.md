# Pre-submission checklist: avoiding desk rejection

Target order: **IJCV** → **Machine Learning** (Springer) → **Neural Computing and Applications**. All three are hybrid:
the subscription route is free; choose it at acceptance (do not tick open access unless a library agreement pays).

Desk editors mostly reject for (1) an unclear contribution in the abstract and first page, (2) poor fit with the
journal, (3) format or policy problems. Every item below targets one of these. Status as of 24 Nov 2026.

Status: **Done** · **To do** (can be done in the repo) · **Authors** (only the authors can do it) · **Wait** (needs round 9)

## A. Format and policy (an easy desk rejection if missed)

| # | Item | Status |
| :--- | :--- | :--- |
| A1 | Springer Nature LaTeX template (`sn-jnl` v3.1), compiles with no errors and no undefined citations | Done (`paper/springer/`) |
| A2 | Length within the limit. IJCV regular article: up to 25 journal pages (draft is 31). *Machine Learning*: no limit found | To do |
| A3 | Author names, ORCIDs, affiliations, corresponding e-mail (placeholders still in `main.tex`) | Authors |
| A4 | Every "To be completed" in Declarations filled: funding, competing interests, author contributions (CRediT) | Authors |
| A5 | Use-of-generative-AI statement matching what was actually done (Springer requires it; AI cannot be an author) | Authors |
| A6 | Data and code availability statements; repository public and with a licence | Licence: Authors |
| A7 | Read the journal's submission guidelines signed in (the pages need a Springer login) and check: declarations, blind review, reference style, figure formats | Authors |
| A8 | *Machine Learning* only: 1–2 page contribution information sheet | To do (after choosing the journal) |
| A9 | Figures referenced in the text, readable at print size, colour-blind safe | To do |
| A10 | No conference version overlap, or a stated extension if one exists | Done (none) |
| A11 | Abstract within the journal's word limit (~250) | Done (check after edits) |

## B. Contribution clear on page 1

| # | Item | Status |
| :--- | :--- | :--- |
| B1 | Abstract states the question, the controlled design, the main numbers and the scope limits | Done (updated abstract in `main.tex`) |
| B2 | First paragraph of the introduction says what is new: a controlled cause (the zero-start gate), not a new state of the art | To do |
| B3 | Contributions as a numbered list at the end of the introduction, each tied to a result | To do |
| B4 | Positioning against the closest work (Farazi et al. 2026; Li et al. 2026; Balaji et al. 2026; VideoDistill) in the introduction, not only in related work | To do (source: `docs/NOVELTY.md` §6, `docs/CONTRADICTIONS.md`) |
| B5 | Absolute accuracy addressed early: our 74–84 worst-group vs 91 for label-using methods (SCER); we isolate a cause, we do not compete | To do |
| B6 | Title carries the scope (single input-level gate, zero start) | Done ("A Zero-Initialised Injection Gate Makes Shortcut Learning More Likely…") |

## C. Fit with the journal

| # | Item | Status |
| :--- | :--- | :--- |
| C1 | Cite 3–5 recent papers from the target journal on robustness, multimodal models or vision-language connectors | To do |
| C2 | Cover letter: contribution in three sentences, why this journal, what is new over Farazi et al., no conflicts, not under review elsewhere | To do (draft needs author details) |
| C3 | Pre-submission inquiry to the IJCV editor-in-chief (abstract + two lines on fit) | Authors (draft ready on request; we do not send it) |
| C4 | Suggested and excluded reviewers (with reasons), if the system asks | Authors |

## D. Evidence a reviewer will check

| # | Item | Status |
| :--- | :--- | :--- |
| D1 | Main result: H1 in all four cells after Holm, 15 paired seeds, effect sizes and intervals | Done |
| D2 | Robustness of H1: label-free selection, unpaired tests, confirmation seeds alone | Done |
| D3 | Mechanism controls: open, half-open, LR, clipping, weight decay, 10× alignment | Done |
| D4 | Gate start sweep (0.01, 0.1) | Wait (`r9-eps01`, `r9-eps1`) |
| D5 | Pathway tests on 5 seeds (removed, zero features, shuffled) and signed gate gradient | Wait (`r9-ablate`) |
| D6 | Norm-capped fixed gate on binary tasks and VQA-CP; text-only VQA-CP baseline | Wait (`r9-capped`, `r9-vqacp`) |
| D7 | Negative and limiting results reported up front: per-layer gates (no gap), VQA-CP (reversed), SigLIP-large Waterbirds (narrowed) | Done in README and `RESULTS_DISCUSSION.md`; To do in manuscript after round 9 |
| D8 | Limitations section: scale (4B, 4-bit), short training, frozen encoder, binary readout, seeds as random replicates | Done (text); To do (cut for length) |
| D9 | Reproducibility: pinned versions, job scripts, per-run results, tests passing | Done (32 tests) |

## E. Polish

| # | Item | Status |
| :--- | :--- | :--- |
| E1 | Related-work section in the manuscript (from `docs/LITERATURE.md`) | To do |
| E2 | Remove writing-kit notes from the generated sections; check `convert.py` output by eye | To do |
| E3 | Language check (one native-level read) | Authors |
| E4 | Every number in the text matches `results_study/` | To do (after round 9) |
| E5 | References: complete authors, venues, years; arXiv entries updated to published versions where available | To do |

## F. Submission strategy

| # | Item | Status |
| :--- | :--- | :--- |
| F1 | Post the manuscript on arXiv before or at submission (priority; lowers the cost of a desk rejection) | Authors |
| F2 | If desk-rejected, accept Springer's transfer offer or resubmit to the next journal in the order above within a week | Plan |
| F3 | Keep the paper-2 material (injection norm, VQA-CP) out of paper 1's claims if it becomes its own paper | Decide after round 9 |
