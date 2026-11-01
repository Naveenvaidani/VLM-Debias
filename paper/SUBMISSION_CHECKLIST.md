# Pre-submission checklist (what a Q1 reviewer is likely to raise)

Status as of 2026-11-01. "Done" = addressed in code, results or text; "Running" = Kaggle job in progress;
"Authors" = needs a decision only the authors can make.

| # | Reviewer concern | Where addressed | Status |
| :---: | :--- | :--- | :--- |
| 1 | "The gate closes" is inaccurate: it is initialised at 0 and never opens | Wording in README, INTRODUCTION, TITLE_ABSTRACT, RESULTS_DISCUSSION (R3, D2) | Done |
| 2 | Learned and fixed gates differ in two ways (trainability, location of the zero init) | Connector table; METHODOLOGY §6; `bottleneck_open` control | Done (text) · Running (control) |
| 3 | SigLIP-large: gate shut (0.0006) but WGA unaffected, so a shut gate is not sufficient | Reframed as a risk factor (D5, title "Induce"); `inject_ratio` logged | Done (text) · Running (measurement) |
| 4 | Fixed gate 44.5 at 8 epochs (1 seed) | Flagged as unresolved; seeds 1–4 | Running |
| 5 | "Connector type does not matter" overclaims (TOST fails; CelebA MLP +5) | "No significant difference detected; equivalence not established" everywhere | Done |
| 6 | Unequal seeds (15 vs 13) | Watermark seeds 8 and 14 were never launched, not excluded; being added | Done (text) · Running (seeds) |
| 7 | Is pairing by seed justified? | METHODOLOGY §6; Welch and Mann–Whitney agree (`results_study/robustness_checks.md`) | Done |
| 8 | Model selection uses group labels | Label-free rule reported; gate effect larger with it (+17.7 / +24.1) | Done |
| 9 | Learning rate not tuned per connector | Sweep 5e-5 / 1e-3 | Running |
| 10 | Phi-4-mini Watermark underpowered (p = 0.044) | Seeds 5–9 added (10 total) | Done (runs) · analysis pending |
| 11 | CelebA non-standard (8k train) | Stated in README §6, METHODOLOGY, R7 | Done |
| 12 | Robustness bought with average accuracy? | Fixed gate raises average accuracy too; ID ≈ 98% unchanged | Done |
| 13 | No standard group-robust baseline at the full model | Group DRO (`--loss groupdro`), MLP and both gates, 2 tasks | Running |
| 14 | Generative-AI disclosure (Springer / Elsevier / IEEE policy) | Template in METHODOLOGY §7 | Authors (adjust and include) |
| 15 | Author names, affiliations, ORCIDs | README citation, paper front matter | Authors |
| 16 | Bibliography: full author lists, preprint IDs | `paper/fetch_authors.py` → `authors.json`; all 93 entries have full authors and eprint IDs | Done |
| 17 | Licence, pinned requirements, data/code availability | `requirements-study.txt` pinned; availability statements in METHODOLOGY §7 | Done · licence: Authors |
| 18 | Causal wording in the title | Recommended title now "Induce"; outline renamed | Done |

## Still open after these runs (would strengthen a TPAMI / IJCV submission)

- Behavioural test on a released Flamingo-style model (force IDEFICS-9B gates open, compare WGA).
- A natural spurious-cue benchmark (SpuriVerse or a VQA-CP subset).
- One larger or full-precision LLM run (limited by T4 memory; otherwise state as a limitation).
- Position the contribution against Farazi et al. (2026): they show gates stay closed; we show what the
  model then relies on, and that it turns robustness into a seed lottery.
