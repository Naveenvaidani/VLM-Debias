# Pre-submission checklist (what a Q1 reviewer is likely to raise, 38 items)

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
| 16 | Bibliography: full author lists, preprint IDs | `paper/fetch_authors.py` → `authors.json`; all 101 entries have full authors and eprint IDs | Done |
| 17 | Licence, pinned requirements, data/code availability | `requirements-study.txt` pinned; availability statements in METHODOLOGY §7 | Done · licence: Authors |
| 18 | Causal wording in the title | Recommended title now "Induce"; outline renamed | Done |

## Second pass (after reading Flamingo, Farazi et al. 2026, SpuriVerse, SpurLens and the NeurIPS checklist)

| # | Reviewer concern | Where addressed | Status |
| :---: | :--- | :--- | :--- |
| 19 | Logged gate gradient is the post-clipping value; global clipping couples the gate to every other update | Raw gradient logged (`grad_raw`); `--clip_exclude_gate` control; R3 text | Done (code, text) · Running (control) |
| 20 | "Flamingo-style" overclaims: Flamingo gates every 4th–7th LM layer and trains 500k steps; gates grow | Renamed "input-level gated cross-attention (Flamingo-type gate)"; scope paragraph in D2; limitation | Done · 3k-step alignment running |
| 21 | Flamingo: removing the gate costs 4.2 points and destabilises training | D2: our fixed gate keeps the identity via zero-initialised output; no instability observed | Done |
| 22 | Is the learned pathway *behaviourally* closed (Farazi's criterion), not just a small parameter? | Test-time ablation of the visual contribution on every selected checkpoint | Done (code) · Running |
| 23 | No correction across the four H1 tests | Holm: three survive (0.0012, 0.0012, 0.011), Phi Watermark fails (0.044) | Done |
| 24 | "Pre-declared" without a registry; hypothesis formed after the pilot | Timeline with commit hashes; confirmation-only test: p = 9.5e-4 and 0.0012 | Done |
| 25 | Linear DFR on the same features beats every VLM: why use a VLM? | D2 framing: study explains a VLM failure; the gap supports the transmission account | Done |
| 26 | Weight decay on the gate in stage 2 | Negligible (relative 2e-6 per step); `--gate_no_wd` control | Done · Running |
| 27 | One shortcut strength | Watermark p = 0.9 and 0.99, three connectors | Queued |
| 28 | SpurLens: bias comes from the vision encoder | Encoder frozen and shared, so it cannot explain the gate effect; complementary account in D2 | Done |
| 29 | Dataset/model licences and pre-training overlap (NeurIPS checklist item 12) | METHODOLOGY ethics: licences listed; overlap shared by all arms | Done |
| 30 | Compute including preliminary work (checklist item 8) | Estimate for rounds 1–5 stated; preliminary/failed sessions | Authors (add estimate) |

## Third pass: novelty (closest prior work searched on arXiv, OpenAlex and the web; see `docs/NOVELTY.md`)

| # | Reviewer concern | Where addressed | Status |
| :---: | :--- | :--- | :--- |
| 31 | "Zero-initialised gating is LLaMA-Adapter / Flamingo; what is new?" | NOVELTY §2: we do not claim the gate; we claim its controlled effect on shortcut reliance and seed variance | Done |
| 32 | "This is modality imbalance [wu2022greedy]" | LITERATURE C4, D2: a concrete architectural instance in LLM-based VLMs, measured on spurious groups; the seed lottery matches modality competition | Done |
| 33 | "If the gate is shut, how does the model see the watermark?" | D2 paragraph; answered by injection ratio and pathway ablation | Done (text) · Running |
| 34 | "Not Flamingo: injection is input-level only" | Per-layer `xattn_deep` (gated blocks before layers 0/8/16/24, parameter-matched), learned vs fixed, 5 seeds | Running (round 6) |
| 35 | "4-bit only" | fp16-weight replication of the gate comparison, 3 seeds | Running (round 6) |
| 36 | "Do released models have small gates?" | All Llama 3.2 Vision attention gates < 0.012 (11B) and < 0.003 (90B); IDEFICS-80B 9/20 | Done |
| 37 | Gradient modulation (OGM-GE) as a modality-imbalance baseline | Listed as optional | Open |
| 38 | Forcing released-model gates open; SpuriVerse | NOVELTY §4: not interpretable (distribution shift) / incompatible protocol; future work | Decided: not pursued |

Optional, not queued: native zero-shot on CelebA and Watermark (needs task images on the account that runs it);
gate excluded from clipping *and* no weight decay together; multi-layer injection.

## Still open after these runs (would strengthen a TPAMI / IJCV submission)

- Behavioural test on a released Flamingo-style model (force IDEFICS-9B gates open, compare WGA).
- A natural spurious-cue benchmark (SpuriVerse or a VQA-CP subset).
- One larger or full-precision LLM run (limited by T4 memory; otherwise state as a limitation).
- Position the contribution against Farazi et al. (2026): they show gates stay closed; we show what the
  model then relies on, and that it turns robustness into a seed lottery.
