# Writing and submission rules, and where the manuscript stands

Sources: the IJCV submission guidelines (pasted from the journal site by the authors, 24 Nov 2026), the IJCV special
issue call (*Multimodal Large Language Models for Unified Comprehension and Generation*, deadline 1 Jan 2027), Jennifer
Widom's *Tips for Writing Technical Papers* (https://cs.stanford.edu/people/widom/paper-writing.html), the NeurIPS 2024
and CVPR 2025 reviewer guidelines (https://neurips.cc/Conferences/2024/ReviewerGuidelines,
https://cvpr.thecvf.com/Conferences/2025/ReviewerGuidelines), and the desk-rejection research in
[`DESK_REJECT_CHECKLIST.md`](DESK_REJECT_CHECKLIST.md). Springer's AI and preprint policy pages and the IJCV journal
page could not be loaded by our tools; the pasted guidelines cover the AI rules.

Status: **Done** · **To do** · **Authors** (only the authors can do it) · **Wait** (needs round 9)

## 1. IJCV formatting rules (from the journal's guidelines)

| Rule | Status |
| :--- | :--- |
| LaTeX with the Springer Nature template, option `[iicol]`; upload source and compiled PDF | Done (`paper/springer/main.tex`, `sn-jnl` v3.1, two-column) |
| No subfolders in the uploaded LaTeX source | To do at upload: flatten `sections/` and `figures/` (one folder) |
| Title page: names, affiliations with city and country, active e-mail of the corresponding author, ORCIDs | Authors: corresponding e-mail and full surname of the first author |
| Abstract 150 to 250 words, no undefined abbreviations or unspecified references | Done (check the count after the final edit) |
| 4 to 6 keywords | Done (6) |
| Headings: decimal, at most three levels | Done |
| Abbreviations defined at first use | To do: one pass |
| Name-and-year citations; reference list alphabetical; DOIs as full links | Done for style (`sn-basic` author-year); To do: DOI links for all entries that have one |
| "Statements and Declarations" section (competing interests mandatory; incomplete submissions are returned) | Done (heading and structure); Authors: funding and competing interests text |
| Data availability statement | Done |
| LLM use documented in the Methods section (copy-editing alone need not be declared) | Done (Methods, "Use of AI tools"); Authors: confirm the wording matches what was done |
| Acknowledgements on the title page; author contributions | Authors |
| Tables numbered, cited in order, captioned, editable | Done (LaTeX tables) |
| Figures: vector with embedded fonts, sans-serif lettering 8 to 12 pt, no titles inside, 84 or 174 mm wide, patterns as well as colour, captions without a final full stop, figure files named Fig1, Fig2, ... | Done except file names (rename at upload) |
| Supplementary material referred to as "Online Resource" | To do (the full protocol and tables in `paper/METHODOLOGY.md` and `RESULTS_DISCUSSION.md`) |
| No simultaneous submission; no salami slicing; cite the literature that supports each claim | Done (one paper) |
| Suggested reviewers must be independent, with institutional e-mails | Authors |
| No page limit stated in the guidelines | Current length: 17 pages, two-column |

## 2. Writing and presentation (Widom; reviewer guidelines)

| Practice | Status |
| :--- | :--- |
| Introduction answers: what is the problem, why it matters, why it is hard, why it is unsolved, what we do and found, with limits | Done (Introduction ¶1 to ¶5) |
| Bulleted contributions in the introduction | Done (5 contributions) |
| The main contribution is visible by page 3 | Done (two-column page 2) |
| Conclusion short, concrete, not a copy of the abstract | Done |
| Define each symbol once before use | Done in Methods (gate, $\Delta_t$, WGA, injection ratio) |
| Avoid "etc.", vague "this"/"it", decorative emphasis | Done in the humanizer pass |
| Limitations stated openly (reviewers are told to reward this) | Done (own section) |
| Not beating the state of the art is not a reason to reject, if novelty and insight are clear | Addressed: we state we isolate a cause, with the SCER comparison |
| Enough detail to reproduce | Done (Methods, code, pinned versions, per-run results) |

## 3. Claims and evidence

| Rule | Status |
| :--- | :--- |
| Every number traceable to a result file | Done for the manuscript sections; To do: final check after round 9 |
| Every claim about other work cited, and the cited work read | Done for the closest work (read in full, `docs/NOVELTY.md` §6); six context papers judged from abstracts are cited only for what their abstracts state |
| Confirmatory and exploratory tests separated | Done (Methods, Statistics) |
| Negative and limiting results reported | Done (Results, "Boundaries"; Limitations) |

## 4. References

| Rule | Status |
| :--- | :--- |
| No duplicate entries | Done (one duplicate, Places, removed) |
| Every arXiv ID and DOI resolves to the cited title | In progress (automatic check against arXiv and Crossref) |
| Cite the published version where one exists | To do after the check |
| Complete author lists (APA 7: up to 20 authors) | To do: `paper/fetch_authors.py` fills them |

## 5. The template list the authors pasted (generic reviewer suggestions)

| Suggestion | Applies to IJCV? | Status |
| :--- | :--- | :--- |
| Quantitative results in the abstract | Yes | Done |
| Bulleted contributions, also in the introduction | Yes | Done |
| Recent literature (2024 to 2026) | Yes | Done (closest work from 2025 and 2026, IJCV papers from 2024 to 2026) |
| Limitations section before the conclusion | Yes | Done |
| 10 to 12 pages | No: the authors confirmed it was a template; IJCV states no page limit | Not applied |
| High-resolution figures | Yes | Done (vector PDF) |
| Editable tables | Yes | Done |
| Equations in a math editor | Yes | Done (LaTeX) |
| Societal impact section | Yes | Done ("Practical and societal implications") |
