"""Convert the paper-kit markdown (paper/*.md) into LaTeX sections for the Springer Nature template.

    python paper/springer/convert.py      (needs pypandoc_binary)

Draft notes in the markdown (italic preambles, paragraph labels such as "¶1 — Hook") are dropped; citation keys
written as [key1; key2] become \\cite{key1,key2}. Edit the generated .tex only after the markdown is final, or edit
the markdown and re-run this script.
"""
import os
import re

import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
CITE = re.compile(r"\[((?:[a-z][a-z0-9]*\d{4}[a-z0-9]*)(?:\s*[;,]\s*[a-z][a-z0-9]*\d{4}[a-z0-9]*)*)\]")


def clean(md: str, drop_until_rule: bool = True, drop_from: str = None) -> str:
    if drop_until_rule and "\n---\n" in md:  # the italic note before the first rule is a writing aid
        md = md.split("\n---\n", 1)[1]
    if drop_from and drop_from in md:
        md = md.split(drop_from, 1)[0]
    md = re.sub(r"^\*\*¶\d+ —[^*]*\*\*\s*$", "", md, flags=re.M)        # paragraph labels
    md = re.sub(r"^---\s*$", "", md, flags=re.M)                       # horizontal rules
    md = re.sub(r"^#\s.*$", "", md, count=1, flags=re.M)               # the file's own top heading
    md = re.sub(r"^(#{2,6})\s+(?:Part [IVX]+ — |\d+(?:\.\d+)*\.\s+|[A-Z]\d+\.\s+)?", lambda m: m.group(1) + " ", md, flags=re.M)
    md = md.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2009", " ")
    md = CITE.sub(lambda m: "\\cite{" + ",".join(k.strip() for k in re.split(r"[;,]", m.group(1))) + "}", md)
    md = re.sub(r"\]\((?:\.\./)*docs/figures/final/", "](figures/", md)
    md = re.sub(r"\]\((?:\.\./)*results_study/h1_gate_seeds\.png", "](figures/h1_seeds_qwen.png", md)
    return md


def convert(md: str) -> str:
    tex = pypandoc.convert_text(md, "latex", format="markdown-auto_identifiers+raw_tex", extra_args=["--wrap=preserve"])
    return tex.replace("\\textbackslash cite\\{", "\\cite{").replace("\\textbackslash{}cite\\{", "\\cite{")


def main():
    jobs = {"introduction": ("INTRODUCTION.md", {}),
            "related_work": ("RELATED_WORK.md", {}),
            "methods": ("METHODOLOGY.md", {"drop_from": "## 7. Statements required at submission"}),
            "results_discussion": ("RESULTS_DISCUSSION.md", {})}
    for name, (src, kw) in jobs.items():
        md = clean(open(os.path.join(PAPER, src), encoding="utf-8").read(), **kw)
        tex = convert(md)
        tex = re.sub(r"\\cite\\\{([^}]*)\\\}", r"\\cite{\1}", tex)
        open(os.path.join(HERE, "sections", name + ".tex"), "w", encoding="utf-8").write(tex)
        print(name, len(tex), "chars,", tex.count("\\cite{"), "citations")


if __name__ == "__main__":
    main()
