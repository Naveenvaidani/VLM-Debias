"""Convert the paper-kit markdown (paper/*.md) into LaTeX sections for the Springer Nature template.

    python paper/springer/convert.py      (needs pypandoc_binary)

Draft notes in the markdown (italic preambles, paragraph labels such as "¶1 — Hook") are dropped; citation keys
written as [key1; key2] become \\cite{key1,key2}. Edit the generated .tex only after the markdown is final, or edit
the markdown and re-run this script.
"""
import os
import re
import shutil

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
    # pandoc needs a blank line before a list that follows a paragraph line
    md = re.sub(r"(?m)^(?!\s*$)(?!\s*(?:[-*+]|\d+\.)\s)(.+)\n(?=\s*(?:[-*+]|\d+\.)\s)", r"\1\n\n", md)
    md = re.sub(r"(?<![\w$\\])d_z(?![\w])", r"$d_z$", md)
    md = CITE.sub(lambda m: "\\citep{" + ",".join(k.strip() for k in re.split(r"[;,]", m.group(1))) + "}", md)
    md = re.sub(r"\]\((?:\.\./)*docs/figures/final/", "](figures/", md)
    md = re.sub(r"\]\((?:\.\./)*results_study/h1_gate_seeds\.png", "](figures/h1_seeds_qwen.png", md)
    return md


def convert(md: str) -> str:
    tex = pypandoc.convert_text(md, "latex", format="markdown-auto_identifiers+raw_tex", extra_args=["--wrap=preserve"])
    return tex


def longtable_to_table(tex: str) -> str:
    """Two-column layouts ([iicol]) do not allow longtable: rewrite pandoc's longtables as full-width floats."""
    out, pos = [], 0
    while True:
        a = tex.find("\\begin{longtable}[]{", pos)
        if a < 0:
            out.append(tex[pos:]); break
        b = tex.find("\\end{longtable}", a)
        block = tex[a:b]
        cap_start = block.find("\\caption")
        colspec = block[len("\\begin{longtable}[]{"):cap_start].rstrip()[:-1]
        cap_end = block.find("\\tabularnewline", cap_start)
        caption = block[cap_start:cap_end]
        head = block[block.find("\\toprule\\noalign{}") + len("\\toprule\\noalign{}"):block.find("\\midrule\\noalign{}")]
        body = block[block.find("\\endlastfoot") + len("\\endlastfoot"):]
        out.append(tex[pos:a])
        out.append("\\begin{table*}[t]\n\\centering\n" + caption + "\n\\footnotesize\n\\begin{tabular}{" + colspec +
                   "}\n\\toprule\n" + head.strip() + "\n\\midrule\n" + body.strip() + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}")
        pos = b + len("\\end{longtable}")
    return "".join(out)


def main():
    jobs = {"introduction": ("INTRODUCTION.md", {}),
            "related_work": ("RELATED_WORK.md", {}),
            "methods": ("manuscript/03_methods.md", {}),
            "results": ("manuscript/04_results.md", {}),
            "discussion": ("manuscript/05_discussion.md", {}),
            "implications": ("manuscript/06_implications.md", {}),
            "limitations": ("manuscript/07_limitations.md", {}),
            "conclusion": ("manuscript/08_conclusion.md", {})}
    for name, (src, kw) in jobs.items():
        md = clean(open(os.path.join(PAPER, src), encoding="utf-8").read(), **kw)
        tex = longtable_to_table(convert(md))
        tex = re.sub(r"\\(cite[pt]?)\\\{([^}]*)\\\}", r"\\\1{\2}", tex)
        open(os.path.join(HERE, "sections", name + ".tex"), "w", encoding="utf-8").write(tex)
        print(name, len(tex), "chars,", tex.count("\\cite{"), "citations")
    # keep the template's bibliography identical to the verified source list built by paper/build_refs.py
    shutil.copyfile(os.path.join(PAPER, "references.bib"), os.path.join(HERE, "references.bib"))


if __name__ == "__main__":
    main()
