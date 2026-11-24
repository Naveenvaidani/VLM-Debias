"""Build the flat IJCV upload folder: no subfolders, figures renamed Fig1, Fig2, ... in order of first appearance.

Also packs Online Resource 1 (every per-run result table and summary from results_study/) as ESM_1.zip.
Compiles the flat copy to check it builds on its own, then zips it.
Usage: python paper/springer/make_upload.py   (run paper/springer/convert.py first)
"""
import glob
import os
import re
import shutil
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "upload")


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    main_tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
    order = re.findall(r"\\input\{sections/([^}]+)\}", main_tex)
    texts = {n: open(os.path.join(HERE, "sections", n + ".tex"), encoding="utf-8").read() for n in order}

    # figure numbering follows the order in which figures appear in the compiled text
    figs = []
    for n in order:
        for f in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}", texts[n]):
            if f not in figs:
                figs.append(f)
    rename = {f: f"Fig{i + 1}{os.path.splitext(f)[1]}" for i, f in enumerate(figs)}
    for f, new in rename.items():
        shutil.copyfile(os.path.join(HERE, "figures", f), os.path.join(OUT, new))
    for n in order:
        t = texts[n]
        for f, new in rename.items():
            t = t.replace("{figures/" + f + "}", "{" + new + "}")
        open(os.path.join(OUT, n + ".tex"), "w", encoding="utf-8").write(t)
    open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write(main_tex.replace(r"\input{sections/", r"\input{"))
    for f in ("sn-jnl.cls", "sn-basic.bst", "references.bib"):
        shutil.copyfile(os.path.join(HERE, f), os.path.join(OUT, f))

    # Online Resource 1: per-run results (CSV, JSON, Markdown summaries), with a short README
    esm = os.path.join(OUT, "ESM_1.zip")
    with zipfile.ZipFile(esm, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Online Resource 1. Per-run results and analysis summaries for every experiment in the "
                                 "article (one row per training run; accuracies in % or as fractions as labelled). "
                                 "Code and per-example predictions: https://github.com/Naveenvaidani/VLM-Debias\n")
        for p in sorted(glob.glob(os.path.join(ROOT, "results_study", "**", "*"), recursive=True)):
            if os.path.isfile(p) and p.endswith((".csv", ".json", ".md")):
                z.write(p, os.path.relpath(p, os.path.join(ROOT, "results_study")))

    # check that the flat copy compiles by itself
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(OUT, "main.log"), encoding="utf-8", errors="replace").read()
    errors = [l for l in log.splitlines() if l.startswith("!")]
    undefined = len(re.findall(r"undefined", log))
    print("figures:", ", ".join(f"{k} -> {v}" for k, v in rename.items()))
    print("compile errors:", len(errors), "| 'undefined' warnings:", undefined)
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    print("pages:", pages.group(1) if pages else "?")

    for f in glob.glob(os.path.join(OUT, "main.*")):
        if not f.endswith((".tex", ".pdf", ".bbl")):
            os.remove(f)
    bundle = os.path.join(HERE, "IJCV_upload.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), f)
    print("wrote", bundle, "with", len(os.listdir(OUT)), "files (flat)")


if __name__ == "__main__":
    main()
