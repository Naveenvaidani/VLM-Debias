"""Fetch full author lists for paper/build_refs.py from the arXiv API (by eprint id) or Crossref (by DOI).

Writes paper/authors.json ({key: ["Last, First", ...]}); build_refs.py uses it when present and falls back to
"first author and others". Usage: python paper/fetch_authors.py
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_refs import R  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "VLM-Debias reference builder (mailto:jeyadev2006@gmail.com)"}
ATOM = "{http://www.w3.org/2005/Atom}"


def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", url, e, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return None


def last_first(name):
    parts = name.split()
    return name if len(parts) < 2 else f"{parts[-1]}, {' '.join(parts[:-1])}"


def main():
    out = {}
    ax = {r[10]: r[0] for r in R if r[10]}
    ids = list(ax)
    for i in range(0, len(ids), 20):
        xml = get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(ids[i:i + 20]), "max_results": 20}))
        for e in ET.fromstring(xml).findall(f"{ATOM}entry"):
            aid = e.find(f"{ATOM}id").text.rsplit("/", 1)[-1].split("v")[0]
            if aid in ax:
                out[ax[aid]] = [last_first(a.find(f"{ATOM}name").text.strip()) for a in e.findall(f"{ATOM}author")]
        time.sleep(3)
    for r in R:
        key, doi = r[0], r[9]
        if key in out or not doi:
            continue
        data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        if data:
            authors = json.loads(data)["message"].get("author", [])
            names = [f"{a['family']}, {a.get('given', '')}".strip(", ") for a in authors if "family" in a]
            if names:
                out[key] = names
        time.sleep(1)
    # sanity: first author's surname must match the verified first author in build_refs.py
    bad = [r[0] for r in R if r[0] in out and r[2].split(",")[0].split()[-1].lower() not in out[r[0]][0].lower()]
    for k in bad:
        print("first-author mismatch, kept abbreviated:", k, out.pop(k)[0], file=sys.stderr)
    json.dump(out, open(os.path.join(HERE, "authors.json"), "w"), indent=1, ensure_ascii=False)
    print(f"full author lists for {len(out)} of {len(R)} entries")


if __name__ == "__main__":
    main()
