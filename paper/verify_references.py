"""Check that every DOI and arXiv ID in paper/references.bib resolves to the cited title (Crossref, arXiv API).
Run from paper/ after writing bibrows.json (keys, titles, DOIs, arXiv IDs). Result: refcheck.json."""
import json, re, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET

rows = json.load(open("bibrows.json"))
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())


def match(a, b):
    a, b = norm(a), norm(b)
    return bool(a and b) and (a[:35] in b or b[:35] in a)


res = []
dois = [r["doi"] for r in rows if r["doi"]]
found = {}
for i in range(0, len(dois), 20):
    q = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"filter": ",".join("doi:" + d for d in dois[i:i + 20]), "rows": 40, "select": "DOI,title"})
    items = None
    for att in range(5):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ref-check/1.0"})
            items = json.load(urllib.request.urlopen(req, timeout=90))["message"]["items"]
            break
        except Exception as e:
            print("crossref retry", att, e, flush=True)
            time.sleep(30 * (att + 1))
    for it in items or []:
        found[it["DOI"].lower()] = (it.get("title") or [""])[0]
    time.sleep(5)
for r in rows:
    if r["doi"]:
        t = found.get(r["doi"].lower())
        res.append((r["key"], "DOI", r["doi"], "NOT FOUND" if t is None else ("OK" if match(t, r["title"]) else "TITLE MISMATCH"), t or ""))

ids = [r["eprint"] for r in rows if r["eprint"]]
NS = {"a": "http://www.w3.org/2005/Atom"}
ax = {}
for i in range(0, len(ids), 25):
    q = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": ",".join(ids[i:i + 25]), "max_results": 30})
    x = None
    for att in range(5):
        try:
            x = ET.fromstring(urllib.request.urlopen(q, timeout=90).read())
            break
        except Exception as e:
            print("arxiv retry", att, e, flush=True)
            time.sleep(20 * (att + 1))
    for e in (x.findall("a:entry", NS) if x is not None else []):
        aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
        ax[aid] = " ".join((e.find("a:title", NS).text or "").split())
    time.sleep(4)
for r in rows:
    if r["eprint"]:
        t = ax.get(r["eprint"])
        res.append((r["key"], "arXiv", r["eprint"], "NOT FOUND" if t is None else ("OK" if match(t, r["title"]) else "TITLE MISMATCH"), t or ""))

json.dump(res, open("refcheck.json", "w"), indent=1)
bad = [x for x in res if x[3] != "OK"]
print(len(res), "identifiers checked;", len(bad), "problems", flush=True)
for b in bad:
    print(b[0], "|", b[1], b[2], "|", b[3], "| found:", b[4][:110], flush=True)
