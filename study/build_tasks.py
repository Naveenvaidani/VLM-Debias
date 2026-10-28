"""
Build the shortcut tasks and the caption-alignment set.

Tasks (binary; y = label, a = shortcut attribute; train has P(a == y) = p, val/test are group-balanced):
  waterbirds     bird type vs background (natural shortcut, from data/waterbirds)
  cooccur        "is there an O?" vs presence of a co-occurring object C (natural COCO shortcut;
                 the (O, C) pair is chosen automatically to have enough counter-examples)
  watermark_p*   "is there a dog?" vs an arbitrary stock-photo watermark (synthetic, Bleeker-style)
  typographic_p* "is there a dog?" vs the word "dog" printed on the image (synthetic, semantic)
  captions       COCO image-caption pairs for stage-1 connector alignment (no task val/test images)

    python study/build_tasks.py --coco_root /kaggle/input/coco-2017-dataset/coco2017 --split train2017
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

YN = ["No", "Yes"]


def object_questions(obj: str):
    return [f"Is there a {obj} in this image?", f"Does this picture contain a {obj}?", f"Can you see a {obj} here?"]


def write_task(out_dir, name, questions, answers, group_names, rows, extra=None):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "metadata.csv"), index=False)
    with open(os.path.join(d, "task.json"), "w") as f:
        json.dump({"name": name, "questions": questions, "answers": answers, "group_names": group_names,
                   **(extra or {})}, f, indent=2)
    df = pd.DataFrame(rows)
    print(f"[{name}] " + ", ".join(f"{s}: {len(df[df.split == s])}" for s in ("train", "val", "test")),
          "| groups per split:", df.groupby(["split", "y", "a"]).size().to_dict(), flush=True)


def split_sizes(n_train, n_val, n_test, p):
    """Per-group counts: train has a == y with probability p; val/test are balanced."""
    tr_major, tr_minor = round(n_train * p / 2), round(n_train * (1 - p) / 2)
    return {"train": {(0, 0): tr_major, (1, 1): tr_major, (0, 1): tr_minor, (1, 0): tr_minor},
            "val": {g: n_val // 4 for g in [(0, 0), (0, 1), (1, 0), (1, 1)]},
            "test": {g: n_test // 4 for g in [(0, 0), (0, 1), (1, 0), (1, 1)]}}


def assign(pools, sizes, rng):
    """pools[(y, a)] -> list of image ids; draws disjoint samples for every split and group."""
    rows = []
    pools = {g: list(rng.permutation(v)) for g, v in pools.items()}
    for split in ("test", "val", "train"):  # fill evaluation splits first
        for g, n in sizes[split].items():
            take, pools[g] = pools[g][:n], pools[g][n:]
            if len(take) < n:
                raise ValueError(f"not enough images for group {g} in {split}: {len(take)} < {n}")
            rows += [{"img_id": int(i), "y": g[0], "a": g[1], "split": split} for i in take]
    return rows


def scaled(n_train, n_val, n_test, p, need_fn, max_scale=1.0):
    """Largest scale in (0, max_scale] for which need_fn(sizes) holds."""
    for scale in np.linspace(max_scale, 0.05, 40):
        s = split_sizes(int(n_train * scale), int(n_val * scale), int(n_test * scale), p)
        if need_fn(s):
            return s, scale
    raise ValueError("dataset too small for any scale")


# --- rendering -----------------------------------------------------------------------------------------

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_watermark(img, rng):
    """Semi-transparent stock-photo mark near the bottom-right corner (arbitrary, label-free cue)."""
    img = img.convert("RGBA")
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(max(12, int(min(w, h) * rng.uniform(0.07, 0.09))))
    text = "© STOCKPIX"
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    x = w - tw - int(w * rng.uniform(0.02, 0.06))
    y = h - th - int(h * rng.uniform(0.02, 0.06))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 170), stroke_width=2, stroke_fill=(0, 0, 0, 170))
    return Image.alpha_composite(img, layer).convert("RGB")


def render_word(img, word, rng):
    """The class word printed at a random position (typographic shortcut)."""
    img = img.convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = _font(max(14, int(min(w, h) * rng.uniform(0.12, 0.2))))
    tw, th = draw.textbbox((0, 0), word, font=font)[2:]
    x = int(rng.uniform(0, max(1, w - tw)))
    y = int(rng.uniform(0, max(1, h - th)))
    colors = [(255, 255, 255), (0, 0, 0), (255, 220, 0), (220, 20, 20)]
    fill = colors[rng.integers(len(colors))]
    stroke = (0, 0, 0) if sum(fill) > 300 else (255, 255, 255)
    draw.text((x, y), word, font=font, fill=fill, stroke_width=max(1, font.size // 12), stroke_fill=stroke)
    return img


# --- tasks ---------------------------------------------------------------------------------------------

class Coco:
    def __init__(self, root, split, download_missing=False):
        with open(os.path.join(root, "annotations", f"instances_{split}.json")) as f:
            inst = json.load(f)
        self.image_dir = os.path.join(root, split)
        self.files = {im["id"]: im["file_name"] for im in inst["images"]}
        self.cat_name = {c["id"]: c["name"] for c in inst["categories"]}
        self.cats = defaultdict(set)
        for a in inst["annotations"]:
            self.cats[a["image_id"]].add(self.cat_name[a["category_id"]])
        self.ids = sorted(self.files)
        self.split = split
        self.download_missing = download_missing
        cap_path = os.path.join(root, "annotations", f"captions_{split}.json")
        self.captions = defaultdict(list)
        if os.path.exists(cap_path):
            with open(cap_path) as f:
                for c in json.load(f)["annotations"]:
                    self.captions[c["image_id"]].append(c["caption"])

    def path(self, img_id):
        p = os.path.join(self.image_dir, self.files[img_id])
        if not os.path.exists(p) and self.download_missing:
            import urllib.request
            os.makedirs(self.image_dir, exist_ok=True)
            urllib.request.urlretrieve(f"http://images.cocodataset.org/{self.split}/{self.files[img_id]}", p)
        return p


def build_object_shortcut_tasks(coco, out, args, rng, kinds=("watermark", "typographic")):
    """watermark_p* and typographic_p*: label = dog present; shortcut overlay controlled per image."""
    pos = [i for i in coco.ids if "dog" in coco.cats[i]]
    neg = [i for i in coco.ids if "dog" not in coco.cats[i]]
    used = set()
    for kind in kinds:
        for p in args.p_list:
            # the overlay is applied per image, so any image can go into either a-group
            def need(s):
                return len(pos) >= sum(n for sp in s for (y, _), n in s[sp].items() if y == 1) and \
                    len(neg) >= sum(n for sp in s for (y, _), n in s[sp].items() if y == 0)
            sizes, scale = scaled(args.n_train, args.n_val, args.n_test, p, need)
            sub = np.random.default_rng(args.seed + int(p * 1000) + (0 if kind == "watermark" else 7))
            rows = []
            pools = {1: list(sub.permutation(pos)), 0: list(sub.permutation(neg))}
            for split in ("test", "val", "train"):  # same images for every p/kind in val/test (seeded)
                for (y, a), n in sizes[split].items():
                    take, pools[y] = pools[y][:n], pools[y][n:]
                    rows += [{"img_id": int(i), "y": y, "a": a, "split": split} for i in take]
            name = f"{kind}_p{p:g}"
            img_dir = os.path.join(out, name, "images")
            os.makedirs(img_dir, exist_ok=True)
            for r in tqdm(rows, desc=f"render {name}"):
                src = coco.path(r["img_id"])
                if r["a"] == 1:
                    dst = os.path.join(img_dir, f"{r['img_id']}.jpg")
                    if not os.path.exists(dst):
                        im = Image.open(src)
                        im = render_watermark(im, sub) if kind == "watermark" else render_word(im, "dog", sub)
                        im.save(dst, quality=92)
                    r["image"] = dst
                else:
                    r["image"] = src
                if r["split"] != "train":
                    used.add(r["img_id"])
            groups = [f"no dog / {'no ' if a == 0 else ''}{'mark' if kind == 'watermark' else 'word'}"
                      if y == 0 else f"dog / {'no ' if a == 0 else ''}{'mark' if kind == 'watermark' else 'word'}"
                      for y in (0, 1) for a in (0, 1)]
            write_task(out, name, object_questions("dog"), YN, groups, rows,
                       {"shortcut": kind, "p": p, "scale": round(float(scale), 3)})
    return used


def build_cooccur_task(coco, out, args, rng):
    """'Is there an O?' where object C co-occurs with O; pick the pair with the most counter-examples."""
    names = sorted(set(coco.cat_name.values()))
    present = {n: set() for n in names}
    for i in coco.ids:
        for n in coco.cats[i]:
            present[n].add(i)
    all_ids = set(coco.ids)
    best = None
    for o in names:
        for c in names:
            if o == c or c == "person":  # person is everywhere; it makes a weak, uninteresting shortcut
                continue
            n11 = len(present[o] & present[c])
            if n11 < 0.5 * len(present[o]):  # require a genuine natural correlation: P(C | O) >= 0.5
                continue
            n10 = len(present[o] - present[c])
            n01 = len(present[c] - present[o])
            score = min(n10, n01, n11 / 4)
            if best is None or score > best[0]:
                best = (score, o, c, n11, n10, n01)
    _, o, c, n11, n10, n01 = best
    print(f"[cooccur] chose O='{o}', C='{c}': O&C={n11}, O&!C={n10}, !O&C={n01}, P(C|O)={n11 / (n11 + n10):.2f}")
    pools = {(1, 1): list(present[o] & present[c]), (1, 0): list(present[o] - present[c]),
             (0, 1): list(present[c] - present[o]), (0, 0): list(all_ids - present[o] - present[c])}
    need = lambda s: all(len(pools[g]) >= sum(s[sp][g] for sp in s) for g in pools)
    sizes, scale = scaled(args.n_train, args.n_val, args.n_test, args.p, need)
    rows = assign(pools, sizes, rng)
    for r in tqdm(rows, desc="cooccur paths"):
        r["image"] = coco.path(r["img_id"])
    groups = [f"{'no ' if y == 0 else ''}{o} / {'no ' if a == 0 else ''}{c}" for y in (0, 1) for a in (0, 1)]
    write_task(out, "cooccur", object_questions(o), YN, groups, rows,
               {"shortcut": "cooccur", "object": o, "context": c, "p": args.p, "scale": round(float(scale), 3)})
    return {r["img_id"] for r in rows if r["split"] != "train"}


def build_waterbirds_task(out, data_dir):
    meta_path = os.path.join(data_dir, "waterbirds", "metadata.csv")
    if not os.path.exists(meta_path):
        print("[waterbirds] skipped: run scripts/prepare_waterbirds.py first")
        return
    m = pd.read_csv(meta_path)
    root = os.path.abspath(os.path.join(data_dir, "waterbirds"))
    split = {0: "train", 1: "val", 2: "test"}
    rows = [{"img_id": i, "image": os.path.join(root, r.img_filename), "y": int(r.y), "a": int(r.place),
             "split": split[int(r.split)]} for i, r in m.iterrows()]
    questions = ["Is the bird in this image a landbird or a waterbird?",
                 "What kind of bird is this, a landbird or a waterbird?",
                 "Classify this bird as a landbird or a waterbird."]
    write_task(out, "waterbirds", questions, ["landbird", "waterbird"],
               ["landbird / land", "landbird / water", "waterbird / land", "waterbird / water"], rows,
               {"shortcut": "background", "p": 0.95})


def build_celeba_task(out, celeba_root, n_train, n_val, rng):
    """CelebA, the standard group-robustness setup (Sagawa et al., 2020): y = Blond_Hair, a = Male.
    Official partitions; the full test set is kept, train and val are subsampled with group proportions
    preserved (val keeps at least 100 images per group) to fit the compute budget."""
    attr = pd.read_csv(os.path.join(celeba_root, "list_attr_celeba.csv"))
    part = pd.read_csv(os.path.join(celeba_root, "list_eval_partition.csv"))
    df = attr.merge(part, on="image_id")
    df["y"] = (df["Blond_Hair"] == 1).astype(int)
    df["a"] = (df["Male"] == 1).astype(int)
    img_dir = next(d for d in (os.path.join(celeba_root, "img_align_celeba", "img_align_celeba"),
                               os.path.join(celeba_root, "img_align_celeba")) if os.path.isdir(d))
    rows = []
    for split, code, n in (("train", 0, n_train), ("val", 1, n_val), ("test", 2, None)):
        d = df[df.partition == code]
        if n and n < len(d):
            take = []
            for (y, a), grp in d.groupby(["y", "a"]):
                k = max(round(n * len(grp) / len(d)), 100 if split == "val" else 1)
                take.append(grp.loc[rng.permutation(grp.index)[:min(k, len(grp))]])
            d = pd.concat(take)
        rows += [{"img_id": r.image_id, "image": os.path.join(img_dir, r.image_id), "y": int(r.y), "a": int(r.a),
                  "split": split} for r in d.itertuples()]
    questions = ["Does this person have blond hair?", "Is this person's hair blond?",
                 "Would you describe this person's hair as blond?"]
    write_task(out, "celeba", questions, YN, ["not blond / female", "not blond / male", "blond / female",
                                              "blond / male"], rows, {"shortcut": "gender", "attribute": "Male"})


def build_captions(coco, out, n, exclude, rng):
    ids = [i for i in coco.ids if coco.captions.get(i) and i not in exclude]
    ids = list(rng.permutation(ids)[:n])
    rows = [{"img_id": int(i), "image": coco.path(i), "captions": json.dumps(coco.captions[i])}
            for i in tqdm(ids, desc="captions")]
    d = os.path.join(out, "captions")
    os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "metadata.csv"), index=False)
    print(f"[captions] {len(rows)} images (excluding {len(exclude)} task val/test images)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_root", required=True)
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--out", default="./data/tasks")
    parser.add_argument("--data_dir", default="./data", help="where scripts/prepare_waterbirds.py wrote waterbirds/")
    parser.add_argument("--n_train", type=int, default=4800)
    parser.add_argument("--n_val", type=int, default=1200)
    parser.add_argument("--n_test", type=int, default=2400)
    parser.add_argument("--p", type=float, default=0.95, help="shortcut strength for cooccur")
    parser.add_argument("--p_list", type=lambda s: [float(x) for x in s.split(",")], default=[0.95],
                        help="shortcut strengths for the synthetic tasks (dose-response)")
    parser.add_argument("--n_captions", type=int, default=10000)
    parser.add_argument("--only", default="waterbirds,cooccur,watermark,typographic",
                        help="comma list of task families to build (captions are always built)")
    parser.add_argument("--celeba_root", default=None, help="folder with list_attr_celeba.csv (Kaggle celeba-dataset)")
    parser.add_argument("--celeba_train", type=int, default=20000)
    parser.add_argument("--celeba_val", type=int, default=5000)
    parser.add_argument("--download_missing", action="store_true", help="fetch COCO images by URL (dev only)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    coco = Coco(args.coco_root, args.split, args.download_missing)
    only = set(args.only.split(","))
    used = build_object_shortcut_tasks(coco, args.out, args, rng, kinds=[k for k in ("watermark", "typographic")
                                                                         if k in only])
    if "cooccur" in only:
        used |= build_cooccur_task(coco, args.out, args, rng)
    if "waterbirds" in only:
        build_waterbirds_task(args.out, args.data_dir)
    if "celeba" in only:
        assert args.celeba_root, "--celeba_root is required for the celeba task"
        build_celeba_task(args.out, args.celeba_root, args.celeba_train, args.celeba_val, rng)
    build_captions(coco, args.out, args.n_captions, used, rng)


if __name__ == "__main__":
    main()
