"""
Build a VQA-CP v2 subset for the connector study (see src/study/vqa.py).

Sampling is by image, so no image appears in more than one split:
  train / val / test_id   from the VQA-CP *train* split (same answer priors as training)
  test_ood                from the VQA-CP *test* split (answer priors per question type differ)
Images used for stage-1 caption alignment are excluded from every evaluation split. COCO 2014 images are
read from the COCO 2017 folders (same image ids; 2017 train + val = 2014 train + val).

    python study/build_vqacp.py --vqacp_dir /kaggle/input/my-vqacp --coco_root /kaggle/input/coco-2017-dataset/coco2017 \
        --exclude_captions data/tasks/captions/metadata.csv --out data/tasks/vqacp
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd


def load(vqacp_dir, split):
    q = json.load(open(os.path.join(vqacp_dir, f"vqacp_v2_{split}_questions.json")))
    a = json.load(open(os.path.join(vqacp_dir, f"vqacp_v2_{split}_annotations.json")))
    a = {x["question_id"]: x for x in a}
    rows = []
    for x in q:
        ann = a[x["question_id"]]
        rows.append({"question_id": x["question_id"], "image_id": x["image_id"], "question": x["question"],
                     "answer": ann["multiple_choice_answer"], "answers": json.dumps([y["answer"] for y in ann["answers"]]),
                     "answer_type": ann["answer_type"], "question_type": ann["question_type"]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vqacp_dir", required=True)
    p.add_argument("--coco_root", required=True, help="folder with train2017/ and val2017/")
    p.add_argument("--exclude_captions", default=None, help="captions metadata.csv; its images are kept out of eval")
    p.add_argument("--out", default="data/tasks/vqacp")
    p.add_argument("--n_train", type=int, default=10000, help="questions")
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test_id", type=int, default=2000)
    p.add_argument("--n_test_ood", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    def path(img_id):
        for d in ("train2017", "val2017"):
            f = os.path.join(args.coco_root, d, f"{img_id:012d}.jpg")
            if os.path.exists(f):
                return f
        return None

    excluded = set()
    if args.exclude_captions and os.path.exists(args.exclude_captions):
        for f in pd.read_csv(args.exclude_captions)["image"]:
            m = re.search(r"(\d{6,})", os.path.basename(str(f)))
            if m:
                excluded.add(int(m.group(1)))
    tr, te = load(args.vqacp_dir, "train"), load(args.vqacp_dir, "test")
    used = set()

    def take(df, n_q, name, avoid):
        imgs = df["image_id"].unique()
        rng.shuffle(imgs)
        chosen, count = [], 0
        per = df.groupby("image_id").size()
        for i in imgs:
            if i in used or i in avoid or path(int(i)) is None:
                continue
            chosen.append(i)
            used.add(i)
            count += per[i]
            if count >= n_q:
                break
        out = df[df["image_id"].isin(set(chosen))].copy()
        out["split"] = name
        return out

    parts = [take(tr, args.n_train, "train", set()),
             take(tr, args.n_val, "val", excluded),
             take(tr, args.n_test_id, "test_id", excluded),
             take(te, args.n_test_ood, "test_ood", excluded)]
    qs = pd.concat(parts, ignore_index=True)
    imgs = sorted(qs["image_id"].unique())
    row = {i: k for k, i in enumerate(imgs)}
    qs["img_row"] = qs["image_id"].map(row)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    pd.DataFrame({"image": [path(int(i)) for i in imgs], "image_id": imgs}).to_csv(
        os.path.join(args.out, "images", "metadata.csv"), index=False)
    qs.to_csv(os.path.join(args.out, "questions.csv"), index=False)
    summary = {s: {"questions": int((qs.split == s).sum()), "images": int(qs[qs.split == s].image_id.nunique()),
                   "answer_types": qs[qs.split == s].answer_type.value_counts().to_dict()} for s in qs.split.unique()}
    summary["excluded_caption_images"] = len(excluded)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
