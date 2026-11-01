"""
Download Waterbirds from the Hugging Face mirror (grodino/waterbirds) and write it in the
original layout: <out>/waterbirds/images/*.jpg + metadata.csv (img_filename, y, place, split).
"""
import argparse
import io
import os

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    args = parser.parse_args()

    raw = snapshot_download("grodino/waterbirds", repo_type="dataset",
                            local_dir=os.path.join(args.data_dir, "waterbirds_hf"), allow_patterns=["data/*.parquet"])
    root = os.path.join(args.data_dir, "waterbirds")
    os.makedirs(os.path.join(root, "images"), exist_ok=True)

    rows = []
    for split, split_id in SPLIT_IDS.items():
        df = pd.read_parquet(os.path.join(raw, "data", f"{split}-00000-of-00001.parquet"))
        for i, row in tqdm(df.iterrows(), total=len(df), desc=split):
            fname = f"images/{split}_{i:05d}.jpg"
            Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB").save(os.path.join(root, fname), quality=95)
            rows.append({"img_filename": fname, "y": int(row["label"]), "place": int(row["place"]), "split": split_id})

    meta = pd.DataFrame(rows)
    meta.to_csv(os.path.join(root, "metadata.csv"), index=False)
    print(meta.groupby(["split", "y", "place"]).size())


if __name__ == "__main__":
    main()
