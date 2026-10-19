"""
Precompute frozen SigLIP patch features for every Waterbirds image.

The vision encoder is frozen during training, so running it once and caching
[N, 196, 768] fp16 patch features removes it from every training step.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipVisionModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.waterbirds import find_waterbirds_root, load_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--vision_model", default="google/siglip-base-patch16-224")
    parser.add_argument("--out", default="./data/siglip_patch_features.npy")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = find_waterbirds_root(args.data_dir)
    meta = load_metadata(root)
    n = len(meta["y"])

    processor = AutoImageProcessor.from_pretrained(args.vision_model)
    encoder = SiglipVisionModel.from_pretrained(args.vision_model, dtype=torch.float16).to(device).eval()
    num_patches = (encoder.config.image_size // encoder.config.patch_size) ** 2
    out = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float16, shape=(n, num_patches, encoder.config.hidden_size)
    )

    for start in tqdm(range(0, n, args.batch_size), desc="SigLIP features"):
        files = meta["img_filename"][start:start + args.batch_size]
        images = [Image.open(os.path.join(root, f)).convert("RGB") for f in files]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device, torch.float16)
        with torch.no_grad():
            feats = encoder(pixel_values=pixel_values).last_hidden_state
        out[start:start + len(files)] = feats.cpu().numpy()

    out.flush()
    print(f"Saved {n} x {num_patches} x {encoder.config.hidden_size} features to {args.out}")


if __name__ == "__main__":
    main()
