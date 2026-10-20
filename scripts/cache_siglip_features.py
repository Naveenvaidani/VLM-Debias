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
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

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
    if "siglip" in args.vision_model.lower():
        encoder = SiglipVisionModel.from_pretrained(args.vision_model, dtype=torch.float16)
    else:
        # e.g. facebook/dinov3-vitb16-pretrain-lvd1689m (gated: accept the license on the model page first)
        encoder = AutoModel.from_pretrained(args.vision_model, dtype=torch.float16)
    encoder = encoder.to(device).eval()
    image_size = encoder.config.image_size
    num_patches = (image_size // encoder.config.patch_size) ** 2
    out = np.lib.format.open_memmap(
        args.out, mode="w+", dtype=np.float16, shape=(n, num_patches, encoder.config.hidden_size)
    )

    for start in tqdm(range(0, n, args.batch_size), desc="SigLIP features"):
        files = meta["img_filename"][start:start + args.batch_size]
        images = [Image.open(os.path.join(root, f)).convert("RGB") for f in files]
        pixel_values = processor(images=images, return_tensors="pt", size={"height": image_size, "width": image_size}
                                 ).pixel_values.to(device, torch.float16)
        with torch.no_grad():
            # keep only the spatial patch tokens (drops CLS / register tokens if the encoder has them)
            feats = encoder(pixel_values=pixel_values).last_hidden_state[:, -num_patches:]
        out[start:start + len(files)] = feats.cpu().numpy()

    out.flush()
    print(f"Saved {n} x {num_patches} x {encoder.config.hidden_size} features to {args.out}")


if __name__ == "__main__":
    main()
