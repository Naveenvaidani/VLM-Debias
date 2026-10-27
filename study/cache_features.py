"""
Cache frozen vision-encoder patch features for every task directory (features.npy aligned with
metadata.csv rows). Optionally also caches an intermediate encoder layer (for the T2 leakage study).

    python study/cache_features.py --tasks_dir data/tasks --vision_model google/siglip-base-patch16-224
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel


def load_encoder(name, device):
    cls = SiglipVisionModel if "siglip" in name.lower() else AutoModel
    dtype = torch.float16 if device == "cuda" else torch.float32  # fp16 matmuls are slow or missing on CPU
    enc = cls.from_pretrained(name, dtype=dtype).to(device).eval()
    return AutoImageProcessor.from_pretrained(name), enc


def cache_dir(task_dir, processor, encoder, device, batch_size, layers):
    meta = pd.read_csv(os.path.join(task_dir, "metadata.csv"))
    size = encoder.config.image_size
    n_patches = (size // encoder.config.patch_size) ** 2
    outs = {}
    for layer in layers:
        name = "features.npy" if layer == -1 else f"features_layer{layer}.npy"
        path = os.path.join(task_dir, name)
        if os.path.exists(path):
            continue
        outs[layer] = (path, np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float16,
                                                       shape=(len(meta), n_patches, encoder.config.hidden_size)))
    if not outs:
        print(f"{task_dir}: cached")
        return
    images = meta["image"].tolist()
    for start in tqdm(range(0, len(images), batch_size), desc=os.path.basename(task_dir)):
        batch = [Image.open(p).convert("RGB") for p in images[start:start + batch_size]]
        px = processor(images=batch, return_tensors="pt", size={"height": size, "width": size}
                       ).pixel_values.to(device, encoder.dtype)
        with torch.no_grad():
            out = encoder(pixel_values=px, output_hidden_states=len(outs) > 1 or -1 not in outs)
        for layer, (_, arr) in outs.items():
            h = out.last_hidden_state if layer == -1 else out.hidden_states[layer]
            arr[start:start + len(batch)] = h[:, -n_patches:].cpu().numpy()  # drop CLS/register tokens if any
    for path, arr in outs.values():
        arr.flush()
        del arr
        os.replace(path + ".tmp", path)
    print(f"{task_dir}: {len(meta)} x {n_patches} patches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_dir", default="./data/tasks")
    parser.add_argument("--vision_model", default="google/siglip-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--layers", type=lambda s: [int(x) for x in s.split(",")], default=[-1],
                        help="-1 = last layer; e.g. -1,6 also caches hidden_states[6]")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, encoder = load_encoder(args.vision_model, device)
    for task_dir in sorted(glob.glob(os.path.join(args.tasks_dir, "*", "metadata.csv"))):
        cache_dir(os.path.dirname(task_dir), processor, encoder, device, args.batch_size, args.layers)


if __name__ == "__main__":
    main()
