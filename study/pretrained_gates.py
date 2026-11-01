"""
Gate values of released Flamingo-style VLMs, read straight from their safetensors checkpoints on the
Hugging Face Hub with HTTP range requests (only the gate tensors are downloaded, a few KB per model).

A gated cross-attention block adds tanh(alpha) * cross_attention(text, image) to the text stream; tanh(alpha)
near 0 means the block passes almost no visual signal (the regime our learned gate falls into).

    python study/pretrained_gates.py --out results_study
"""
import argparse
import json
import os
import re
import struct

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url

REPOS = ["HuggingFaceM4/idefics-9b", "HuggingFaceM4/idefics-9b-instruct", "HuggingFaceM4/idefics-80b",
         "HuggingFaceM4/idefics-80b-instruct"]
GATE = re.compile(r"(alpha|gate)", re.I)
DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": "bf16"}


def read_header(url):
    n = struct.unpack("<Q", requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60).content)[0]
    raw = requests.get(url, headers={"Range": f"bytes=8-{7 + n}"}, timeout=60).content
    return json.loads(raw), 8 + n


def read_tensor(url, base, meta):
    start, end = meta["data_offsets"]
    raw = requests.get(url, headers={"Range": f"bytes={base + start}-{base + end - 1}"}, timeout=60).content
    if meta["dtype"] == "BF16":  # bf16 -> f32 by shifting into the high half
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    api, rows = HfApi(), []
    for repo in REPOS:
        files = [s.rfilename for s in api.model_info(repo).siblings if s.rfilename.endswith(".safetensors")]
        for fname in files:
            url = hf_hub_url(repo, fname)
            header, base = read_header(url)
            for key, meta in header.items():
                if key == "__metadata__" or not GATE.search(key.split(".")[-1]):
                    continue
                v = read_tensor(url, base, meta)
                layer = re.search(r"\.(\d+)\.", key)
                rows.append({"model": repo, "param": key, "layer": int(layer.group(1)) if layer else -1,
                             "kind": key.split(".")[-1], "n": int(v.size),
                             "alpha_mean": float(v.mean()), "tanh_abs_mean": float(np.abs(np.tanh(v)).mean())})
        print(repo, sum(r["model"] == repo for r in rows), "gate tensors", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pretrained_gates.json"), "w") as f:
        json.dump(rows, f, indent=1)
    lines = ["# Gate values of released Flamingo-style models (|tanh(alpha)|, mean over each tensor)\n",
             "| Model | Gate | Layers | Mean | Min | Max | Layers with < 0.05 |", "|---|---|---|---|---|---|---|"]
    for repo in REPOS:
        for kind in sorted({r["kind"] for r in rows if r["model"] == repo}):
            v = np.array([r["tanh_abs_mean"] for r in rows if r["model"] == repo and r["kind"] == kind])
            lines.append(f"| {repo} | {kind} | {len(v)} | {v.mean():.3f} | {v.min():.3f} | {v.max():.3f} | "
                         f"{int((v < 0.05).sum())} |")
    with open(os.path.join(args.out, "pretrained_gates.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
