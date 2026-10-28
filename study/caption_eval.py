"""
Cost of a fix: caption loss (next-token cross-entropy on COCO captions) of a trained model, so connector
retraining (connector-DFR / JTT) can be checked for damage to general image description.

Loads the stage-2 adapter (LoRA + connector); --connector_ckpt optionally swaps in a retrained connector.

    python study/caption_eval.py --llm Qwen/Qwen3.5-4B --connector mlp --adapter run/adapter.pt \
        --connector_ckpt run_dfr/connector.pt --out run_dfr/caption_eval.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import CaptionDataset, PromptBuilder
from src.study.device import autocast, get_device, torch_device
from src.study.vlm import StudyVLM

CAPTION_PROMPT = ["Describe this image briefly."]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--connector", required=True)
    parser.add_argument("--adapter", required=True, help="adapter.pt of a stage-2 run (LoRA + connector)")
    parser.add_argument("--connector_ckpt", default=None, help="retrained connector to swap in")
    parser.add_argument("--captions_dir", default="./data/tasks/captions")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mlp_grid", type=int, default=0,
                        help="MLP tokens = grid x grid (e.g. 6 -> 36, near Q-Former's 32)")
    parser.add_argument("--mlp_pool", type=int, default=1)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch_device(get_device(args.device))
    tok = AutoTokenizer.from_pretrained(args.llm)
    builder = PromptBuilder(tok, CAPTION_PROMPT, ["No", "Yes"])
    data = CaptionDataset(args.captions_dir, seed=0)
    idx = np.linspace(0, len(data) - 1, min(args.n, len(data))).astype(int)  # fixed, evenly spaced subset
    model = StudyVLM(args.llm, args.connector, builder.before_ids, vision_dim=data.features.shape[-1],
                     dtype=args.dtype, device=device, mlp_pool=args.mlp_pool, mlp_grid=args.mlp_grid,
                     quantize_4bit=not args.no_4bit)
    model.load_state_dict(torch.load(args.adapter, map_location="cpu"), strict=False)
    if args.connector_ckpt:
        model.connector.load_state_dict(torch.load(args.connector_ckpt, map_location="cpu"))
    model.eval()
    total, tokens = 0.0, 0
    for feats, caps in DataLoader(Subset(data, idx), batch_size=args.batch_size):
        b = {k: v.to(device) for k, v in builder.caption_batch(list(caps)).items()}
        with autocast(device, model.compute_dtype):
            hidden, _, _ = model(feats.to(device), b["after_ids"], b["after_mask"])
            logits, target = model.logits_at(hidden, b["labels"])
        total += float(F.cross_entropy(logits, target, ignore_index=-100, reduction="sum"))
        tokens += int((target != -100).sum())
    out = {"caption_loss": total / tokens, "tokens": tokens, "n_images": len(idx),
           "adapter": args.adapter, "connector_ckpt": args.connector_ckpt}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"caption loss {out['caption_loss']:.4f} over {tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
