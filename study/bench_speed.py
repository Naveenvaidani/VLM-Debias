"""
Time one training step (LoRA + connector, batch of task prompts) for a connector configuration on
random features, so speed options can be compared before spending GPU hours. Prints one JSON line.

    python study/bench_speed.py --llm Qwen/Qwen3.5-4B --connector mlp --mlp_pool 2 --no_4bit
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import PromptBuilder
from src.study.vlm import StudyVLM

QUESTIONS = ["Is the bird in this image a waterbird?", "Does this photo show a waterbird?",
             "Would you classify the bird here as a waterbird?"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--connector", choices=["mlp", "qformer", "bottleneck"], default="mlp")
    parser.add_argument("--mlp_pool", type=int, default=1)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.llm)
    builder = PromptBuilder(tok, QUESTIONS, ["No", "Yes"])
    row = {k: v for k, v in vars(args).items() if k not in ("llm", "steps")}
    try:
        model = StudyVLM(args.llm, args.connector, builder.before_ids, vision_dim=768, dtype="fp16",
                         gradient_checkpointing=args.gradient_checkpointing, device=device,
                         mlp_pool=args.mlp_pool, quantize_4bit=not args.no_4bit)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for step in range(args.steps + 2):  # first two steps are warm-up (kernel compilation)
            feats = torch.randn(args.batch_size, 196, 768, device=device)
            b = {k: v.to(device) for k, v in builder.answer_batch(
                [step % 2] * args.batch_size, [step % 2] * args.batch_size).items()}
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                hidden, _, _ = model(feats, b["after_ids"], b["after_mask"])
                logits, target = model.logits_at(hidden, b["labels"])
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= 2:
                times.append(time.time() - t0)
        row.update(sec_per_step=round(sum(times) / len(times), 3),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    except torch.OutOfMemoryError:
        row.update(sec_per_step=None, error="OOM")
    print("BENCH " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
