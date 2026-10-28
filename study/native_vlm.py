"""
Native-VLM baseline: the LLM with its own vision tower and connector (no study connector), zero-shot.
Prediction = the more likely of the two answer tokens after the question, as in the study's restricted
prediction, so worst-group accuracy is directly comparable. Needs the task images (not cached features).

    python study/native_vlm.py --llm Qwen/Qwen3.5-4B --task_dir data/tasks/waterbirds --out results/native
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.study.data import TaskSpec, group_metrics


def first_token(tok, word):
    return tok(word, add_special_tokens=False).input_ids[0]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--task_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    spec = TaskSpec(args.task_dir)
    test = spec.meta[spec.meta["split"] == "test"]
    if args.max_samples:
        test = test.iloc[np.linspace(0, len(test) - 1, args.max_samples).astype(int)]
    proc = AutoProcessor.from_pretrained(args.llm)
    model = AutoModelForImageTextToText.from_pretrained(
        args.llm, dtype=torch.float16, device_map={"": "cuda"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    tok = proc.tokenizer
    proc.tokenizer.padding_side = "left"
    answer_ids = [first_token(tok, a) for a in spec.answers]
    assert len(set(answer_ids)) == len(answer_ids), "answer words share their first token"
    labels, groups = test["y"].to_numpy(), (2 * test["y"] + test["a"]).to_numpy()
    results = {"llm": args.llm, "task": spec.name, "n": len(test), "per_template": {}}
    for t, question in enumerate(spec.questions):
        preds = []
        for start in tqdm(range(0, len(test), args.batch_size), desc=f"template {t}"):
            rows = test.iloc[start:start + args.batch_size]
            images = [Image.open(p).convert("RGB") for p in rows["image"]]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    for _ in images]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            batch = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            logits = model(**batch).logits[:, -1, :]
            preds.append(logits[:, answer_ids].argmax(-1).cpu().numpy())
        m = group_metrics(np.concatenate(preds), labels, groups, spec.group_names)
        results["per_template"][t] = m
        print(f"template {t}: avg {m['avg_acc']:.4f} wga {m['worst_group_acc']:.4f}", flush=True)
    keys = results["per_template"][0].keys()
    results["mean_over_templates"] = {k: float(np.mean([results["per_template"][t][k] for t in results["per_template"]]))
                                      for k in keys}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"native_{spec.name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
