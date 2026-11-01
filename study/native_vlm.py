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
    parser.add_argument("--templates", default="", help="comma list of question phrasings to run (default all)")
    parser.add_argument("--calibrate_all", action="store_true", help="calibrate every phrasing (default: first only)")
    args = parser.parse_args()
    spec = TaskSpec(args.task_dir)
    test = spec.meta[spec.meta["split"] == "test"]
    val = spec.meta[spec.meta["split"] == "val"]
    if args.max_samples:
        test = test.iloc[np.linspace(0, len(test) - 1, args.max_samples).astype(int)]
        val = val.iloc[np.linspace(0, len(val) - 1, args.max_samples).astype(int)]
    proc = AutoProcessor.from_pretrained(args.llm)
    model = AutoModelForImageTextToText.from_pretrained(
        args.llm, dtype=torch.float16, device_map={"": "cuda"},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16)).eval()
    tok = proc.tokenizer
    proc.tokenizer.padding_side = "left"
    answer_ids = [first_token(tok, a) for a in spec.answers]
    assert len(set(answer_ids)) == len(answer_ids), "answer words share their first token"

    def margins(rows_df, question):
        """log p(answer 1) - log p(answer 0) for the next token, per image."""
        out = []
        for start in tqdm(range(0, len(rows_df), args.batch_size), desc=question[:30]):
            rows = rows_df.iloc[start:start + args.batch_size]
            images = [Image.open(p).convert("RGB") for p in rows["image"]]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                    for _ in images]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            batch = proc(text=texts, images=images, return_tensors="pt", padding=True).to("cuda")
            logp = model(**batch).logits[:, -1, :].float().log_softmax(-1)
            out.append((logp[:, answer_ids[1]] - logp[:, answer_ids[0]]).cpu().numpy())
        return np.concatenate(out)

    def groups_of(df):
        return (2 * df["y"] + df["a"]).to_numpy()

    labels, groups = test["y"].to_numpy(), groups_of(test)
    results = {"llm": args.llm, "task": spec.name, "n": len(test), "per_template": {}, "calibrated": {}}
    arrays = {}
    wanted = {int(x) for x in args.templates.split(",") if x} or set(range(len(spec.questions)))
    for t, question in enumerate(spec.questions):
        if t not in wanted:
            continue
        m_test = margins(test, question)
        arrays[f"test_t{t}"] = m_test
        m = group_metrics((m_test > 0).astype(int), labels, groups, spec.group_names)
        results["per_template"][t] = m
        print(f"template {t}: avg {m['avg_acc']:.4f} wga {m['worst_group_acc']:.4f}", flush=True)
        if t == 0 or args.calibrate_all:  # one threshold on the margin, chosen on validation (balanced accuracy)
            m_val = margins(val, question)
            arrays[f"val_t{t}"] = m_val
            yv, gv = val["y"].to_numpy(), groups_of(val)

            def bal(th):
                return group_metrics((m_val > th).astype(int), yv, gv, spec.group_names)["balanced_acc"]

            th = float(max(np.quantile(m_val, np.linspace(0.01, 0.99, 197)), key=bal))
            mc = group_metrics((m_test > th).astype(int), labels, groups, spec.group_names)
            results["calibrated"][t] = {"threshold": th, **mc}
            print(f"template {t} calibrated (threshold {th:.3f}): avg {mc['avg_acc']:.4f} "
                  f"wga {mc['worst_group_acc']:.4f}", flush=True)
    keys = next(iter(results["per_template"].values())).keys()
    results["mean_over_templates"] = {k: float(np.mean([results["per_template"][t][k] for t in results["per_template"]]))
                                      for k in keys}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"native_{spec.name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out, f"native_{spec.name}_margins.npz"), **arrays)


if __name__ == "__main__":
    main()
