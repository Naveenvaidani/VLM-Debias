"""
Open-ended visual question answering under changing priors: VQA-CP v2 [Agrawal et al., CVPR 2018].

VQA-CP re-splits VQA v2 so that, for every question type, the answer distribution differs between train and
test (e.g. "what color is the banana" is mostly "yellow" in train and mostly "green" in test). A model that
answers from the question alone (a language prior, i.e. a text-side shortcut) fails on test. This is the
natural-image, open-ended counterpart of the binary shortcut tasks, and it tests the gate mechanism directly:
if the learned gate passes little visual signal, the model should fall back on the question prior.

Task directory (built by study/build_vqacp.py):
    images/metadata.csv   image (path), image_id          -> images/features.npy  [N_img, P, D] fp16
    questions.csv         question_id, img_row, question, answer, answers (JSON list of 10 human answers),
                          answer_type, question_type, split in {train, val, test_id, test_ood}

Answers are generated freely (greedy decoding) and scored with the VQA accuracy, min(#humans who gave the
answer / 3, 1) averaged over the 10 leave-one-out subsets of annotators, after the standard normalisation
(lower case, punctuation, articles, number words), which is a simplified version of the official evaluation script.
"""
import json
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import FEATURES, PromptBuilder

INSTRUCTION = "Answer the question using a single word or phrase."
NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
           "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$:']")


def normalize(ans: str) -> str:
    """Lower case, strip punctuation (keeping decimal points), map number words to digits, drop articles."""
    a = ans.lower().replace("\n", " ").replace("\t", " ").strip()
    a = _PUNCT.sub("", a)
    a = re.sub(r"(?<!\d)\.(?!\d)", "", a)
    words = [NUMBERS.get(w, w) for w in a.split() if w not in ARTICLES]
    return " ".join(words)


def vqa_accuracy(pred: str, human_answers: List[str]) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-annotator-out subsets of min(matches / 3, 1)."""
    p = normalize(pred)
    gts = [normalize(a) for a in human_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return float(np.mean(accs)) if accs else 0.0


class VQASpec:
    def __init__(self, task_dir: str):
        self.dir = task_dir
        self.questions = pd.read_csv(os.path.join(task_dir, "questions.csv"))
        self.features_path = os.path.join(task_dir, "images", FEATURES)
        train = self.questions[self.questions["split"] == "train"]
        # language prior: the most frequent training answer for each question type
        self.prior = {qt: Counter(g["answer"].map(normalize)).most_common(1)[0][0]
                      for qt, g in train.groupby("question_type")}


class VQADataset(Dataset):
    """Yields (patch features, row index into spec.questions)."""
    def __init__(self, spec: VQASpec, split: str):
        self.spec = spec
        self.features = np.load(spec.features_path, mmap_mode="r")
        self.rows = spec.questions.index[spec.questions["split"] == split].to_numpy()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = int(self.spec.questions.at[r, "img_row"])
        return torch.from_numpy(np.array(self.features[img], dtype=np.float32)), int(r)


class VQABuilder(PromptBuilder):
    """PromptBuilder whose question varies per sample. The chat header (`before`) is shared, as required by the
    connectors; `after` = question + instruction + assistant header, tokenised per sample."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, [f"{INSTRUCTION} Q"], ["yes", "no"])
        self.stop_ids = {i for i in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                     tokenizer.convert_tokens_to_ids("<|endoftext|>")) if isinstance(i, int) and i >= 0}

    def question_ids(self, q: str) -> List[int]:
        text = self._format(f"{self.IMG}\n{q.strip()} {INSTRUCTION}")
        b, a = text.split(self.IMG)
        assert self._ids(b) == self.before_ids, "chat header must not depend on the question"
        return self._ids(a)

    def qa_batch(self, questions: List[str], answers: List[str]) -> Dict[str, torch.Tensor]:
        self.after_ids = [self.question_ids(q) for q in questions]
        eos = self.tok.eos_token_id
        targets = [self._ids(a.strip())[:8] + ([eos] if eos is not None else []) for a in answers]
        return self.batch(list(range(len(questions))), targets)


@torch.no_grad()
def generate(model, builder: VQABuilder, feats: torch.Tensor, questions: List[str], device, max_new: int = 6
             ) -> List[str]:
    """Greedy decoding without a KV cache (the connector re-reads the full sequence each step)."""
    prompts = [builder.question_ids(q) for q in questions]
    gen: List[List[int]] = [[] for _ in questions]
    done = [False] * len(questions)
    pad = builder.tok.pad_token_id if builder.tok.pad_token_id is not None else 0
    for _ in range(max_new):
        seqs = [p + g for p, g in zip(prompts, gen)]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
        ids, mask = ids.to(device), mask.to(device)
        hidden, _, _ = model(feats, ids, mask)
        last = mask.sum(1) - 1
        h = hidden[torch.arange(len(seqs), device=hidden.device), last]
        nxt = model.lm_head(h.to(model.lm_head.weight.dtype)).float().argmax(-1).tolist()
        for i, t in enumerate(nxt):
            if done[i]:
                continue
            if t in builder.stop_ids or "\n" in builder.tok.decode([t]):
                done[i] = True
            else:
                gen[i].append(t)
        if all(done):
            break
    return [builder.tok.decode(g, skip_special_tokens=True).strip() for g in gen]


def vqa_metrics(spec: VQASpec, rows: np.ndarray, preds: List[str]) -> Dict[str, float]:
    """Overall and per-answer-type VQA accuracy, worst answer type, and how often the model gives the
    training-prior answer for the question type (language-prior reliance), next to how often that is correct."""
    q = spec.questions.loc[rows]
    acc = np.array([vqa_accuracy(p, json.loads(a)) for p, a in zip(preds, q["answers"])])
    out = {"vqa_acc": float(acc.mean()), "n": int(len(acc))}
    for at, idx in q.groupby("answer_type").indices.items():
        out[f"acc[{at}]"] = float(acc[idx].mean())
    out["worst_answer_type_acc"] = min(v for k, v in out.items() if k.startswith("acc["))
    prior = np.array([spec.prior.get(qt, "") for qt in q["question_type"]])
    pn = np.array([normalize(p) for p in preds])
    out["prior_answer_rate"] = float((pn == prior).mean())
    out["prior_answer_correct_rate"] = float(np.mean([normalize(a) == pr for a, pr in zip(q["answer"], prior)]))
    yn = (q["answer_type"] == "yes/no").to_numpy()
    if yn.any():
        out["yes_rate[yes/no]"] = float((pn[yn] == "yes").mean())
    return out
