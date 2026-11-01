"""
Shortcut tasks for the connector study.

A task directory holds:
    task.json      name, questions (phrasing templates), answers (index = label), group names
    metadata.csv   image, y (label), a (shortcut attribute), split (train/val/test)
    features.npy   frozen vision patch features aligned with metadata rows, [N, P, D] fp16

group = 2 * y + a. Training data has a correlated with y; val/test are group-balanced, so
"majority" groups (a == y) are in-distribution and "minority" groups (a != y) are OOD.
"""
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# which cached feature file to read (STUDY_FEATURES=features_siglip-large.npy for another encoder)
FEATURES = os.environ.get("STUDY_FEATURES", "features.npy")

SPLITS = ("train", "val", "test")
TRAIN_TEMPLATES = (0, 1)  # the last template is held out from training, to test phrasing generalisation


class TaskSpec:
    def __init__(self, task_dir: str):
        with open(os.path.join(task_dir, "task.json")) as f:
            spec = json.load(f)
        self.dir = task_dir
        self.name: str = spec["name"]
        self.questions: List[str] = spec["questions"]
        self.answers: List[str] = spec["answers"]
        self.group_names: List[str] = spec["group_names"]
        self.meta = pd.read_csv(os.path.join(task_dir, "metadata.csv"))
        self.features_path = os.path.join(task_dir, FEATURES)


class TaskDataset(Dataset):
    """Yields (patch features, label, group, template id). Training samples draw a random
    training template; evaluation uses a fixed template."""
    def __init__(self, spec: TaskSpec, split: str, template: Optional[int] = None, seed: int = 0):
        self.spec = spec
        self.features = np.load(spec.features_path, mmap_mode="r")
        rows = spec.meta[spec.meta["split"] == split]
        self.indices = rows.index.to_numpy()
        self.y = rows["y"].to_numpy().astype(np.int64)
        self.group = (2 * rows["y"] + rows["a"]).to_numpy().astype(np.int64)
        self.template = template
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        feats = torch.from_numpy(np.array(self.features[self.indices[i]], dtype=np.float32))
        t = self.template if self.template is not None else int(self.rng.choice(TRAIN_TEMPLATES))
        return feats, int(self.y[i]), int(self.group[i]), t


class PromptBuilder:
    """
    Builds chat-formatted prompts with an image slot: [before][IMAGE][after][answer].
    `before` is identical for every sample (chat header), so prefix connectors insert image
    tokens at one fixed position; `after` (question + assistant header) varies and is right-padded.
    """
    IMG = "<<IMG>>"

    def __init__(self, tokenizer, questions: List[str], answers: List[str]):
        self.tok = tokenizer
        before, after = [], []
        for q in questions:
            text = self._format(f"{self.IMG}\n{q}")
            b, a = text.split(self.IMG)
            before.append(b)
            after.append(a)
        assert len(set(before)) == 1, "chat header must not depend on the question"
        self.before_ids = self._ids(before[0])
        self.after_ids = [self._ids(a) for a in after]
        # answer = first token of the answer word; must differ between classes
        self.answer_ids = [self._ids(ans)[0] for ans in answers]
        assert len(set(self.answer_ids)) == len(self.answer_ids), f"answers share first token: {answers}"

    def _format(self, user_text: str) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        return f"Question: {user_text}\nAnswer:"

    def _ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    def batch(self, templates: List[int], targets: List[List[int]]) -> Dict[str, torch.Tensor]:
        """
        Right-padded `after` ids (question + assistant header + target tokens) with an attention
        mask and next-token labels: labels[i, j] is the token position j must predict, -100 elsewhere.
        Only target tokens get labels, so the prompt itself is never trained on.
        """
        seqs, labels = [], []
        for t, tgt in zip(templates, targets):
            prompt = self.after_ids[t]
            seq = prompt + tgt
            lab = [-100] * len(seq)
            for k, token in enumerate(tgt):
                lab[len(prompt) - 1 + k] = token  # position before each target token predicts it
            seqs.append(seq)
            labels.append(lab)
        length = max(len(s) for s in seqs)
        if getattr(self, "pad_to", None):  # fixed shapes for XLA (no recompilation per batch)
            length = max(length, self.pad_to)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((len(seqs), length), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        lab_t = torch.full((len(seqs), length), -100, dtype=torch.long)
        for i, (s, lab) in enumerate(zip(seqs, labels)):
            ids[i, :len(s)] = torch.tensor(s)
            mask[i, :len(s)] = 1
            lab_t[i, :len(lab)] = torch.tensor(lab)
        return {"after_ids": ids, "after_mask": mask, "labels": lab_t}

    def answer_batch(self, templates: List[int], y: List[int]) -> Dict[str, torch.Tensor]:
        """Classification-as-generation: the target is the first token of the answer word."""
        return self.batch(templates, [[self.answer_ids[int(label)]] for label in y])

    def caption_batch(self, captions: List[str]) -> Dict[str, torch.Tensor]:
        eos = self.tok.eos_token_id
        targets = [self._ids(c.strip())[:48] + ([eos] if eos is not None else []) for c in captions]
        return self.batch([0] * len(captions), targets)


class CaptionDataset(Dataset):
    """Image-caption pairs for connector alignment (stage 1). One random caption per visit."""
    def __init__(self, caption_dir: str, seed: int = 0):
        self.meta = pd.read_csv(os.path.join(caption_dir, "metadata.csv"))
        self.features = np.load(os.path.join(caption_dir, FEATURES), mmap_mode="r")
        self.captions = [json.loads(c) for c in self.meta["captions"]]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        feats = torch.from_numpy(np.array(self.features[i], dtype=np.float32))
        return feats, str(self.rng.choice(self.captions[i]))


def group_metrics(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray, group_names: List[str]) -> Dict:
    """Average / worst-group / per-group / ID-vs-OOD accuracy, balanced accuracy and positive rate."""
    correct = preds == labels
    out = {"avg_acc": float(correct.mean())}
    accs = []
    for g, name in enumerate(group_names):
        m = groups == g
        acc = float(correct[m].mean()) if m.any() else float("nan")
        out[f"acc[{name}]"] = acc
        accs.append(acc)
    out["worst_group_acc"] = float(np.nanmin(accs))
    majority = (groups == 0) | (groups == 3)  # a == y
    out["acc_id"] = float(correct[majority].mean())
    out["acc_ood"] = float(correct[~majority].mean())
    out["delta_ood"] = out["acc_id"] - out["acc_ood"]
    out["rsr"] = out["delta_ood"] / max(out["acc_id"], 1e-8)
    out["balanced_acc"] = float(np.mean([correct[labels == c].mean() for c in np.unique(labels)]))
    out["positive_rate"] = float((preds == 1).mean())
    return out
