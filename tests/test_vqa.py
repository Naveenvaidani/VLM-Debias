import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.study.vqa import VQABuilder, VQASpec, generate, normalize, vqa_accuracy, vqa_metrics
from tests.test_study import FakeTokenizer


class FakeTok(FakeTokenizer):
    def convert_tokens_to_ids(self, t):
        return -1

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "") for i in ids)


class TestVQA(unittest.TestCase):
    def test_normalize_and_accuracy(self):
        self.assertEqual(normalize("Two."), "2")
        self.assertEqual(normalize("The Red Car!"), "red car")
        self.assertEqual(normalize("3.5"), "3.5")
        self.assertAlmostEqual(vqa_accuracy("two", ["2"] * 3 + ["3"] * 7), 0.9)  # 3 of 10: 7 subsets x 1 + 3 x 2/3
        self.assertAlmostEqual(vqa_accuracy("two", ["2"] * 4 + ["3"] * 6), 1.0)
        self.assertAlmostEqual(vqa_accuracy("blue", ["2"] * 10), 0.0)

    def test_qa_batch_labels_only_answer_tokens(self):
        tok = FakeTok()
        b = VQABuilder(tok)
        out = b.qa_batch(["what color is it", "is it big"], ["red", "yes"])
        n_labels = (out["labels"] != -100).sum(1).tolist()
        self.assertEqual(n_labels, [2, 2])  # answer word + eos
        q = b.question_ids("what color is it")
        self.assertEqual(out["labels"][0, len(q) - 1].item(), tok.vocab["red"])

    def test_generate_stops_at_eos(self):
        tok = FakeTok()
        b = VQABuilder(tok)
        red = tok("red").input_ids[0]
        V = 64

        class Stub(torch.nn.Module):
            compute_dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Identity()
                self.lm_head.weight = torch.zeros(1)

            def forward(self, feats, ids, mask):
                nxt = torch.where(ids == red, torch.full_like(ids, tok.eos_token_id), torch.full_like(ids, red))
                return torch.nn.functional.one_hot(nxt, V).float(), None, 0

        out = generate(Stub(), b, torch.zeros(2, 4, 8), ["what color is it", "and this one"], "cpu")
        self.assertEqual(out, ["red", "red"])

    def test_metrics_and_language_prior(self):
        with tempfile.TemporaryDirectory() as d:
            q = pd.DataFrame({"question_id": range(6), "img_row": range(6),
                              "question": ["what color"] * 6,
                              "answer": ["yellow", "yellow", "green", "green", "green", "yes"],
                              "answers": [json.dumps([a] * 10) for a in ["yellow", "yellow", "green", "green", "green", "yes"]],
                              "answer_type": ["other"] * 5 + ["yes/no"], "question_type": ["what color"] * 5 + ["is"],
                              "split": ["train", "train", "test_ood", "test_ood", "test_ood", "test_ood"]})
            q.to_csv(os.path.join(d, "questions.csv"), index=False)
            spec = VQASpec(d)
            self.assertEqual(spec.prior["what color"], "yellow")
            m = vqa_metrics(spec, np.array([2, 3, 4, 5]), ["yellow", "yellow", "green", "yes"])
            self.assertAlmostEqual(m["acc[other]"], 1 / 3)
            self.assertAlmostEqual(m["prior_answer_rate"], 0.5)  # yellow twice out of 4 (no prior for "is")
            self.assertEqual(m["worst_answer_type_acc"], m["acc[other]"])

    def test_build_vqacp_splits_are_image_disjoint_and_exclude_alignment_images(self):
        with tempfile.TemporaryDirectory() as d:
            coco = os.path.join(d, "coco", "train2017")
            os.makedirs(coco)
            for i in range(1, 41):
                open(os.path.join(coco, f"{i:012d}.jpg"), "w").close()
            def write(split, ids):
                qs, an = [], []
                for k, i in enumerate(ids):
                    for j in range(2):
                        qid = i * 10 + j
                        qs.append({"question_id": qid, "image_id": i, "question": "q?", "coco_split": "train2014"})
                        an.append({"question_id": qid, "image_id": i, "multiple_choice_answer": "yes", "answer_type": "yes/no",
                                   "question_type": "is", "answers": [{"answer": "yes"}] * 10, "coco_split": "train2014"})
                json.dump(qs, open(os.path.join(d, f"vqacp_v2_{split}_questions.json"), "w"))
                json.dump(an, open(os.path.join(d, f"vqacp_v2_{split}_annotations.json"), "w"))
            write("train", range(1, 31))
            write("test", range(20, 41))  # overlaps train images, as in VQA-CP
            pd.DataFrame({"image": [os.path.join(coco, f"{i:012d}.jpg") for i in (35, 36)]}).to_csv(os.path.join(d, "cap.csv"), index=False)
            out = os.path.join(d, "out")
            subprocess.run([sys.executable, "study/build_vqacp.py", "--vqacp_dir", d, "--coco_root", os.path.join(d, "coco"),
                            "--exclude_captions", os.path.join(d, "cap.csv"), "--out", out, "--n_train", "20",
                            "--n_val", "4", "--n_test_id", "4", "--n_test_ood", "8"], check=True, capture_output=True,
                           cwd=os.path.join(os.path.dirname(__file__), ".."))
            qs = pd.read_csv(os.path.join(out, "questions.csv"))
            per = qs.groupby("image_id")["split"].nunique()
            self.assertTrue((per == 1).all())
            self.assertFalse(qs[qs.split != "train"].image_id.isin([35, 36]).any())
            self.assertEqual(set(qs.split), {"train", "val", "test_id", "test_ood"})
            img = pd.read_csv(os.path.join(out, "images", "metadata.csv"))
            self.assertEqual(len(img), qs.image_id.nunique())
            self.assertTrue((img.iloc[qs.img_row].image_id.values == qs.image_id.values).all())


if __name__ == "__main__":
    unittest.main()
