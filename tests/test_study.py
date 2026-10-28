import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.study.connectors import build_connector
from src.study.data import PromptBuilder, group_metrics


class FakeTokenizer:
    """Whitespace tokenizer with a trivial chat template, enough to test prompt assembly."""
    chat_template = "fake"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return f"<user> {messages[0]['content']} <assistant>"

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 2) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()


class TestStudy(unittest.TestCase):
    def test_connectors_shapes_and_budget(self):
        patches, text = torch.randn(2, 196, 768), torch.randn(2, 9, 2560)
        out_mlp, _ = build_connector("mlp", 768, 2560)(patches)
        out_q, attn_q = build_connector("qformer", 768, 2560)(patches)
        out_b, attn_b = build_connector("bottleneck", 768, 2560)(patches, text)
        self.assertEqual(out_mlp.shape, (2, 196, 2560))
        self.assertEqual(out_q.shape, (2, 32, 2560))
        self.assertEqual(attn_q.shape, (2, 32, 196))
        self.assertEqual(out_b.shape, text.shape)
        self.assertEqual(attn_b.shape, (2, 9, 196))
        counts = [sum(p.numel() for p in build_connector(c, 768, 2560).parameters()) for c in
                  ("mlp", "qformer", "bottleneck")]
        self.assertLess(max(counts) / min(counts), 1.5, f"connector budgets drifted: {counts}")

    def test_bottleneck_starts_as_identity(self):
        conn = build_connector("bottleneck", 64, 32)
        text = torch.randn(1, 5, 32)
        fused, _ = conn(torch.randn(1, 196, 64), text)
        self.assertTrue(torch.allclose(fused, text))

    def test_fixed_gate_bottleneck_identity_and_open(self):
        conn = build_connector("bottleneck_fixed", 64, 32)
        text = torch.randn(1, 5, 32)
        fused, _ = conn(torch.randn(1, 196, 64), text)
        self.assertTrue(torch.allclose(fused, text))
        self.assertNotIn("gate", dict(conn.named_parameters()))
        fused.sum().backward()
        self.assertGreater(conn.out_proj.weight.grad.abs().sum().item(), 0)

    def test_prompt_builder_labels_only_targets(self):
        pb = PromptBuilder(FakeTokenizer(), ["is there a dog", "any dog at all"], ["No", "Yes"])
        b = pb.answer_batch([0, 1], [1, 0])
        labels, mask = b["labels"], b["after_mask"]
        self.assertEqual(int((labels != -100).sum()), 2)  # exactly one target per sample
        for i, y in enumerate([1, 0]):
            pos = int((labels[i] != -100).nonzero())
            self.assertEqual(int(labels[i, pos]), pb.answer_ids[y])
            self.assertEqual(int(b["after_ids"][i, pos + 1]), pb.answer_ids[y])  # next token is the answer
            self.assertEqual(int(mask[i].sum()), pos + 2)  # right padding after the answer
        cap = pb.caption_batch(["a dog on grass"])
        self.assertEqual(int((cap["labels"] != -100).sum()), 5)  # 4 words + eos

    def test_group_metrics(self):
        labels = np.array([0, 0, 1, 1])
        groups = np.array([0, 1, 2, 3])
        m = group_metrics(np.array([0, 1, 1, 1]), labels, groups, ["a", "b", "c", "d"])
        self.assertAlmostEqual(m["worst_group_acc"], 0.0)
        self.assertAlmostEqual(m["acc_id"], 1.0)
        self.assertAlmostEqual(m["acc_ood"], 0.5)
        self.assertAlmostEqual(m["balanced_acc"], 0.75)
        self.assertAlmostEqual(m["positive_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
