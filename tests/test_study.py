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

    def test_open_gate_bottleneck_identity_learnable_and_open(self):
        conn = build_connector("bottleneck_open", 64, 32)
        text = torch.randn(1, 5, 32)
        fused, _ = conn(torch.randn(1, 196, 64), text)
        self.assertTrue(torch.allclose(fused, text))
        self.assertIn("gate", dict(conn.named_parameters()))
        self.assertGreater(torch.tanh(conn.gate).item(), 0.99)

    def test_fixed_gate_bottleneck_identity_and_open(self):
        conn = build_connector("bottleneck_fixed", 64, 32)
        text = torch.randn(1, 5, 32)
        fused, _ = conn(torch.randn(1, 196, 64), text)
        self.assertTrue(torch.allclose(fused, text))
        self.assertNotIn("gate", dict(conn.named_parameters()))
        fused.sum().backward()
        self.assertGreater(conn.out_proj.weight.grad.abs().sum().item(), 0)

    def test_logits_at_fixed_shape_path_matches(self):
        import src.study.vlm as vlm
        from types import SimpleNamespace
        torch.manual_seed(0)
        fake = SimpleNamespace(lm_head=torch.nn.Linear(8, 11, bias=False))
        hidden = torch.randn(3, 5, 8)
        labels = torch.full((3, 5), -100)
        labels[0, 2], labels[1, 0], labels[2, 4] = 4, 7, 1
        ref_logits, ref_target = vlm.StudyVLM.logits_at(fake, hidden, labels)
        orig = vlm.is_xla
        vlm.is_xla = lambda d: True
        try:
            g_logits, g_target = vlm.StudyVLM.logits_at(fake, hidden, labels, one_per_row=True)
            f_logits, f_target = vlm.StudyVLM.logits_at(fake, hidden, labels)
        finally:
            vlm.is_xla = orig
        self.assertTrue(torch.allclose(ref_logits, g_logits) and torch.equal(ref_target, g_target))
        ce = torch.nn.functional.cross_entropy
        self.assertTrue(torch.allclose(ce(ref_logits, ref_target), ce(f_logits, f_target, ignore_index=-100)))

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

    def test_group_dro_upweights_worst_group(self):
        from src.study import engine

        class Stub(torch.nn.Module):
            compute_dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.dro_q, self.dro_eta = torch.ones(4) / 4, 1.0

            def forward(self, feats, after_ids, after_mask):
                return feats, None, 0

            def logits_at(self, hidden, labels, one_per_row=True):
                return hidden, labels.max(1).values

        class Builder:
            def answer_batch(self, templates, ys):
                return {"after_ids": torch.zeros(len(ys), 1, dtype=torch.long),
                        "after_mask": torch.ones(len(ys), 1, dtype=torch.long),
                        "labels": torch.tensor(ys).unsqueeze(1)}

        model = Stub()
        feats = torch.tensor([[5.0, 0], [5.0, 0], [0, 5.0], [5.0, 0]])  # last sample (group 3) is wrong
        y, g = torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 2, 3])
        loss = engine.step_loss(model, Builder(), feats, y, torch.zeros(4), "cpu", groups=g)
        self.assertEqual(int(model.dro_q.argmax()), 3)
        self.assertAlmostEqual(float(model.dro_q.sum()), 1.0, places=5)
        self.assertGreater(float(loss), float(torch.nn.functional.cross_entropy(feats, y)))

    def test_deep_gated_connector_identity_params_and_gates(self):
        torch.manual_seed(0)
        h, patches = torch.randn(2, 5, 2560), torch.randn(2, 196, 768)
        for name, start in (("xattn_deep", 0.0), ("xattn_deep_fixed", 1.0)):
            conn = build_connector(name, 768, 2560)
            self.assertEqual(conn.kind, "deep")
            self.assertEqual(len(conn.blocks), 4)
            self.assertEqual(conn.gate_values(), [start] * 4)
            for blk in conn.blocks:  # every block starts as the identity
                self.assertTrue(torch.allclose(blk(patches, h)[0], h))
            n = sum(p.numel() for p in conn.parameters())
            single = sum(p.numel() for p in build_connector("bottleneck", 768, 2560).parameters())
            self.assertLess(abs(n - single) / single, 0.1)  # parameter-matched to the input-level block
        self.assertEqual(len(build_connector("xattn_deep", 768, 2560).learned_gates()), 4)
        self.assertEqual(build_connector("xattn_deep_fixed", 768, 2560).learned_gates(), [])

    def test_clip_and_log_reports_raw_gate_grad_and_can_exclude_gate(self):
        from src.study.engine import clip_and_log
        conn = build_connector("bottleneck", 8, 16)
        model = type("M", (), {"connector": conn})()
        params = list(conn.parameters())
        for p in params:
            p.grad = torch.ones_like(p)
        conn.gate.grad = torch.full_like(conn.gate, 50.0)
        raw, norm = clip_and_log(model, params, 1.0)
        self.assertAlmostEqual(raw, 50.0)  # logged before clipping
        self.assertGreater(norm, 50.0)
        self.assertLess(float(conn.gate.grad), 1.0)  # clipped with everything else
        for p in params:
            p.grad = torch.ones_like(p)
        conn.gate.grad = torch.full_like(conn.gate, 50.0)
        raw, norm = clip_and_log(model, params, 1.0, exclude_gate=True)
        self.assertAlmostEqual(float(conn.gate.grad), 50.0)  # gate untouched
        others = torch.cat([p.grad.flatten() for p in params if p is not conn.gate])
        self.assertAlmostEqual(float(others.norm()), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
