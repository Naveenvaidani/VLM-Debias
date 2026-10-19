import math
import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.debiased_vlm import LocalFeatureBottleneck, MLPPrefixProjector
from src.models.siglip_scratch import SiglipVisionEncoderScratch
from src.models.qwen_scratch import Qwen2ForCausalLMScratch
from src.models.debiased_vlm_scratch import DebiasedVLMScratch
from src.models.lora import LoRALinear, apply_lora_to_llm
from src.training.loss import DebiasedVLMLoss
from src.data.waterbirds import group_metrics


class TestVLMShield(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_patches = 196
        self.vision_dim = 768
        self.llm_dim = 1536
        self.seq_len = 16

    def test_local_feature_bottleneck(self):
        bottleneck = LocalFeatureBottleneck(vision_dim=self.vision_dim, llm_dim=self.llm_dim, num_heads=8)
        vision_patches = torch.randn(self.batch_size, self.num_patches, self.vision_dim)
        text_queries = torch.randn(self.batch_size, self.seq_len, self.llm_dim)

        fused_features, attn_weights = bottleneck(vision_patches, text_queries)

        self.assertEqual(fused_features.shape, (self.batch_size, self.seq_len, self.llm_dim))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.seq_len, self.num_patches))
        self.assertTrue(torch.allclose(attn_weights.sum(-1), torch.ones(self.batch_size, self.seq_len), atol=1e-5))

    def test_bottleneck_starts_as_identity(self):
        # Zero-initialised gate: the pretrained LLM sees unmodified text embeddings at step 0
        bottleneck = LocalFeatureBottleneck(vision_dim=64, llm_dim=32, num_heads=4)
        text = torch.randn(1, 5, 32)
        fused, _ = bottleneck(torch.randn(1, 196, 64), text)
        self.assertTrue(torch.allclose(fused, text))

    def test_bottleneck_output_depends_on_image(self):
        bottleneck = LocalFeatureBottleneck(vision_dim=64, llm_dim=32, num_heads=4)
        with torch.no_grad():
            bottleneck.gate.fill_(1.0)
        text = torch.randn(1, 5, 32)
        a, _ = bottleneck(torch.randn(1, 196, 64), text)
        b, _ = bottleneck(torch.randn(1, 196, 64), text)
        self.assertFalse(torch.allclose(a, b))

    def test_mlp_prefix_projector(self):
        proj = MLPPrefixProjector(vision_dim=self.vision_dim, llm_dim=64)
        out = proj(torch.randn(self.batch_size, self.num_patches, self.vision_dim))
        self.assertEqual(out.shape, (self.batch_size, 49, 64))

    def test_siglip_scratch(self):
        encoder = SiglipVisionEncoderScratch(num_layers=2)
        patches = encoder(torch.randn(self.batch_size, 3, 224, 224))
        self.assertEqual(patches.shape, (self.batch_size, 196, 768))

    def test_qwen_scratch(self):
        llm = Qwen2ForCausalLMScratch(vocab_size=1000, num_layers=2)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        self.assertEqual(llm(input_ids).logits.shape, (self.batch_size, self.seq_len, 1000))

    def test_debiased_vlm_scratch_forward(self):
        model = DebiasedVLMScratch(vocab_size=1000, num_vision_layers=2, num_llm_layers=2)
        pixel_values = torch.randn(self.batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))

        outputs, attn_maps = model(pixel_values, input_ids)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_len, 1000))
        self.assertEqual(attn_maps.shape[-1], 196)

    def test_lora_freezes_base_and_matches_device(self):
        llm = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8))
        llm.q_proj = nn.Linear(8, 8)
        count = apply_lora_to_llm(llm, rank=2, target_modules=("q_proj",))
        self.assertEqual(count, 1)
        trainable = {n for n, p in llm.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {"q_proj.lora_A", "q_proj.lora_B"})
        self.assertIsInstance(llm.q_proj, LoRALinear)
        # B is zero-initialised: adapter starts as a no-op
        x = torch.randn(3, 8)
        self.assertTrue(torch.allclose(llm.q_proj(x), llm.q_proj.original_linear(x)))

    def test_loss_ntp(self):
        loss_fn = DebiasedVLMLoss(lambda_entropy=0.1)
        logits = torch.randn(self.batch_size, self.seq_len, 100)
        labels = torch.randint(0, 100, (self.batch_size, self.seq_len))
        attn_maps = torch.softmax(torch.randn(self.batch_size, self.seq_len, 196), dim=-1)

        total_loss, loss_dict = loss_fn(logits, labels, attn_maps)
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        for key in ("total_loss", "loss_task", "loss_entropy"):
            self.assertIn(key, loss_dict)
        self.assertGreater(loss_dict["total_loss"], 0.0)

    def test_loss_entropy_bounded(self):
        loss_fn = DebiasedVLMLoss(lambda_entropy=1.0)
        logits, labels = torch.randn(2, 2), torch.tensor([0, 1])
        collapsed = torch.zeros(2, 4, 196)
        collapsed[..., 0] = 1.0
        uniform = torch.full((2, 4, 196), 1 / 196)
        _, d_collapsed = loss_fn(logits, labels, collapsed)
        _, d_uniform = loss_fn(logits, labels, uniform)
        self.assertTrue(math.isfinite(d_collapsed["loss_entropy"]))
        self.assertAlmostEqual(d_collapsed["loss_entropy"], 1.0, places=4)
        self.assertAlmostEqual(d_uniform["loss_entropy"], 0.0, places=4)

    def test_loss_shortcut_suppression(self):
        loss_fn = DebiasedVLMLoss(lambda_entropy=0.0, lambda_suppression=1.0)
        attn = torch.full((1, 3, 4), 0.25)
        mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        _, d = loss_fn(torch.randn(1, 2), torch.tensor([1]), attn, shortcut_patch_mask=mask)
        self.assertAlmostEqual(d["loss_shortcut_supp"], 0.25, places=5)

    def test_group_metrics(self):
        labels = np.array([0, 0, 1, 1])
        groups = np.array([0, 1, 2, 3])
        preds = np.array([0, 1, 1, 1])  # wrong only on landbird/water
        m = group_metrics(preds, labels, groups)
        self.assertAlmostEqual(m["avg_acc"], 0.75)
        self.assertAlmostEqual(m["worst_group_acc"], 0.0)
        self.assertAlmostEqual(m["acc_id"], 1.0)
        self.assertAlmostEqual(m["acc_ood"], 0.5)
        self.assertAlmostEqual(m["rsr"], 0.5)


if __name__ == "__main__":
    unittest.main()
