import unittest
import torch
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.debiased_vlm import LocalFeatureBottleneck, DebiasedVLM
from src.models.siglip_scratch import SiglipVisionEncoderScratch
from src.models.qwen_scratch import Qwen2ForCausalLMScratch
from src.models.debiased_vlm_scratch import DebiasedVLMScratch
from src.training.loss import DebiasedVLMLoss

class TestVLMShield(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_patches = 196
        self.vision_dim = 768
        self.llm_dim = 1536
        self.seq_len = 16
        self.vocab_size = 151936

    def test_local_feature_bottleneck(self):
        bottleneck = LocalFeatureBottleneck(
            vision_dim=self.vision_dim,
            llm_dim=self.llm_dim,
            num_heads=8
        )
        vision_patches = torch.randn(self.batch_size, self.num_patches, self.vision_dim)
        text_queries = torch.randn(self.batch_size, self.seq_len, self.llm_dim)
        
        fused_features, attn_weights = bottleneck(vision_patches, text_queries)
        
        self.assertEqual(fused_features.shape, (self.batch_size, self.seq_len, self.llm_dim))
        self.assertTrue(attn_weights.shape[-1] == self.num_patches)

    def test_siglip_scratch(self):
        encoder = SiglipVisionEncoderScratch(num_layers=2)
        pixel_values = torch.randn(self.batch_size, 3, 224, 224)
        patches = encoder(pixel_values)
        self.assertEqual(patches.shape, (self.batch_size, 196, 768))

    def test_qwen_scratch(self):
        llm = Qwen2ForCausalLMScratch(vocab_size=1000, num_layers=2)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        outputs = llm(input_ids)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_len, 1000))

    def test_debiased_vlm_scratch_forward(self):
        model = DebiasedVLMScratch(vocab_size=1000, num_vision_layers=2, num_llm_layers=2)
        pixel_values = torch.randn(self.batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        
        outputs, attn_maps = model(pixel_values, input_ids)
        self.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_len, 1000))
        self.assertTrue(attn_maps.shape[-1] == 196)

    def test_debiased_vlm_loss(self):
        loss_fn = DebiasedVLMLoss(lambda_sparsity=0.1)
        logits = torch.randn(self.batch_size, self.seq_len, 100)
        labels = torch.randint(0, 100, (self.batch_size, self.seq_len))
        attn_maps = torch.softmax(torch.randn(self.batch_size, self.seq_len, 196), dim=-1)
        
        total_loss, loss_dict = loss_fn(logits, labels, attn_maps)
        
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertIn("total_loss", loss_dict)
        self.assertIn("loss_ntp", loss_dict)
        self.assertIn("loss_sparsity", loss_dict)
        self.assertGreater(loss_dict["total_loss"], 0.0)

if __name__ == "__main__":
    unittest.main()
