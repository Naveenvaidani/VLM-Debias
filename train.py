import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.models.debiased_vlm import DebiasedVLM
from src.models.debiased_vlm_scratch import DebiasedVLMScratch
from src.models.lora import apply_lora_to_llm
from src.training.loss import DualGroundingLoss

class ShortcutTrapDataset(Dataset):
    """
    Synthetic / MS-COCO style dataset with 85% spurious trigger correlation.
    Patch 0 contains a watermark artifact in 85% of samples.
    """
    def __init__(self, num_samples: int = 100, seq_len: int = 16, vocab_size: int = 1000, spurious_ratio: float = 0.85):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.spurious_ratio = spurious_ratio
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        pixel_values = torch.randn(3, 224, 224)
        shortcut_mask = torch.zeros(196)
        
        # Inject watermark trigger into patch 0 (top-left) with 85% probability
        if torch.rand(1).item() < self.spurious_ratio:
            pixel_values[:, :16, :16] = 5.0
            shortcut_mask[0] = 1.0 # Spurious patch flag
            
        # Target object features in center patches
        pixel_values[:, 100:116, 100:116] = -3.0
        
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'labels': labels,
            'shortcut_mask': shortcut_mask
        }

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss_accum = 0.0
    ntp_loss_accum = 0.0
    entropy_loss_accum = 0.0
    supp_loss_accum = 0.0
    
    for batch in tqdm(dataloader, desc="Training Epoch"):
        optimizer.zero_grad()
        
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        shortcut_mask = batch['shortcut_mask'].to(device)
        
        outputs, attention_maps = model(pixel_values, input_ids)
        loss, loss_dict = loss_fn(outputs.logits, labels, attention_maps, shortcut_patch_mask=shortcut_mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss_accum += loss_dict['total_loss']
        ntp_loss_accum += loss_dict['loss_ntp']
        entropy_loss_accum += loss_dict['loss_entropy']
        supp_loss_accum += loss_dict['loss_shortcut_supp']
        
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss_accum / num_batches,
        'loss_ntp': ntp_loss_accum / num_batches,
        'loss_entropy': entropy_loss_accum / num_batches,
        'loss_supp': supp_loss_accum / num_batches
    }

def main():
    parser = argparse.ArgumentParser(description="Train VLM-SHIELD with Dual Grounding Loss & LoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_entropy", type=float, default=0.1, help="Entropy penalty weight")
    parser.add_argument("--lambda_suppression", type=float, default=0.5, help="Shortcut suppression penalty weight")
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA (r=16) to Qwen language model")
    parser.add_argument("--use_scratch", action="store_true", help="Train pure PyTorch model built completely from scratch")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== VLM-SHIELD Dual Grounding Training ===")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Use Scratch: {args.use_scratch} | Apply LoRA: {args.use_lora}")

    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.use_scratch:
        vocab_size = 1000
        model = DebiasedVLMScratch(vocab_size=vocab_size, num_vision_layers=4, num_llm_layers=4).to(device)
    else:
        vocab_size = 151936
        model = DebiasedVLM(load_pretrained=True).to(device)
        
    if args.use_lora:
        adapted_layers = apply_lora_to_llm(model.llm, rank=16, lora_alpha=32.0)
        print(f"Applied LoRA adapters (r=16) to {adapted_layers} linear projection layers.")
        
    loss_fn = DualGroundingLoss(
        lambda_entropy=args.lambda_entropy, 
        lambda_suppression=args.lambda_suppression
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    dataset = ShortcutTrapDataset(num_samples=40, vocab_size=vocab_size, spurious_ratio=0.85)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, loss_fn, device)
        print(f"Epoch [{epoch}/{args.epochs}] -> Total Loss: {metrics['total_loss']:.4f} | NTP: {metrics['loss_ntp']:.4f} | Entropy Loss: {metrics['loss_entropy']:.4f} | Shortcut Supp: {metrics['loss_supp']:.4f}")
        
    checkpoint_path = os.path.join(args.save_dir, "debiased_vlm_dual_grounding.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nModel checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
