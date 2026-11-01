"""
VLM-SHIELD demo: classify a bird image (landbird / waterbird) with a trained checkpoint
and show the answer token's cross-attention over SigLIP patches as a heatmap.

    python app.py --run bottleneck_balanced
"""
import argparse
import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

from src.data.waterbirds import CLASS_NAMES
from src.models.debiased_vlm import DebiasedVLM


def load_system(run: str, vision_model: str = "google/siglip-base-patch16-224"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join("runs", run, "best.pt"), map_location="cpu")
    train_args = ckpt["args"]
    model = DebiasedVLM(llm_model_name_or_path=train_args["llm"], mode=train_args["mode"],
                        lora_rank=train_args["lora_rank"], lora_alpha=2.0 * train_args["lora_rank"], device=device)
    model.load_state_dict(ckpt["state"], strict=False)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(vision_model)
    encoder = SiglipVisionModel.from_pretrained(vision_model, dtype=torch.float16).to(device).eval()
    return model, processor, encoder, device


def attention_overlay(image: Image.Image, attn: np.ndarray) -> Image.Image:
    """Min-max normalise the 14x14 patch attention, upsample, and blend a JET colormap over the image."""
    side = int(np.sqrt(attn.shape[-1]))
    grid = attn.reshape(side, side)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    w, h = image.size
    heat = cv2.GaussianBlur(cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC), (15, 15), 0)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2.addWeighted(np.array(image.convert("RGB")), 0.6, heat_rgb, 0.4, 0))


def build_interface(model, processor, encoder, device):
    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return {}, None
        pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device, torch.float16)
        feats = encoder(pixel_values=pixel_values).last_hidden_state.float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits, attn = model(feats)
        probs = torch.softmax(logits.float(), -1)[0].tolist()
        overlay = attention_overlay(image, attn[0, -1].float().cpu().numpy()) if attn is not None else None
        return dict(zip(CLASS_NAMES, probs)), overlay

    with gr.Blocks(title="VLM-SHIELD") as demo:
        gr.Markdown("# VLM-SHIELD\nLandbird / waterbird classification through a cross-attention bottleneck, "
                    "with the answer token's attention over image patches.")
        with gr.Row():
            image_input = gr.Image(type="pil", label="Bird image")
            with gr.Column():
                label = gr.Label(label="Prediction")
                heatmap = gr.Image(label="Answer-token attention")
        gr.Button("Classify", variant="primary").click(predict, inputs=image_input, outputs=[label, heatmap])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="bottleneck_balanced", help="Run directory under ./runs")
    args = parser.parse_args()
    demo = build_interface(*load_system(args.run))
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
