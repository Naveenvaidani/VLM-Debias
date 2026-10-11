"""
Local Foundation Model Downloader Script
Downloads Google SigLIP Vision Encoder and Qwen2.5-1.5B Language Model locally.
"""
import os
import argparse
import sys

def download_models(output_dir: str = "./local_models", siglip_id: str = "google/siglip-base-patch16-224", qwen_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    print(f"=== VLM-SHIELD Foundation Model Downloader ===")
    print(f"Target Directory: {os.path.abspath(output_dir)}")
    
    os.makedirs(output_dir, exist_ok=True)
    siglip_dir = os.path.join(output_dir, "siglip-base-patch16-224")
    qwen_dir = os.path.join(output_dir, "Qwen2.5-1.5B-Instruct")
    
    try:
        from transformers import AutoModel, AutoImageProcessor, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required. Install via `pip install transformers`.")
        sys.exit(1)
        
    print(f"\n[1/2] Downloading Vision Encoder: {siglip_id}...")
    try:
        siglip_model = AutoModel.from_pretrained(siglip_id)
        siglip_model.save_pretrained(siglip_dir)
        
        try:
            siglip_image_processor = AutoImageProcessor.from_pretrained(siglip_id)
            siglip_image_processor.save_pretrained(siglip_dir)
        except Exception as proc_e:
            print(f"Image processor load note: {proc_e}")
            
        print(f"Successfully downloaded & saved SigLIP to {siglip_dir}")
    except Exception as e:
        print(f"Failed to download {siglip_id}: {e}")
        
    print(f"\n[2/2] Downloading Language Model: {qwen_id}...")
    try:
        qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_id, trust_remote_code=True)
        qwen_model = AutoModelForCausalLM.from_pretrained(qwen_id, trust_remote_code=True)
        
        qwen_model.save_pretrained(qwen_dir)
        qwen_tokenizer.save_pretrained(qwen_dir)
        print(f"Successfully downloaded & saved Qwen2.5-1.5B to {qwen_dir}")
    except Exception as e:
        print(f"Failed to download {qwen_id}: {e}")

    print("\nDownload process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained foundation models locally.")
    parser.add_argument("--output_dir", type=str, default="./local_models", help="Directory to store downloaded models.")
    parser.add_argument("--siglip_id", type=str, default="google/siglip-base-patch16-224", help="HuggingFace model ID for SigLIP.")
    parser.add_argument("--qwen_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="HuggingFace model ID for Qwen.")
    args = parser.parse_args()
    
    download_models(args.output_dir, args.siglip_id, args.qwen_id)
