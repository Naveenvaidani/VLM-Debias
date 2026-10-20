"""
Pre-download the foundation models into the Hugging Face cache (no remote code is executed).
"""
import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained foundation models.")
    parser.add_argument("--siglip_id", default="google/siglip-base-patch16-224")
    parser.add_argument("--qwen_id", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()

    for repo_id, patterns in ((args.siglip_id, ["*.json", "*.safetensors", "*.model"]), (args.qwen_id, None)):
        path = snapshot_download(repo_id, allow_patterns=patterns)
        print(f"{repo_id} -> {path}")


if __name__ == "__main__":
    main()
