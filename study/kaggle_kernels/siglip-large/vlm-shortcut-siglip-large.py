"""Kaggle GPU kernel: larger vision encoder. SigLIP-large/16-256 features (256 patches x 1024), Qwen3.5-4B,
the four connectors on Waterbirds, 3 seeds; stage-1 alignment redone (the connectors' input size changes)."""
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"
EXPORT = "/kaggle/working/results"
FEATURES = "features_siglip-large.npy"


def run(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def find(pattern):
    out = subprocess.run(f"find -L /kaggle/input -path '{pattern}'", shell=True, capture_output=True,
                         text=True).stdout.split()
    return out[0] if out else None


data_dir = os.path.dirname(os.path.dirname(os.path.dirname(find(f"*tasks/captions/{FEATURES}"))))
prev = find("*/qwen35-4b_siglip-large/align/mlp/connector.pt")  # an earlier session of this kernel
resume = prev.split("/qwen35-4b_siglip-large/")[0] if prev else ""
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
run(f"{sys.executable} study/kaggle_study.py --llm Qwen/Qwen3.5-4B --tasks waterbirds --seeds 3 "
    f"--connectors mlp,qformer,bottleneck,bottleneck_fixed --gpus 2 --align_steps 300 --epochs 4 --mlp_pool 2 "
    f"--features {FEATURES} --vision_model google/siglip-large-patch16-256 --run_tag _siglip-large "
    f"--job_budget_min 140 --total_budget_min 660 --data_from {data_dir} --export_dir {EXPORT} "
    + (f"--resume_from {resume}" if resume else ""), cwd=WORK)
