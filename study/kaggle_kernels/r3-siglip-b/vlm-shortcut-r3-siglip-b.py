"""SigLIP-large features: Waterbirds seed 4 (four connectors), then Watermark gate arms seeds 0-1."""
LLM = "Qwen/Qwen3.5-4B"
FEAT = "features_siglip-large.npy"
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"
EXPORT = "/kaggle/working/results"


def run(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=check, **kw)


def find(pattern):
    out = subprocess.run(f"find -L /kaggle/input -path '{pattern}'", shell=True, capture_output=True,
                         text=True).stdout.split()
    return out[0] if out else None


data_dir = os.path.dirname(os.path.dirname(os.path.dirname(find(f"*tasks/captions/{FEAT}"))))
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b main https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{LLM}')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
base = f"{sys.executable} study/kaggle_study.py --llm {LLM} --gpus 2 --epochs 4 --mlp_pool 2 --data_from {data_dir} --export_dir {EXPORT} "
run(base + "--tasks waterbirds --seed_start 4 --seeds 5 --connectors mlp,qformer,bottleneck,bottleneck_fixed --align_steps 300 --gradient_checkpointing --features features_siglip-large.npy --vision_model google/siglip-large-patch16-256 --run_tag _siglip-large --skip_smoke --job_budget_min 140 --total_budget_min 300", check=False, cwd=WORK)  # one setting failing must not stop the next
run(base + "--tasks watermark_p0.95 --seeds 2 --connectors bottleneck,bottleneck_fixed --align_steps 300 --gradient_checkpointing --features features_siglip-large.npy --vision_model google/siglip-large-patch16-256 --run_tag _siglip-large --skip_smoke --job_budget_min 120 --total_budget_min 330", check=False, cwd=WORK)  # one setting failing must not stop the next
