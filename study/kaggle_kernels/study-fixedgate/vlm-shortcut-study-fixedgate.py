"""Kaggle GPU kernel: fixed-gate cross-attention bottleneck on both tasks, 3 seeds (Qwen3.5-4B NF4, 2x T4).

Speed setup from the step benchmark (s/step, batch 16, T4): causal-conv1d kernel, MLP tokens pooled
196 -> 49, no gradient checkpointing, 4 epochs -> MLP 4.0, Q-Former 2.5, bottleneck 1.7 (was MLP 13.7).
"""
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"
EXPORT = "/kaggle/working/results"


def run(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def find(pattern):
    out = subprocess.run(f"find -L /kaggle/input -path '{pattern}'", shell=True, capture_output=True,
                         text=True).stdout.split()
    return out[0] if out else None


data = find("*tasks/captions/metadata.csv")
assert data, "prepared data not found under /kaggle/input"
data_dir = os.path.dirname(os.path.dirname(os.path.dirname(data)))  # .../data
print("prepared data:", data_dir, flush=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"{sys.executable} -c 'import causal_conv1d'")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"{sys.executable} study/kaggle_study.py --llm Qwen/Qwen3.5-4B --tasks waterbirds,watermark_p0.95 --seeds 3 --connectors bottleneck_fixed "
    f"--gpus 2 --align_steps 300 --epochs 4 --mlp_pool 2 --job_budget_min 110 --total_budget_min 660 "
    f"--data_from {data_dir} --export_dir {EXPORT}", cwd=WORK)
