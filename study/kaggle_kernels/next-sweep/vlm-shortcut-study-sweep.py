"""Kaggle GPU kernel: robustness of the connector comparison to training settings (Waterbirds, seed 0).

  _lr5e-5, _lr1e-3   connector learning rate sweep (main study used 2e-4), all four connectors
  _long             8 epochs instead of 4, MLP and fixed-gate cross-attention
"""
import os
import shutil
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


data_dir = os.path.dirname(os.path.dirname(os.path.dirname(find("*tasks/captions/metadata.csv"))))
resume = "/kaggle/tmp/resume"
for tag in ("_lr5e-5", "_lr1e-3", "_long"):
    for c in ("mlp", "qformer", "bottleneck", "bottleneck_fixed"):
        os.makedirs(f"{resume}/qwen35-4b{tag}/align/{c}", exist_ok=True)
        shutil.copy(find(f"*/qwen35-4b/align/{c}/connector.pt"), f"{resume}/qwen35-4b{tag}/align/{c}/connector.pt")
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
base = (f"{sys.executable} study/kaggle_study.py --llm Qwen/Qwen3.5-4B --tasks waterbirds --seeds 1 --gpus 2 "
        f"--align_steps 300 --mlp_pool 2 --job_budget_min 200 --data_from {data_dir} --export_dir {EXPORT} "
        f"--resume_from {resume}")
for extra in ("--connectors mlp,qformer,bottleneck,bottleneck_fixed --epochs 4 --connector_lr 5e-5 --run_tag _lr5e-5 "
              "--total_budget_min 150",
              "--connectors mlp,qformer,bottleneck,bottleneck_fixed --epochs 4 --connector_lr 1e-3 --run_tag _lr1e-3 "
              "--total_budget_min 150",
              "--connectors mlp,bottleneck_fixed --epochs 8 --run_tag _long --total_budget_min 210"):
    subprocess.run(f"{base} {extra}", shell=True, cwd=WORK)  # one setting failing must not stop the others
