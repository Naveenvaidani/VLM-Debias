"""Learnable gate started at tanh = 0.1 with default W_o, Watermark, seeds 5-9."""
LLM = "Qwen/Qwen3.5-4B"
FEAT = "metadata.csv"
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
run("pip install -q 'transformers==5.22.1' 'peft==0.20.0' 'bitsandbytes==0.50.2' accelerate 'flash-linear-attention==0.6.0' pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b main https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{LLM}')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
base = f"{sys.executable} study/kaggle_study.py --llm {LLM} --gpus 2 --epochs 4 --mlp_pool 2 --data_from {data_dir} --export_dir {EXPORT} "
run(base + "--tasks watermark_p0.95 --seed_start 5 --seeds 10 --connectors bottleneck_eps1d --align_steps 300 --run_tag _wo --skip_smoke --job_budget_min 70 --total_budget_min 600", check=False, cwd=WORK)
