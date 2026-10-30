"""Kaggle GPU kernel: frozen LLM (connector-only training) on Waterbirds, 3 connectors x 3 seeds.

Separates the connector's effect from LoRA's: reuses the first run's aligned connectors so the only
change from the main study is that the LLM is not adapted.
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
align = os.path.dirname(os.path.dirname(find("*/qwen35-4b/align/mlp/connector.pt")))
resume = "/kaggle/tmp/resume"
shutil.copytree(align, f"{resume}/qwen35-4b_frozen/align")  # same stage-1 connectors as the main study
for prev in ("qwen35-4b_frozen",):  # results of an earlier session of this kernel, if attached
    hit = find(f"*/{prev}/waterbirds/*/s*/results.json")
    if hit:
        shutil.copytree(hit.split(f"/{prev}/")[0] + f"/{prev}", f"{resume}/{prev}", dirs_exist_ok=True)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"{sys.executable} study/kaggle_study.py --llm Qwen/Qwen3.5-4B --tasks waterbirds --seeds 3 --gpus 2 "
    f"--align_steps 300 --epochs 4 --mlp_pool 2 --job_budget_min 110 --total_budget_min 660 --freeze_llm "
    f"--run_tag _frozen --data_from {data_dir} --export_dir {EXPORT} --resume_from {resume}", cwd=WORK)
