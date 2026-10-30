"""Kaggle GPU kernel: seeds 3 and 4 for the four main connectors on both tasks (brings every config to 5 seeds),
reusing the stage-1 connectors of the main study."""
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
resume = "/kaggle/tmp/resume/qwen35-4b/align"
for c in ("mlp", "qformer", "bottleneck", "bottleneck_fixed"):
    os.makedirs(f"{resume}/{c}", exist_ok=True)
    shutil.copy(find(f"*/qwen35-4b/align/{c}/connector.pt"), f"{resume}/{c}/connector.pt")
finished = subprocess.run("find -L /kaggle/input -path '*/qwen35-4b/*/*/s[34]/results.json'", shell=True,
                          capture_output=True, text=True).stdout.split()
for prev in finished:  # an earlier session of this kernel: skip what it finished
    dst = "/kaggle/tmp/resume/qwen35-4b/" + prev.split("/qwen35-4b/")[1]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(prev, dst)
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
run(f"{sys.executable} study/kaggle_study.py --llm Qwen/Qwen3.5-4B --tasks waterbirds,watermark_p0.95 --seeds 5 --seed_start 3 "
    f"--connectors mlp,qformer,bottleneck,bottleneck_fixed --gpus 2 --align_steps 300 --epochs 4 --mlp_pool 2 "
    f"--job_budget_min 110 --total_budget_min 660 --data_from {data_dir} --export_dir {EXPORT} "
    f"--resume_from /kaggle/tmp/resume", cwd=WORK)
