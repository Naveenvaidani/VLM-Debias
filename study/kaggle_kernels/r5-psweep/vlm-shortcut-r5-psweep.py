"""Dose-response: Watermark at shortcut strength p = 0.9 and 0.99 (p = 0.95 is the main task), MLP and both gates, seeds 0-2."""
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
run("pip install -q 'transformers>=5.20' peft bitsandbytes accelerate flash-linear-attention pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b main https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{LLM}')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
base = f"{sys.executable} study/kaggle_study.py --llm {LLM} --gpus 2 --epochs 4 --mlp_pool 2 --data_from {data_dir} --export_dir {EXPORT} "
# Build Watermark at shortcut strengths 0.9 and 0.99 into a separate folder (build_tasks always rebuilds the
# caption set, so the original captions and p = 0.95 task are kept and only the two new task folders are added).
import glob, shutil
ann = subprocess.run("find -L /kaggle/input -name instances_train2017.json", shell=True, capture_output=True, text=True).stdout.split()
coco_root = os.path.dirname(os.path.dirname(ann[0]))
merged = "/kaggle/tmp/data_merged"
shutil.copytree(data_dir, merged, dirs_exist_ok=True)
run(f"{sys.executable} study/build_tasks.py --coco_root {coco_root} --out /kaggle/tmp/newtasks --only watermark --p_list 0.9,0.99", cwd=WORK)
for t in ("watermark_p0.9", "watermark_p0.99"):
    shutil.copytree(f"/kaggle/tmp/newtasks/{t}", f"{merged}/tasks/{t}", dirs_exist_ok=True)
base = base.replace(f"--data_from {data_dir}", f"--data_from {merged}")
run(base + "--tasks watermark_p0.9,watermark_p0.99 --seeds 3 --connectors mlp,bottleneck,bottleneck_fixed --align_steps 300 --skip_smoke --job_budget_min 70 --run_tag _psweep --total_budget_min 560", check=False, cwd=WORK)
