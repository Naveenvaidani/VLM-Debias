"""VQA-CP v2 (open-ended answers, natural COCO images, changing priors): learned vs fixed gate, Qwen3.5-4B,
seeds 0-2. Builds a 8k/1k/2k/3k-question subset, caches SigLIP-base features, aligns both connectors for 300 steps,
then trains and evaluates each gate on its own GPU (free-form generation, with and without the visual pathway)."""
LLM = "Qwen/Qwen3.5-4B"
import glob
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/tmp/VLM-Debias"
EXPORT = "/kaggle/working/results"
START = time.time()


def run(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=check, **kw)


def find(pattern):
    out = subprocess.run(f"find -L /kaggle/input -path '{pattern}'", shell=True, capture_output=True, text=True).stdout.split()
    return out[0] if out else None


vqacp_dir = os.path.dirname(find("*vqacp_v2_train_questions.json"))
coco_root = os.path.dirname(os.path.dirname(find("*train2017/000000000009.jpg")))
captions = os.path.dirname(find("*tasks/captions/metadata.csv"))
run("nvidia-smi --query-gpu=name,memory.total --format=csv")
run("pip uninstall -y torchao")
run("pip install -q 'transformers==5.22.1' 'peft==0.20.0' 'bitsandbytes==0.50.2' accelerate 'flash-linear-attention==0.6.0' pyarrow tabulate")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b main https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{LLM}')\"")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
task = f"{WORK}/data/tasks/vqacp"
run(f"{sys.executable} study/build_vqacp.py --vqacp_dir {vqacp_dir} --coco_root {coco_root} "
    f"--exclude_captions {captions}/metadata.csv --out {task} --n_train 8000", cwd=WORK)
run(f"{sys.executable} study/cache_features.py --tasks_dir {task}", cwd=WORK)
shutil.copy(f"{task}/summary.json", f"{EXPORT}/vqacp_summary.json")
shutil.copy(f"{task}/questions.csv", f"{EXPORT}/vqacp_questions.csv")
runs = f"{WORK}/runs_study/qwen35-4b_vqacp"
CONNS = ["bottleneck", "bottleneck_fixed"]
procs = []
for gpu, c in enumerate(CONNS):  # 300-step alignment on COCO captions, one connector per GPU
    os.makedirs(f"{runs}/align/{c}", exist_ok=True)
    procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} {sys.executable} study/align.py --llm {LLM} --connector {c} "
                                  f"--captions_dir {captions} --out {runs}/align/{c} --steps 300 --seed 0 "
                                  f"> {runs}/align/{c}/align.log 2>&1", shell=True, cwd=WORK))
[p.wait() for p in procs]
for c in CONNS:
    shutil.copytree(f"{runs}/align/{c}", f"{EXPORT}/qwen35-4b_vqacp/align/{c}", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.pt"))


def worker(gpu, c):
    cmds = []
    for seed in range(3):
        d = f"{runs}/{c}/s{seed}"
        cmds.append(f"mkdir -p {d} && [ $(( $(date +%s) - {int(START)} )) -lt 33000 ] && "
                    f"{sys.executable} study/train_vqa.py --llm {LLM} --connector {c} --task_dir {task} "
                    f"--align_ckpt {runs}/align/{c}/connector.pt --seed {seed} --run_dir {d} --time_budget_min 110 "
                    f"> {d}/train.log 2>&1; mkdir -p {EXPORT}/qwen35-4b_vqacp/{c}/s{seed}; "
                    f"cp {d}/*.json {d}/train.log {EXPORT}/qwen35-4b_vqacp/{c}/s{seed}/ 2>/dev/null")
    return subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{' ; '.join(cmds)}'", shell=True, cwd=WORK)


procs = [worker(g, c) for g, c in enumerate(CONNS)]
[p.wait() for p in procs]
run(f"tail -n 2 {runs}/*/s*/train.log", check=False)
