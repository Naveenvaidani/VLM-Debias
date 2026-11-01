"""Kaggle TPU pilot: does the study run on a TPU core (torch_xla, bf16, no 4-bit), and how fast?

Phi-4-mini-instruct (3.8B, standard attention) with the pooled MLP connector on Waterbirds: 20 alignment
steps, then 1 epoch on 512 training images with capped evaluation. Prints seconds per step.
"""
import os
import subprocess
import sys
import time

WORK = "/kaggle/tmp/VLM-Debias"
LLM = "microsoft/Phi-4-mini-instruct"


def run(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, **kw)


data = subprocess.run("find -L /kaggle/input -path '*tasks/captions/metadata.csv'", shell=True, capture_output=True,
                      text=True).stdout.split()[0]
tasks_dir = os.path.dirname(os.path.dirname(data))
run(f"{sys.executable} -c 'import torch; print(\"torch\", torch.__version__)'")
if run(f"{sys.executable} -c 'import torch_xla; print(\"torch_xla\", torch_xla.__version__)'", check=False).returncode:
    v = subprocess.run(f"{sys.executable} -c 'import torch; print(torch.__version__.split(\"+\")[0])'", shell=True,
                       capture_output=True, text=True).stdout.strip()
    run(f"pip install -q 'torch_xla[tpu]=={v}' -f https://storage.googleapis.com/libtpu-releases/index.html")
run("pip install -q 'transformers>=5.20' peft accelerate pyarrow")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
os.makedirs(f"{WORK}/data", exist_ok=True)
run(f"ln -sfn {tasks_dir} {WORK}/data/tasks")
env = dict(os.environ, PJRT_DEVICE="TPU", TOKENIZERS_PARALLELISM="false")
common = f"--llm {LLM} --connector mlp --mlp_pool 2 --no_4bit --device xla --dtype bf16"

t0 = time.time()
run(f"{sys.executable} study/align.py {common} --out runs_tpu/align --steps 20 --batch_size 16",
    cwd=WORK, env=env)
print(f"ALIGN 20 steps: {(time.time() - t0) / 60:.1f} min (includes model load and XLA compilation)", flush=True)
t0 = time.time()
run(f"{sys.executable} study/train_task.py {common} --task_dir data/tasks/waterbirds --align_ckpt runs_tpu/align/connector.pt "
    f"--run_dir runs_tpu/wb --epochs 1 --batch_size 16 --max_train_samples 512 --max_eval_samples 256", cwd=WORK, env=env)
print(f"TASK 512 images x 1 epoch + capped eval: {(time.time() - t0) / 60:.1f} min", flush=True)
run(f"cat {WORK}/runs_tpu/align/align.json | head -c 400; echo; grep -h 'epoch' {WORK}/runs_tpu/wb/*.log 2>/dev/null | tail -3", check=False)
