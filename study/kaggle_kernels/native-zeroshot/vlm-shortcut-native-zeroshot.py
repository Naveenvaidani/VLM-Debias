"""Kaggle GPU kernel: Qwen3.5-4B with its own vision tower, zero-shot on Waterbirds (first phrasing), raw and
calibrated (answer threshold chosen on validation)."""
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"
EXPORT = "/kaggle/working/results"


def run(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' bitsandbytes accelerate flash-linear-attention pyarrow")
run("MAX_JOBS=4 pip install -q --no-build-isolation causal-conv1d")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
os.makedirs(EXPORT, exist_ok=True)
run(f"pip freeze > {EXPORT}/environment.txt; git -C {WORK} rev-parse HEAD >> {EXPORT}/environment.txt")
run(f"{sys.executable} scripts/prepare_waterbirds.py", cwd=WORK)
run(f"{sys.executable} -c \"import sys; sys.path.insert(0, 'study'); import build_tasks; "
    f"build_tasks.build_waterbirds_task('data/tasks', 'data')\"", cwd=WORK)
run(f"{sys.executable} study/native_vlm.py --llm Qwen/Qwen3.5-4B --task_dir data/tasks/waterbirds --templates 0 --out {EXPORT}",
    cwd=WORK)
