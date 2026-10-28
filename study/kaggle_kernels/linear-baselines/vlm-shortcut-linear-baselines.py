"""Kaggle CPU kernel: ERM probe, DFR and group-balanced probe on the frozen SigLIP features (no GPU)."""
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"


def run(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


data = subprocess.run("find -L /kaggle/input -path '*tasks/captions/metadata.csv'", shell=True, capture_output=True,
                      text=True).stdout.split()[0]
tasks_dir = os.path.dirname(os.path.dirname(data))
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} study/linear_baselines.py --tasks_dir {tasks_dir} --tasks waterbirds,watermark_p0.95 --pools mean,meanmax,grid7 "
    f"--out /kaggle/working/results", cwd=WORK)
