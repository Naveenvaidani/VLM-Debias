"""Kaggle CPU kernel: rebuild the waterbirds + watermark tasks and captions, cache SigLIP-large/16-256 features."""
import glob
import os
import subprocess
import sys

WORK = "/kaggle/tmp/VLM-Debias"


def run(cmd, **kw):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


ann = glob.glob("/kaggle/input/**/annotations/instances_train2017.json", recursive=True) or subprocess.run(
    "find -L /kaggle/input -name instances_train2017.json", shell=True, capture_output=True, text=True).stdout.split()
assert ann, "COCO annotations not found under /kaggle/input"
coco_root = os.path.dirname(os.path.dirname(ann[0]))
run("pip uninstall -y torchao")
run("pip install -q 'transformers>=5.20' pyarrow")
run(f"rm -rf {WORK} && git clone --depth 1 -b study/connector-shortcuts https://github.com/Naveenvaidani/VLM-Debias.git {WORK}")
run(f"{sys.executable} study/kaggle_study.py --tasks waterbirds,watermark_p0.95 --coco_root {coco_root} "
    f"--vision_model google/siglip-large-patch16-256 --features features_siglip-large.npy "
    f"--prepare_only --export_dir /kaggle/working", cwd=WORK)
run("du -sh /kaggle/working/data/tasks/*")
