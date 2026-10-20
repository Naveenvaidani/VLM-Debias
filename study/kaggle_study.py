"""
Run the connector-shortcut study on a multi-GPU machine (built for Kaggle 2x T4, fp16).

  1. data: Waterbirds + COCO-based shortcut tasks + caption set, frozen SigLIP features
  2. smoke test (fails fast before spending GPU hours)
  3. stage 1: align each connector on captions (one job per connector)
  4. stage 2: train + test every (task, connector, seed); one job per GPU from a shared queue
  5. aggregate; small result files are exported after every job so a killed session loses nothing

    python study/kaggle_study.py --llm Qwen/Qwen3.5-4B --seeds 3 --export_dir /kaggle/working/results
"""
import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONNECTORS = ("mlp", "qformer", "bottleneck")


def tail(path, n=40):
    if os.path.exists(path):
        lines = open(path, errors="replace").read().replace("\r", "\n").splitlines()
        print(f"--- tail of {path} ---\n" + "\n".join(lines[-n:]) + "\n--- end ---", flush=True)


def sh(cmd, gpu=None, log=None):
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("$", " ".join(cmd), f"[gpu {gpu}]" if gpu is not None else "", flush=True)
    out = open(log, "a") if log else None
    code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT if out else None).returncode
    if out:
        out.close()
    if code != 0 and log:
        tail(log)
    return code


def llm_tag(llm):
    return llm.split("/")[-1].lower().replace(".", "")


class Pool:
    """Runs (name, cmd, done_file) jobs on `gpus` GPUs; skips jobs whose done_file exists."""
    def __init__(self, gpus, start, total_budget_min, job_budget_min):
        self.gpus, self.start = gpus, start
        self.total_budget_min, self.job_budget_min = total_budget_min, job_budget_min
        self.failures = []

    def run(self, jobs, on_done=None):
        q = queue.Queue()
        for j in jobs:
            q.put(j)

        def worker(gpu):
            while True:
                try:
                    name, cmd, done, log = q.get_nowait()
                except queue.Empty:
                    return
                if os.path.exists(done):
                    print(f"skip {name} (done)", flush=True)
                    continue
                if (time.time() - self.start) / 60 + self.job_budget_min > self.total_budget_min:
                    print(f"skip {name}: not enough session time left", flush=True)
                    continue
                ok = sh(cmd, gpu=gpu, log=log) == 0 and os.path.exists(done)
                print(f"[{(time.time() - self.start) / 60:.0f} min] {'done' if ok else 'FAILED'} {name} (gpu {gpu})",
                      flush=True)
                if not ok:
                    self.failures.append(name)
                if on_done:
                    on_done(name)

        threads = [threading.Thread(target=worker, args=(g,)) for g in range(self.gpus)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--coco_root", default="/kaggle/input/coco-2017-dataset/coco2017")
    parser.add_argument("--tasks", default="waterbirds,cooccur,watermark_p0.95,typographic_p0.95")
    parser.add_argument("--connectors", default=",".join(CONNECTORS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--align_steps", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--job_budget_min", type=float, default=70, help="per training job")
    parser.add_argument("--total_budget_min", type=float, default=630, help="Kaggle kills sessions at 12 h")
    parser.add_argument("--export_dir", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    args = parser.parse_args()
    py, start = sys.executable, time.time()
    tag = llm_tag(args.llm)
    runs = os.path.join(ROOT, "runs_study", tag)
    os.makedirs(runs, exist_ok=True)
    ckpt = ["--gradient_checkpointing"] if args.gradient_checkpointing else []
    tasks = args.tasks.split(",")
    connectors = args.connectors.split(",")

    # 1. data
    if "waterbirds" in tasks and not os.path.exists(os.path.join(ROOT, "data", "waterbirds", "metadata.csv")):
        assert sh([py, "scripts/prepare_waterbirds.py"]) == 0
    if not os.path.exists(os.path.join(ROOT, "data", "tasks", "captions", "metadata.csv")):
        families = sorted({t.split("_p")[0] for t in tasks})
        assert sh([py, "study/build_tasks.py", "--coco_root", args.coco_root, "--only", ",".join(families)]) == 0
    assert sh([py, "study/cache_features.py", "--tasks_dir", "data/tasks"], gpu=0) == 0

    def export(name):
        if not args.export_dir:
            return
        for path in glob.glob(os.path.join(runs, "**", "*"), recursive=True):
            rel = os.path.relpath(path, runs)
            if os.path.isfile(path) and (path.endswith((".json", ".npz", ".log", ".md", ".csv", ".png"))):
                dst = os.path.join(args.export_dir, tag, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(path, dst)

    # 2. smoke: a few steps of alignment + a tiny task run per connector type
    if not args.skip_smoke:
        for c in connectors:
            d = os.path.join(runs, "smoke", c)
            log = os.path.join(runs, "smoke.log")
            assert sh([py, "study/align.py", "--llm", args.llm, "--connector", c, "--out", d, "--steps", "5"] + ckpt,
                      gpu=0, log=log) == 0, f"smoke align failed ({c})"
            assert sh([py, "study/train_task.py", "--llm", args.llm, "--connector", c, "--task_dir",
                       f"data/tasks/{tasks[0]}", "--align_ckpt", os.path.join(d, "connector.pt"),
                       "--run_dir", d, "--epochs", "1", "--max_train_samples", "64"] + ckpt,
                      gpu=0, log=log) == 0, f"smoke train failed ({c})"
        shutil.rmtree(os.path.join(runs, "smoke"), ignore_errors=True)

    pool = Pool(args.gpus, start, args.total_budget_min, args.job_budget_min)

    # 3. stage 1 alignment
    align_jobs = []
    for c in connectors:
        d = os.path.join(runs, "align", c)
        os.makedirs(d, exist_ok=True)
        align_jobs.append((f"align/{c}", [py, "study/align.py", "--llm", args.llm, "--connector", c, "--out", d,
                                          "--steps", str(args.align_steps)] + ckpt,
                           os.path.join(d, "connector.pt"), os.path.join(d, "align.log")))
    pool.run(align_jobs, on_done=export)

    # 4. stage 2 task runs (seed-major order, so partial sessions still cover every config)
    task_jobs = []
    for seed in range(args.seeds):
        for task in tasks:
            for c in connectors:
                d = os.path.join(runs, task, c, f"s{seed}")
                os.makedirs(d, exist_ok=True)
                task_jobs.append((f"{task}/{c}/s{seed}", [
                    py, "study/train_task.py", "--llm", args.llm, "--connector", c,
                    "--task_dir", f"data/tasks/{task}", "--align_ckpt", os.path.join(runs, "align", c, "connector.pt"),
                    "--run_dir", d, "--seed", str(seed), "--epochs", str(args.epochs),
                    "--time_budget_min", str(args.job_budget_min - 10)] + ckpt,
                    os.path.join(d, "results.json"), os.path.join(d, "train.log")))
    pool.run(task_jobs, on_done=export)

    # 5. aggregate
    sh([py, "study/aggregate.py", "--runs_dir", runs, "--out", os.path.join(runs, "summary")])
    export("summary")
    print(f"Finished in {(time.time() - start) / 60:.0f} min. Failed: {pool.failures or 'none'}")


if __name__ == "__main__":
    main()
