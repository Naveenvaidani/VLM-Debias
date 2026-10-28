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
import shlex
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
    parser.add_argument("--seed_start", type=int, default=0, help="first seed to run (e.g. 2 to finish seed 2 only)")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--align_steps", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--job_budget_min", type=float, default=70, help="per training job")
    parser.add_argument("--total_budget_min", type=float, default=630, help="Kaggle kills sessions at 12 h")
    parser.add_argument("--export_dir", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--mlp_pool", type=int, default=1, help="avg-pool MLP patch tokens (2 -> 49 tokens)")
    parser.add_argument("--mlp_grid", type=int, default=0, help="adaptive MLP token grid (6 -> 36 tokens)")
    parser.add_argument("--no_4bit", action="store_true", help="fp16 LLM weights instead of NF4")
    parser.add_argument("--data_from", default=None,
                        help="prepared data/ dir from a data-prep session (skips building tasks and features)")
    parser.add_argument("--freeze_llm", action="store_true", help="train the connector only (no LoRA on the LLM)")
    parser.add_argument("--align_per_seed", action="store_true",
                        help="separate stage-1 alignment per seed, so seed variance covers both stages")
    parser.add_argument("--vision_model", default="google/siglip-base-patch16-224")
    parser.add_argument("--features", default="features.npy", help="cached feature file name (one per encoder)")
    parser.add_argument("--stage3", default="",
                        help="comma list of dfr,jtt,prompt: after each run, retrain the connector (dfr, jtt) or "
                             "re-evaluate with a debiasing instruction (prompt)")
    parser.add_argument("--instruction", default="Answer based only on the main object; ignore the background "
                                                  "and any overlaid text or watermarks.")
    parser.add_argument("--balanced_sampling", action="store_true")
    parser.add_argument("--connector_lr", type=float, default=None, help="stage-2 connector learning rate")
    parser.add_argument("--lora_lr", type=float, default=None, help="stage-2 LoRA learning rate")
    parser.add_argument("--attn_entropy_weight", type=float, default=0.0)
    parser.add_argument("--align_ltd", type=float, default=0.0, help="LTD weight for stage-1 alignment")
    parser.add_argument("--stage3_epochs", type=int, default=3)
    parser.add_argument("--run_tag", default="", help="suffix for the runs folder, e.g. _frozen")
    parser.add_argument("--prepare_only", action="store_true",
                        help="build tasks + features, export features/metadata to --export_dir/data and stop")
    parser.add_argument("--resume_from", default=None,
                        help="exported results dir of an earlier session (e.g. a kernel output attached as input): "
                             "finished jobs and aligned connectors are copied in and skipped")
    args = parser.parse_args()
    py, start = sys.executable, time.time()
    tag = llm_tag(args.llm)
    tag += args.run_tag
    os.environ["STUDY_FEATURES"] = args.features  # read by src/study/data.py in every child process
    runs = os.path.join(ROOT, "runs_study", tag)
    os.makedirs(runs, exist_ok=True)
    ckpt = ["--gradient_checkpointing"] if args.gradient_checkpointing else []
    ckpt += ["--mlp_pool", str(args.mlp_pool)] + (["--mlp_grid", str(args.mlp_grid)] if args.mlp_grid else []) + (["--no_4bit"] if args.no_4bit else [])
    tasks = args.tasks.split(",")
    connectors = args.connectors.split(",")

    # 1. data (copied from a data-prep session when available: images are not needed, only features)
    if args.data_from:
        shutil.copytree(args.data_from, os.path.join(ROOT, "data"), dirs_exist_ok=True)
    wb_meta = os.path.join(ROOT, "data", "waterbirds", "metadata.csv")
    if not args.data_from and "waterbirds" in tasks and not os.path.exists(wb_meta):
        assert sh([py, "scripts/prepare_waterbirds.py"]) == 0
    if not os.path.exists(os.path.join(ROOT, "data", "tasks", "captions", "metadata.csv")):
        families = sorted({t.split("_p")[0] for t in tasks})
        assert sh([py, "study/build_tasks.py", "--coco_root", args.coco_root, "--only", ",".join(families)]) == 0
    assert sh([py, "study/cache_features.py", "--tasks_dir", "data/tasks", "--vision_model", args.vision_model,
               "--out_name", args.features], gpu=0) == 0
    if args.prepare_only:
        for path in glob.glob(os.path.join(ROOT, "data", "tasks", "*", "*")):
            if path.endswith((args.features, ".csv", ".json")):
                dst = os.path.join(args.export_dir, "data", os.path.relpath(path, os.path.join(ROOT, "data")))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(path, dst)
        print(f"data exported in {(time.time() - start) / 60:.0f} min")
        return

    def export(name):
        if not args.export_dir:
            return
        for path in glob.glob(os.path.join(runs, "**", "*"), recursive=True):
            rel = os.path.relpath(path, runs)
            if os.path.isfile(path) and (path.endswith((".json", ".npz", ".log", ".md", ".csv", ".png")) or \
                    (path.endswith("connector.pt") and ("/align/" in path or "bottleneck" in path))):
                dst = os.path.join(args.export_dir, tag, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(path, dst)

    # resume: bring back finished jobs and aligned connectors from an earlier session
    if args.resume_from:
        src = os.path.join(args.resume_from, tag)
        n = 0
        for path in glob.glob(os.path.join(src, "**", "*"), recursive=True):
            if os.path.isfile(path):
                dst = os.path.join(runs, os.path.relpath(path, src))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(path, dst)
                n += 1
        print(f"resumed {n} files from {src}", flush=True)
        if glob.glob(os.path.join(src, "*", "*", "s*", "results.json")):
            args.skip_smoke = True  # a previous session already passed the smoke test

    # 2. smoke: a few steps of alignment + a tiny task run per connector type
    if not args.skip_smoke:
        for c in connectors:
            d = os.path.join(runs, "smoke", c)
            log = os.path.join(runs, "smoke.log")
            assert sh([py, "study/align.py", "--llm", args.llm, "--connector", c, "--out", d, "--steps", "5"] + ckpt +
                      (["--ltd_weight", str(args.align_ltd)] if args.align_ltd else []),
                      gpu=0, log=log) == 0, f"smoke align failed ({c})"
            assert sh([py, "study/train_task.py", "--llm", args.llm, "--connector", c, "--task_dir",
                       f"data/tasks/{tasks[0]}", "--align_ckpt", os.path.join(d, "connector.pt"),
                       "--run_dir", d, "--epochs", "1", "--max_train_samples", "32",
                       "--max_eval_samples", "64", "--save_adapter"] + ckpt +
                      (["--balanced_sampling"] if args.balanced_sampling else []) +
                      (["--attn_entropy_weight", str(args.attn_entropy_weight)] if args.attn_entropy_weight else []),
                      gpu=0, log=log) == 0, f"smoke train failed ({c})"
            smoke3 = {"dfr": ["--train_on", "val_balanced", "--epochs", "1"], "jtt": ["--train_on", "jtt", "--epochs", "1"],
                      "prompt": ["--epochs", "0", "--instruction", args.instruction]}
            for st in [x for x in args.stage3.split(",") if x]:
                assert sh([py, "study/train_task.py", "--llm", args.llm, "--connector", c, "--task_dir",
                           f"data/tasks/{tasks[0]}", "--init_from", os.path.join(d, "adapter.pt"),
                           "--run_dir", os.path.join(d, st), "--max_train_samples", "32",
                           "--max_eval_samples", "64"] + smoke3[st] + ckpt, gpu=0, log=log) == 0, f"smoke {st} failed ({c})"
        shutil.rmtree(os.path.join(runs, "smoke"), ignore_errors=True)

    pool = Pool(args.gpus, start, args.total_budget_min, args.job_budget_min)

    # 3. stage 1 alignment (one per connector, or one per connector and seed)
    def align_dir(c, seed):
        return os.path.join(runs, "align", c, f"s{seed}") if args.align_per_seed else os.path.join(runs, "align", c)

    align_jobs = []
    for seed in (range(args.seed_start, args.seeds) if args.align_per_seed else [0]):
        for c in connectors:
            d = align_dir(c, seed)
            os.makedirs(d, exist_ok=True)
            align_jobs.append((os.path.relpath(d, runs), [
                py, "study/align.py", "--llm", args.llm, "--connector", c, "--out", d,
                "--steps", str(args.align_steps), "--seed", str(seed)] + ckpt +
                (["--ltd_weight", str(args.align_ltd)] if args.align_ltd else []),
                os.path.join(d, "connector.pt"), os.path.join(d, "align.log")))
    pool.run(align_jobs, on_done=export)

    # 4. stage 2 task runs (seed-major order, so partial sessions still cover every config)
    task_jobs = []
    for seed in range(args.seed_start, args.seeds):
        for task in tasks:
            for c in connectors:
                d = os.path.join(runs, task, c, f"s{seed}")
                os.makedirs(d, exist_ok=True)
                base = [py, "study/train_task.py", "--llm", args.llm, "--connector", c,
                        "--task_dir", f"data/tasks/{task}", "--align_ckpt", os.path.join(align_dir(c, seed), "connector.pt"),
                        "--run_dir", d, "--seed", str(seed), "--epochs", str(args.epochs),
                        "--time_budget_min", str(args.job_budget_min - 10)] + ckpt + \
                       (["--freeze_llm"] if args.freeze_llm else []) + \
                       (["--balanced_sampling"] if args.balanced_sampling else []) + \
                       (["--connector_lr", str(args.connector_lr)] if args.connector_lr else []) + \
                       (["--lr", str(args.lora_lr)] if args.lora_lr else []) + \
                       (["--attn_entropy_weight", str(args.attn_entropy_weight)] if args.attn_entropy_weight else [])
                stages = [s for s in args.stage3.split(",") if s]
                if not stages:
                    task_jobs.append((f"{task}/{c}/s{seed}", base, os.path.join(d, "results.json"),
                                      os.path.join(d, "train.log")))
                    continue
                # base run (skipped if already done), then connector-only retraining from its adapter
                steps = [] if os.path.exists(os.path.join(d, "adapter.pt")) else [base + ["--save_adapter"]]
                mode = {"dfr": ["--train_on", "val_balanced", "--epochs", str(args.stage3_epochs)],
                        "jtt": ["--train_on", "jtt", "--epochs", str(args.stage3_epochs)],
                        "prompt": ["--epochs", "0", "--instruction", args.instruction]}
                for st in stages:
                    d3 = os.path.join(runs, task, f"{c}_{st}", f"s{seed}")
                    os.makedirs(d3, exist_ok=True)
                    steps.append([py, "study/train_task.py", "--llm", args.llm, "--connector", c,
                                  "--task_dir", f"data/tasks/{task}", "--init_from", os.path.join(d, "adapter.pt"),
                                  "--run_dir", d3, "--seed", str(seed)] + mode[st] + ckpt)
                # cost of the fix: caption loss of the stage-2 model and of each retrained connector
                cap = [py, "study/caption_eval.py", "--llm", args.llm, "--connector", c,
                       "--adapter", os.path.join(d, "adapter.pt"), "--mlp_pool", str(args.mlp_pool)]
                steps.append(cap + ["--out", os.path.join(d, "caption_eval.json")])
                for st in [x for x in stages if x in ("dfr", "jtt")]:
                    d3 = os.path.join(runs, task, f"{c}_{st}", f"s{seed}")
                    steps.append(cap + ["--connector_ckpt", os.path.join(d3, "connector.pt"),
                                        "--out", os.path.join(d3, "caption_eval.json")])
                chain = " && ".join(" ".join(shlex.quote(x) for x in cmd) for cmd in steps)
                last = os.path.join(runs, task, f"{c}_{stages[-1]}", f"s{seed}")
                task_jobs.append((f"{task}/{c}/s{seed}+{'+'.join(stages)}", ["bash", "-c", chain],
                                  os.path.join(last, "results.json"), os.path.join(d, "train.log")))
    pool.run(task_jobs, on_done=export)

    # 5. aggregate
    sh([py, "study/aggregate.py", "--runs_dir", runs, "--out", os.path.join(runs, "summary")])
    export("summary")
    print(f"Finished in {(time.time() - start) / 60:.0f} min. Failed: {pool.failures or 'none'}")


if __name__ == "__main__":
    main()
