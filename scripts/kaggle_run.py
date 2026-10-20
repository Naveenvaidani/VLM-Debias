"""
Run the whole Waterbirds experiment on a multi-GPU machine (built for Kaggle's 2x T4).

Steps: prepare data -> cache frozen vision features -> smoke test -> train + evaluate every
(connector, sampling, seed) configuration, one job per GPU in parallel -> aggregate over seeds.

T4 GPUs have no bf16, so everything runs in fp16 with loss scaling. fp16 numerics differ from the
bf16 laptop runs, so compare architectures *within* one environment, not across environments.

    python scripts/kaggle_run.py --seeds 3 --llm Qwen/Qwen3.5-4B
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = [("mlp_prefix", False), ("bottleneck", False), ("mlp_prefix", True), ("bottleneck", True)]


def run_name(mode: str, balanced: bool, seed: int) -> str:
    return f"{mode}{'_balanced' if balanced else ''}" + (f"_s{seed}" if seed else "")


def sh(cmd, gpu=None, log=None):
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out = open(log, "a") if log else None
    print("$", " ".join(cmd), f"[gpu {gpu}]" if gpu is not None else "", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)
    if out:
        out.close()
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vision_model", default="google/siglip-base-patch16-224")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--time_budget_min", type=float, default=75, help="per-run training budget")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="needed on GPUs with <= 8 GB")
    args = parser.parse_args()
    py = sys.executable
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    start = time.time()

    # 1. Data and cached vision features (once)
    if not os.path.exists(os.path.join(ROOT, "data", "waterbirds", "metadata.csv")):
        assert sh([py, "scripts/prepare_waterbirds.py"]) == 0, "data preparation failed"
    if not os.path.exists(os.path.join(ROOT, "data", "siglip_patch_features.npy")):
        assert sh([py, "scripts/cache_siglip_features.py", "--vision_model", args.vision_model], gpu=0) == 0, \
            "feature caching failed"

    ckpt = ["--gradient_checkpointing"] if args.gradient_checkpointing else []

    # 2. Fail fast: a tiny training run for each connector before committing hours of GPU time
    if not args.skip_smoke:
        for mode in ("bottleneck", "mlp_prefix"):
            code = sh([py, "train.py", "--llm", args.llm, "--mode", mode, "--dtype", "fp16", "--epochs", "1",
                       "--max_train_samples", "160", "--batch_size", str(args.batch_size),
                       "--run_name", f"smoke_{mode}"] + ckpt, gpu=0, log=os.path.join(ROOT, "runs", "smoke.log"))
            assert code == 0, f"smoke test failed for {mode}: see runs/smoke.log"

    # 3. One job per (config, seed); each GPU runs jobs from the shared queue
    jobs = queue.Queue()
    for seed in range(args.seeds):
        for mode, balanced in CONFIGS:
            jobs.put((mode, balanced, seed))
    failures = []

    def worker(gpu: int):
        while True:
            try:
                mode, balanced, seed = jobs.get_nowait()
            except queue.Empty:
                return
            name = run_name(mode, balanced, seed)
            run_dir = os.path.join(ROOT, "runs", name)
            if os.path.exists(os.path.join(run_dir, "test_results.pt")):
                print(f"skip {name}", flush=True)
                continue
            log = os.path.join(ROOT, "runs", f"{name}.log")
            train_cmd = [py, "train.py", "--llm", args.llm, "--mode", mode, "--dtype", "fp16",
                         "--epochs", str(args.epochs), "--patience", str(args.patience),
                         "--batch_size", str(args.batch_size), "--time_budget_min", str(args.time_budget_min),
                         "--seed", str(seed), "--run_name", name]
            train_cmd += ckpt
            if balanced:
                train_cmd.append("--balanced")
            ok = sh(train_cmd, gpu=gpu, log=log) == 0 and sh([py, "evaluate.py", "--single", name], gpu=gpu, log=log) == 0
            if not ok:
                failures.append(name)
            print(f"[{(time.time() - start) / 60:.0f} min] {'done' if ok else 'FAILED'} {name} (gpu {gpu})", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(args.gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4. Figures for seed 0, then mean +- std over seeds
    sh([py, "evaluate.py"], gpu=0)
    sh([py, "scripts/aggregate_seeds.py"])
    print(f"Finished in {(time.time() - start) / 60:.0f} min. Failed runs: {failures or 'none'}")


if __name__ == "__main__":
    main()
