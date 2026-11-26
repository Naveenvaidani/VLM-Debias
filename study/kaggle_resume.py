"""Watch a Kaggle study kernel; if it stops before finishing, resume it on another account.

Steps, all through the Kaggle API:
  1. wait until the source kernel reaches a terminal status
  2. download its output (results, logs and aligned connectors) to kaggle_out/<dir>
  3. if the session finished every job, stop; otherwise
  4. upload the exported results as a private dataset on the target account
  5. push a copy of the kernel to the target account that runs the same command with --resume_from,
     so finished jobs and aligned connectors are skipped and only the missing jobs run

Accounts: a name maps to ~/.kaggle_<name>_token (KAGGLE_API_TOKEN) or, for "jeyadevr", the legacy
~/Downloads/kaggle.json copied to a temporary config dir.

    python study/kaggle_resume.py --src dad --kernel rajeshkumar1974/vlm-shortcut-r13-lr300 \
        --dir r13-lr300 --target naveen
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = os.path.join(ROOT, ".venv", "bin", "kaggle")
KERNELS = os.path.join(ROOT, "study", "kaggle_kernels")
USERS = {"dad": "rajeshkumar1974", "gp1": "mrronjakowski", "gp2": "selvamani148", "jaden": "jakeljaden",
         "mom": "umamaheswari1981", "naveen": "naveenvaidani", "shriniketh": "shrinikethr",
         "jeyadevr": "jeyadevr"}


def log(msg):
    print(time.strftime("%H:%M"), msg, flush=True)


def kaggle(account, *args, timeout=600):
    env = dict(os.environ)
    tmp = None
    if account == "jeyadevr":
        tmp = tempfile.mkdtemp()
        shutil.copy(os.path.expanduser("~/Downloads/kaggle.json"), os.path.join(tmp, "kaggle.json"))
        os.chmod(os.path.join(tmp, "kaggle.json"), 0o600)
        env.pop("KAGGLE_API_TOKEN", None)
        env.update(KAGGLE_CONFIG_DIR=tmp, HOME=tmp)
    else:
        with open(os.path.expanduser(f"~/.kaggle_{account}_token")) as f:
            env["KAGGLE_API_TOKEN"] = f.read().strip()
    try:
        out = subprocess.run([KAGGLE, *args], env=env, capture_output=True, text=True, timeout=timeout)
        return "\n".join(line for line in (out.stdout + out.stderr).splitlines() if "outdated" not in line)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def status(account, ref):
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", kaggle(account, "kernels", "status", ref, timeout=90))
    return m.group(1) if m else "UNKNOWN"


def wait(account, ref, poll):
    while True:
        s = status(account, ref)
        if s in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            return s
        time.sleep(poll)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="account running the kernel")
    p.add_argument("--kernel", required=True, help="owner/slug of the running kernel")
    p.add_argument("--dir", required=True, help="kernel folder under study/kaggle_kernels")
    p.add_argument("--target", required=True, help="account to resume on")
    p.add_argument("--poll", type=int, default=600)
    p.add_argument("--dry_run", action="store_true", help="download and check only, do not upload or push")
    p.add_argument("--force_resume", action="store_true", help="resume even if the session finished (testing)")
    p.add_argument("--kernel_dir", default=None, help="kernel folder if different from --dir")
    p.add_argument("--suffix", default="resume", help="suffix for the resumed kernel and dataset slugs")
    a = p.parse_args()

    slug = a.kernel.split("/")[1]
    s = wait(a.src, a.kernel, a.poll)
    log(f"{a.kernel} ended: {s}")

    out = os.path.join(ROOT, "kaggle_out", a.dir)
    download(a.src, a.kernel, out)
    results = os.path.join(out, "results")
    done = [d for d, _, f in os.walk(results) if "results.json" in f]
    logtxt = ""
    for name in os.listdir(out):
        if name.endswith(".log"):
            logtxt += open(os.path.join(out, name), errors="ignore").read()
    skipped = logtxt.count("not enough session time")
    log(f"downloaded: {len(done)} finished runs, {skipped} jobs skipped for time")
    if s == "COMPLETE" and skipped == 0 and not a.force_resume:
        log("nothing to resume")
        return
    have = os.path.isdir(results) and any(f.endswith("connector.pt") for _, _, fs in os.walk(results) for f in fs)
    if not have:
        log("session kept no usable output; rerunning the whole job on " + a.target)
    if a.dry_run:
        log("dry run: would resume on " + a.target)
        return

    # 4. private dataset holding the exported results as one archive
    owner = USERS[a.target]
    ds_slug = f"{slug}-{a.suffix}-data"[:50]
    if have:
        ds = tempfile.mkdtemp()
        with tarfile.open(os.path.join(ds, "resume.tar.gz"), "w:gz") as t:
            t.add(results, arcname="results")
        json.dump({"title": ds_slug, "id": f"{owner}/{ds_slug}", "licenses": [{"name": "CC0-1.0"}]},
                  open(os.path.join(ds, "dataset-metadata.json"), "w"))
        log(kaggle(a.target, "datasets", "create", "-p", ds, timeout=3600)[-300:])
        shutil.rmtree(ds, ignore_errors=True)
        st = ""
        for _ in range(120):
            st = kaggle(a.target, "datasets", "status", f"{owner}/{ds_slug}", timeout=90).lower()
            if "ready" in st:
                break
            time.sleep(30)
        log(f"dataset {owner}/{ds_slug}: {st.strip()}")

    # 5. kernel copy on the target account with --resume_from
    src_dir = os.path.join(KERNELS, a.kernel_dir or a.dir)
    dst_dir = os.path.join(KERNELS, f"{a.kernel_dir or a.dir}-{a.suffix}")
    shutil.rmtree(dst_dir, ignore_errors=True)
    os.makedirs(dst_dir)
    meta = json.load(open(os.path.join(src_dir, "kernel-metadata.json")))
    code_file = meta["code_file"]
    new_slug = f"{slug}-{a.suffix}"
    meta.update(id=f"{owner}/{new_slug}", title=new_slug.replace("-", " "), code_file=f"{new_slug}.py",
                kernel_sources=[f"{owner}/vlm-shortcut-data-prep"],
                dataset_sources=[f"{owner}/{ds_slug}"] if have else [])
    json.dump(meta, open(os.path.join(dst_dir, "kernel-metadata.json"), "w"))
    code = open(os.path.join(src_dir, code_file)).read()
    resume = '''
import glob, tarfile
RES = None
for _t in glob.glob("/kaggle/input/**/resume.tar*", recursive=True):
    tarfile.open(_t).extractall("/kaggle/tmp/resume")
    RES = "/kaggle/tmp/resume/results"
if RES is None:
    _hits = glob.glob("/kaggle/input/**/results/*/align", recursive=True)
    RES = os.path.dirname(os.path.dirname(_hits[0])) if _hits else None
print("resume from", RES, flush=True)
RESUME = f" --resume_from {RES}" if RES else ""
'''
    lines = code.splitlines()
    first_run = next(i for i, l in enumerate(lines) if l.startswith("run(base"))
    lines.insert(first_run, resume)
    code = "\n".join(lines) + "\n"
    code = re.sub(r'(run\(base \+ "[^"]*")', r'\1 + RESUME', code)
    open(os.path.join(dst_dir, f"{new_slug}.py"), "w").write(code)
    prep = f"{owner}/vlm-shortcut-data-prep"
    if wait(a.target, prep, 120) != "COMPLETE":
        log(f"data prep {prep} did not complete; kernel not pushed")
        return
    msg = kaggle(a.target, "kernels", "push", "-p", dst_dir, timeout=600)
    log(msg)
    if "successfully pushed" not in msg:
        log(f"PUSH FAILED for {owner}/{new_slug}")
        return 1
    log(f"pushed {owner}/{new_slug}; waiting for it")
    s2 = wait(a.target, f"{owner}/{new_slug}", a.poll)
    log(f"{owner}/{new_slug} ended: {s2}")
    out2 = os.path.join(ROOT, "kaggle_out", f"{a.dir}-{a.suffix}")
    download(a.target, f"{owner}/{new_slug}", out2)
    done2 = [d for d, _, f in os.walk(os.path.join(out2, "results")) if "results.json" in f]
    log(f"resumed session: {len(done2)} finished runs in {out2}")


def download(account, ref, out):
    os.makedirs(out, exist_ok=True)
    for attempt in range(5):
        msg = kaggle(account, "kernels", "output", ref, "-p", out, "--force",
                     "--file-pattern", r".*\.(json|csv|txt|log|npz|md|pt)$", timeout=3600)
        log(msg[-200:])
        if "TIMEOUT" not in msg and "rror" not in msg:
            return
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
