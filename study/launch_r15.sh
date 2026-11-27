#!/usr/bin/env bash
# Round 15: wait for the weekly Kaggle GPU reset (Sat 28 Nov 00:00 UTC), push each kernel on its account, then start
# a watcher that resumes it on the target account if the 12 h session ends early.
#   job          source      target
set -u
cd "$(dirname "$0")/.."
JOBS="r15-celeba-a:naveen:shriniketh
r15-celeba-b:shriniketh:naveen
r15-9b-wb-a:dad:jeyadevr
r15-9b-wb-b:mom:jaden
r15-lora-wm:gp1:jeyadevr
r15-frac-wm:gp2:mom
r15-eps1d-wm:jaden:gp2"
declare -A OWNER=([dad]=rajeshkumar1974 [gp1]=mrronjakowski [gp2]=selvamani148 [jaden]=jakeljaden
                  [mom]=umamaheswari1981 [naveen]=naveenvaidani [shriniketh]=shrinikethr)
RESET=$(date -d "2026-11-28 00:05 UTC" +%s)
log() { echo "$(date +%H:%M) $*"; }
while [ "$(date +%s)" -lt "$RESET" ]; do sleep 300; done
log "reset reached"
for line in $JOBS; do
  IFS=: read -r job src tgt <<< "$line"
  for try in $(seq 1 36); do  # up to 6 h of retries if both GPU slots are busy
    out=$(KAGGLE_API_TOKEN=$(cat ~/.kaggle_${src}_token) timeout 300 .venv/bin/kaggle kernels push \
          -p study/kaggle_kernels/$job 2>&1 | grep -v Warning)
    log "$job on $src (try $try): $out"
    echo "$out" | grep -q "successfully pushed" && break
    sleep 600
  done
  if echo "$out" | grep -q "successfully pushed"; then
    sleep 60
    setsid systemd-inhibit --what=sleep:idle --who=kaggle-watch --why=r15 .venv/bin/python study/kaggle_resume.py \
      --src "$src" --kernel "${OWNER[$src]}/vlm-shortcut-$job" --dir "$job" --target "$tgt" \
      > "kaggle_out/watch_$job.log" 2>&1 < /dev/null &
    log "$job watcher started (resume target $tgt)"
  else
    log "$job PUSH FAILED after retries"
  fi
done
log "all pushes attempted"
wait
