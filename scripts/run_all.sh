#!/usr/bin/env bash
# Train the four Waterbirds runs (baseline / bottleneck x ERM / group-balanced), then evaluate.
# Usage: scripts/run_all.sh <llm> <per-run time budget in minutes>
set -euo pipefail
LLM=${1:-Qwen/Qwen3.5-4B}
BUDGET=${2:-0}
PY=${PY:-.venv/bin/python}

for mode in mlp_prefix bottleneck; do
  for balanced in "" "--balanced"; do
    name="${mode}${balanced:+_balanced}"
    if [ -f "runs/${name}/DONE" ]; then echo "skip ${name}"; continue; fi
    "$PY" train.py --llm "$LLM" --mode "$mode" $balanced --epochs 8 --patience 2 --gradient_checkpointing \
      --time_budget_min "$BUDGET" --run_name "$name"
    touch "runs/${name}/DONE"
  done
done
"$PY" evaluate.py
