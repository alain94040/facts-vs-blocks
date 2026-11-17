#!/bin/zsh
# Repeat tasks 4c and 5 on the main model, committing after each.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
commit() {
  git add repeat results/repeat
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}
echo "$(date '+%T') task 4c"
$PY repeat/task4c.py --model main --seed 0 > logs/repeat_task4c_main.log 2>&1
commit "repeat task 4c: covariance-matched null and mean-removed weight comparison (Qwen2.5-7B)"
echo "$(date '+%T') task 5"
$PY repeat/task5.py --model main --seed 0 > logs/repeat_task5_main.log 2>&1
commit "repeat task 5: selected configurations on the REPORT halves (Qwen2.5-7B)"
echo "$(date '+%T') all done"
