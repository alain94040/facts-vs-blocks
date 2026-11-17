#!/bin/zsh
# Repeat tasks 3, 4 and 4b on the main model, one after the other, committing after each.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
commit() {
  git add repeat results/repeat
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}
echo "$(date '+%T') task 3"
$PY repeat/task3.py --model main --seed 0 > logs/repeat_task3_main.log 2>&1
commit "repeat task 3: dose (partial and gain) for the best layer and block (Qwen2.5-7B, SEARCH)"
echo "$(date '+%T') task 4"
$PY repeat/task4.py --model main --seed 0 > logs/repeat_task4_main.log 2>&1
commit "repeat task 4: swap(i, j) interchangeability on 50 TEXT SEARCH chunks (Qwen2.5-7B)"
echo "$(date '+%T') task 4b"
$PY repeat/task4b.py --model main --seed 0 > logs/repeat_task4b_main.log 2>&1 || { commit "repeat task 4b: positive control or run failed (see STOP.md / log)"; exit 1; }
commit "repeat task 4b: weight comparison of layer pairs (Qwen2.5-7B), positive control, link to task 4"
echo "$(date '+%T') all done"
