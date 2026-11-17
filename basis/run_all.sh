#!/bin/zsh
# Basis tasks 1 and 2 on 7B, then task 3 on 14B, committing after each task.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
commit() {
  git add basis results/basis
  git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
}
echo "$(date '+%T') task 1"
$PY basis/task1.py --model main --seed 0 > logs/basis_task1.log 2>&1
commit "basis task 1: kept energy of own, shared and rotated-null bases per zone (Qwen2.5-7B)"
echo "$(date '+%T') task 2"
$PY basis/task2.py --model main --seed 0 > logs/basis_task2.log 2>&1
commit "basis task 2: family projected on own, shared and null bases, SEARCH evaluation (Qwen2.5-7B)"
echo "$(date '+%T') task 3"
$PY basis/task3_outliers.py --seed 0 > logs/basis_task3_outliers.log 2>&1
$PY basis/task1.py --model confirm --seed 0 > logs/basis_task3_energy.log 2>&1
$PY basis/task2.py --model confirm --configs base own:4 shared:4 --seed 0 > logs/basis_task3_edits.log 2>&1
commit "basis task 3: energies and short edit check on Qwen2.5-14B"
echo "$(date '+%T') all done"
