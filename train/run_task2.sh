#!/bin/zsh
# Remaining Task 2 runs, sequential. main-2k seed 0 is run first, alone, before this script.
# Stops on the first error. Run under caffeinate -i.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
mkdir -p logs
run() {  # run_name seed stream [extra args]
  local name=$1 seed=$2 stream=$3; shift 3
  echo "$(date '+%F %T') start $name" >> logs/queue.log
  $PY train/train.py --task task2 --run_name $name --seed $seed --stream $stream --log_every 100 "$@" > logs/$name.log 2>&1
  echo "$(date '+%F %T') done  $name" >> logs/queue.log
}
run main-2k-s1    1 main-2k
run main-8k-s0    0 main-8k
run main-8k-s1    1 main-8k
run main-32k-s0   0 main-32k
run main-32k-s1   1 main-32k
run ctrl-8k-x4-s0 0 ctrl-8k-x4
run main-0-s0     0 main-0
run main-0-s1     1 main-0
# Fork trunk: first 20% of main-8k's steps, story tokens only (identical in both fork streams).
TRUNK_STEPS=$($PY -c "import json;print(json.load(open('data/cache/streams/fork-A.json'))['trunk_steps'])")
run fork-trunk-s0 0 fork-A --stop_after $TRUNK_STEPS --no_eval
run fork-A-s0     0 fork-A --init_from checkpoints/task2/fork-trunk-s0/step_${TRUNK_STEPS}.pt
run fork-B-s0     0 fork-B --init_from checkpoints/task2/fork-trunk-s0/step_${TRUNK_STEPS}.pt
$PY analysis/gate2.py --seed 0 > logs/gate2.log 2>&1 || true
echo "$(date '+%F %T') all done" >> logs/queue.log
