#!/bin/zsh
# Task 2b batch, after P3 (= main-2k seed 0) has passed Gate 2. Order fixed by PLAN.md Task 2b:
# main-8k s0, main-32k s0, ctrl-8k-x4 s0, main-0 s0, fork trunk, fork-A, fork-B, then seed 1 runs.
# For every run: a 50-step check (loss must decrease; tokens/s recorded), the full run, Gate 2 on
# that run, then a commit of results/task2/. Stops at the first failure of any kind.
# Run under caffeinate -i.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
mkdir -p logs
note() { echo "$(date '+%F %T') $*" >> logs/queue.log; }

smoke() {  # name seed stream [extra args]; 50 steps, loss must decrease
  local name=$1 seed=$2 stream=$3; shift 3
  $PY train/train.py --task task2 --run_name smoke-$name --seed $seed --stream $stream \
      --no_ckpt --no_eval "$@" > logs/smoke-$name.log 2>&1
  $PY -c "
import json,sys; m=json.load(open('results/task2/smoke-$name/metrics.json'))
ok = m['steps_run'] == 50 and m['loss_last10_mean'] < m['loss_first10_mean']
print('smoke $name', m['loss_first10_mean'], '->', m['loss_last10_mean'], 'tok/s', round(m['tokens_per_second']), 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)" >> logs/queue.log
}

commit() {
  git add results/task2
  git commit -q -m "task 2: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
}

run() {  # name seed stream [extra args]
  local name=$1 seed=$2 stream=$3; shift 3
  note "start $name"
  $PY train/train.py --task task2 --run_name $name --seed $seed --stream $stream --log_every 100 "$@" \
      > logs/$name.log 2>&1
  note "done  $name"
}

gated() {  # name seed stream: smoke, run, gate, commit
  local name=$1 seed=$2 stream=$3
  smoke $name $seed $stream --stop_after 50
  run $name $seed $stream
  if ! $PY analysis/gate2.py --seed 0 --runs $name >> logs/queue.log 2>&1; then
    note "GATE 2 FAIL $name"; commit "$name, Gate 2 FAIL"; exit 1
  fi
  note "gate2 pass $name"
  commit "$name (Gate 2 pass)"
}

gated main-8k-s0     0 main-8k
gated main-32k-s0    0 main-32k
gated ctrl-8k-x4-s0  0 ctrl-8k-x4
gated main-0-s0      0 main-0

# Fork trunk: the first 20% of main-8k's steps, story tokens only (identical in both fork streams).
TRUNK=$($PY -c "import json;print(json.load(open('results/task2/streams/fork-A/metrics.json'))['trunk_steps'])")
CK=checkpoints/task2/fork-trunk-s0/step_${TRUNK}.pt
smoke fork-trunk-s0 0 fork-A --stop_after 50
run fork-trunk-s0 0 fork-A --stop_after $TRUNK --no_eval
test -f $CK
commit "fork trunk ($TRUNK steps)"
for F in A B; do
  smoke fork-$F-s0 0 fork-$F --init_from $CK --stop_after $((TRUNK + 50))
  run fork-$F-s0 0 fork-$F --init_from $CK
  if ! $PY analysis/gate2.py --seed 0 --runs fork-$F-s0 >> logs/queue.log 2>&1; then
    note "GATE 2 FAIL fork-$F-s0"; commit "fork-$F-s0, Gate 2 FAIL"; exit 1
  fi
  note "gate2 pass fork-$F-s0"
  commit "fork-$F-s0 (Gate 2 pass)"
done

gated main-2k-s1     1 main-2k
gated main-8k-s1     1 main-8k
gated main-32k-s1    1 main-32k
gated main-0-s1      1 main-0

$PY analysis/gate2.py --seed 0 >> logs/queue.log 2>&1 && note "GATE 2 PASS all runs" || note "GATE 2 FAIL (all runs)"
commit "all runs done, Gate 2 over all runs"
note "all done"
