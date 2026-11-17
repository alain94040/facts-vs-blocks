#!/bin/zsh
# Task 4b batch: 50-step check on fork-A (F-all), then the starting model and the six variants.
set -e
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
$PY train/finetune.py --init fork-A-s0 --variant F-all --stream ft-B1k --seed 0 --run_name smoke-F-all \
    --stop_after 50 --no_eval --log_every 10 > logs/ft-smoke.log 2>&1
$PY -c "
import json,sys; m=json.load(open('results/task4b/smoke-F-all/metrics.json'))
ok = m['steps_run'] == 50 and m['loss_last10_mean'] < m['loss_first10_mean']
print('smoke', m['loss_first10_mean'], '->', m['loss_last10_mean'], 'tok/s', round(m['tokens_per_second']), 'OK' if ok else 'FAIL')
sys.exit(0 if ok else 1)"
for v in none F-all F-attn F-mlp F-mlp34 F-L0 F-emb; do
  echo "$(date '+%T') start $v"
  $PY train/finetune.py --init fork-A-s0 --variant $v --stream ft-B1k --seed 0 > logs/ft-$v.log 2>&1
done
echo "$(date '+%T') all done"
