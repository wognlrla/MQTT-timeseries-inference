#!/bin/sh
cd /srv/team/task6
/srv/team/venv/bin/python run.py > run.log 2>&1 || { echo RUN_FAILED >> run.log; exit 1; }
: > cost.jsonl
for m in gru chronos-bolt-tiny chronos-bolt-small chronos-bolt-base; do
  /srv/team/venv/bin/python cost.py $m 288 2>>cost.err | tail -1 >> cost.jsonl
done
echo PIPELINE_DONE >> run.log
