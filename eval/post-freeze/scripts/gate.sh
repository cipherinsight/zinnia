#!/usr/bin/env bash
# A queue job that holds its slot until the scaling-npb driver opens the gate: the DS-1000 timing
# must not start while the NPBench/Pythran count jobs still run on CPUs 32-63 (siblings of 0-31).
# Usage (job line): timed bash gate.sh   (the slot's CPU list is appended and ignored)
RUN=/home/zhantong/zinnia-eval-run
until [[ -e $RUN/scaling-npb/gate-open ]]; do sleep 30; done
echo "gate passed $(date -Is)"
