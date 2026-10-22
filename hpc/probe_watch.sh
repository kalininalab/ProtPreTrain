#!/bin/bash
# Probe a pretraining run's kept checkpoints while it trains. Runs on the submit node as a scheduler-universe job
# (hpc/watch.sub); every round it queues one GPU job per new kept checkpoint (benchmark.py dynamics probe
# --per_checkpoint), each of which probes that checkpoint and refreshes <out>/dynamics/dynamics.{csv,md,png}.
#
#   probe_watch.sh <out> <config> <interval seconds> [dynamics probe arguments...]
#
# Exits once every run of <config> has written pretrain.json and nothing is left to queue.
set -uo pipefail

OUT="${1:?usage: probe_watch.sh <out> <config> <interval> [args...]}"
CONFIG="${2:?config}"
INTERVAL="${3:?interval}"
shift 3

cd "$(dirname "${BASH_SOURCE[0]}")/.."
QUEUE="hpc/runs/watch_$(basename "$OUT")_${CONFIG//[^A-Za-z0-9_-]/}.txt"

while true; do
    # read before queueing: a run that finished before this round has had its last checkpoint seen by it
    runs=("$OUT"/dynamics/"$CONFIG"_s*/)
    finished=("$OUT"/dynamics/"$CONFIG"_s*/pretrain.json)
    echo "=== $(date -Is): ${#finished[@]} of ${#runs[@]} runs finished"
    python3 scripts/benchmark.py --out "$OUT" dynamics probe --config "$CONFIG" --per_checkpoint --condor "$QUEUE" "$@"
    if [[ -s "$QUEUE" ]]; then
        # markers only once the jobs are really queued, so a failed submit is retried next round
        if condor_submit -a "runfile=$PWD/$QUEUE" hpc/gpu.sub; then
            xargs -r touch < "$QUEUE.markers"
        fi
        rm -f "$QUEUE" "$QUEUE.markers"
    elif [[ -e "${finished[0]}" && ${#finished[@]} -eq ${#runs[@]} ]]; then
        echo "every run finished and every kept checkpoint is queued"
        exit 0
    fi
    sleep "$INTERVAL"
done
