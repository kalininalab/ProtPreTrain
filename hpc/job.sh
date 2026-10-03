#!/bin/bash
# Wrapper every STEP condor job runs: set up the environment, then exec the real command.
#
#   arguments = <mlflow store | -> <command...>
#   e.g.        /home/s8ilsena/step/runs/demo/mlflow.db python train.py --dataset afdb_rep_v4 ...
#
# The first argument is this job's MLflow SQLite store ("-" for jobs that log nothing). One store per job, because
# SQLite locking is not safe across nodes on NFS; scripts/aggregate_results.py --db_glob reads them all and
# scripts/merge_mlflow.py merges them for `mlflow ui`. A relative path is taken relative to the cluster root.
#
# Environment variables are exported here because submit files cannot use `environment =` on this cluster (it
# would stop JOB_TRANSFORM_AddHomeEnv from injecting HOME, which +WantGPUHomeMounted depends on).
set -euo pipefail

HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HPC_DIR")"
ROOT="$(dirname "$REPO")"
cd "$REPO"

STORE="${1:?usage: job.sh <mlflow store | -> <command...>}"
shift

# The repo is mounted, not installed in the image
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# The container runs as our uid, which has no passwd entry in the image: getpass.getuser() (used by torch and
# others) checks these before the password database
export USER="${USER:-step}"
export LOGNAME="$USER"

# Caches on the shared filesystem: the container filesystem is thrown away after every job
export TORCH_HOME="$ROOT/cache/torch"
export HF_HOME="$ROOT/cache/hf"
export MPLCONFIGDIR="$ROOT/cache/mpl"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

if [[ "$STORE" != "-" ]]; then
    [[ "$STORE" == /* ]] || STORE="$ROOT/$STORE"
    mkdir -p "$(dirname "$STORE")"
    export MLFLOW_TRACKING_URI="sqlite:///$STORE"
fi
export MLFLOW_DISABLE_AGENT_HINT=1

# Condor captures stdout to a file
export PYTHONUNBUFFERED=1
export TQDM_MININTERVAL=30

# Torch otherwise sizes its thread pools to the whole machine, not the cores we requested
THREADS="${_CONDOR_REQUEST_CPUS:-${OMP_NUM_THREADS:-4}}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "=== $(date -Is) on $(hostname) ==="
echo "repo:    $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "store:   ${MLFLOW_TRACKING_URI:-none}"
echo "threads: $THREADS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "gpu:     none"
echo "cmd:     $*"
echo "=========================================="

exec "$@"
