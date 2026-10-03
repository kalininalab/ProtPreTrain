#!/usr/bin/env bash
# One-time setup on the submit node (conduit / conduit2). Light enough for its 2 GB memory limit: git and mkdir only.
#
#   ssh conduit
#   git clone -b hpc git@github.com:kalininalab/ProtPreTrain.git /tmp/step-bootstrap
#   bash /tmp/step-bootstrap/hpc/setup_cluster.sh
#
# Safe to re-run: existing directories are kept and an existing clone is fast-forwarded.
set -euo pipefail

ROOT="${STEP_ROOT:-/home/$USER/step}"
# SSH, not HTTPS: the submit nodes fail the TLS handshake with github.com
REPO_URL="${STEP_REPO_URL:-git@github.com:kalininalab/ProtPreTrain.git}"
BRANCH="${STEP_BRANCH:-hpc}"

mkdir -p "$ROOT"/{data,runs,bench,cache,runlogs}

if [[ -d "$ROOT/ProtPreTrain/.git" ]]; then
    echo "==> $ROOT/ProtPreTrain already a clone, pulling"
    git -C "$ROOT/ProtPreTrain" pull --ff-only
else
    echo "==> cloning $REPO_URL ($BRANCH) into $ROOT/ProtPreTrain"
    git clone --branch "$BRANCH" "$REPO_URL" "$ROOT/ProtPreTrain"
fi

# The code reads data/<db> relative to the repo; keep the bulky data outside the clone
ln -sfn "$ROOT/data" "$ROOT/ProtPreTrain/data"

cat <<MSG

Ready: $ROOT
  repo     $ROOT/ProtPreTrain   (submit from here)
  data     $ROOT/data           (foldcomp databases; downstream raw files go in data/<task>/raw)
  runs     $ROOT/runs           (checkpoints, per-job MLflow stores, summaries)
  bench    $ROOT/bench          (scripts/benchmark.py --out)
  runlogs  $ROOT/runlogs        (condor .out/.err/.log)

Next: cd $ROOT/ProtPreTrain && condor_submit -a 'runfile=hpc/runs/smoke.txt' hpc/gpu.sub
If ROOT is not /home/s8ilsena/step, edit \`root\` in hpc/common.sub and the paths in hpc/runs/*.txt.
MSG
