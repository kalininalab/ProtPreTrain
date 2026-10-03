# Environment image for running STEP on the Saarland HPC (conduit; HTCondor docker universe).
#
# The image carries *only* the environment, installed from uv.lock - no code. The repo is bind-mounted from the
# shared filesystem at job time (hpc/job.sh puts it on PYTHONPATH), so a code change needs a `git pull` on the
# cluster, not a rebuild. Rebuild (hpc/build.sh) only when pyproject.toml / uv.lock change.

# `base`: torch brings its own CUDA 13 libraries as pip wheels. What this image is for is the
# NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES defaults that let the container runtime inject the driver -
# HTCondor submit files here cannot set `environment` (see hpc/README.md), so those have to come from the image.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

# Interpreter and venv live under /opt, world-readable: HTCondor runs the container as the submitting user's uid,
# which has no home or passwd entry inside the image.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Exactly the locked environment (torch 2.14.0+cu130 etc.), without the project itself (it is mounted).
COPY pyproject.toml uv.lock README.md .python-version /src/
RUN cd /src && uv sync --frozen --no-dev --no-install-project --no-cache \
    && rm -rf /src && chmod -R a+rX /opt/python /opt/venv
RUN python -c "import torch, torch_geometric, lightning, mlflow, foldcomp; print('torch', torch.__version__)"

WORKDIR /workspace
