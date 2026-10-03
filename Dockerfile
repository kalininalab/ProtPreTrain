# Official PyTorch base image, pinned to the torch 2.14.0+cu126 stack in uv.lock
FROM pytorch/pytorch:2.14.0-cuda12.6-cudnn9-runtime
WORKDIR /app

# uv-managed install (pyproject.toml + uv.lock are the single source of truth)
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/
COPY pyproject.toml uv.lock /app/
RUN apt-get update && apt-get install -y git gnupg2 gcc g++ \
    && uv sync --frozen --no-group dev --no-install-project

COPY ./step ./step
COPY finetune.py ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "finetune.py"]
