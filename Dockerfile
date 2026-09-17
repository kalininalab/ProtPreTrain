# Use an official PyTorch base image (pinned: matches the verified torch 2.5.1+cu121 stack)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
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
