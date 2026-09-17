# Use an official PyTorch base image (pinned: matches the verified torch 2.5.1+cu121 stack)
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y git gnupg2 gcc g++
RUN pip install --no-cache-dir -r requirements.txt
COPY ./step ./step
COPY finetune.py ./

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "finetune.py"]
