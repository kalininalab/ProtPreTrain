# Use an official PyTorch base image
FROM pytorch/pytorch:latest
WORKDIR /app

COPY requirements.txt /app/
RUN apt-get update && apt-get install -y git gnupg2 gcc g++
RUN pip install --no-cache-dir -r requirements.txt
COPY ./step ./step
COPY finetune.py ./
COPY train.py ./

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "finetune.py"]
