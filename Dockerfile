# Use an official PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app
RUN apt-get update && apt-get install -y git gnupg2 gcc g++

# Copy the entire contents of the 'step/' folder into the container's working directory
ADD step/ /app/step
COPY finetune.py /app/
COPY train.py /app/

# Install any additional dependencies you might need for your project
# For example, if you have a 'requirements.txt' file, you can use it to install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Set an environment variable to disable buffering of Python output to allow logs to appear immediately
ENV PYTHONUNBUFFERED=1
