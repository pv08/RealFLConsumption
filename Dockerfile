FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

COPY . .

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
