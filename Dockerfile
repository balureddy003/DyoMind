# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git python3 python3-pip wget curl cmake build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python and vLLM
RUN pip3 install --upgrade pip && \
    pip3 install "vllm[openai]" torch torchvision

# HuggingFace cache support
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Expose OpenAI-compatible API server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "mistralai/Mistral-7B-Instruct-v0.2", "--port", "8000"]