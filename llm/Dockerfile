# Dockerfile for LLM service with vLLM and Qwen2.5-8B
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /llm-app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install vLLM and dependencies
# vllm 0.6.3.post1 requires torch==2.4.0 and supports Qwen2ForCausalLM
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    vllm==0.6.3.post1 \
    fastapi==0.115.5 \
    uvicorn[standard]==0.32.1 \
    pydantic==2.10.3 \
    transformers==4.48.0

# Copy application code
COPY app/ /llm-app/app/

# Create model cache directory
RUN mkdir -p /llm-app/models

# Expose port
EXPOSE 8001

# Start FastAPI server
WORKDIR /llm-app/app
CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8001"]
