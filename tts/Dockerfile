
# Dockerfile for TTS FastAPI app with GPU support and training capabilities
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /tts-app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    ffmpeg git build-essential \
    libsndfile1 portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy requirements files
COPY app/requirements.txt ./requirements_tts.txt
COPY chatterbox/requirements.txt ./requirements_chatterbox.txt

# Install Python dependencies
RUN pip install --no-build-isolation "numpy<2.0" cython setuptools wheel && \
    pip install --no-build-isolation "pkuseg==0.0.25" && \
    pip install -r requirements_tts.txt && \
    pip install -r requirements_chatterbox.txt && \
    pip install fastapi uvicorn jiwer torchaudio safetensors huggingface_hub

# Copy chatterbox code for training
COPY chatterbox/ /tts-app/chatterbox/

# Copy training service
COPY src/ /tts-app/src/

# Copy app code
COPY app/*.py /tts-app/app/

# Create output directories
RUN mkdir -p /tts-app/models/checkpoints /tts-app/output /tts-app/data

# Expose port
EXPOSE 8000

# Start FastAPI server
WORKDIR /tts-app/app
CMD ["uvicorn", "tts_api:app", "--host", "0.0.0.0", "--port", "8000"]
