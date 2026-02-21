FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /stt-app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcudnn8 \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        portaudio19-dev \
        libportaudio2 \
        libsndfile1 \
        ffmpeg \
        libglib2.0-0 \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN python3 -m pip install -r requirements.txt

# Download the model to models/kb-whisper-large (matches runtime path)
RUN mkdir -p models && \
    huggingface-cli download KBLab/kb-whisper-large \
        --local-dir models/kb-whisper-large \
        --local-dir-use-symlinks False \
        --revision standard && \
    huggingface-cli download KBLab/wav2vec2-large-voxrex-swedish \
        --local-dir models/wav2vec2-large-voxrex-swedish \
        --local-dir-use-symlinks False \
        --revision main

# Copy the rest of your application code
COPY ./src/ .

# Create output directory for WhisperX
RUN mkdir -p output

# fix error with whisperX
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true 
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["fastapi", "run", "./app/api_handler.py", "--port", "5080"]