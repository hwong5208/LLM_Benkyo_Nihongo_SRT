# Use an official NVIDIA CUDA image as a parent image
# We need CUDA and cuDNN for Faster-Whisper
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set non-interactive to avoid apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies including Python and FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies natively
# Depending on requirements, we can copy them first to leverage Docker cache
COPY requirements.txt .

# We need to ensure we install torch with CUDA support, and faster-whisper
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir python-dotenv

# Copy the current directory contents into the container at /app
COPY pipeline_cuda.py .
COPY .env.example .

# Add a volume for input/output files and workspace caching
VOLUME ["/data"]

# Example usage:
# docker run --gpus all -v /local/dir:/data llm_srt python3 pipeline_cuda.py --input /data/video.mp4 --output-dir /data/output

ENTRYPOINT ["python3", "pipeline_cuda.py"]
