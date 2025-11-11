# Multi-stage Dockerfile for VLA-GR Navigation Framework

# Stage 1: Base image with dependencies
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Install PyTorch and dependencies
FROM base AS builder

WORKDIR /tmp

# Install PyTorch with CUDA support
RUN pip3 install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Stage 3: Final application image
FROM builder AS app

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install the package
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data /app/outputs

# Set up entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose ports (for visualization, API, etc.)
EXPOSE 8000 6006

# Set default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python3", "demo.py"]

# Stage 4: Development image with additional tools
FROM app AS dev

# Install development dependencies
COPY requirements-dev.txt .
RUN pip3 install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
