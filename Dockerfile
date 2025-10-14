# Base CUDA image
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Bangkok

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    build-essential \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Set working directory
WORKDIR /workspace/her2-attention-classifier

# Copy your project folder
COPY . .

# Copy the exported Conda environment
COPY environment.yml .

# Accept Anaconda channel ToS before creating the environment
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda env create -f environment.yml

# Activate environment by default
SHELL ["conda", "run", "-n", "her2-class", "/bin/bash", "-c"]

# Start in bash for interactive dev
CMD ["/bin/bash"]
