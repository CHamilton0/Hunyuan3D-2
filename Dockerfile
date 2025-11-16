# Use Python 3.10 with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set timezone and disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies including OpenGL libraries for nvdiffrast
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libegl1-mesa \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tzdata \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA architecture for compilation (supported by CUDA 12.1)
# RTX 5090 requires compute capability 8.9
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA="1"
ENV NVCC_GENCODE="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90"

# Set up display for headless OpenGL (required for nvdiffrast)
ENV DISPLAY=:99
ENV PYOPENGL_PLATFORM=egl

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Fix version compatibility issues first
RUN pip install "numpy<2.0" \
    "torch>=2.2.0" \
    "transformers==4.48.0" \
    "diffusers==0.30.3"

# Install remaining requirements (excluding conflicting packages)
RUN pip install -r requirements.txt

# Install nvdiffrast from source (not available on PyPI)
RUN git clone https://github.com/NVlabs/nvdiffrast.git && \
    cd nvdiffrast && \
    pip install . && \
    cd .. && \
    rm -rf nvdiffrast

# Copy the entire project
COPY . .

# Install the package
RUN pip install -e . 

# Build and install custom extensions properly with correct CUDA architecture
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1

# Clean any previous builds
RUN cd hy3dgen/texgen/custom_rasterizer && \
    rm -rf build/ dist/ *.egg-info/ && \
    python3 setup.py clean --all && \
    python3 setup.py install && \
    cd ../../..

RUN cd hy3dgen/texgen/differentiable_renderer && \
    rm -rf build/ dist/ *.egg-info/ && \
    python3 setup.py clean --all && \
    python3 setup.py install && \
    cd ../../..

# Verify custom extensions are accessible
RUN python3 -c "try: import custom_rasterizer; print('custom_rasterizer imported successfully')\nexcept ImportError as e: print(f'custom_rasterizer import failed: {e}')" || true

RUN pip install sentencepiece

# Expose default port for API server
EXPOSE 8080

# Use ENTRYPOINT for the base command and CMD for default arguments
ENTRYPOINT ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8080"]
CMD []
