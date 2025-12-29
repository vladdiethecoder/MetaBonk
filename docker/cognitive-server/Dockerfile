# Cognitive Server for MetaBonk
# Runs SGLang with Phi-3-Vision for multi-instance inference.

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    git \
    wget \
    ca-certificates \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CUDA wheels via PyTorch index for cu121)
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.2.0 torchvision==0.17.0 && \
    python3 -m pip install --no-cache-dir \
      transformers==4.38.0 \
      accelerate==0.27.0 \
      "sglang[all]==0.2.9" \
      pyzmq==25.1.2 \
      pillow==10.2.0 \
      numpy==1.26.4 \
      pydantic==2.6.1 \
      einops==0.7.0

WORKDIR /app

# Server code
COPY cognitive_server.py /app/
COPY temporal_processor.py /app/
COPY zmq_bridge.py /app/
COPY rl_integration.py /app/

# Expose ZMQ port (host networking is typical, but expose for completeness)
EXPOSE 5555

CMD ["python3", "cognitive_server.py", \
     "--model-path", "/models", \
     "--tp-size", "1", \
     "--quantization", "awq", \
     "--kv-cache-dtype", "fp8", \
     "--max-running-requests", "64", \
     "--zmq-port", "5555"]

