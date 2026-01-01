# Cognitive Server for MetaBonk.
# Runs SGLang with Phi-3-Vision for multi-instance inference.

FROM nvidia/cuda:13.1.0-devel-ubuntu24.04

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

# Python deps
# sglang pins a compatible torch/transformers/flashinfer stack for Blackwell (sm_120) GPUs.
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir \
      sglang==0.5.6.post2

# Prefer a newer Transformers build so Phi-3-Vision can take advantage of faster
# attention backends when available. (Trust-remote-code model code lives in the
# mounted model dir and is patched for cache compatibility.)
RUN python3 -m pip install --no-cache-dir \
      transformers==4.57.1

# Runtime dependency for sgl_kernel (SGLang kernel ops).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnuma1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Server code
COPY cognitive_server.py /app/
COPY temporal_processor.py /app/
COPY zmq_bridge.py /app/
COPY rl_integration.py /app/
COPY pyairports /app/pyairports

# Expose ZMQ port (host networking is typical, but expose for completeness)
EXPOSE 5555

CMD ["python3", "cognitive_server.py", \
     "--model-path", "/models/Phi-3-vision-128k-instruct", \
     "--tp-size", "1", \
     "--sglang-port", "30000", \
     "--max-running-requests", "64", \
     "--zmq-port", "5555"]
