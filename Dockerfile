# ---------- Stage 1: CUDA 12.4 wheel ----------
FROM python:3.10-slim AS wheels
ARG CUDA_WHL=cu124
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /wheels
RUN python -m pip install --upgrade pip
RUN pip download \
    --only-binary=:all: \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/${CUDA_WHL} \
    -d /wheels \
    llama-cpp-python runpod \
    paramiko cryptography pynacl bcrypt

# ---------- Stage 2: final (CUDA runtime base) ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/runpod-volume/models/your-model.gguf

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates libgomp1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=wheels /wheels /wheels
RUN pip install --no-index --find-links=/wheels \
      llama-cpp-python runpod paramiko cryptography pynacl bcrypt && \
    rm -rf /wheels

COPY rp_handler.py ./rp_handler.py
CMD ["python3", "-u", "rp_handler.py"]
