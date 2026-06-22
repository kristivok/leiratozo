# NVIDIA CUDA 12.6 + cuDNN runtime Ubuntu 22.04 – GPU-s futtatáshoz
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

# Rendszercsomagok: Python 3.11, ffmpeg, build eszközök
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Csak a requirements-t másoljuk be először (Docker layer cache)
COPY requirements.txt /app/

# Venv létrehozása és csomagok telepítése
# PyTorch CUDA 11.8 index URL-ről (cu118 build)
RUN python3.11 -m venv venv \
    && venv/bin/pip install --upgrade pip setuptools wheel \
    && venv/bin/pip install --no-cache-dir \
       torch==2.6.0+cu118 torchaudio==2.6.0+cu118 \
       --index-url https://download.pytorch.org/whl/cu118 \
    && venv/bin/pip install --no-cache-dir faster-whisper \
    && venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/app/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# A projektfájlok bemásolása (venv és cache nélkül – ld. .dockerignore)
COPY . /app

EXPOSE 58515

CMD ["python3.11", "app.py"]
