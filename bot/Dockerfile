# CUDA 12 + cuDNN 9 runtime: required by ctranslate2 (faster-whisper) for GPU inference.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Ubuntu 24.04 ships a default "ubuntu" user (uid 1000); the models volume is only read.
USER ubuntu
EXPOSE 8000
# No access log: the WS URL carries DEMO_KEY in the query string.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
