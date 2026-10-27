FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Weights and caches live on the mounted volume, so restarts do not re-download.
ENV HF_HOME=/models/espeech/hf \
    MODELS_DIR=/models/espeech \
    PYTHONUNBUFFERED=1
EXPOSE 9000
CMD ["python", "server.py"]
