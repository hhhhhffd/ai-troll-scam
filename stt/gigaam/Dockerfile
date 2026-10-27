# Same base as tts/espeech: the torch layers are shared, so this image costs little disk.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Without the [torch] extra: the base image's torch 2.8 satisfies GigaAM (>= 2.6).
ARG GIGAAM_COMMIT=7447938d791c4f3e643386ee22c33777004293a5
RUN pip install --no-cache-dir "gigaam @ git+https://github.com/salute-developers/GigaAM.git@${GIGAAM_COMMIT}" aiohttp==3.12.15

WORKDIR /app
COPY server.py .
ENV PYTHONUNBUFFERED=1
EXPOSE 9000
CMD ["python", "server.py"]
