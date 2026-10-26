#!/usr/bin/env bash
# Downloads all models into ./models (mounted into containers). Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

LLM_MODEL="${LLM_MODEL:-qwen3:4b}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
PIPER_VOICE="${PIPER_VOICE:-ru_RU-irina-medium}"

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "  ok   $dest"
    return
  fi
  echo "  get  $dest"
  mkdir -p "$(dirname "$dest")"
  curl -fL --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

echo "==> LLM: $LLM_MODEL (via Ollama container)"
docker compose up -d llm
until docker compose exec -T llm ollama list >/dev/null 2>&1; do sleep 2; done
docker compose exec -T llm ollama pull "$LLM_MODEL"

echo "==> Whisper: faster-whisper-$WHISPER_MODEL"
hf="https://huggingface.co/Systran/faster-whisper-$WHISPER_MODEL/resolve/main"
for f in config.json model.bin tokenizer.json vocabulary.txt; do
  fetch "$hf/$f" "models/whisper/faster-whisper-$WHISPER_MODEL/$f"
done

echo "==> Piper voice: $PIPER_VOICE"
# Voice id format: <lang>_<REGION>-<name>-<quality>, e.g. ru_RU-irina-medium
IFS=- read -r locale name quality <<<"$PIPER_VOICE"
lang="${locale%%_*}"
piper="https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$name/$quality"
fetch "$piper/$PIPER_VOICE.onnx" "models/piper/$PIPER_VOICE.onnx"
fetch "$piper/$PIPER_VOICE.onnx.json" "models/piper/$PIPER_VOICE.onnx.json"

echo "==> Done"
