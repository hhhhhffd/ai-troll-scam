#!/usr/bin/env bash
# Downloads all models into ./models (mounted into containers). Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

LLM_MODEL="${LLM_MODEL:-qwen3:4b-instruct}"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
# Systran hosts the classic sizes; large-v3-turbo is converted by dropbox-dash (faster-whisper's own alias).
if [ "$WHISPER_MODEL" = large-v3-turbo ]; then
  WHISPER_REPO="${WHISPER_REPO:-dropbox-dash/faster-whisper-large-v3-turbo}"
else
  WHISPER_REPO="${WHISPER_REPO:-Systran/faster-whisper-$WHISPER_MODEL}"
fi
# Personas pick their voice from this set (see bot/personas/*.md).
PIPER_VOICES="${PIPER_VOICES:-ru_RU-irina-medium ru_RU-dmitri-medium ru_RU-ruslan-medium ru_RU-denis-medium}"
MODELS_HOST_DIR="${MODELS_HOST_DIR:-./models}"

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "  ok   $dest"
    return
  fi
  echo "  get  $dest"
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# Files that exist only in some repos (vocabulary.txt vs .json, preprocessor_config.json for v3).
fetch_optional() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "  ok   $dest"
  elif curl -fsI -L "$url" >/dev/null 2>&1; then
    fetch "$url" "$dest"
  fi
}

echo "==> LLM: $LLM_MODEL (via Ollama container)"
docker compose up -d llm
until docker compose exec -T llm /usr/local/bin/ollama list >/dev/null 2>&1; do sleep 2; done
docker compose exec -T llm /usr/local/bin/ollama pull "$LLM_MODEL"

echo "==> Whisper: faster-whisper-$WHISPER_MODEL"
hf="https://huggingface.co/$WHISPER_REPO/resolve/main"
for f in config.json model.bin tokenizer.json; do
  fetch "$hf/$f" "$MODELS_HOST_DIR/whisper/faster-whisper-$WHISPER_MODEL/$f"
done
for f in vocabulary.txt vocabulary.json preprocessor_config.json; do
  fetch_optional "$hf/$f" "$MODELS_HOST_DIR/whisper/faster-whisper-$WHISPER_MODEL/$f"
done

for voice in $PIPER_VOICES; do
  echo "==> Piper voice: $voice"
  # Voice id format: <lang>_<REGION>-<name>-<quality>, e.g. ru_RU-irina-medium
  IFS=- read -r locale name quality <<<"$voice"
  lang="${locale%%_*}"
  piper="https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/$locale/$name/$quality"
  fetch "$piper/$voice.onnx" "$MODELS_HOST_DIR/piper/$voice.onnx"
  fetch "$piper/$voice.onnx.json" "$MODELS_HOST_DIR/piper/$voice.onnx.json"
done

echo "==> Done"
