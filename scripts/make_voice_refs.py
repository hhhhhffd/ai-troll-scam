# /// script
# requires-python = ">=3.10"
# dependencies = ["google-genai"]
# ///
"""Generate synthetic reference clips for the TTS sidecars with Gemini TTS.

Writes ${MODELS_HOST_DIR}/voices/<voice>/<emotion>.wav (mono 24 kHz) + .txt (exact transcript).
Usage: uv run scripts/make_voice_refs.py [--voice granny] [--emotion laughs] [--force]
Reads GEMINI_API_KEY and MODELS_HOST_DIR from the environment or the repo .env.
"""
import argparse
import os
import shutil
import sys
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

# Free tier allows only ~10 requests/day per TTS model; pass --model to use another one.
DEFAULT_MODEL = "gemini-3.8-flash-tts"
SAMPLE_RATE = 24000
# Reference clips must be <= 12 s; very short ones clone poorly.
MIN_SECONDS, MAX_SECONDS = 4.0, 11.0

# voice -> (Gemini voice name, speaker description)
VOICES = {
    "granny": ("Gacrux", "an elderly Russian woman of about 74, warm, slightly quavering, unhurried"),
    "ded": ("Algenib", "an elderly Russian man of about 75, hoarse, slow, gruff but kind"),
}

# emotion -> (style instruction, phrase). Phrases avoid gendered past forms so both voices fit.
EMOTIONS = {
    "neutral": ("in a plain, even conversational tone",
                "Алло, слушаю вас внимательно, говорите, пожалуйста, чуть погромче, я плохо слышу."),
    "calm": ("very calm, soothing and quiet",
             "Не волнуйтесь, не надо так спешить, давайте спокойно разберёмся, что у вас случилось."),
    "warm": ("warm, affectionate and kind, with a smile in the voice",
             "Ой, милый мой, как хорошо, что вы позвонили, а то я тут совсем одна сижу целый день."),
    "irritated": ("irritated, impatient",
                  "Так, подождите, не тараторьте. Я ничего не понимаю, говорите нормально, кто вы такой?"),
    "angry": ("angry and indignant, loud, sharp",
              "Да как вам не стыдно! Я всю жизнь честно работала, а вы мне тут голову морочите, жулики!"),
    "laughs": ("amused, laughing out loud in the middle of the phrase, with real audible laughter",
               "Ха-ха-ха! Ой, не могу, ну вы и насмешили меня, ха-ха, ой, держите меня, сейчас упаду!"),
    "sighs": ("tired, with a deep audible sigh at the start and in the middle",
              "Ох... да что ж такое... Ох, ноги совсем не ходят, давление опять, и в доме никого нет."),
    "confused": ("confused and hesitant, puzzled, with pauses and trailing off",
                 "Что-что? Погодите, я не поняла... Какой банк? Мне ничего такого не приходило, наверное."),
    "sad": ("sad, quiet and tearful, voice trembling",
            "Внуки давно не звонят, одна я осталась, поговорить-то не с кем, вот и сижу у окошка."),
    "surprised": ("very surprised and astonished, rising intonation",
                  "Да что вы говорите?! Не может быть! Прямо сейчас, в мою квартиру? Ничего себе новости!"),
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def synth(client: genai.Client, model: str, voice_name: str, speaker: str, style: str, phrase: str) -> bytes:
    prompt = f"Say in a {style} voice, like {speaker}, at a natural pace:\n{phrase}"
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name))
            ),
        ),
    )
    return resp.candidates[0].content.parts[0].inline_data.data  # raw PCM16 24 kHz mono


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", choices=list(VOICES), help="only this voice")
    ap.add_argument("--emotion", choices=list(EMOTIONS), help="only this emotion")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true", help="regenerate existing clips")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    load_env_file(root / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    models_dir = Path(os.environ.get("MODELS_HOST_DIR", "./models"))
    if not models_dir.is_absolute():
        models_dir = root / models_dir
    client = genai.Client(api_key=key)

    for voice, (voice_name, speaker) in VOICES.items():
        if args.voice and voice != args.voice:
            continue
        out = models_dir / "voices" / voice
        out.mkdir(parents=True, exist_ok=True)
        for emotion, (style, phrase) in EMOTIONS.items():
            if args.emotion and emotion != args.emotion:
                continue
            wav_path = out / f"{emotion}.wav"
            if wav_path.exists() and not args.force:
                continue
            for attempt in range(2):
                try:
                    pcm = synth(client, args.model, voice_name, speaker, style, phrase)
                    seconds = len(pcm) / 2 / SAMPLE_RATE
                    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
                        raise ValueError(f"duration {seconds:.1f}s out of range")
                    break
                except Exception as e:  # sporadic 5xx or out-of-range duration: retry
                    if "RESOURCE_EXHAUSTED" in str(e):
                        sys.exit(f"{voice}/{emotion}: quota exhausted for {args.model}; try --model or wait")
                    print(f"{voice}/{emotion}: attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]}",
                          file=sys.stderr, flush=True)
                    time.sleep(10)
            else:
                sys.exit(f"{voice}/{emotion}: giving up")
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(pcm)
            (out / f"{emotion}.txt").write_text(phrase + "\n", encoding="utf-8")
            print(f"{voice}/{emotion}: {seconds:.1f}s", flush=True)
            time.sleep(3)

    if not args.voice or args.voice == "granny":
        default = models_dir / "voices" / "default"
        if default.exists():
            shutil.rmtree(default)
        shutil.copytree(models_dir / "voices" / "granny", default)


if __name__ == "__main__":
    main()
