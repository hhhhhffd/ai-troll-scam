"""Synthesize test phrases with ElevenLabs into 16 kHz mono WAVs /tmp/u0.wav, /tmp/u1.wav, ...

Run inside the bot container (it has the key): docker compose exec -T bot python - "фраза" ... < scripts/make_test_speech.py
"""

import os, sys, wave
import httpx

# A stock male voice, so the caller doesn't sound like the granny.
VOICE = os.environ.get("TEST_CALLER_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
for i, text in enumerate(sys.argv[1:]):
    r = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=pcm_16000",
        headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
        json={"text": text, "model_id": "eleven_multilingual_v2", "language_code": "ru"},
        timeout=60,
    )
    r.raise_for_status()
    with wave.open(f'/tmp/u{i}.wav', 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(r.content)
    print(f'/tmp/u{i}.wav', len(r.content) / 2 / 16000)
