#!/usr/bin/env python3
"""Russian TTS shoot-out: same granny phrases through every available TTS, timings + blind listening page.

Runs inside the bot container (it has the deps, the keys and the compose network):
  docker compose exec -T bot python - [--only higgs,espeech] [--voice granny] < scripts/tts_bench.py
Writes /models/tts-bench/<timestamp>/ (= ${MODELS_HOST_DIR}/tts-bench on the host): one WAV per
engine x phrase under an anonymous name, index.html for blind A/B, key.json + results.md.

Backends: local sidecars (higgs, espeech; tts/CONTRACT.md) if reachable, Gemini TTS if GEMINI_API_KEY,
ElevenLabs v4 if ELEVENLABS_API_KEY+ELEVENLABS_VOICE_ID, Fish if FISH_API_KEY. Each cloud request
is one short phrase, so a full run costs ~10 requests per cloud engine.
"""

import argparse
import asyncio
import html
import json
import os
import random
import re
import statistics
import time
import wave
from pathlib import Path

import aiohttp

RATE = 24000
PHRASES = [
    "Алло? Алло, кто это?",
    "[confused] Подождите, какой ещё банк? Откуда вы знаете мой номер?",
    "[irritated] Так. Сначала вы Алексей, теперь Максим. Вы там по очереди трубку передаёте?",
    "[angry] Это вы меня не перебивайте! Я пока не поняла, кто мне звонит.",
    "[laughs] Ой, насмешили. Код ему продиктуй, как же.",
    "[sighs] Сорок лет бухгалтером отработала, а такого ещё не слышала.",
    "[warm] Ну вот, другое дело. Когда по-человечески говорят, и разговаривать приятно.",
    "Kaspi, говорите? У меня в Алматы отделение прямо через дорогу, я сама туда схожу и всё узнаю.",
]
_TAG_RE = re.compile(r"\[[^\]]*\]\s*")


def pcm_from_maybe_wav(data: bytes) -> bytes:
    """Strip a RIFF header if the backend sent WAV instead of raw PCM."""
    if not data.startswith(b"RIFF"):
        return data
    pos = 12
    while pos + 8 <= len(data):
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if data[pos : pos + 4] == b"data":
            return data[pos + 8 : pos + 8 + size]
        pos += 8 + size
    raise ValueError("WAV without data chunk")


async def timed_stream(chunks) -> tuple[bytes, float, float]:
    """Collects an async iterator of byte chunks; returns (pcm, time to first audio, total time)."""
    start = time.monotonic()
    first = None
    buf = bytearray()
    async for chunk in chunks:
        if chunk and first is None:
            first = time.monotonic() - start
        buf.extend(chunk)
    return bytes(buf), first or 0.0, time.monotonic() - start


async def sidecar(session, url, voice, text):
    async with session.post(f"{url}/v1/audio/speech", json={"input": text, "voice": voice}) as r:
        if r.status != 200:
            raise RuntimeError(f"{r.status}: {(await r.text())[:200]}")
        async for chunk in r.content.iter_chunked(4096):
            yield chunk


async def elevenlabs(session, _voice, text):
    url = f"https://api.elevenlabs.io/v1/text-to-dialogue/stream?output_format=pcm_{RATE}"
    body = {
        "model_id": os.environ.get("ELEVENLABS_MODEL", "eleven_v4_turbo"),
        "language_code": "ru",
        "inputs": [{"text": text, "voice_id": os.environ["ELEVENLABS_VOICE_ID"]}],
    }
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    async with session.post(url, json=body, headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f"{r.status}: {(await r.text())[:200]}")
        async for chunk in r.content.iter_chunked(4096):
            yield chunk


async def fish(session, _voice, text):
    headers = {
        "Authorization": f"Bearer {os.environ['FISH_API_KEY']}",
        "model": os.environ.get("FISH_MODEL", "s2.1-pro-free"),
    }
    body = {"text": text, "format": "pcm", "sample_rate": RATE, "latency": "balanced"}
    if os.environ.get("FISH_REFERENCE_ID"):
        body["reference_id"] = os.environ["FISH_REFERENCE_ID"]
    async with session.post("https://api.fish.audio/v1/tts", json=body, headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f"{r.status}: {(await r.text())[:200]}")
        async for chunk in r.content.iter_chunked(4096):
            yield chunk


async def gemini_tts(_session, _voice, text):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Gacrux")
            )
        ),
    )
    prompt = f"Скажи голосом пожилой женщины лет семидесяти пяти, живо, по телефону: {text}"
    stream = await client.aio.models.generate_content_stream(
        model=os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"), contents=prompt, config=config
    )
    async for response in stream:
        for part in response.candidates[0].content.parts if response.candidates else []:
            if part.inline_data and part.inline_data.data:
                yield pcm_from_maybe_wav(part.inline_data.data)


async def sidecar_up(session, url) -> bool:
    try:
        async with session.get(f"{url}/status", timeout=aiohttp.ClientTimeout(total=2)) as r:
            return r.status == 200
    except (aiohttp.ClientError, TimeoutError):
        return False


async def available_backends(session):
    backends = {}
    for name, env in (("higgs", "HIGGS_URL"), ("espeech", "ESPEECH_URL")):
        url = os.environ.get(env, f"http://tts-{name}:9000")
        if await sidecar_up(session, url):
            backends[name] = (lambda s, v, t, url=url: sidecar(s, url, v, t), url)
    if os.environ.get("GEMINI_API_KEY"):
        backends["gemini-tts"] = (gemini_tts, None)
    if os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("ELEVENLABS_VOICE_ID"):
        backends["elevenlabs"] = (elevenlabs, None)
    if os.environ.get("FISH_API_KEY"):
        backends["fish"] = (fish, None)
    return backends


async def sidecar_control(session, url, action) -> None:
    async with session.post(f"{url}/{action}", timeout=aiohttp.ClientTimeout(total=600)) as r:
        if r.status != 200:
            raise RuntimeError(f"{url}/{action}: {r.status} {(await r.text())[:200]}")


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def listening_page(items: list[dict]) -> str:
    rows = []
    for i, phrase in enumerate(PHRASES):
        clips = [it for it in items if it["phrase"] == i]
        random.shuffle(clips)
        audio = "".join(
            f'<figure><figcaption>{html.escape(c["file"].removesuffix(".wav"))}</figcaption>'
            f'<audio controls preload="none" src="{c["file"]}"></audio></figure>'
            for c in clips
        )
        rows.append(f"<section><h2>{i + 1}. {html.escape(_TAG_RE.sub('', phrase))}</h2>{audio}</section>")
    return (
        '<!doctype html><meta charset="utf-8"><title>TTS blind test</title>'
        "<style>body{font:16px system-ui;max-width:900px;margin:24px auto;padding:0 16px}"
        "section{margin-bottom:28px}figure{display:inline-block;margin:6px 12px 6px 0}"
        "figcaption{font:13px monospace;color:#666}</style>"
        "<h1>Слепое сравнение голосов</h1><p>Названия клипов анонимные; ответ — в key.json.</p>"
        + "".join(rows)
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated backend names")
    p.add_argument("--voice", default="granny")
    p.add_argument("--keep-loaded", action="store_true", help="don't unload sidecars afterwards")
    args = p.parse_args()

    out = Path("/models/tts-bench") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    items, results = [], {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(sock_connect=5, sock_read=120)) as session:
        backends = await available_backends(session)
        if args.only:
            backends = {k: v for k, v in backends.items() if k in args.only.split(",")}
        print("backends:", ", ".join(backends) or "none")
        for name, (synth, url) in backends.items():
            # Sidecars share the GPU: load one, measure, unload before the next.
            if url:
                print(f"== {name}: loading")
                await sidecar_control(session, url, "load")
            stats = results.setdefault(name, {"ttfa": [], "rtf": [], "errors": 0})
            try:
                for i, phrase in enumerate(PHRASES):
                    try:
                        pcm, ttfa, total = await timed_stream(synth(session, args.voice, phrase))
                    except Exception as e:
                        stats["errors"] += 1
                        print(f"  {name} #{i + 1} FAILED: {str(e)[:200]}")
                        if "429" in str(e):  # quota exhausted: the rest would fail the same way
                            break
                        continue
                    seconds = len(pcm) / 2 / RATE
                    if seconds == 0:
                        stats["errors"] += 1
                        print(f"  {name} #{i + 1} returned no audio")
                        continue
                    stats["ttfa"].append(ttfa)
                    stats["rtf"].append(total / seconds)
                    file = f"{random.getrandbits(32):08x}.wav"
                    write_wav(out / file, pcm)
                    items.append({"file": file, "engine": name, "phrase": i})
                    print(f"  {name} #{i + 1}: first audio {ttfa:.2f}s, total {total:.2f}s for {seconds:.1f}s")
            finally:
                if url and not args.keep_loaded:
                    await sidecar_control(session, url, "unload")

    lines = ["| TTS | first audio p50 | max | RTF p50 | errors |", "|---|---|---|---|---|"]
    for name, s in results.items():
        if s["ttfa"]:
            lines.append(
                f"| {name} | {statistics.median(s['ttfa']):.2f} s | {max(s['ttfa']):.2f} s"
                f" | {statistics.median(s['rtf']):.2f} | {s['errors']} |"
            )
        else:
            lines.append(f"| {name} | — | — | — | {s['errors']} |")
    table = "\n".join(lines)
    print(table)
    (out / "results.md").write_text(f"# TTS bench {out.name}, voice={args.voice}\n\n{table}\n")
    (out / "key.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    (out / "index.html").write_text(listening_page(items))
    print(f"written to {out} (host: $MODELS_HOST_DIR/tts-bench/{out.name})")


asyncio.run(main())
