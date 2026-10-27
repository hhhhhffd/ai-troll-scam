"""GigaAM-v3 speech recognition sidecar.

Same control endpoints as the TTS sidecars (tts/CONTRACT.md): POST /load, POST /unload, GET /status.
POST /v1/transcribe with a WAV body (one speech segment, <= 25 s) -> {"text": "..."}.
The e2e model already returns punctuation and normalized text.
"""

import asyncio
import gc
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

log = logging.getLogger("gigaam")

MODEL_NAME = os.environ.get("GIGAAM_MODEL", "v3_e2e_rnnt")
DOWNLOAD_ROOT = os.environ.get("GIGAAM_DIR", "/models/gigaam")
# One worker: GPU inference is serialized and the model is only touched from it.
executor = ThreadPoolExecutor(max_workers=1)
state: dict = {}  # "model" when loaded
lock = asyncio.Lock()


def _load() -> None:
    import gigaam

    model = gigaam.load_model(MODEL_NAME, device="cuda", download_root=DOWNLOAD_ROOT)
    state["model"] = model
    # The first pass initializes CUDA kernels; do it here, not on the caller's turn.
    with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm") as f:
        _write_silence(f.name)
        model.transcribe(f.name)


def _write_silence(path: str) -> None:
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(32000))


def _unload() -> None:
    import torch

    state.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _transcribe(wav: bytes) -> str:
    # GigaAM reads audio through ffmpeg from a path; /dev/shm keeps it off the disk.
    with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm") as f:
        f.write(wav)
        f.flush()
        return state["model"].transcribe(f.name).text.strip()


async def handle_load(_request: web.Request) -> web.Response:
    async with lock:
        if not state:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(executor, _load)
            except Exception as e:
                log.exception("load failed")
                await loop.run_in_executor(executor, _unload)
                return web.Response(status=500, text=f"load failed: {type(e).__name__}: {e}"[:300])
    return web.json_response({"loaded": True})


async def handle_unload(_request: web.Request) -> web.Response:
    async with lock:
        await asyncio.get_running_loop().run_in_executor(executor, _unload)
    return web.json_response({"loaded": False})


async def handle_status(_request: web.Request) -> web.Response:
    return web.json_response({"loaded": bool(state)})


async def handle_transcribe(request: web.Request) -> web.Response:
    wav = await request.read()
    if not wav.startswith(b"RIFF"):
        return web.Response(status=400, text="expected a WAV body")
    async with lock:
        if not state:
            return web.Response(status=409, text="not loaded")
        try:
            text = await asyncio.get_running_loop().run_in_executor(executor, _transcribe, wav)
        except Exception as e:
            log.exception("transcription failed")
            return web.Response(status=500, text=f"{type(e).__name__}: {e}"[:300])
    return web.json_response({"text": text})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.add_routes(
        [
            web.post("/load", handle_load),
            web.post("/unload", handle_unload),
            web.get("/status", handle_status),
            web.post("/v1/transcribe", handle_transcribe),
        ]
    )
    web.run_app(app, host="0.0.0.0", port=9000, access_log=None)
