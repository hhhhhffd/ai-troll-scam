"""ESpeech (F5-TTS) sidecar implementing tts/CONTRACT.md."""
import asyncio
import gc
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from aiohttp import web

log = logging.getLogger("espeech")

MODELS = Path(os.environ.get("MODELS_DIR", "/models/espeech"))
VOICES = Path(os.environ.get("VOICES_DIR", "/voices"))
NFE_STEP = int(os.environ.get("NFE_STEP", "16"))
SAMPLE_RATE = 24000
# Keep each synthesis call short: first audio arrives sooner and F5 stays stable.
MAX_PIECE_CHARS = 100

MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
EMOTIONS = {"neutral", "calm", "warm", "irritated", "angry", "laughs", "sighs", "confused", "sad", "surprised"}
TAG_RE = re.compile(r"\[([^\]]*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
VOICE_RE = re.compile(r"^[\w-]+$")

# One worker: GPU synthesis is serialized, and the model objects are only touched from it.
executor = ThreadPoolExecutor(max_workers=1)
state: dict = {}  # model, vocoder, accentizer, refs; empty = unloaded
lock = asyncio.Lock()


def _load() -> None:
    import torch
    from f5_tts.infer.utils_infer import load_model, load_vocoder
    from f5_tts.model import DiT
    from ruaccent import RUAccent

    model = load_model(DiT, MODEL_CFG, str(MODELS / "model/espeech_tts_rlv2.pt"),
                       vocab_file=str(MODELS / "model/vocab.txt"), device="cuda")
    vocoder = load_vocoder(device="cuda")
    accentizer = RUAccent()
    # CPU onnx: keeps VRAM for the DiT.
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
                    workdir=str(MODELS / "ruaccent"))
    state.update(model=model, vocoder=vocoder, accentizer=accentizer, refs={})
    # Warm up CUDA kernels so the first call is not slow.
    _synthesize_piece(_get_ref("default", "neutral"), "Проверка связи.")
    torch.cuda.empty_cache()


def _unload() -> None:
    import torch

    state.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _accent(text: str) -> str:
    # A '+' means the caller already placed stress marks.
    if "+" in text:
        return text
    try:
        return state["accentizer"].process_all(text)
    except ValueError as e:
        # RUAccent's homograph ONNX model rejects some inputs (missing token_type_ids); speaking
        # without stress marks beats dropping the phrase.
        log.warning("accentizer failed, using plain text: %s", e)
        return text


def _find_ref(voice: str, emotion: str) -> Path | None:
    for v in (voice, "default"):
        for e in (emotion, "neutral"):
            wav = VOICES / v / f"{e}.wav"
            if wav.is_file() and wav.with_suffix(".txt").is_file():
                return wav
    return None


def _get_ref(voice: str, emotion: str) -> tuple[str, str]:
    """Return (preprocessed ref wav, accented ref text); cached per reference file."""
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text

    wav = _find_ref(voice, emotion)
    if wav is None:
        raise LookupError(f"no reference audio for voice '{voice}'")
    if wav not in state["refs"]:
        text = _accent(wav.with_suffix(".txt").read_text(encoding="utf-8").strip())
        state["refs"][wav] = preprocess_ref_audio_text(str(wav), text, show_info=lambda *_: None)
    return state["refs"][wav]


def _synthesize_piece(ref: tuple[str, str], text: str) -> bytes:
    import torchaudio
    from f5_tts.infer.utils_infer import infer_process

    ref_audio, ref_text = ref
    wave, sr, _ = infer_process(ref_audio, ref_text, _accent(text), state["model"], state["vocoder"],
                                nfe_step=NFE_STEP, show_info=lambda *_: None, progress=_NoProgress)
    if wave is None or len(wave) == 0:
        return b""
    if sr != SAMPLE_RATE:
        import torch
        wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, SAMPLE_RATE).numpy()
    return (np.clip(wave, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class _NoProgress:
    @staticmethod
    def tqdm(it, *_, **__):
        return it


def split_segments(text: str) -> list[tuple[str, str]]:
    """'[angry] a [laughs] b' -> [('angry','a'), ('laughs','b')]; unknown tags are dropped."""
    segments, emotion, pos = [], "neutral", 0
    for m in TAG_RE.finditer(text):
        segments.append((emotion, text[pos:m.start()]))
        tag = m.group(1).strip().lower()
        if tag in EMOTIONS:
            emotion = tag
        pos = m.end()
    segments.append((emotion, text[pos:]))
    return [(e, t.strip()) for e, t in segments if re.search(r"\w", t)]


def split_pieces(text: str) -> list[str]:
    """Greedily group sentences into pieces of at most MAX_PIECE_CHARS."""
    pieces: list[str] = []
    for sentence in SENTENCE_RE.split(text):
        if pieces and len(pieces[-1]) + len(sentence) < MAX_PIECE_CHARS:
            pieces[-1] += " " + sentence
        else:
            pieces.append(sentence)
    return pieces


async def handle_load(request: web.Request) -> web.Response:
    async with lock:
        if state:
            return web.json_response({"loaded": True})
        try:
            await asyncio.get_running_loop().run_in_executor(executor, _load)
        except Exception as e:
            log.exception("load failed")
            await asyncio.get_running_loop().run_in_executor(executor, _unload)
            return web.Response(status=500, text=f"load failed: {type(e).__name__}: {e}"[:300])
    return web.json_response({"loaded": True})


async def handle_unload(request: web.Request) -> web.Response:
    async with lock:
        await asyncio.get_running_loop().run_in_executor(executor, _unload)
    return web.json_response({"loaded": False})


async def handle_status(request: web.Request) -> web.Response:
    return web.json_response({"loaded": bool(state)})


async def handle_speech(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
        text, voice = body["input"], body.get("voice") or "default"
        if not isinstance(text, str) or not isinstance(voice, str) or not VOICE_RE.match(voice):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return web.Response(status=400, text="expected JSON {input: str, voice?: str}")
    segments = split_segments(text)
    if not segments:
        return web.Response(status=400, text="empty input")
    if not state:
        return web.Response(status=409, text="not loaded")

    loop = asyncio.get_running_loop()
    resp = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
    prepared = False
    async with lock:
        if not state:  # unloaded while waiting for the lock
            return web.Response(status=409, text="not loaded")
        try:
            for emotion, segment in segments:
                ref = await loop.run_in_executor(executor, _get_ref, voice, emotion)
                for piece in split_pieces(segment):
                    if request.transport is None or request.transport.is_closing():
                        return resp
                    pcm = await loop.run_in_executor(executor, _synthesize_piece, ref, piece)
                    if not prepared:
                        await resp.prepare(request)
                        prepared = True
                    await resp.write(pcm)
        except (ConnectionResetError, asyncio.CancelledError):
            log.info("client disconnected")
            raise
        except Exception as e:
            log.exception("synthesis failed")
            if not prepared:
                status = 400 if isinstance(e, LookupError) else 500
                return web.Response(status=status, text=f"{type(e).__name__}: {e}"[:300])
            return resp  # headers already sent: end the stream early
    if not prepared:
        await resp.prepare(request)
    await resp.write_eof()
    return resp


def make_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.post("/load", handle_load),
        web.post("/unload", handle_unload),
        web.get("/status", handle_status),
        web.post("/v1/audio/speech", handle_speech),
    ])
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    web.run_app(make_app(), host="0.0.0.0", port=9000)
