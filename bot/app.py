"""HTTP/WS entry point: health, personas and the single-call `/ws/call` endpoint."""

import hmac
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from openai import AsyncOpenAI, OpenAIError

import pipeline
from personas import DEFAULT_PERSONA, load_personas
from services.shared_models import load_models

DEMO_KEY = os.environ.get("DEMO_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# loguru defaults to DEBUG, which would put user transcripts in the logs.
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

if not DEMO_KEY:
    logger.warning("DEMO_KEY is empty: anyone who can reach the site can place calls")

personas = load_personas()
# Only one call at a time (single GPU). No await between check and set, so this is race-free.
busy = False


async def warm_up_llm() -> None:
    """One tiny request so Ollama has the model loaded before the first call."""
    client = AsyncOpenAI(base_url=pipeline.LLM_BASE_URL, api_key="ollama", timeout=120)
    try:
        await client.chat.completions.create(
            model=pipeline.LLM_MODEL,
            messages=[{"role": "user", "content": "Привет"}],
            max_tokens=1,
            extra_body={"reasoning_effort": "none"},
        )
    except OpenAIError as e:
        logger.error(f"LLM warm-up failed (is {pipeline.LLM_MODEL} pulled?): {e!r}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The server only starts listening after this, so /api/health answering == models are ready.
    models_dir = pipeline.MODELS_DIR
    logger.info("Loading Whisper and Piper")
    load_models(
        models_dir / "whisper" / f"faster-whisper-{WHISPER_MODEL}",
        models_dir / "piper",
        {p.voice for p in personas.values()},
    )
    await warm_up_llm()
    logger.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "busy": busy}


@app.get("/api/engines")
async def list_engines():
    return [
        {"id": "local", "name": "Локально (Whisper + Qwen + Piper)", "available": True},
        {"id": "gemini", "name": "Gemini Live", "available": bool(pipeline.GEMINI_API_KEY)},
    ]


@app.get("/api/personas")
async def list_personas():
    return [p.public() for p in personas.values()]


@app.websocket("/ws/call")
async def ws_call(
    websocket: WebSocket, key: str = "", persona: str = DEFAULT_PERSONA, engine: str = "local"
):
    global busy
    await websocket.accept()

    if DEMO_KEY and not hmac.compare_digest(key.encode(), DEMO_KEY.encode()):
        await websocket.send_json({"type": "unauthorized"})
        await websocket.close(code=4401)
        return

    # Unknown engine falls back to local, like an unknown persona.
    if engine != "gemini":
        engine = "local"
    if engine == "gemini" and not pipeline.GEMINI_API_KEY:
        await websocket.send_json({"type": "error", "message": "Gemini недоступен: на сервере не задан GEMINI_API_KEY"})
        await websocket.close(code=4400)
        return

    if busy:
        await websocket.send_json({"type": "busy"})
        await websocket.close(code=4409)
        return

    busy = True
    try:
        # Unknown persona falls back to the default one rather than failing the call.
        selected = personas.get(persona) or personas[DEFAULT_PERSONA]
        logger.info(f"Call started, persona={selected.id}, engine={engine}")
        await pipeline.run_call(websocket, selected, engine)
    except Exception:
        logger.exception("Call failed")
    finally:
        busy = False
        logger.info("Call finished")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass  # already closed by the transport or the client
