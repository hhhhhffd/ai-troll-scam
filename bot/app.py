"""HTTP/WS entry point: health, personas and the single-call `/ws/call` endpoint."""

import hmac
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

import engines
import pipeline
from personas import DEFAULT_PERSONA, load_personas
from services.gpu import GpuError, build_gpu_manager
from services.shared_models import load_models

DEMO_KEY = os.environ.get("DEMO_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
# Engine whose GPU models are loaded at startup, so its first call starts without waiting.
PRELOAD_ENGINE = os.environ.get("PRELOAD_ENGINE", "local")

# loguru defaults to DEBUG, which would put user transcripts in the logs.
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

if not DEMO_KEY:
    logger.warning("DEMO_KEY is empty: anyone who can reach the site can place calls")

personas = load_personas()
gpu = build_gpu_manager(engines.SIDECARS)
# Only one call at a time (single GPU). No await between check and set, so this is race-free.
busy = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The server only starts listening after this, so /api/health answering == models are ready.
    models_dir = pipeline.MODELS_DIR
    load_models(
        models_dir / "whisper" / f"faster-whisper-{WHISPER_MODEL}",
        models_dir / "piper",
        {p.voice for p in personas.values()},
    )
    await gpu.reset()
    preload = engines.ENGINES_BY_ID.get(PRELOAD_ENGINE)
    if preload and await engines.problem(preload) is None:
        logger.info(f"Preloading GPU models for engine {preload.id}")
        try:
            await gpu.prepare(engines.gpu_needs(preload))
        except GpuError as e:
            logger.error(f"Preload failed: {e}")
    logger.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "busy": busy, "gpu": gpu.loaded}


@app.get("/api/engines")
async def list_engines():
    result = []
    for engine in engines.ENGINES:
        reason = await engines.problem(engine)
        item = {"id": engine.id, "name": engine.name, "available": reason is None}
        if reason:
            item["note"] = reason
        result.append(item)
    return result


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
    selected_engine = engines.ENGINES_BY_ID.get(engine) or engines.ENGINES_BY_ID["local"]
    if reason := await engines.problem(selected_engine):
        await websocket.send_json({"type": "error", "message": f"{selected_engine.name} недоступен: {reason}"})
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
        try:
            # Swapping models can take a while; the client keeps "dialing" until `ready`.
            await gpu.prepare(engines.gpu_needs(selected_engine))
        except GpuError as e:
            logger.error(f"GPU prepare for {selected_engine.id} failed: {e}")
            await websocket.send_json({"type": "error", "message": f"Не удалось подготовить модели: {e}"})
            await websocket.close(code=4400)
            return
        logger.info(f"Call started, persona={selected.id}, engine={selected_engine.id}")
        await pipeline.run_call(websocket, selected, selected_engine)
    except Exception:
        logger.exception("Call failed")
    finally:
        busy = False
        logger.info("Call finished")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass  # already closed by the transport or the client
