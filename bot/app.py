"""HTTP/WS entry point: health, the persona's name and the single-call `/ws/call` endpoint."""

import hmac
import os
import sys
from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

import pipeline
from persona import load_persona

DEMO_KEY = os.environ.get("DEMO_KEY", "")

# loguru defaults to DEBUG, which would put user transcripts in the logs.
logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO"))

if not DEMO_KEY:
    logger.warning("DEMO_KEY is empty: anyone who can reach the site can place calls")

persona = load_persona()
# Only one call at a time. No await between check and set, so this is race-free.
busy = False


async def warm_local_llm() -> None:
    """Load the fallback model into VRAM for good: a cold load takes ~15 s, too long for a turn."""
    root = pipeline.LLM_BASE_URL.removesuffix("/").removesuffix("/v1")
    body = {
        "model": pipeline.LLM_MODEL,
        "messages": [{"role": "user", "content": "Привет"}],
        "stream": False,
        "keep_alive": -1,
        "options": {"num_predict": 1},
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post(f"{root}/api/chat", json=body) as r:
            if r.status != 200:
                raise RuntimeError(f"Ollama {r.status}: {(await r.text())[:200]}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if pipeline.LOCAL_LLM_FALLBACK:
        try:
            await warm_local_llm()
            logger.info(f"Local fallback LLM {pipeline.LLM_MODEL} loaded")
        except Exception as e:
            # Calls still work on Groq; only the last fallback is gone.
            logger.error(f"Local fallback LLM unavailable: {e!r}")
    if missing := pipeline.missing_keys():
        logger.error(f"Calls are disabled until set: {', '.join(missing)}")
    logger.info("Ready")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok", "busy": busy}


@app.get("/api/persona")
async def get_persona():
    return {"name": persona.name}


@app.websocket("/ws/call")
async def ws_call(websocket: WebSocket, key: str = ""):
    global busy
    await websocket.accept()

    if DEMO_KEY and not hmac.compare_digest(key.encode(), DEMO_KEY.encode()):
        await websocket.send_json({"type": "unauthorized"})
        await websocket.close(code=4401)
        return

    if missing := pipeline.missing_keys():
        await websocket.send_json({"type": "error", "message": f"Не заданы {', '.join(missing)}"})
        await websocket.close(code=4400)
        return

    if busy:
        await websocket.send_json({"type": "busy"})
        await websocket.close(code=4409)
        return

    busy = True
    try:
        logger.info("Call started")
        await pipeline.run_call(websocket, persona)
    except Exception:
        logger.exception("Call failed")
    finally:
        busy = False
        logger.info("Call finished")
        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass  # already closed by the transport or the client
