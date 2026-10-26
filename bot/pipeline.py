"""One phone call = one Pipecat pipeline: ws in -> VAD/STT -> LLM -> TTS -> ws out."""

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from personas import Persona
from services.call_io import INPUT_SAMPLE_RATE, RawPcmSerializer, UiEventsProcessor
from services.shared_models import SharedPiperTTSService, SharedWhisperSTTService

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "100"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.8"))
LLM_HISTORY_MESSAGES = int(os.environ.get("LLM_HISTORY_MESSAGES", "16"))
MAX_CALL_SECONDS = int(os.environ.get("MAX_CALL_SECONDS", "600"))
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "60"))

# Piper's native rate; sent to the client in `ready` so no resampling is needed.
OUTPUT_SAMPLE_RATE = 22050

# Pause after the user stops talking before the bot answers (latency vs. cutting the user off).
VAD_STOP_SECS = 0.3
USER_SPEECH_TIMEOUT = 0.4

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
# Anything that is not a letter, digit, whitespace or common punctuation (emoji, markdown, symbols).
_UNSPEAKABLE_RE = re.compile(r"[^\w\s.,!?;:\-—–…'\"«»()]", re.UNICODE)


async def make_speakable(text: str, _text_type: str) -> str:
    """Strip reasoning blocks, markdown and emoji: Piper would read or choke on them."""
    text = _THINK_RE.sub("", text)
    return _UNSPEAKABLE_RE.sub("", text).replace("_", " ")


async def send_json(websocket: WebSocket, message: dict) -> None:
    try:
        await websocket.send_text(json.dumps(message, ensure_ascii=False))
    except Exception as e:  # the client may already be gone
        logger.warning(f"Could not send {message.get('type')}: {e}")


async def run_call(websocket: WebSocket, persona: Persona) -> None:
    """Run a call on an already accepted websocket until it ends; returns when it is over."""

    async def hangup():
        await worker.cancel(reason="client hangup")

    stt = SharedWhisperSTTService(
        device="cuda",
        compute_type="float16",
        settings=SharedWhisperSTTService.Settings(
            model=str(MODELS_DIR / "whisper" / f"faster-whisper-{os.environ.get('WHISPER_MODEL', 'small')}"),
            language=Language.RU,
        ),
    )
    llm = OLLamaLLMService(
        base_url=LLM_BASE_URL,
        settings=OLLamaLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=persona.prompt,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            extra={"reasoning_effort": "none"},
        ),
    )
    tts = SharedPiperTTSService(
        download_dir=MODELS_DIR / "piper",
        settings=SharedPiperTTSService.Settings(voice=persona.voice),
        length_scale=persona.length_scale,
        pitch_cents=persona.pitch_cents,
        tremolo_hz=persona.tremolo_hz,
        tremolo_depth=persona.tremolo_depth,
        text_transforms=[("*", make_speakable)],
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            serializer=RawPcmSerializer(OUTPUT_SAMPLE_RATE, hangup),
            allowed_origins=[],
        ),
    )

    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
            user_turn_strategies=UserTurnStrategies(
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT)]
            ),
            user_idle_timeout=IDLE_TIMEOUT_SECONDS,
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,
            UiEventsProcessor(),
            transport.output(),
            assistant_agg,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=INPUT_SAMPLE_RATE, audio_out_sample_rate=OUTPUT_SAMPLE_RATE
        ),
        enable_rtvi=False,
        idle_timeout_secs=None,  # idleness is handled by the user aggregator + MAX_CALL_SECONDS
    )

    history_limit = persona.history or LLM_HISTORY_MESSAGES

    async def end_call(reason: str):
        logger.info(f"Ending call: {reason}")
        await send_json(websocket, {"type": "end", "reason": reason})
        await worker.cancel(reason=reason)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # The LLM must know what it already said, so the greeting goes into the context too.
        context.add_message({"role": "assistant", "content": persona.greeting})
        await worker.queue_frame(TTSSpeakFrame(persona.greeting))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel(reason="client disconnected")

    @user_agg.event_handler("on_user_turn_message_added")
    async def on_user_message(agg, message):
        logger.debug(f"User: {message.content}")
        await send_json(
            websocket, {"type": "transcript", "role": "user", "text": message.content}
        )

    @user_agg.event_handler("on_user_turn_idle")
    async def on_user_idle(agg):
        await end_call("idle")

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(agg, message):
        logger.debug(f"Bot: {message.content}")
        messages = context.get_messages()
        if len(messages) <= history_limit:
            return
        kept = messages[-history_limit:]
        # Providers expect the dialog to start with a user message.
        while kept and kept[0]["role"] != "user":
            kept = kept[1:]
        context.set_messages(kept)

    async def limit_duration():
        await asyncio.sleep(MAX_CALL_SECONDS)
        await end_call("max_duration")

    await send_json(
        websocket,
        {"type": "ready", "sampleRate": OUTPUT_SAMPLE_RATE, "persona": persona.id},
    )

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    limit_task = asyncio.create_task(limit_duration())
    try:
        await runner.run()
    finally:
        limit_task.cancel()
