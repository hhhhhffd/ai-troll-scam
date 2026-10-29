"""One phone call = one Pipecat pipeline.

ws in -> Groq Whisper -> LLM (Groq models, local Ollama model last) -> ElevenLabs v4 -> ws out.
"""

import asyncio
import json
import os
import re

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
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from persona import Persona
from services.call_io import INPUT_SAMPLE_RATE, RawPcmSerializer, UiEventsProcessor
from services.groq_fallback_llm import GroqFallbackLLMService
from services.text_tags import strip_angle_tags

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_v4_turbo")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
# Comma-separated fallback list "model[:reasoning_effort]"; the effort defaults to GROQ_REASONING_EFFORT.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
# "none" for Qwen; gpt-oss only accepts low/medium/high.
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "none")
GROQ_MODELS = [
    (name, effort or GROQ_REASONING_EFFORT)
    for name, _, effort in (item.strip().partition(":") for item in GROQ_MODEL.split(","))
    if name
]
# 1 = the local Ollama model answers when every Groq model is rate-limited.
LOCAL_LLM_FALLBACK = os.environ.get("CLOUD_LLM_LOCAL_FALLBACK", "0") == "1"
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.8"))
LLM_MAX_TOKENS = int(os.environ.get("CLOUD_LLM_MAX_TOKENS", "200"))
HISTORY_MESSAGES = int(os.environ.get("CLOUD_HISTORY_MESSAGES", "30"))
MAX_CALL_SECONDS = int(os.environ.get("MAX_CALL_SECONDS", "600"))
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "60"))
# Loudness a sound needs to count as speech, 0..1 = -110..-10 LUFS (Pipecat default 0.6 = -50 LUFS).
# Higher so a TV or people talking across the room is not heard, only a voice close to the mic.
VAD_MIN_VOLUME = float(os.environ.get("VAD_MIN_VOLUME", "0.75"))

OUTPUT_SAMPLE_RATE = 24000  # ElevenLabs pcm_24000; sent to the client in `ready`

# Eleven v4 acts inline audio tags in square brackets; it reads anything else in brackets or <> aloud.
EMOTION_PROMPT = """

Голос. Твои реплики озвучивает выразительный синтез речи. Перед фразой, где интонация действительно меняется, можно поставить тег в квадратных скобках, только из этого списка: [curious] [confused] [annoyed] [shouts] [whispers] [sighs] [laughs] [thoughtful] [short pause]. Тег описывает, как произнести следующие слова, и сам не произносится. Пример: «Постойте. Вы сказали, что перевод уже прошёл, а теперь говорите, что только пройдёт. [annoyed] Так какой из двух вариантов?».
Тег нужен примерно в каждой третьей-четвёртой реплике, остальные звучат спокойно и вообще без тегов. Не начинай реплики с тега по привычке, не больше двух тегов на реплику, никогда не отвечай одним тегом без слов. Тег идёт после причины: сначала звонящий давит или путается, потом ты раздражаешься; [shouts] только как короткая вспышка, когда звонящий уже не первый раз давит, хамит или требует одно и то же (на первое требование не кричи), и не чаще раза за несколько реплик; дальше возвращайся к обычному тону. Не пиши слова капсом. Никаких других скобок, угловых скобок вроде <кричит>, ремарок и эмодзи."""

# Pause after the user stops talking before the bot answers (latency vs. cutting the user off).
VAD_STOP_SECS = 0.3
USER_SPEECH_TIMEOUT = 0.4
# While the granny speaks only this many transcribed words interrupt her, so a sneeze or a TV
# in the background (VAD alone fires on any sound) doesn't cut her off; when she is silent one word starts a turn.
INTERRUPT_MIN_WORDS = 3

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def missing_keys() -> list[str]:
    """Env keys a call can't run without."""
    keys = {
        "GROQ_API_KEY": GROQ_API_KEY,
        "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
        "ELEVENLABS_VOICE_ID": ELEVENLABS_VOICE_ID,
    }
    return [name for name, value in keys.items() if not value]


async def drop_stage_directions(text: str, _text_type: str) -> str:
    """ElevenLabs keeps its "[...]" audio tags but must never receive "<кричит>"-style ones."""
    return strip_angle_tags(_THINK_RE.sub("", text))


async def send_json(websocket: WebSocket, message: dict) -> None:
    try:
        await websocket.send_text(json.dumps(message, ensure_ascii=False))
    except Exception as e:  # the client may already be gone
        logger.warning(f"Could not send {message.get('type')}: {e}")


def build_services(persona: Persona):
    """Returns (stt, llm, tts, user aggregator params)."""
    stt = GroqSTTService(
        api_key=GROQ_API_KEY,
        settings=GroqSTTService.Settings(model=GROQ_STT_MODEL, language=Language.RU),
    )
    llm = GroqFallbackLLMService(
        api_key=GROQ_API_KEY,
        models=GROQ_MODELS,
        local_base_url=LLM_BASE_URL if LOCAL_LLM_FALLBACK else None,
        local_model=LLM_MODEL if LOCAL_LLM_FALLBACK else None,
        settings=GroqFallbackLLMService.Settings(
            model=GROQ_MODELS[0][0],
            system_instruction=persona.prompt + EMOTION_PROMPT,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        ),
    )
    tts = ElevenLabsDialogueTTSService(
        api_key=ELEVENLABS_API_KEY,
        sample_rate=OUTPUT_SAMPLE_RATE,
        settings=ElevenLabsDialogueTTSService.Settings(
            model=ELEVENLABS_MODEL, voice=ELEVENLABS_VOICE_ID, language=Language.RU
        ),
        text_transforms=[("*", drop_stage_directions)],
    )
    # Segmented STT: Silero finds the end of speech.
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS, min_volume=VAD_MIN_VOLUME)),
        user_turn_strategies=UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(min_words=INTERRUPT_MIN_WORDS)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT)]
        ),
        user_idle_timeout=IDLE_TIMEOUT_SECONDS,
    )
    return stt, llm, tts, user_params


async def run_call(websocket: WebSocket, persona: Persona) -> None:
    """Run a call on an already accepted websocket until it ends; returns when it is over."""

    async def hangup():
        await worker.cancel(reason="client hangup")

    stt, llm, tts, user_params = build_services(persona)
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(context, user_params=user_params)

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

    async def end_call(reason: str):
        logger.info(f"Ending call: {reason}")
        await send_json(websocket, {"type": "end", "reason": reason})
        await worker.cancel(reason=reason)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # The assistant aggregator commits TTSSpeakFrame text to the context once it is spoken.
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
        if len(messages) <= HISTORY_MESSAGES:
            return
        kept = messages[-HISTORY_MESSAGES:]
        # Providers expect the dialog to start with a user message.
        while kept and kept[0]["role"] != "user":
            kept = kept[1:]
        context.set_messages(kept)

    async def limit_duration():
        await asyncio.sleep(MAX_CALL_SECONDS)
        await end_call("max_duration")

    await send_json(websocket, {"type": "ready", "sampleRate": OUTPUT_SAMPLE_RATE})

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    limit_task = asyncio.create_task(limit_duration())
    try:
        await runner.run()
    finally:
        limit_task.cancel()
