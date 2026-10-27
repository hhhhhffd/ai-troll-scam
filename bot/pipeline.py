"""One phone call = one Pipecat pipeline.

Engine "local": ws in -> VAD/STT -> LLM -> TTS -> ws out.
Engine "gemini": ws in -> Gemini Live (speech-to-speech) -> ws out.
Cascade engines (cloud, fish, higgs, espeech): ws in -> STT -> cloud LLM -> expressive TTS -> ws out.
"""

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import WebSocket
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.elevenlabs.dialogue.tts import ElevenLabsDialogueTTSService
from pipecat.services.fish.tts import FishAudioTTSService
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import engines
from engines import Engine
from personas import Persona
from services.call_io import INPUT_SAMPLE_RATE, RawPcmSerializer, UiEventsProcessor
from services.gigaam_stt import GigaamSTTService
from services.local_tts import SidecarTTSService
from services.shared_models import SharedPiperTTSService, SharedWhisperSTTService

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "100"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.8"))
LLM_HISTORY_MESSAGES = int(os.environ.get("LLM_HISTORY_MESSAGES", "16"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-live")
GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
# "none" for Qwen; gpt-oss only accepts low/medium/high.
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "none")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3-turbo")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "flux-general-multi")
ELEVENLABS_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_v4_turbo")
FISH_MODEL = os.environ.get("FISH_MODEL", "s2.1-pro-free")
CLOUD_LLM_MAX_TOKENS = int(os.environ.get("CLOUD_LLM_MAX_TOKENS", "200"))
# Cloud LLMs handle long contexts cheaply, so cascades remember (almost) the whole call.
CLOUD_HISTORY_MESSAGES = int(os.environ.get("CLOUD_HISTORY_MESSAGES", "80"))
MAX_CALL_SECONDS = int(os.environ.get("MAX_CALL_SECONDS", "600"))
IDLE_TIMEOUT_SECONDS = int(os.environ.get("IDLE_TIMEOUT_SECONDS", "60"))

# Native output rates of the engines; sent to the client in `ready` so no resampling is needed.
OUTPUT_SAMPLE_RATE = 22050  # Piper
GEMINI_OUTPUT_SAMPLE_RATE = 24000
CASCADE_OUTPUT_SAMPLE_RATE = 24000  # ElevenLabs pcm_24000, Fish pcm, sidecars (tts/CONTRACT.md)

# Expressive TTS engines act out these tags (each TTS maps them to its own syntax) and never say them.
EMOTION_PROMPT = """

Голос. Твои реплики озвучивает выразительный синтез речи. Перед фразой, где интонация действительно меняется, можно поставить один тег эмоции в квадратных скобках, только из этого списка: [calm] [warm] [irritated] [angry] [laughs] [sighs] [confused] [sad] [surprised]. Пример: «[irritated] Так, подождите. Кто вы такой?». Ставь теги редко и только по поводу, без тега фраза звучит спокойно. Никаких других скобок и ремарок."""

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


def _local_stt(kind: str):
    """Segmented STT on this machine's GPU: shared Whisper or the GigaAM sidecar."""
    if kind == "gigaam":
        return GigaamSTTService(base_url=engines.SIDECARS["gigaam"])
    return SharedWhisperSTTService(
        device="cuda",
        compute_type="float16",
        settings=SharedWhisperSTTService.Settings(
            model=str(MODELS_DIR / "whisper" / f"faster-whisper-{os.environ.get('WHISPER_MODEL', 'large-v3-turbo')}"),
            language=Language.RU,
        ),
    )


def build_local(persona: Persona):
    """Cascade engine. Returns (stages between input and UI events, user_agg, assistant_agg, context)."""
    stt = _local_stt(engines.LOCAL_STT)
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
    return [stt, user_agg, llm, tts], user_agg, assistant_agg, context


def build_gemini(persona: Persona):
    """Gemini Live engine: the model does STT, dialog and TTS; same return shape as build_local."""
    llm = GeminiLiveLLMService(
        api_key=GEMINI_API_KEY,
        settings=GeminiLiveLLMService.Settings(
            model=GEMINI_MODEL,
            # Native-audio models pick the language themselves, so pin it in the prompt.
            system_instruction=f"{persona.prompt}\n\nВсегда говори только по-русски.",
            voice=persona.gemini_voice,
            language=Language.RU,
        ),
    )
    # Greeting: the bot speaks first. The model gets it as the opening user turn.
    context = LLMContext(
        [
            {
                "role": "user",
                "content": f"Ты снял трубку. Скажи вслух дословно только это и замолчи: «{persona.greeting}»",
            }
        ]
    )
    # Gemini's server-side VAD drives turns and barge-in; Silero only supplies the local
    # user-turn frames that the idle timeout needs.
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_idle_timeout=IDLE_TIMEOUT_SECONDS,
        ),
    )
    return [user_agg, llm], user_agg, assistant_agg, context


def _cascade_stt(engine: Engine):
    """Returns (stt service, user aggregator params) for the configured cascade STT."""
    stt = engines.cascade_stt(engine)
    if stt == "deepgram":
        service = DeepgramFluxSTTService(
            api_key=engines.DEEPGRAM_API_KEY,
            settings=DeepgramFluxSTTService.Settings(
                model=DEEPGRAM_MODEL, language_hints=[Language.RU]
            ),
        )
        # Flux decides when a turn starts and ends (and interrupts the bot on StartOfTurn).
        return service, LLMUserAggregatorParams(
            user_turn_strategies=service.recommended_user_turn_strategies(enable_interruptions=True),
            user_idle_timeout=IDLE_TIMEOUT_SECONDS,
        )
    if stt == "groq":
        service = GroqSTTService(
            api_key=engines.GROQ_API_KEY,
            settings=GroqSTTService.Settings(model=GROQ_STT_MODEL, language=Language.RU),
        )
    else:
        service = _local_stt(stt)
    # Segmented STT: Silero finds the end of speech, like the local engine.
    return service, LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=USER_SPEECH_TIMEOUT)]
        ),
        user_idle_timeout=IDLE_TIMEOUT_SECONDS,
    )


def _cascade_llm(prompt: str):
    if engines.CLOUD_LLM_PROVIDER == "groq":
        return GroqLLMService(
            api_key=engines.GROQ_API_KEY,
            settings=GroqLLMService.Settings(
                model=GROQ_MODEL,
                system_instruction=prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=CLOUD_LLM_MAX_TOKENS,
                extra={"reasoning_effort": GROQ_REASONING_EFFORT},
            ),
        )
    return GoogleLLMService(
        api_key=engines.GEMINI_API_KEY,
        settings=GoogleLLMService.Settings(
            model=GEMINI_TEXT_MODEL,
            system_instruction=prompt,
            temperature=LLM_TEMPERATURE,
            max_tokens=CLOUD_LLM_MAX_TOKENS,
        ),
    )


def _cascade_tts(engine: Engine, persona: Persona):
    if engine.tts == "elevenlabs":
        return ElevenLabsDialogueTTSService(
            api_key=engines.ELEVENLABS_API_KEY,
            sample_rate=CASCADE_OUTPUT_SAMPLE_RATE,
            settings=ElevenLabsDialogueTTSService.Settings(
                model=ELEVENLABS_MODEL, voice=engines.ELEVENLABS_VOICE_ID, language=Language.RU
            ),
        )
    if engine.tts == "fish":
        return FishAudioTTSService(
            api_key=engines.FISH_API_KEY,
            sample_rate=CASCADE_OUTPUT_SAMPLE_RATE,
            # Fish rejects an empty reference_id; None means its default voice.
            settings=FishAudioTTSService.Settings(model=FISH_MODEL, voice=engines.FISH_REFERENCE_ID or None),
        )
    return SidecarTTSService(base_url=engines.SIDECARS[engine.tts], voice=persona.id)


def build_cascade(persona: Persona, engine: Engine):
    """STT -> cloud LLM -> expressive TTS; same return shape as build_local."""
    stt, user_params = _cascade_stt(engine)
    llm = _cascade_llm(persona.prompt + EMOTION_PROMPT)
    tts = _cascade_tts(engine, persona)
    context = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(context, user_params=user_params)
    return [stt, user_agg, llm, tts], user_agg, assistant_agg, context


async def run_call(websocket: WebSocket, persona: Persona, engine: Engine) -> None:
    """Run a call on an already accepted websocket until it ends; returns when it is over."""

    async def hangup():
        await worker.cancel(reason="client hangup")

    gemini = engine.id == "gemini"
    if gemini:
        output_sample_rate = GEMINI_OUTPUT_SAMPLE_RATE
        stages, user_agg, assistant_agg, context = build_gemini(persona)
        history_limit = None  # Gemini keeps its own session history
    elif engine.id == "local":
        output_sample_rate = OUTPUT_SAMPLE_RATE
        stages, user_agg, assistant_agg, context = build_local(persona)
        history_limit = persona.history or LLM_HISTORY_MESSAGES
    else:
        output_sample_rate = CASCADE_OUTPUT_SAMPLE_RATE
        stages, user_agg, assistant_agg, context = build_cascade(persona, engine)
        history_limit = CLOUD_HISTORY_MESSAGES

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=INPUT_SAMPLE_RATE,
            audio_out_sample_rate=output_sample_rate,
            serializer=RawPcmSerializer(output_sample_rate, hangup),
            allowed_origins=[],
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            *stages,
            UiEventsProcessor(),
            transport.output(),
            assistant_agg,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=INPUT_SAMPLE_RATE, audio_out_sample_rate=output_sample_rate
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
        if gemini:
            await worker.queue_frame(LLMRunFrame())
        else:
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
        if history_limit is None or len(messages) <= history_limit:
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
        {"type": "ready", "sampleRate": output_sample_rate, "persona": persona.id},
    )

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    limit_task = asyncio.create_task(limit_duration())
    try:
        await runner.run()
    finally:
        limit_task.cancel()
