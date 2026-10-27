"""Voice engines offered in the UI: what each one is built from and what it needs to run.

`local` and `gemini` are the original engines. The others are cascades
STT -> cloud LLM -> TTS that differ by TTS; STT and LLM providers come from env.
"""

import os
from dataclasses import dataclass

import aiohttp

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
FISH_API_KEY = os.environ.get("FISH_API_KEY", "")
FISH_REFERENCE_ID = os.environ.get("FISH_REFERENCE_ID", "")

# auto = Deepgram Flux if its key is set, else Groq Whisper, else local Whisper on the GPU.
CASCADE_STT = os.environ.get("CASCADE_STT", "auto")
CLOUD_LLM_PROVIDER = os.environ.get("CLOUD_LLM_PROVIDER", "gemini")

SIDECARS = {
    "higgs": os.environ.get("HIGGS_URL", "http://tts-higgs:9000"),
    "espeech": os.environ.get("ESPEECH_URL", "http://tts-espeech:9000"),
}


@dataclass(frozen=True)
class Engine:
    id: str
    name: str
    tts: str  # piper | gemini_live | elevenlabs | fish | higgs | espeech


ENGINES = [
    Engine("local", "Локально (Whisper + Qwen + Piper)", "piper"),
    Engine("gemini", "Gemini Live", "gemini_live"),
    Engine("cloud", "Облако (ElevenLabs v4)", "elevenlabs"),
    Engine("fish", "Fish S2 Pro (облако)", "fish"),
    Engine("higgs", "Higgs TTS 3 (своя GPU)", "higgs"),
    Engine("espeech", "ESpeech (своя GPU)", "espeech"),
]
ENGINES_BY_ID = {e.id: e for e in ENGINES}
CASCADE_TTS = {"elevenlabs", "fish", "higgs", "espeech"}


def cascade_stt() -> str:
    if CASCADE_STT != "auto":
        return CASCADE_STT
    if DEEPGRAM_API_KEY:
        return "deepgram"
    if GROQ_API_KEY:
        return "groq"
    return "whisper"


def gpu_needs(engine: Engine) -> frozenset[str]:
    """GPU resources (see services/gpu.py) that must be resident during a call on `engine`."""
    if engine.id == "local":
        return frozenset({"whisper", "ollama"})
    if engine.tts not in CASCADE_TTS:
        return frozenset()
    needs = {engine.tts} if engine.tts in SIDECARS else set()
    if cascade_stt() == "whisper":
        needs.add("whisper")
    return frozenset(needs)


def _llm_problem() -> str | None:
    if CLOUD_LLM_PROVIDER == "gemini":
        return None if GEMINI_API_KEY else "нужен GEMINI_API_KEY"
    if CLOUD_LLM_PROVIDER == "groq":
        return None if GROQ_API_KEY else "нужен GROQ_API_KEY"
    return f"неизвестный CLOUD_LLM_PROVIDER={CLOUD_LLM_PROVIDER}"


def _stt_problem(full_cloud: bool) -> str | None:
    stt = cascade_stt()
    if stt == "deepgram":
        return None if DEEPGRAM_API_KEY else "нужен DEEPGRAM_API_KEY"
    if stt == "groq":
        return None if GROQ_API_KEY else "нужен GROQ_API_KEY"
    if stt == "whisper":
        return "нужен DEEPGRAM_API_KEY или GROQ_API_KEY" if full_cloud else None
    return f"неизвестный CASCADE_STT={stt}"


async def _sidecar_up(url: str) -> bool:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1.5)) as session:
            async with session.get(f"{url}/status") as r:
                return r.status == 200
    except (aiohttp.ClientError, TimeoutError):
        return False


async def problem(engine: Engine) -> str | None:
    """Why `engine` can't take a call right now (shown in the UI), or None if it can."""
    if engine.id == "local":
        return None
    if engine.id == "gemini":
        return None if GEMINI_API_KEY else "нужен GEMINI_API_KEY"
    if engine.tts == "elevenlabs" and not (ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        return "нужны ELEVENLABS_API_KEY и ELEVENLABS_VOICE_ID"
    if engine.tts == "fish" and not FISH_API_KEY:
        return "нужен FISH_API_KEY"
    if reason := _llm_problem() or _stt_problem(full_cloud=engine.tts == "elevenlabs"):
        return reason
    if engine.tts in SIDECARS and not await _sidecar_up(SIDECARS[engine.tts]):
        return f"не запущен контейнер tts-{engine.tts} (профиль {engine.tts})"
    return None
