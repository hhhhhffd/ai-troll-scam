"""Whisper and Piper services that reuse models loaded once at startup.

WhisperSTTService/PiperTTSService load their model inside the constructor, i.e. on every
call. Loading Whisper onto the GPU takes seconds, so we load once and hand the same
instance to every per-call service.
"""

import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator
from functools import partial
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
from piper import PiperVoice, SynthesisConfig
from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.piper import tts as piper_tts_module
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

_whisper_model: WhisperModel | None = None
_piper_voices: dict[str, PiperVoice] = {}


def load_models(whisper_path: Path, piper_dir: Path, voices: set[str]) -> None:
    global _whisper_model
    _whisper_model = WhisperModel(str(whisper_path), device="cuda", compute_type="float16")
    # WhisperSTTService does not expose beam_size; greedy decoding is noticeably faster
    # and accurate enough for short phone phrases.
    _whisper_model.transcribe = partial(_whisper_model.transcribe, beam_size=1)
    for voice in voices:
        _piper_voices[voice] = PiperVoice.load(piper_dir / f"{voice}.onnx")


class _CachedPiperVoice:
    """Stands in for `PiperVoice` inside pipecat's piper module: `load` returns the shared voice."""

    @staticmethod
    def load(model_path, *_args, **_kwargs) -> PiperVoice:
        return _piper_voices[Path(model_path).stem]


class SharedWhisperSTTService(WhisperSTTService):
    def _load(self):
        assert _whisper_model is not None, "load_models() was not called"
        self._model = _whisper_model
        unsupported = self._unsupported_language()
        if unsupported:
            raise ValueError(unsupported)


class SharedPiperTTSService(PiperTTSService):
    """Piper with a preloaded voice plus an "elderly voice" effect.

    - length_scale slows speech down (native Piper parameter).
    - pitch_cents: the audio is declared to be at a proportionally lower sample rate and the
      stateful resampler in pipecat converts it to the output rate. That lowers the pitch
      (and slows it a bit more); no extra processing step, no chunk-boundary clicks.
    - tremolo: amplitude modulation, phase kept continuous across chunks of one utterance.
    """

    def __init__(
        self,
        *,
        length_scale: float = 1.0,
        pitch_cents: float = 0,
        tremolo_hz: float = 0,
        tremolo_depth: float = 0,
        **kwargs,
    ):
        # PiperTTSService.__init__ calls the module-level `PiperVoice.load`; swap it for the
        # cached voice just for the duration of construction.
        original = piper_tts_module.PiperVoice
        piper_tts_module.PiperVoice = _CachedPiperVoice
        try:
            super().__init__(**kwargs)
        finally:
            piper_tts_module.PiperVoice = original
        self._syn_config = SynthesisConfig(length_scale=length_scale)
        self._pitch_factor = 2 ** (pitch_cents / 1200)
        self._tremolo_hz = tremolo_hz
        self._tremolo_depth = tremolo_depth

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        voice = self._voice
        rate = round(voice.config.sample_rate * self._pitch_factor)
        samples_done = 0

        def next_chunk(it):
            return next(it, None)

        async def chunks() -> AsyncIterator[bytes]:
            nonlocal samples_done
            iterator = iter(voice.synthesize(text, self._syn_config))
            while (chunk := await asyncio.to_thread(next_chunk, iterator)) is not None:
                audio = chunk.audio_int16_bytes
                if self._tremolo_depth > 0:
                    samples = np.frombuffer(audio, dtype=np.int16)
                    t = (samples_done + np.arange(len(samples))) / rate
                    gain = 1 - self._tremolo_depth * (0.5 + 0.5 * np.sin(2 * math.pi * self._tremolo_hz * t))
                    audio = (samples * gain).astype(np.int16).tobytes()
                    samples_done += len(samples)
                yield audio

        try:
            await self.start_tts_usage_metrics(text)
            async for frame in self._stream_audio_frames_from_iterator(
                chunks(), in_sample_rate=rate, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
