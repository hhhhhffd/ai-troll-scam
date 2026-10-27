"""TTS over a local sidecar (Higgs, ESpeech): streamed raw PCM, see tts/CONTRACT.md."""

from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService

SIDECAR_SAMPLE_RATE = 24000


class SidecarTTSService(TTSService):
    def __init__(self, *, base_url: str, voice: str, **kwargs):
        super().__init__(
            sample_rate=SIDECAR_SAMPLE_RATE,
            push_start_frame=True,
            push_stop_frames=True,
            # The sidecar picks the model itself; voice selects its reference set.
            settings=TTSSettings(model=None, voice=voice, language=None),
            **kwargs,
        )
        self._url = f"{base_url}/v1/audio/speech"
        self._voice = voice
        self._session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def cleanup(self):
        await super().cleanup()
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if self._session is None:
            # No total timeout: a long phrase streams for as long as it is spoken.
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(sock_connect=5, sock_read=30))
        try:
            await self.start_tts_usage_metrics(text)
            async with self._session.post(self._url, json={"input": text, "voice": self._voice}) as r:
                if r.status != 200:
                    error = (await r.text())[:300]
                    logger.error(f"{self} sidecar error {r.status}: {error}")
                    yield ErrorFrame(error=f"TTS sidecar {r.status}: {error}")
                    return
                leftover = b""
                async for chunk in r.content.iter_chunked(self.chunk_size):
                    # Chunk boundaries may split a 16-bit sample.
                    data = leftover + chunk
                    cut = len(data) - len(data) % 2
                    data, leftover = data[:cut], data[cut:]
                    if data:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(data, self.sample_rate, 1, context_id=context_id)
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.error(f"{self} sidecar unreachable: {e!r}")
            yield ErrorFrame(error=f"TTS sidecar unreachable: {e!r}")
        finally:
            await self.stop_ttfb_metrics()
