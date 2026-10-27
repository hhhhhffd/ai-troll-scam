"""Segmented STT over the GigaAM-v3 sidecar (stt/gigaam): one WAV segment in, punctuated text out."""

from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


class GigaamSTTService(SegmentedSTTService):
    def __init__(self, *, base_url: str, **kwargs):
        super().__init__(settings=STTSettings(model=None, language=Language.RU), **kwargs)
        self._url = f"{base_url}/v1/transcribe"
        self._session: aiohttp.ClientSession | None = None

    def can_generate_metrics(self) -> bool:
        return True

    async def cleanup(self):
        await super().cleanup()
        if self._session:
            await self._session.close()
            self._session = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        try:
            await self.start_processing_metrics()
            async with self._session.post(self._url, data=audio) as r:
                if r.status != 200:
                    error = (await r.text())[:300]
                    logger.error(f"{self} sidecar error {r.status}: {error}")
                    yield ErrorFrame(error=f"GigaAM {r.status}: {error}")
                    return
                text = (await r.json())["text"]
            await self.stop_processing_metrics()
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.error(f"{self} sidecar unreachable: {e!r}")
            yield ErrorFrame(error=f"GigaAM unreachable: {e!r}")
            return
        if text:
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language=Language.RU)
