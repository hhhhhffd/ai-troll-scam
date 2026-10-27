"""Keeps only the GPU models the current engine needs in VRAM (12 GB fits one heavy model).

Each engine declares a set of resource names. Before a call `prepare()` unloads every loaded
resource the engine does not need, then loads the missing ones. Calls are serialized by the
single-call lock in app.py, so prepare() never runs twice at once.
"""

import asyncio
import os
from collections.abc import Awaitable, Callable

import aiohttp
from loguru import logger

from services import shared_models

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llm:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b-instruct")
# Loading a 10 GB TTS checkpoint from disk can take a minute or more.
SIDECAR_LOAD_TIMEOUT = float(os.environ.get("SIDECAR_LOAD_TIMEOUT", "300"))


class GpuError(Exception):
    """A model could not be loaded; the message is shown to the caller."""


class Resource:
    def __init__(self, name: str, load: Callable[[], Awaitable[None]], unload: Callable[[], Awaitable[None]]):
        self.name = name
        self.load = load
        self.unload = unload


def _ollama_root() -> str:
    return LLM_BASE_URL.removesuffix("/").removesuffix("/v1")


async def _ollama_request(path: str, body: dict) -> None:
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{_ollama_root()}{path}", json=body) as r:
            if r.status != 200:
                raise GpuError(f"Ollama ответил {r.status}: {(await r.text())[:200]}")


async def _load_ollama() -> None:
    # A real one-token chat, not just a load: the first generation after loading is slow otherwise.
    await _ollama_request(
        "/api/chat",
        {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": "Привет"}],
            "stream": False,
            "keep_alive": -1,
            "options": {"num_predict": 1},
        },
    )


async def _unload_ollama() -> None:
    await _ollama_request("/api/generate", {"model": LLM_MODEL, "prompt": "", "keep_alive": 0})


async def _load_whisper() -> None:
    await asyncio.to_thread(shared_models.load_whisper)


async def _unload_whisper() -> None:
    await asyncio.to_thread(shared_models.unload_whisper)


def _sidecar(base_url: str) -> tuple[Callable[[], Awaitable[None]], Callable[[], Awaitable[None]]]:
    """A TTS container whose launcher starts/stops the model server (see tts/launcher.py)."""

    async def call(action: str, timeout: float) -> None:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post(f"{base_url}/{action}") as r:
                    if r.status != 200:
                        raise GpuError(f"{base_url}/{action}: {r.status} {(await r.text())[:300]}")
        except (aiohttp.ClientError, TimeoutError) as e:
            raise GpuError(f"TTS-сервер {base_url} недоступен ({action}): {e!r}") from e

    async def load() -> None:
        await call("load", SIDECAR_LOAD_TIMEOUT)

    async def unload() -> None:
        await call("unload", 60)

    return load, unload


class GpuManager:
    def __init__(self, resources: list[Resource]):
        self._resources = {r.name: r for r in resources}
        self._loaded: set[str] = set()

    @property
    def loaded(self) -> list[str]:
        return sorted(self._loaded)

    async def reset(self) -> None:
        """Unload everything: after a bot restart Ollama and the sidecars may still hold VRAM."""
        for resource in self._resources.values():
            try:
                await resource.unload()
            except Exception as e:  # an unreachable sidecar holds no VRAM either
                logger.warning(f"Startup unload of {resource.name} skipped: {e}")
        self._loaded.clear()

    async def prepare(self, needed: frozenset[str]) -> None:
        # Unload first: loading the new model while the old one is resident would not fit.
        for name in sorted(self._loaded - needed):
            logger.info(f"GPU: unloading {name}")
            try:
                await self._resources[name].unload()
            except Exception as e:
                raise GpuError(f"Не удалось выгрузить {name}: {e}") from e
            self._loaded.discard(name)
        for name in sorted(needed - self._loaded):
            logger.info(f"GPU: loading {name}")
            try:
                await self._resources[name].load()
            except GpuError:
                raise
            except Exception as e:
                raise GpuError(f"Не удалось загрузить {name}: {e!r}") from e
            self._loaded.add(name)


def build_gpu_manager(sidecars: dict[str, str]) -> GpuManager:
    resources = [
        Resource("whisper", _load_whisper, _unload_whisper),
        Resource("ollama", _load_ollama, _unload_ollama),
    ]
    for name, url in sidecars.items():
        resources.append(Resource(name, *_sidecar(url)))
    return GpuManager(resources)
