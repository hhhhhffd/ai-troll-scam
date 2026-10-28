"""Groq LLM over several models: a rate-limited or failing model hands the same turn to the next one.

Free-tier limits (tokens per minute and per day) are per model, so a list multiplies what a call can use.
The local Ollama model can close the list; it is OpenAI-compatible, so it rides the same request code.
"""

from loguru import logger
from openai import APIStatusError
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.groq.llm import GroqLLMService


class GroqFallbackLLMService(GroqLLMService):
    def __init__(
        self,
        *,
        models: list[tuple[str, str]],
        local_base_url: str | None = None,
        local_model: str | None = None,
        **kwargs,
    ):
        """`models`: Groq (model, reasoning_effort) in order of preference, tried from the first on every
        turn; `local_model` at `local_base_url` (Ollama) is the last resort."""
        super().__init__(**kwargs)
        self._backends = [(self._client, model, effort) for model, effort in models]
        if local_model:
            local = self.create_client(api_key="ollama", base_url=local_base_url)
            self._backends.append((local, local_model, "none"))

    async def get_chat_completions(self, context: LLMContext):
        for i, (client, model, effort) in enumerate(self._backends):
            self._client = client
            self._settings.model = model
            self._settings.reasoning_effort = effort
            try:
                return await super().get_chat_completions(context)
            except APIStatusError as e:
                # 4xx other than 429 is our request's fault; another model won't fix it.
                if i == len(self._backends) - 1 or (e.status_code < 500 and e.status_code != 429):
                    raise
                logger.warning(f"{self}: {model} failed ({e.status_code}), trying {self._backends[i + 1][1]}")
