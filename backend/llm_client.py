"""
LLM client abstraction for FactLens.

Supports both Google Gemini (google-genai SDK v2+) and OpenAI behind a unified interface.
Provider is selected via settings.llm_provider ("gemini" or "openai").

All calls are async, with retry logic via tenacity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences that some models wrap JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


class LLMClient:
    """
    Unified async LLM client.
    Instantiate once and reuse (connection-pool friendly).
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider
        self._gemini_client = None
        self._openai_client = None
        self._initialized = False

    def _init(self) -> None:
        if self._initialized:
            return
        if self._provider == "gemini":
            from google import genai
            self._gemini = genai.Client(api_key=settings.gemini_api_key)
        elif self._provider == "openai":
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {self._provider}")
        self._initialized = True

    # ── Text generation ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> str:
        """Generate text. Returns raw string (caller parses JSON if needed)."""
        self._init()

        if self._provider == "gemini":
            return await self._generate_gemini(
                system_prompt, user_prompt,
                model or settings.active_extraction_model,
                temperature, max_tokens,
            )
        else:
            return await self._generate_openai(
                system_prompt, user_prompt,
                model or settings.active_extraction_model,
                temperature, max_tokens,
            )

    async def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._gemini.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            ),
        )
        return response.text

    async def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if "gpt-4o" in model:
            kwargs["response_format"] = {"type": "json_object"}
        response = await self._openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    # ── JSON helpers ───────────────────────────────────────────────────────────

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.1,
        expect_array: bool = False,
    ) -> Any:
        """
        Generate and parse JSON. Returns parsed Python object.
        Handles markdown fences that some models add.
        """
        raw = await self.generate(
            system_prompt, user_prompt, model=model, temperature=temperature
        )
        cleaned = _strip_json_fences(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Try to find JSON array/object within the response
            match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            logger.warning(f"JSON parse failed. Raw response snippet: {raw[:300]}")
            raise ValueError(f"LLM returned invalid JSON: {e}") from e

    # ── Embeddings ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        self._init()

        if self._provider == "gemini":
            return await self._embed_gemini(texts)
        else:
            return await self._embed_openai(texts)

    async def _embed_gemini(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()

        def _batch_embed():
            # The new SDK supports a list of contents in one call
            response = self._gemini.models.embed_content(
                model=settings.embedding_model,
                contents=texts,
            )
            # response.embeddings is a list of ContentEmbedding objects
            return [emb.values for emb in response.embeddings]

        return await loop.run_in_executor(None, _batch_embed)

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = await self._openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# Global singleton — safe to share across async tasks
llm_client = LLMClient()
