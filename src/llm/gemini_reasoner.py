"""
STRATUM-F1 — Gemini Reasoning Backend

A free-tier alternative to the Claude backend. Uses Google's Gemini models
(``gemini-2.0-flash`` by default, which is on the free tier) to turn a race
briefing into a live strategy recommendation.

Get a free API key at https://aistudio.google.com/apikey and set it as
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``).
"""

import logging
import os
from typing import Optional

from .reasoner import ReasoningResult, LLMUnavailableError, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


class GeminiReasoner:
    """Calls Google Gemini to turn a race briefing into a recommendation.

    Mirrors the interface of ``StrategyReasoner`` (a ``reason(prompt)`` method
    returning a ``ReasoningResult``) so it can be swapped in transparently.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the Gemini reasoner.

        Args:
            model: Gemini model ID. Defaults to a free-tier flash model.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.
            system_prompt: Persona/system prompt for the strategist.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._client = self._build_client()
        # Populated after stream_reason() completes, for token accounting.
        self.last_result: Optional[ReasoningResult] = None

    @staticmethod
    def _build_client():
        """Construct the Gemini client, raising a clear error if unavailable."""
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMUnavailableError(
                "The 'google-genai' package is not installed. Run "
                "'pip install google-genai' to enable Gemini reasoning."
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMUnavailableError(
                "No Gemini API key found. Get a free key at "
                "https://aistudio.google.com/apikey and set GEMINI_API_KEY."
            )

        try:
            return genai.Client(api_key=api_key)
        except Exception as exc:  # pragma: no cover - env dependent
            raise LLMUnavailableError(
                f"Failed to initialize the Gemini client: {exc}"
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the SDK is importable (does not validate the key)."""
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False
        return True

    def reason(self, prompt: str) -> ReasoningResult:
        """Send a race briefing to Gemini and return its recommendation.

        Args:
            prompt: The markdown race briefing from StrategyPromptBuilder.

        Returns:
            A ReasoningResult with the strategist's text and token usage.
        """
        logger.info("Requesting live strategy reasoning from %s", self.model)

        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )

        usage = getattr(response, "usage_metadata", None)
        return ReasoningResult(
            text=response.text or "",
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )

    def stream_reason(self, prompt: str):
        """Stream the recommendation token-by-token.

        Yields text chunks as they arrive. After the generator is exhausted,
        ``self.last_result`` holds the full text and token usage.

        Args:
            prompt: The markdown race briefing from StrategyPromptBuilder.

        Yields:
            Incremental text chunks (str).
        """
        logger.info("Streaming live strategy reasoning from %s", self.model)
        from google.genai import types

        self.last_result = None
        parts = []
        last_usage = None

        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        for chunk in stream:
            if chunk.text:
                parts.append(chunk.text)
                yield chunk.text
            if getattr(chunk, "usage_metadata", None):
                last_usage = chunk.usage_metadata

        self.last_result = ReasoningResult(
            text="".join(parts),
            model=self.model,
            input_tokens=getattr(last_usage, "prompt_token_count", None),
            output_tokens=getattr(last_usage, "candidates_token_count", None),
        )
