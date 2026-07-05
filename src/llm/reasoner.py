"""
STRATUM-F1 — Live GenAI Reasoning Layer

Sends the structured "Race Briefing" produced by ``StrategyPromptBuilder`` to
Claude and returns a natural-language strategic recommendation. This closes the
loop between the simulation engine and human-readable tactical reasoning.

Credentials are resolved by the Anthropic SDK from the environment
(``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``, or an ``ant auth login``
profile) — no key is hardcoded here.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"

# Which backend get_reasoner() uses when none is specified.
# Override with the STRATUM_LLM_PROVIDER environment variable.
DEFAULT_PROVIDER = "gemini"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Formula 1 Race Strategist embedded on the pit wall. "
    "You have exact knowledge of the tyre degradation curves and access to a "
    "simulation engine that has already projected future race times for every "
    "viable pit window. Reason like a race engineer: weigh traffic and dirty "
    "air, the tyre cliff, undercut/overcut threats from nearby rivals, and the "
    "optimizer's projections. Give a short step-by-step rationale, then commit "
    "to a single clear call: PIT NOW or STAY OUT."
)


class LLMUnavailableError(RuntimeError):
    """Raised when the Anthropic SDK or credentials are not available."""


@dataclass
class ReasoningResult:
    """The outcome of a live reasoning call."""

    text: str
    model: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class StrategyReasoner:
    """Calls Claude to turn a race briefing into a live strategy recommendation."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 16000,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the reasoner.

        Args:
            model: Claude model ID. Defaults to the latest Opus.
            effort: Thinking/effort level (low | medium | high | max).
            max_tokens: Maximum tokens in the response.
            system_prompt: Persona/system prompt for the strategist.
        """
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._client = self._build_client()
        # Populated after stream_reason() completes, for token accounting.
        self.last_result: Optional[ReasoningResult] = None

    @staticmethod
    def _build_client():
        """Construct the Anthropic client, raising a clear error if unavailable."""
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMUnavailableError(
                "The 'anthropic' package is not installed. Run "
                "'pip install anthropic' to enable live LLM reasoning."
            ) from exc

        try:
            # Resolves credentials from the environment or an `ant auth` profile.
            return anthropic.Anthropic()
        except Exception as exc:  # pragma: no cover - env dependent
            raise LLMUnavailableError(
                f"Failed to initialize the Anthropic client: {exc}"
            ) from exc

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the SDK is importable (does not validate credentials)."""
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def reason(self, prompt: str) -> ReasoningResult:
        """Send a race briefing to Claude and return its recommendation.

        Args:
            prompt: The markdown race briefing from StrategyPromptBuilder.
                The system persona is applied separately, so pass the briefing
                without a prepended system prompt.

        Returns:
            A ReasoningResult with the strategist's text and token usage.
        """
        logger.info("Requesting live strategy reasoning from %s", self.model)

        import anthropic  # local import; presence guaranteed by _build_client

        try:
            # Stream to stay under HTTP timeouts while adaptive thinking runs.
            with self._client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except anthropic.APIStatusError as exc:
            logger.error("Anthropic API error (%s): %s", exc.status_code, exc.message)
            raise
        except anthropic.APIConnectionError as exc:
            logger.error("Network error contacting Anthropic: %s", exc)
            raise

        text = "".join(
            block.text for block in message.content if block.type == "text"
        )

        return ReasoningResult(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
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
        self.last_result = None
        parts: list[str] = []

        with self._client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                yield text
            message = stream.get_final_message()

        self.last_result = ReasoningResult(
            text="".join(parts),
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )


def get_reasoner(provider: Optional[str] = None):
    """Return a reasoner for the chosen provider.

    Args:
        provider: "gemini" (free), "claude"/"anthropic", or None to read the
            STRATUM_LLM_PROVIDER environment variable (default: gemini).

    Returns:
        A reasoner exposing ``reason(prompt) -> ReasoningResult``.
    """
    provider = (provider or os.environ.get("STRATUM_LLM_PROVIDER") or DEFAULT_PROVIDER).lower()

    if provider in ("gemini", "google"):
        from .gemini_reasoner import GeminiReasoner
        return GeminiReasoner()
    if provider in ("claude", "anthropic"):
        return StrategyReasoner()

    raise LLMUnavailableError(
        f"Unknown LLM provider '{provider}'. Use 'gemini' or 'claude'."
    )
