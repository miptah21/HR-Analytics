"""
Multi-Provider LLM Client with Automatic Fallback (G-17)

Provides a resilient LLM interface that chains through multiple API keys
and providers to maximize availability. The fallback order is:

    Gemini (primary) → Gemini (backup) → Groq → Groq_2 → Groq_3

Each provider attempt is logged. If all providers fail, a safe default
message is returned so the prediction endpoint never errors on LLM issues.

Usage:
    from src.llm import generate_text
    result = await generate_text("Your prompt here")
"""
import os
import asyncio
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger("hr-llm")


@dataclass
class LLMProvider:
    """A single LLM provider configuration."""
    name: str
    provider_type: str  # "gemini" or "groq"
    api_key: str
    model: str


def _load_providers() -> list[LLMProvider]:
    """Load all available LLM providers from environment variables.

    Returns providers in fallback priority order:
        Gemini keys first (fastest for google-genai), then Groq keys.
    """
    providers: list[LLMProvider] = []

    # Groq providers (insanely fast inference, prioritized for speed)
    groq_keys = [
        ("GROQ_API_KEY", "Groq-Primary"),
        ("GROQ_API_KEY_2", "Groq-Backup-1"),
        ("GROQ_API_KEY_3", "Groq-Backup-2"),
    ]
    for env_var, name in groq_keys:
        key = os.environ.get(env_var, "").strip()
        if key and key != "your-groq-api-key-here":
            providers.append(LLMProvider(
                name=name,
                provider_type="groq",
                api_key=key,
                model="llama-3.1-8b-instant",  # Smaller 8B model for maximum speed
            ))

    # Gemini providers (Google Generative AI) - Moved to backup due to rate limits
    gemini_keys = [
        ("GEMINI_API_KEY", "Gemini-Primary"),
        ("GEMINI_API_KEY_2", "Gemini-Backup"),
    ]
    for env_var, name in gemini_keys:
        key = os.environ.get(env_var, "").strip()
        if key and key != "your-gemini-api-key-here":
            providers.append(LLMProvider(
                name=name,
                provider_type="gemini",
                api_key=key,
                model="gemini-2.5-flash",
            ))

    if not providers:
        logger.warning("No LLM API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env")

    return providers


# Module-level provider chain (loaded once at import)
_providers: list[LLMProvider] | None = None


def _get_providers() -> list[LLMProvider]:
    """Lazy-load providers on first use (after .env is loaded)."""
    global _providers
    if _providers is None:
        _providers = _load_providers()
        if _providers:
            names = [p.name for p in _providers]
            logger.info("LLM fallback chain: %s", " → ".join(names))
    return _providers


def _call_gemini(provider: LLMProvider, prompt: str) -> str:
    """Call Google Gemini API (synchronous)."""
    from google import genai

    client = genai.Client(api_key=provider.api_key)
    response = client.models.generate_content(
        model=provider.model,
        contents=prompt,
    )
    return response.text.strip()


def _call_groq(provider: LLMProvider, prompt: str) -> str:
    """Call Groq API via OpenAI-compatible endpoint (synchronous)."""
    import httpx

    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": provider.model,
            "messages": [
                {"role": "system", "content": "You are an expert HR Business Partner."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 256,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_provider(provider: LLMProvider, prompt: str) -> str:
    """Route to the correct provider implementation."""
    if provider.provider_type == "gemini":
        return _call_gemini(provider, prompt)
    elif provider.provider_type == "groq":
        return _call_groq(provider, prompt)
    else:
        raise ValueError(f"Unknown provider type: {provider.provider_type}")


async def generate_text(
    prompt: str,
    fallback_message: str = "AI strategy unavailable. Schedule a 1:1 check-in with the employee.",
) -> str:
    """Generate text using the LLM fallback chain.

    Tries each provider in order. If a provider fails (rate limit, auth error,
    network timeout), it logs the failure and tries the next one. If all
    providers fail, returns the fallback_message.

    Args:
        prompt: The text prompt to send to the LLM.
        fallback_message: Safe default if all providers fail.

    Returns:
        Generated text from the first successful provider, or fallback_message.
    """
    providers = _get_providers()

    if not providers:
        return "Configure GEMINI_API_KEY or GROQ_API_KEY for AI-powered strategies."

    last_error: Exception | None = None

    for provider in providers:
        start = time.monotonic()
        try:
            result = await asyncio.to_thread(_call_provider, provider, prompt)
            elapsed = time.monotonic() - start
            logger.info(
                "LLM success: provider=%s elapsed=%.2fs chars=%d",
                provider.name, elapsed, len(result),
            )
            return result

        except Exception as e:
            elapsed = time.monotonic() - start
            last_error = e

            # Classify the error for better logging
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                reason = "RATE_LIMITED"
            elif "401" in err_str or "403" in err_str or "auth" in err_str.lower():
                reason = "AUTH_ERROR"
            elif "timeout" in err_str.lower():
                reason = "TIMEOUT"
            else:
                reason = "ERROR"

            logger.warning(
                "LLM %s: provider=%s elapsed=%.2fs error=%s — trying next...",
                reason, provider.name, elapsed, err_str[:200],
            )

    # All providers exhausted
    logger.error(
        "All %d LLM providers failed. Last error: %s",
        len(providers), last_error,
    )
    return fallback_message


def get_provider_status() -> list[dict]:
    """Return status of all configured providers (for health/debug endpoints)."""
    providers = _get_providers()
    return [
        {
            "name": p.name,
            "type": p.provider_type,
            "model": p.model,
            "key_prefix": p.api_key[:8] + "..." if p.api_key else "N/A",
        }
        for p in providers
    ]
