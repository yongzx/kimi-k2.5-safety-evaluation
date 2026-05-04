"""OpenRouter API client."""

import requests
import time
from typing import Optional
from config import OPENROUTER_BASE_URL, require_openrouter_api_key


def call_model(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
    provider: str | None = None,
    reasoning_effort: str | None = None,
    reasoning_max_tokens: int | None = None,
    seed: int | None = None,
) -> str:
    """
    Call a model via OpenRouter API.

    Args:
        model: Model identifier (e.g., "moonshotai/kimi-k2.5")
        messages: List of {"role": str, "content": str} dicts
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Maximum tokens in response
        max_retries: Number of retries on failure
        provider: Optional provider slug to force (e.g., "Moonshot")
        reasoning_effort: Optional reasoning effort level for models that support it
                         ("none", "minimal", "low", "medium", "high", "xhigh")
        reasoning_max_tokens: Optional reasoning token budget for models that
                             support `reasoning.max_tokens`
        seed: Optional random seed for providers that support deterministic sampling

    Returns:
        Model response text

    Raises:
        ValueError: If API call fails after retries
    """
    openrouter_api_key = require_openrouter_api_key()
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/kimi-analysis",
        "X-Title": "Kimi Mental Health Eval",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if seed is not None:
        payload["seed"] = seed

    # Force specific provider if requested
    if provider:
        payload["provider"] = {
            "order": [provider],
            "allow_fallbacks": False,
        }

    # Enable reasoning if requested
    if reasoning_effort or reasoning_max_tokens is not None:
        reasoning_payload = {}
        if reasoning_effort:
            reasoning_payload["effort"] = reasoning_effort
        if reasoning_max_tokens is not None:
            reasoning_payload["max_tokens"] = reasoning_max_tokens
        payload["reasoning"] = reasoning_payload

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise ValueError(f"API call failed after {max_retries} attempts: {e}")
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
            time.sleep(wait_time)

    raise ValueError("Unexpected: exited retry loop without returning")
