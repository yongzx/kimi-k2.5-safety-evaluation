"""Configuration for Kimi mental health evaluation."""

import os
from pathlib import Path

# API Configuration
_DEFAULT_KEY_PATH = Path.home() / "Documents/Projects/keys/openrouter_api_key"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY and _DEFAULT_KEY_PATH.exists():
    OPENROUTER_API_KEY = _DEFAULT_KEY_PATH.read_text().strip()


def require_openrouter_api_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in your environment (recommended via `codex-env`) "
            f"or place it in {_DEFAULT_KEY_PATH}."
        )
    return OPENROUTER_API_KEY


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models
EVAL_MODEL = "openai/gpt-5.2"  # Model being evaluated
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"  # For classifying free-form responses

# Provider mapping - force specific providers for models
# Use None to let OpenRouter choose automatically
MODEL_PROVIDERS = {
    "moonshotai/kimi-k2.5": os.getenv("PSYCHOSISBENCH_OPENROUTER_PROVIDER") or None,
    "google/gemini-3-pro-preview": None,   # Let OpenRouter choose
}

# Reasoning effort mapping - enable reasoning for specific models
# Use None for models that don't need explicit reasoning.
# Value can be:
#   - effort string: "none", "minimal", "low", "medium", "high", "xhigh"
#   - dict payload merged into OpenRouter `reasoning`, e.g.:
#       {"effort": "xhigh"}
#       {"max_tokens": 100000}
MODEL_REASONING = {
    "openai/gpt-5.2:reasoning": "medium",  # GPT-5.2 with default reasoning
}

# Per-model output token budget override for eval model calls.
# If a model is absent here, run_eval.py falls back to 65536.
MODEL_MAX_TOKENS = {}

# Output directory
OUTPUT_DIR = Path(__file__).parent / "results"
