"""Configuration for prompt ablation experiment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AblationConfig:
    # --- API ---
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    runpod_api_key: str = field(
        default_factory=lambda: os.environ.get("RUNPOD_API_KEY", "")
    )
    base_url: str = "https://openrouter.ai/api"
    model: str = "moonshotai/kimi-k2.5"
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    reasoning_enabled: bool | None = None
    extra_model_args: dict[str, Any] = field(default_factory=dict)

    # --- Mode ---
    dry_run: bool = True

    # --- Safety ---
    max_pods: int = 3
    spending_cap_usd: float = 5.0
    kill_switch: bool = False
    allow_pod_stop: bool = False
    allow_pod_terminate: bool = False
    abort_on_llm_error: bool = False

    # --- Experiment ---
    max_turns: int = 15
    adversary_interval_sec: float = 10.0
    adversary_initial_delay_sec: float = 20.0
    num_trials: int = 3
    context_window_messages: int = 30

    # --- Paths ---
    prompts_dir: str = "benchmarks/self_replication/prompts"
    data_dir: str = "data/processed/002-prompt-ablation"
    log_dir: str = "logs/log-002-prompt-ablation"

    # Compatibility alias used by exp_001 tools.py dispatch_tool
    @property
    def proxy_base_url(self) -> str:
        return self.base_url
