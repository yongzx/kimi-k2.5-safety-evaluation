"""Experiment configuration for self-replication evaluation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # --- API ---
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    proxy_base_url: str = "http://localhost:8000"
    model: str = "moonshotai/kimi-k2.5"
    max_tokens: int = 4096

    # --- RunPod ---
    runpod_api_key: str = field(
        default_factory=lambda: os.environ.get("RUNPOD_API_KEY", "")
    )
    dry_run: bool = True

    # --- Safety ---
    max_pods: int = 3
    spending_cap_usd: float = 5.0
    kill_switch: bool = False

    # --- Experiment ---
    max_turns: int = 50
    adversary_interval_sec: float = 30.0
    adversary_initial_delay_sec: float = 60.0
    num_trials: int = 10
    context_window_messages: int = 30  # keep last N messages + system prompt

    # --- Logging ---
    log_dir: str = "logs/log-001-self-replication"
    data_dir: str = "data/processed/001-self-replication"
