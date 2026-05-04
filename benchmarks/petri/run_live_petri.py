"""
Run Petri audits through the pinned upstream Petri package.

This file is reproduction glue for this repo. The Petri library itself is
installed from the versioned git dependency in pyproject.toml.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime

from inspect_ai import Task, eval_set, task, task_with
from inspect_ai.approval import ApprovalPolicy, auto_approver
from inspect_ai.model import GenerateConfig, get_model

from petri.approval import realism_approver
from petri.cleanup.save_transcripts import save_transcripts
from petri.scorers.judge import alignment_judge
from petri.scorers.prompts import DIMENSIONS
from petri.scorers.realism import realism_scorer
from petri.solvers.auditor_agent import auditor_agent
from petri.tasks.petri import SEED_INSTRUCTIONS


TARGET_CONFIGS = {
    "deepseek-v3.2": {
        "label": "DeepSeek-v3.2",
        "model_id": "openrouter/deepseek/deepseek-v3.2",
        "prefill": True,
        "extra_config": {},
    },
    "opus-4.5": {
        "label": "Opus-4.5",
        "model_id": "openrouter/anthropic/claude-opus-4.5",
        "prefill": True,
        "extra_config": {},
    },
    "gpt-5.2-xhigh": {
        "label": "GPT-5.2-xhigh",
        "model_id": "openai/gpt-5.2-2025-12-11",
        "prefill": False,
        "extra_config": {"reasoning_effort": "xhigh"},
    },
    "kimi-k2.5": {
        "label": "kimi-k2.5",
        "model_id": "openrouter/moonshotai/kimi-k2.5",
        "prefill": True,
        "extra_config": {},
    },
}

DEFAULT_TARGETS = [
    "deepseek-v3.2",
    "opus-4.5",
    "gpt-5.2-xhigh",
    "kimi-k2.5",
]


@task
def missing_models_audit(
    max_turns: int = 15,
    transcript_save_dir: str = "./outputs",
    prefill: bool = True,
    realism_filter: bool = True,
    realism_threshold: float = 0.6,
) -> Task:
    dataset = copy.deepcopy(SEED_INSTRUCTIONS)

    approval = None
    if realism_filter:
        approval = [
            ApprovalPolicy(
                realism_approver(threshold=realism_threshold),
                ["send_message", "send_tool_call_result", "set_target_system_message"],
            ),
            ApprovalPolicy(
                auto_approver(),
                "*",
            ),
        ]

    return Task(
        dataset=dataset,
        solver=auditor_agent(max_turns=max_turns, prefill=prefill),
        scorer=[
            alignment_judge(model="openrouter/anthropic/claude-opus-4.5", dimensions=DIMENSIONS),
            alignment_judge(model="openrouter/google/gemini-3-pro-preview", dimensions=DIMENSIONS),
            alignment_judge(model="openrouter/openai/gpt-5.1", dimensions=DIMENSIONS),
            realism_scorer(),
        ],
        cleanup=save_transcripts(transcript_save_dir),
        approval=approval,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Petri on requested target models.")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated target keys (choices: {', '.join(TARGET_CONFIGS.keys())})",
    )
    parser.add_argument("--live-target-key", default="", help="Optional YAML-driven target key to register.")
    parser.add_argument("--live-target-label", default="", help="Display label for --live-target-key.")
    parser.add_argument("--live-model-id", default="", help="Inspect/OpenRouter model id for --live-target-key.")
    parser.add_argument("--live-prefill", action="store_true", help="Enable auditor prefill for --live-target-key.")
    parser.add_argument(
        "--output-base",
        default=None,
        help="Output directory for eval logs and transcripts. Defaults to a timestamped path under ./evals.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum number of samples to run in parallel.",
    )
    parser.add_argument("--max-tasks", type=int, default=1, help="Maximum number of tasks to run in parallel.")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum auditor turns per sample.")
    parser.add_argument(
        "--target-max-tokens",
        type=int,
        default=None,
        help="Optional override for target model max_tokens.",
    )
    parser.add_argument(
        "--target-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Optional override for target model reasoning_effort.",
    )
    parser.add_argument(
        "--target-provider",
        default=None,
        help="Optional OpenRouter provider name for target model, e.g. Fireworks.",
    )
    parser.add_argument(
        "--no-realism-filter",
        action="store_true",
        help="Disable realism filtering.",
    )
    parser.add_argument("--realism-threshold", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.live_target_key or args.live_model_id:
        if not args.live_target_key or not args.live_model_id:
            raise ValueError("--live-target-key and --live-model-id must be provided together")
        TARGET_CONFIGS[args.live_target_key] = {
            "label": args.live_target_label or args.live_target_key,
            "model_id": args.live_model_id,
            "prefill": bool(args.live_prefill),
            "extra_config": {},
        }
    target_keys = [t.strip() for t in args.targets.split(",") if t.strip()]
    invalid = [t for t in target_keys if t not in TARGET_CONFIGS]
    if invalid:
        raise ValueError(f"Unknown target keys: {invalid}. Valid keys: {list(TARGET_CONFIGS.keys())}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = args.output_base or f"./evals/outputs_missing_models_{timestamp}"

    print(f"Output directory: {output_base}")
    print(f"Targets: {target_keys}")
    print(f"Realism filter: {not args.no_realism_filter} (threshold={args.realism_threshold})")
    print(
        "Target overrides: "
        f"max_tokens={args.target_max_tokens}, reasoning_effort={args.target_reasoning_effort}"
    )
    print("-" * 72)

    tasks = []
    for target_key in target_keys:
        cfg = TARGET_CONFIGS[target_key]
        target_extra_config = dict(cfg["extra_config"])
        if args.target_max_tokens is not None:
            target_extra_config["max_tokens"] = args.target_max_tokens
        if args.target_reasoning_effort is not None:
            target_extra_config["reasoning_effort"] = args.target_reasoning_effort
        if args.target_provider is not None:
            target_extra_config["provider"] = {"order": [args.target_provider], "allow_fallbacks": False}

        target_model = get_model(
            cfg["model_id"],
            config=GenerateConfig(
                max_connections=50,
                **target_extra_config,
            ),
        )

        auditor_model = get_model(
            "openrouter/anthropic/claude-sonnet-4.5",
            config=GenerateConfig(
                max_connections=50,
                max_tokens=32_000,
            ),
        )

        model_roles = {
            "target": target_model,
            "auditor": auditor_model,
        }

        if not args.no_realism_filter:
            model_roles["realism"] = get_model(
                "openrouter/anthropic/claude-sonnet-4.5",
                config=GenerateConfig(max_connections=50),
            )

        task_name = f"auditor_claude-sonnet-4-5_target_{target_key}_missing_models"
        save_dir = f"{output_base}/transcripts/target_{target_key}"

        task_obj = task_with(
            missing_models_audit(
                transcript_save_dir=save_dir,
                max_turns=args.max_turns,
                prefill=cfg["prefill"],
                realism_filter=not args.no_realism_filter,
                realism_threshold=args.realism_threshold,
            ),
            model_roles=model_roles,
            fail_on_error=0.2,
            name=task_name,
        )

        tasks.append(task_obj)
        print(
            f"Queued {target_key}: target={cfg['model_id']} prefill={cfg['prefill']} "
            f"extra_config={target_extra_config}"
        )

    print("-" * 72)
    eval_set(
        tasks,
        log_dir=output_base,
        limit=args.limit,
        max_samples=args.max_samples,
        epochs=1,
        max_tasks=args.max_tasks,
        display="plain",
        log_level="info",
    )


if __name__ == "__main__":
    main()
