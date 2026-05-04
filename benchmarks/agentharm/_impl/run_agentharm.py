#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def _validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "split",
        "sample_limit",
        "agent",
        "max_messages",
        "refusal_judge_model",
        "semantic_judge_model",
        "models",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    if cfg["split"] not in {"val", "test_private", "test_public"}:
        raise ValueError("split must be one of: val, test_private, test_public")

    if not isinstance(cfg["sample_limit"], int) or cfg["sample_limit"] <= 0:
        raise ValueError("sample_limit must be a positive integer")

    if not isinstance(cfg["max_messages"], int) or cfg["max_messages"] <= 0:
        raise ValueError("max_messages must be a positive integer")

    models = cfg["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    names: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each model entry must be a mapping")
        for key in ["name", "model"]:
            if key not in item or not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"Each model entry requires non-empty string '{key}'")
        if "reasoning_tokens" in item and (
            not isinstance(item["reasoning_tokens"], int) or item["reasoning_tokens"] <= 0
        ):
            raise ValueError("reasoning_tokens must be a positive integer when provided")
        if "max_tokens" in item and (
            not isinstance(item["max_tokens"], int) or item["max_tokens"] <= 0
        ):
            raise ValueError("max_tokens must be a positive integer when provided")
        if "model_args" in item and not isinstance(item["model_args"], dict):
            raise ValueError("model_args must be a mapping when provided")
        if item["name"] in names:
            raise ValueError(f"Duplicate model name: {item['name']}")
        names.add(item["name"])


def _select_models(models: list[dict[str, Any]], requested: list[str] | None) -> list[dict[str, Any]]:
    if not requested:
        return models
    wanted = {x.strip() for x in requested if x.strip()}
    selected = [m for m in models if m["name"] in wanted]
    missing = sorted(wanted - {m["name"] for m in selected})
    if missing:
        raise ValueError(f"Unknown model names in --models: {', '.join(missing)}")
    return selected


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AgentHarm via inspect-evals")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names from config to run (default: all)",
    )
    p.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Override sample_limit from config",
    )
    p.add_argument(
        "--task",
        choices=["harmful", "benign"],
        default="harmful",
        help="AgentHarm task type (default: harmful)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed override for generation",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id (default: UTC timestamp)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print execution plan without calling APIs",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY is not set.", file=sys.stderr)
        print("Use `codex-env -- python benchmarks/agentharm/_impl/run_agentharm.py ...` or export the key.", file=sys.stderr)
        return 2

    requested = [x for x in args.models.split(",") if x.strip()] if args.models else None
    selected_models = _select_models(cfg["models"], requested)

    sample_limit = args.sample_limit if args.sample_limit is not None else int(cfg["sample_limit"])
    if sample_limit <= 0:
        raise ValueError("sample-limit must be a positive integer")

    run_id = _make_run_id(args.run_id)
    root = Path(__file__).resolve().parent.parent
    run_dir = root / "results" / "evals" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_utc(),
        "config_path": str(args.config),
        "settings": {
            "task": args.task,
            "split": cfg["split"],
            "sample_limit": sample_limit,
            "agent": cfg["agent"],
            "agent_kwargs": cfg.get("agent_kwargs", {}),
            "max_messages": int(cfg["max_messages"]),
            "refusal_judge_model": cfg["refusal_judge_model"],
            "semantic_judge_model": cfg["semantic_judge_model"],
            "seed": args.seed,
        },
        "models": selected_models,
        "dry_run": bool(args.dry_run),
        "runs": [],
    }

    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")
    print(f"Split: {cfg['split']} | sample_limit: {sample_limit} | max_messages: {cfg['max_messages']}")

    if args.dry_run:
        for model_cfg in selected_models:
            print(
                f"[dry-run] {model_cfg['name']}: model={model_cfg['model']} "
                f"reasoning_effort={model_cfg.get('reasoning_effort')}"
            )
        manifest["completed_at"] = _now_utc()
        with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"Wrote manifest: {run_dir / 'manifest.json'}")
        return 0

    # Lazy imports so dry-run works even before env setup.
    from inspect_ai import eval as inspect_eval
    from inspect_evals.agentharm import agentharm, agentharm_benign

    had_error = False

    for model_cfg in selected_models:
        model_name = model_cfg["name"]
        model_id = model_cfg["model"]
        reasoning_effort = model_cfg.get("reasoning_effort")
        reasoning_tokens = model_cfg.get("reasoning_tokens")
        max_tokens = model_cfg.get("max_tokens")
        model_args = model_cfg.get("model_args") or {}

        model_log_dir = run_dir / model_name
        model_log_dir.mkdir(parents=True, exist_ok=True)

        run_item: dict[str, Any] = {
            "name": model_name,
            "model": model_id,
            "reasoning_effort": reasoning_effort,
            "reasoning_tokens": reasoning_tokens,
            "max_tokens": max_tokens,
            "model_args": model_args,
            "log_dir": str(model_log_dir),
            "started_at": _now_utc(),
            "status": "started",
        }
        manifest["runs"].append(run_item)

        print(f"[run] {model_name} -> {model_id}")

        try:
            task_builder = agentharm_benign if args.task == "benign" else agentharm
            task = task_builder(
                agent=cfg["agent"],
                split=cfg["split"],
                refusal_judge=cfg["refusal_judge_model"],
                semantic_judge=cfg["semantic_judge_model"],
                agent_kwargs=cfg.get("agent_kwargs") or None,
            )

            eval_kwargs: dict[str, Any] = {}
            if isinstance(reasoning_effort, str) and reasoning_effort:
                eval_kwargs["reasoning_effort"] = reasoning_effort
            if isinstance(reasoning_tokens, int) and reasoning_tokens > 0:
                eval_kwargs["reasoning_tokens"] = reasoning_tokens
            if isinstance(max_tokens, int) and max_tokens > 0:
                eval_kwargs["max_tokens"] = max_tokens
            if args.seed is not None:
                eval_kwargs["seed"] = args.seed

            logs = inspect_eval(
                task,
                model=model_id,
                model_args=model_args,
                log_dir=str(model_log_dir),
                limit=sample_limit,
                message_limit=int(cfg["max_messages"]),
                max_tasks=1,
                **eval_kwargs,
            )

            run_item["status"] = "success"
            run_item["completed_at"] = _now_utc()
            run_item["eval_logs"] = [str(log.location) for log in logs if getattr(log, "location", None)]
            print(f"[ok] {model_name} complete")
        except Exception as exc:  # noqa: BLE001
            had_error = True
            run_item["status"] = "error"
            run_item["completed_at"] = _now_utc()
            run_item["error"] = str(exc)
            print(f"[error] {model_name}: {exc}", file=sys.stderr)

        with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    manifest["completed_at"] = _now_utc()
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Summary helper: python benchmarks/agentharm/_impl/summarize_results.py --run-dir {run_dir}")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
