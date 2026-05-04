#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.utils.config import build_context, enable_only_benchmark
from benchmarks.utils.orchestrator import run_enabled_benchmarks
from benchmarks.agentharm.runner import run as run_agentharm
from benchmarks.evaluation_awareness.runner import run as run_eval_awareness
from benchmarks.petri.runner import run as run_petri
from benchmarks.psychosisbench.runner import run as run_psychosisbench
from benchmarks.self_replication.runner import run as run_self_replication


REGISTRY = {
    "self_replication": run_self_replication,
    "evaluation_awareness": run_eval_awareness,
    "petri": run_petri,
    "agentharm": run_agentharm,
    "psychosisbench": run_psychosisbench,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured reproduction benchmarks.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/kimi_k25_paper_reprod.yaml")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--only", choices=sorted(REGISTRY), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ctx = build_context(REPO_ROOT, config_path, args.run_id)
    if args.only:
        object.__setattr__(ctx, "config", enable_only_benchmark(ctx.config, args.only))
    summary = run_enabled_benchmarks(ctx, REGISTRY)
    print(f"Run ID: {summary['run_id']}")
    print(f"Artifacts: {ctx.run_dir}")
    for name, result in summary["benchmarks"].items():
        print(f"- {name}: {result.get('status', 'unknown')} -> {result.get('artifact_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
