from __future__ import annotations

from typing import Any, Callable

from benchmarks.utils.artifacts import now_utc, write_json
from benchmarks.utils.config import RunContext


BenchmarkFn = Callable[[RunContext, dict[str, Any]], dict[str, Any]]


def run_enabled_benchmarks(ctx: RunContext, registry: dict[str, BenchmarkFn]) -> dict[str, Any]:
    benchmarks = ctx.config.get("benchmarks") or {}
    if not isinstance(benchmarks, dict):
        raise ValueError("benchmarks must be a mapping")

    results: dict[str, Any] = {
        "run_id": ctx.run_id,
        "config_path": str(ctx.config_path),
        "started_at": now_utc(),
        "target_model": ctx.config.get("target_model") or {},
        "benchmarks": {},
    }

    for name, bench_cfg in benchmarks.items():
        if not isinstance(bench_cfg, dict):
            raise ValueError(f"benchmark config for {name} must be a mapping")
        if not bench_cfg.get("enabled", False):
            continue
        if name not in registry:
            raise ValueError(f"Unknown benchmark: {name}")
        results["benchmarks"][name] = registry[name](ctx, bench_cfg)

    results["completed_at"] = now_utc()
    write_json(ctx.run_dir / "aggregate_summary.json", results)
    return results
