from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunContext:
    repo_root: Path
    run_id: str
    output_root: Path
    log_root: Path
    config_path: Path
    config: dict[str, Any]

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def run_log_dir(self) -> Path:
        return self.log_root / f"log-{self.run_id}"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def build_context(repo_root: Path, config_path: Path, override_run_id: str | None = None) -> RunContext:
    cfg = load_yaml(config_path)
    run_cfg = cfg.get("run") or {}
    if not isinstance(run_cfg, dict):
        raise ValueError("run must be a mapping")
    run_id = override_run_id or str(run_cfg.get("id") or "reproduction-run")
    output_root = repo_root / str(run_cfg.get("output_dir") or "data/processed")
    log_root = repo_root / str(run_cfg.get("log_dir") or "logs")
    ctx = RunContext(
        repo_root=repo_root,
        run_id=run_id,
        output_root=output_root,
        log_root=log_root,
        config_path=config_path,
        config=cfg,
    )
    ctx.run_dir.mkdir(parents=True, exist_ok=True)
    ctx.run_log_dir.mkdir(parents=True, exist_ok=True)
    return ctx


def enable_only_benchmark(cfg: dict[str, Any], benchmark: str) -> dict[str, Any]:
    patched = deepcopy(cfg)
    benchmarks = patched.get("benchmarks") or {}
    if benchmark not in benchmarks:
        raise ValueError(f"Unknown benchmark in config: {benchmark}")
    for name, bench_cfg in benchmarks.items():
        if not isinstance(bench_cfg, dict):
            raise ValueError(f"benchmark config for {name} must be a mapping")
        bench_cfg["enabled"] = name == benchmark
    return patched
