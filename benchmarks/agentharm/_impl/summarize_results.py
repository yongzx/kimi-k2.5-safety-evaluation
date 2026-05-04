#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize AgentHarm inspect eval logs")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory under results/evals (default: latest by mtime)",
    )
    return p.parse_args()


def _latest_run_dir(evals_root: Path) -> Path:
    candidates = [p for p in evals_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {evals_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _metric_value(metrics: dict[str, Any], preferred_names: list[str]) -> float | int | str | None:
    if not metrics:
        return None

    # Exact key match first.
    for name in preferred_names:
        if name in metrics:
            metric = metrics[name]
            return getattr(metric, "value", None)

    # Then try case-insensitive and suffix matching.
    lowered = {k.lower(): k for k in metrics.keys()}
    for name in preferred_names:
        name_low = name.lower()
        if name_low in lowered:
            metric = metrics[lowered[name_low]]
            return getattr(metric, "value", None)
        for key, metric in metrics.items():
            key_low = key.lower()
            metric_name = str(getattr(metric, "name", "")).lower()
            if key_low.endswith(f"/{name_low}") or metric_name == name_low:
                return getattr(metric, "value", None)

    return None


def _find_metric_bundle(log: Any) -> dict[str, Any]:
    results = getattr(log, "results", None)
    if not results:
        return {}
    scores = getattr(results, "scores", []) or []
    if not scores:
        return {}

    # AgentHarm uses combined_scorer; use first score entry.
    first = scores[0]
    return getattr(first, "metrics", {}) or {}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "-" if v is None else str(v)


def main() -> int:
    args = parse_args()

    # Lazy import so this script can be discovered without dependencies.
    from inspect_ai.log import list_eval_logs, read_eval_log

    root = Path(__file__).resolve().parent.parent
    evals_root = root / "results" / "evals"

    run_dir = args.run_dir if args.run_dir else _latest_run_dir(evals_root)
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []

    for model_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        logs = list_eval_logs(str(model_dir), recursive=True, descending=True)
        if not logs:
            rows.append(
                {
                    "name": model_dir.name,
                    "status": "no_logs",
                    "model": None,
                    "log_path": None,
                    "completed_samples": None,
                    "total_samples": None,
                    "avg_score": None,
                    "avg_refusals": None,
                    "avg_score_non_refusals": None,
                    "avg_full_score": None,
                }
            )
            continue

        latest = logs[0]
        log = read_eval_log(latest.name, header_only=True)

        metrics = _find_metric_bundle(log)
        results = getattr(log, "results", None)
        spec = getattr(log, "eval", None)

        rows.append(
            {
                "name": model_dir.name,
                "status": str(getattr(log, "status", "unknown")),
                "model": str(getattr(spec, "model", "")) if spec else None,
                "log_path": latest.name,
                "completed_samples": getattr(results, "completed_samples", None) if results else None,
                "total_samples": getattr(results, "total_samples", None) if results else None,
                "avg_score": _metric_value(metrics, ["avg_score"]),
                "avg_refusals": _metric_value(metrics, ["avg_refusals"]),
                "avg_score_non_refusals": _metric_value(metrics, ["avg_score_non_refusals"]),
                "avg_full_score": _metric_value(metrics, ["avg_full_score"]),
            }
        )

    summary = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "manifest": manifest,
        "models": rows,
    }

    summary_json_path = run_dir / "summary.json"
    summary_md_path = run_dir / "summary.md"
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append(f"# AgentHarm Summary: {run_dir.name}")
    lines.append("")
    lines.append("| model_name | status | completed | total | avg_score | avg_refusals | avg_score_non_refusals | avg_full_score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    str(row["status"]),
                    _fmt(row["completed_samples"]),
                    _fmt(row["total_samples"]),
                    _fmt(row["avg_score"]),
                    _fmt(row["avg_refusals"]),
                    _fmt(row["avg_score_non_refusals"]),
                    _fmt(row["avg_full_score"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Log Paths")
    lines.append("")
    for row in rows:
        lines.append(f"- `{row['name']}`: `{row['log_path']}`")

    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {summary_json_path}")
    print(f"Wrote: {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
