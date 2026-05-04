"""
Summarize per-dimension Petri judge scores from Inspect eval logs.

Outputs CSV columns:
model,dimension,mean_raw,stderr_raw,mean_normalized,stderr_normalized,n_samples
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from petri.scorers.prompts import DIMENSIONS


VALID_DIMENSIONS = set(DIMENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dimension scores from Inspect eval logs.")
    parser.add_argument("--log-dir", required=True, help="Directory containing Inspect eval logs.")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Path to write CSV. Defaults to <log-dir>/dimension_summary.csv",
    )
    return parser.parse_args()


def model_label_from_task_name(task_name: str) -> str:
    match = re.search(r"target_(.+?)_missing_models", task_name)
    if not match:
        return task_name
    target_key = match.group(1)
    return {
        "deepseek-v3.2": "DeepSeek-v3.2",
        "opus-4.5": "Opus-4.5",
        "gpt-5.2-xhigh": "GPT-5.2-xhigh",
        "kimi-k2.5": "kimi-k2.5",
    }.get(target_key, target_key)


def stderr(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean_v = sum(values) / n
    variance = sum((v - mean_v) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_csv = Path(args.out_csv) if args.out_csv else log_dir / "dimension_summary.csv"

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    log_infos = list_eval_logs(str(log_dir))
    if not log_infos:
        raise FileNotFoundError(f"No eval logs found in {log_dir}")

    for info in log_infos:
        log = read_eval_log(info.name)
        model_label = model_label_from_task_name(log.eval.task)
        for sample in log.samples or []:
            if not sample.scores:
                continue
            for score in sample.scores.values():
                value = score.value
                if not isinstance(value, dict):
                    continue
                for dimension, raw in value.items():
                    if dimension in VALID_DIMENSIONS and isinstance(raw, (int, float)):
                        grouped[(model_label, dimension)].append(float(raw))

    rows: list[dict[str, object]] = []
    for (model, dimension), values in grouped.items():
        n = len(values)
        if n == 0:
            continue
        mean_raw = sum(values) / n
        stderr_raw = stderr(values)
        rows.append(
            {
                "model": model,
                "dimension": dimension,
                "mean_raw": mean_raw,
                "stderr_raw": stderr_raw,
                "mean_normalized": (mean_raw - 1.0) / 9.0,
                "stderr_normalized": stderr_raw / 9.0,
                "n_samples": n,
            }
        )

    rows.sort(key=lambda r: (str(r["model"]).lower(), str(r["dimension"]).lower()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "dimension",
                "mean_raw",
                "stderr_raw",
                "mean_normalized",
                "stderr_normalized",
                "n_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
