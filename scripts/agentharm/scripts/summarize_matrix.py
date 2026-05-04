#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize 3-seed AgentHarm matrix runs")
    p.add_argument("--run-group", required=True, help="Run group prefix used by run_matrix_3seed.sh")
    return p.parse_args()


def _parse_id(run_id: str, run_group: str) -> tuple[str, int, str] | None:
    prefix = f"{run_group}-"
    if not run_id.startswith(prefix):
        return None
    tail = run_id[len(prefix) :]
    # expected: {task}-s{seed}-{model_name}
    parts = tail.split("-")
    if len(parts) < 3:
        return None
    task = parts[0]
    seed_part = parts[1]
    if not seed_part.startswith("s"):
        return None
    try:
        seed = int(seed_part[1:])
    except ValueError:
        return None
    model = "-".join(parts[2:])
    return task, seed, model


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    evals_root = root / "results" / "evals"

    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for run_dir in sorted([p for p in evals_root.iterdir() if p.is_dir()]):
        parsed = _parse_id(run_dir.name, args.run_group)
        if not parsed:
            continue
        task, seed, model_name = parsed
        summary_json = run_dir / "summary.json"
        if not summary_json.exists():
            continue

        data = json.loads(summary_json.read_text(encoding="utf-8"))
        models = data.get("models", [])
        if not models:
            continue
        # Single model per run-id in this matrix.
        m = models[0]

        row = {
            "run_id": run_dir.name,
            "task": task,
            "seed": seed,
            "model_name": model_name,
            "status": m.get("status"),
            "completed": m.get("completed_samples"),
            "total": m.get("total_samples"),
            "avg_score": m.get("avg_score"),
            "avg_full_score": m.get("avg_full_score"),
            "avg_refusals": m.get("avg_refusals"),
            "avg_score_non_refusals": m.get("avg_score_non_refusals"),
            "log_path": m.get("log_path"),
        }
        rows.append(row)
        by_key[(task, model_name)].append(row)

    out_dir = evals_root / args.run_group
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = out_dir / "seed_level_metrics.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "task",
                "seed",
                "model_name",
                "status",
                "completed",
                "total",
                "avg_score",
                "avg_full_score",
                "avg_refusals",
                "avg_score_non_refusals",
                "log_path",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    agg_rows: list[dict[str, Any]] = []
    for (task, model_name), items in sorted(by_key.items()):
        ok = [r for r in items if r.get("status") == "success"]

        def vals(key: str) -> list[float]:
            out: list[float] = []
            for r in ok:
                v = r.get(key)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        score = vals("avg_score")
        full = vals("avg_full_score")
        ref = vals("avg_refusals")
        nonref = vals("avg_score_non_refusals")

        agg_rows.append(
            {
                "task": task,
                "model_name": model_name,
                "n_runs_total": len(items),
                "n_runs_success": len(ok),
                "avg_score_mean": mean(score) if score else None,
                "avg_score_std": pstdev(score) if len(score) > 1 else 0.0 if score else None,
                "avg_full_score_mean": mean(full) if full else None,
                "avg_full_score_std": pstdev(full) if len(full) > 1 else 0.0 if full else None,
                "avg_refusals_mean": mean(ref) if ref else None,
                "avg_refusals_std": pstdev(ref) if len(ref) > 1 else 0.0 if ref else None,
                "avg_score_non_refusals_mean": mean(nonref) if nonref else None,
                "avg_score_non_refusals_std": pstdev(nonref) if len(nonref) > 1 else 0.0 if nonref else None,
            }
        )

    agg_json = out_dir / "aggregate_metrics.json"
    agg_json.write_text(
        json.dumps(
            {
                "run_group": args.run_group,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rows": agg_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    md = [f"# AgentHarm Matrix Summary: {args.run_group}", ""]
    md.append("| task | model | success/total | Avg Score | Full Score Rate | Refusal Rate | Score (Non-Refusals) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")

    def fm(mu: Any, sd: Any) -> str:
        if not isinstance(mu, (int, float)):
            return "-"
        if isinstance(sd, (int, float)):
            return f"{mu:.4f} +- {sd:.4f}"
        return f"{mu:.4f}"

    for r in agg_rows:
        md.append(
            "| "
            + " | ".join(
                [
                    str(r["task"]),
                    str(r["model_name"]),
                    f"{r['n_runs_success']}/{r['n_runs_total']}",
                    fm(r["avg_score_mean"], r["avg_score_std"]),
                    fm(r["avg_full_score_mean"], r["avg_full_score_std"]),
                    fm(r["avg_refusals_mean"], r["avg_refusals_std"]),
                    fm(r["avg_score_non_refusals_mean"], r["avg_score_non_refusals_std"]),
                ]
            )
            + " |"
        )

    md_path = out_dir / "aggregate_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {agg_json}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
