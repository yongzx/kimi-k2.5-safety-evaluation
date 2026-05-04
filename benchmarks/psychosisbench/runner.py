from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.utils.artifacts import now_utc, write_json
from benchmarks.utils.config import RunContext


MODEL_ORDER = [
    ("moonshotai/kimi-k2.5", "Kimi K2.5", "#E6A46F"),
    ("anthropic/claude-opus-4.5", "Claude Opus 4.5", "#0F3F30"),
    ("openai/gpt-5.2", "GPT-5.2", "#6E8795"),
    ("deepseek/deepseek-v3.2", "DeepSeek V3.2", "#707423"),
]
T95_DF2 = 4.303


def run(ctx: RunContext, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ctx.run_dir / "psychosisbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("mode") or "live")
    if mode == "live":
        seed_rows = _run_live(ctx, cfg, out_dir)
        aggregate = _aggregate(seed_rows)
        if not any(int(row.get("dcs_n") or 0) > 0 for row in seed_rows):
            raise RuntimeError("psychosisbench live runner produced zero scored DCS turns")
        source = str(out_dir / "live_results")
    else:
        raise ValueError(f"Unsupported psychosisbench mode: {mode}; this repo is live-only and does not bundle result artifacts")

    _write_seed_tsv(out_dir / "results.tsv", seed_rows)
    _write_aggregate_tsv(out_dir / "aggregate.tsv", aggregate)
    _plot_figure36(out_dir / "figures", aggregate)
    write_json(
        out_dir / "manifest.json",
        {
            "benchmark": "psychosisbench",
            "mode": mode,
            "created_at": now_utc(),
            "source": source,
            "paper_figures": {
                "figure36": "figures/figure36_psychosisbench_delusion_main_panels.{png,pdf}",
            },
            "metric_formulas": {
                "dcs": "mean DCS score over scored turns; denominator scope: per seed",
                "hes": "mean HES score over scored turns; denominator scope: per seed",
                "sis": "SIS intervention count / SIS scored turns; denominator scope: per seed",
                "aggregate_mean": "mean per-seed metric across seeds; denominator scope: Kimi seeds",
                "aggregate_std": "sample standard deviation across seeds",
            },
        },
    )
    write_json(out_dir / "scores.json", {"seed_level": seed_rows, "aggregate": aggregate})
    return {"status": "complete", "artifact_dir": str(out_dir), "mode": mode}

def _run_live(ctx: RunContext, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    target = ctx.config.get("target_model") or {}
    model = str(cfg.get("model") or target.get("model") or "moonshotai/kimi-k2.5")
    seeds = [int(s) for s in (cfg.get("seeds") or [0])]
    max_tokens = int(cfg.get("max_tokens", 65536))
    reasoning_tokens = cfg.get("reasoning_tokens", target.get("reasoning_tokens", 100000))
    provider = str(cfg.get("openrouter_provider") or target.get("openrouter_provider") or "")
    test_mode = bool(cfg.get("test_mode", False))
    case_limit = cfg.get("case_limit")
    turn_limit = cfg.get("turn_limit")
    tool_root = ctx.repo_root / "scripts/psychosisbench"
    live_root = out_dir / "live_results"
    live_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    for seed in seeds:
        output_path = live_root / f"eval_delusion_moonshotai-kimi-k2_5_seed{seed}.json"
        log_file = ctx.run_log_dir / f"psychosisbench_delusion_s{seed}.log"
        cmd = [
            sys.executable,
            "run_eval.py",
            "--eval",
            "delusion",
            "--model",
            model,
            "--seed",
            str(seed),
            "--max-tokens",
            str(max_tokens),
            "--output",
            str(output_path),
        ]
        if reasoning_tokens is not None:
            cmd.extend(["--reasoning-max-tokens", str(int(reasoning_tokens))])
        if test_mode:
            cmd.append("--test")
        if case_limit is not None:
            cmd.extend(["--case-limit", str(int(case_limit))])
        if turn_limit is not None:
            cmd.extend(["--turn-limit", str(int(turn_limit))])

        with log_file.open("w", encoding="utf-8") as log:
            if provider:
                env["PSYCHOSISBENCH_OPENROUTER_PROVIDER"] = provider
            proc = subprocess.run(cmd, cwd=tool_root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"psychosisbench live run failed; see {log_file}")
        rows.append(_parse_result_json(output_path))
    return rows


def _parse_result_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata") or {}
    delusion = (payload.get("summaries") or {}).get("delusion") or {}
    dcs = delusion.get("dcs") or {}
    hes = delusion.get("hes") or {}
    sis = delusion.get("sis") or {}
    return {
        "model": metadata.get("eval_model"),
        "seed": metadata.get("eval_seed"),
        "status": "complete",
        "dcs": dcs.get("mean_score"),
        "hes": hes.get("mean_score"),
        "sis": sis.get("intervention_rate"),
        "n_results": len(payload.get("results") or []),
        "dcs_n": dcs.get("scored_turns"),
        "hes_n": hes.get("scored_turns"),
        "sis_n": sis.get("scored_turns"),
        "output_path": str(path),
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model = str(row["model"])
        grouped.setdefault(model, {"dcs": [], "hes": [], "sis": []})
        for metric in ["dcs", "hes", "sis"]:
            if isinstance(row.get(metric), (int, float)):
                grouped[model][metric].append(float(row[metric]))

    out: list[dict[str, Any]] = []
    for model, vals in sorted(grouped.items()):
        item: dict[str, Any] = {"model": model, "n_runs": len(vals["dcs"])}
        for metric in ["dcs", "hes", "sis"]:
            item[f"{metric}_mean"] = mean(vals[metric]) if vals[metric] else None
            item[f"{metric}_std"] = pstdev(vals[metric]) if len(vals[metric]) > 1 else 0.0 if vals[metric] else None
        out.append(item)
    return out

def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)


def _write_seed_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["model", "seed", "status", "dcs", "hes", "sis", "n_results", "dcs_n", "hes_n", "sis_n", "output_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})


def _write_aggregate_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["model", "n_runs", "dcs_mean", "dcs_std", "hes_mean", "hes_std", "sis_mean", "sis_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})


def _plot_figure36(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    by_model = {str(row["model"]): row for row in rows}
    models = [spec for spec in MODEL_ORDER if spec[0] in by_model]
    if not models:
        models = [(str(row["model"]), str(row["model"]), "#E6A46F") for row in rows]

    panels = [
        ("Delusion Confirmation", "dcs", "lower is safer"),
        ("Harm Enablement", "hes", "lower is safer"),
        ("Safety Intervention Rate", "sis", "higher is safer"),
    ]
    labels = [label for _model, label, _color in models]
    colors = [color for _model, _label, color in models]
    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    for ax, (title, metric, subtitle) in zip(axes, panels):
        means = np.array([float(by_model[model].get(f"{metric}_mean") or 0.0) for model, _label, _color in models])
        ci95 = []
        for model, _label, _color in models:
            row = by_model[model]
            n_runs = int(row.get("n_runs") or 1)
            std = float(row.get(f"{metric}_std") or 0.0)
            ci95.append(T95_DF2 * std / math.sqrt(max(n_runs, 1)))
        err = np.array(ci95)
        ax.bar(x, means, color=colors, width=0.72, linewidth=0)
        ax.errorbar(x, means, yerr=err, fmt="none", ecolor="#1F1F1F", elinewidth=1.2, capsize=3)
        for idx, value in enumerate(means):
            ax.text(idx, value + err[idx] + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=12)
        ax.set_title(f"{title}\n({subtitle})", fontsize=16, fontweight="bold", pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Score", fontweight="bold")
        ax.set_ylim(0.0, max((means + err).tolist() + [1.0]) * 1.18)
        ax.grid(True, alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Figure 36: PsychosisBench Delusion Results", fontsize=20, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "figure36_psychosisbench_delusion_main_panels.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_dir / "figure36_psychosisbench_delusion_main_panels.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
