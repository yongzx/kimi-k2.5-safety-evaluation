from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import yaml

from benchmarks.utils.artifacts import now_utc, write_json
from benchmarks.utils.config import RunContext


TASK_LABELS = {
    "harmful": "Harmful behaviors",
    "benign": "Benign behaviors",
}

MODEL_ORDER = ["kimi_k25_100k", "opus_46_high", "gpt52_xhigh", "deepseek_v32_100k"]
MODEL_LABELS = {
    "gpt52_xhigh": "GPT-5.2",
    "kimi_k25_100k": "Kimi K2.5",
    "opus_46_high": "Claude Opus 4.5",
    "deepseek_v32_100k": "DeepSeek V3.2",
}
CONDITION_ORDER = ["No attack", "Forced tool call", "Template attack"]
CONDITION_STYLE = {
    "No attack": {"color": "#f5b7b1"},
    "Forced tool call": {"color": "#ec7063"},
    "Template attack": {"color": "#c0392b"},
}


def run(ctx: RunContext, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ctx.run_dir / "agentharm"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("mode") or "live")
    if mode == "live":
        rows = _run_live(ctx, cfg, out_dir)
        source = str(out_dir / "live_evals")
    else:
        raise ValueError(f"Unsupported agentharm mode: {mode}; this repo is live-only and does not bundle result artifacts")

    summary = _aggregate(rows)
    if not any(int(row.get("n_runs_success") or 0) > 0 for row in summary):
        raise RuntimeError("agentharm live runner produced zero successful runs")
    _write_seed_metrics(out_dir / "seed_level_metrics.csv", rows)
    _write_aggregate_tsv(out_dir / "aggregate_metrics.tsv", summary)
    figure_dir = out_dir / "figures"
    _plot_paper_figures(out_dir, figure_dir, summary)
    write_json(
        out_dir / "manifest.json",
        {
            "benchmark": "agentharm",
            "mode": mode,
            "created_at": now_utc(),
            "source": source,
            "paper_figures": {
                "figure33": "figures/figure33_agentharm_refusal_rate.{png,pdf}",
                "figure34": "figures/figure34_agentharm_harm_score.{png,pdf}",
                "figure35": "figures/figure35_agentharm_kimi_harm_by_category.{png,pdf}",
            },
            "metric_formulas": {
                "avg_score_mean": "mean(avg_score over successful seeds); denominator scope: per task/model across seeds",
                "avg_refusals_mean": "mean(avg_refusals over successful seeds); denominator scope: per task/model across seeds",
                "avg_score_non_refusals_mean": "mean(avg_score_non_refusals over successful seeds); denominator scope: per task/model across seeds",
                "std": "population standard deviation over successful seeds",
                "figure_ci95": "live fallback uses std across seeds as error bars",
            },
        },
    )
    write_json(out_dir / "scores.json", {"seed_level": rows, "aggregate": summary})
    return {"status": "complete", "artifact_dir": str(out_dir), "mode": mode}

def _run_live(ctx: RunContext, cfg: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    target = ctx.config.get("target_model") or {}
    model_name = str(cfg.get("model_name") or target.get("name") or "kimi_k25_100k").replace("-", "_").replace(".", "_")
    model_id = str(cfg.get("model") or target.get("model") or "moonshotai/kimi-k2.5")
    if not model_id.startswith("openrouter/"):
        model_id = f"openrouter/{model_id}"
    seeds = [int(s) for s in (cfg.get("seeds") or [0])]
    tasks = [str(t) for t in (cfg.get("tasks") or ["harmful"])]
    sample_limit = int(cfg.get("sample_limit", 1))
    provider = str(cfg.get("openrouter_provider") or target.get("openrouter_provider") or "")

    tool_root = ctx.repo_root / "benchmarks/agentharm"
    evals_root = tool_root / "results/evals"
    live_root = out_dir / "live_evals"
    live_root.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "live_config.yaml"
    live_cfg = {
        "split": str(cfg.get("split") or "test_public"),
        "sample_limit": sample_limit,
        "agent": str(cfg.get("agent") or "default"),
        "agent_kwargs": cfg.get("agent_kwargs") or {"tool_choice": "auto"},
        "max_messages": int(cfg.get("max_messages", 20)),
        "refusal_judge_model": str(cfg.get("refusal_judge_model") or "openrouter/openai/gpt-4o-2024-08-06"),
        "semantic_judge_model": str(cfg.get("semantic_judge_model") or "openrouter/openai/gpt-4o-2024-08-06"),
        "models": [
            {
                "name": model_name,
                "model": model_id,
                "reasoning_tokens": int(cfg.get("reasoning_tokens", target.get("reasoning_tokens", 100000))),
                "model_args": {
                    "reasoning_enabled": bool(cfg.get("reasoning_enabled", target.get("reasoning_enabled", True))),
                    **(
                        {"provider": {"order": [provider], "allow_fallbacks": False}}
                        if provider
                        else {}
                    ),
                },
            }
        ],
    }
    config_path.write_text(yaml.safe_dump(live_cfg, sort_keys=False), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    env = os.environ.copy()
    for task in tasks:
        if task not in TASK_LABELS:
            raise ValueError(f"agentharm task must be one of {sorted(TASK_LABELS)}; got {task}")
        for seed in seeds:
            run_id = f"{ctx.run_id}-{task}-s{seed}-{model_name}"
            log_file = ctx.run_log_dir / f"agentharm_{task}_s{seed}.log"
            cmd = [
                sys.executable,
                "_impl/run_agentharm.py",
                "--config",
                str(config_path),
                "--models",
                model_name,
                "--sample-limit",
                str(sample_limit),
                "--task",
                task,
                "--seed",
                str(seed),
                "--run-id",
                run_id,
            ]
            with log_file.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, cwd=tool_root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise RuntimeError(f"agentharm live run failed; see {log_file}")

            summarize_cmd = [sys.executable, "_impl/summarize_results.py", "--run-dir", str(evals_root / run_id)]
            with log_file.open("a", encoding="utf-8") as log:
                proc = subprocess.run(summarize_cmd, cwd=tool_root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise RuntimeError(f"agentharm summarizer failed; see {log_file}")

            src_run = evals_root / run_id
            dst_run = live_root / run_id
            if dst_run.exists():
                shutil.rmtree(dst_run)
            shutil.copytree(src_run, dst_run)
            summary_path = dst_run / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            model_rows = payload.get("models") or []
            if not model_rows:
                raise RuntimeError(f"agentharm summary has no model rows: {summary_path}")
            m = model_rows[0]
            rows.append(
                {
                    "run_id": run_id,
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
            )
    return rows

def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["task"]), str(row["model_name"])), []).append(row)

    out: list[dict[str, Any]] = []
    for (task, model_name), items in sorted(grouped.items()):
        ok = [r for r in items if str(r.get("status")) == "success"]

        def stats(key: str) -> tuple[float | None, float | None]:
            vals = [float(r[key]) for r in ok if isinstance(r.get(key), (int, float))]
            if not vals:
                return None, None
            return mean(vals), pstdev(vals) if len(vals) > 1 else 0.0

        item: dict[str, Any] = {
            "task": task,
            "condition": TASK_LABELS.get(task, task),
            "model_name": model_name,
            "n_runs_total": len(items),
            "n_runs_success": len(ok),
        }
        for key in ["avg_score", "avg_full_score", "avg_refusals", "avg_score_non_refusals"]:
            mu, sd = stats(key)
            item[f"{key}_mean"] = mu
            item[f"{key}_std"] = sd
        out.append(item)
    return out


def _write_seed_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
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
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else str(value)


def _write_aggregate_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "condition",
        "model_name",
        "n_runs_total",
        "n_runs_success",
        "avg_score_mean",
        "avg_score_std",
        "avg_refusals_mean",
        "avg_refusals_std",
        "avg_score_non_refusals_mean",
        "avg_score_non_refusals_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fieldnames})


def _plot_paper_figures(out_dir: Path, figure_dir: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    refusal_csv = out_dir / "agentharm_main_panels.csv"
    harmscore_csv = out_dir / "agentharm_main_panels_harmscore.csv"
    category_csv = out_dir / "agentharm_harmscore_by_category.csv"
    if refusal_csv.exists():
        _plot_main_panel_from_csv(
            refusal_csv,
            figure_dir / "figure33_agentharm_refusal_rate",
            title="Refusal Rate Across Models (AgentHarm)",
            ylabel="Refusal Rate",
        )
    if harmscore_csv.exists():
        _plot_main_panel_from_csv(
            harmscore_csv,
            figure_dir / "figure34_agentharm_harm_score",
            title="Harm Score Across Models (AgentHarm)",
            ylabel="Average Score",
        )
    if category_csv.exists():
        _plot_category_from_csv(
            category_csv,
            figure_dir / "figure35_agentharm_kimi_harm_by_category",
            title="Kimi K2.5 Harm Score by Task Category",
        )
    if not refusal_csv.exists() or not harmscore_csv.exists() or not category_csv.exists():
        _plot_live_fallback(figure_dir, aggregate_rows)


def _plot_main_panel_from_csv(csv_path: Path, out_stem: Path, title: str, ylabel: str) -> None:
    stats: dict[tuple[str, str], dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            condition = row["condition"]
            model = row["model_name"]
            if condition not in CONDITION_ORDER or model not in MODEL_ORDER:
                continue
            stats[(condition, model)] = {
                "mean": float(row["mean"]),
                "ci_low": float(row["ci95_low"]),
                "ci_high": float(row["ci95_high"]),
            }
    if not stats:
        return

    x = np.arange(len(MODEL_ORDER))
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    width = 0.72 / len(CONDITION_ORDER)
    for i, condition in enumerate(CONDITION_ORDER):
        offsets = x + (i - (len(CONDITION_ORDER) - 1) / 2.0) * width
        means = np.array([stats[(condition, model)]["mean"] for model in MODEL_ORDER])
        lo = np.array([stats[(condition, model)]["mean"] - stats[(condition, model)]["ci_low"] for model in MODEL_ORDER])
        hi = np.array([stats[(condition, model)]["ci_high"] - stats[(condition, model)]["mean"] for model in MODEL_ORDER])
        ax.bar(
            offsets,
            means,
            width=width,
            label=condition,
            color=CONDITION_STYLE[condition]["color"],
            edgecolor="none",
            yerr=np.vstack([lo, hi]),
            capsize=4,
            alpha=0.95,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        for j, value in enumerate(means):
            ax.text(offsets[j], value + hi[j] + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=12)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=16, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.975)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.53, 0.895), frameon=False, ncol=len(CONDITION_ORDER))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    _save_figure(fig, out_stem)


def _plot_category_from_csv(csv_path: Path, out_stem: Path, title: str) -> None:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "category": row["category"],
                    "mean": float(row["mean"]),
                    "ci_low": float(row["ci95_low"]),
                    "ci_high": float(row["ci95_high"]),
                }
            )
    if not rows:
        return
    x = np.arange(len(rows))
    means = np.array([r["mean"] for r in rows])
    lo = np.array([r["mean"] - r["ci_low"] for r in rows])
    hi = np.array([r["ci_high"] - r["mean"] for r in rows])
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bars = ax.bar(x, means, color="#ec7063", edgecolor="#444444", linewidth=0.7, yerr=np.vstack([lo, hi]), capsize=4)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, means[i] + hi[i] + 0.015, f"{means[i]:.2f}", ha="center", va="bottom", fontsize=11)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=20)
    ax.set_ylabel("Average Score", fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([r["category"] for r in rows], rotation=16, ha="right")
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.25, axis="y", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, out_stem)


def _plot_live_fallback(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kimi_rows = [r for r in rows if "kimi" in str(r["model_name"]).lower()]
    if not kimi_rows:
        return
    labels = [str(r["condition"]) for r in kimi_rows]
    harm = [float(r.get("avg_score_mean") or 0) for r in kimi_rows]
    refusal = [float(r.get("avg_refusals_mean") or 0) for r in kimi_rows]
    x = range(len(kimi_rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], harm, width=width, label="harm score", color="#E6A46F")
    ax.bar([i + width / 2 for i in x], refusal, width=width, label="refusal rate", color="#6E8795")
    ax.set_xticks(list(x), labels, rotation=20, ha="right")
    ax.set_ylim(0, max(harm + refusal + [1.0]) * 1.15)
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(frameon=False)
    ax.set_title("agentharm Kimi K2.5 summary")
    fig.tight_layout()
    _save_figure(fig, out_dir / "agentharm_live_kimi_summary")


def _save_figure(fig: Any, out_stem: Path) -> None:
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
