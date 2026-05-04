from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.utils.artifacts import now_utc, write_json
from benchmarks.utils.config import RunContext


MODEL_SPECS = [
    ("kimi-k2.5-100k-reasoning", "Kimi K2.5", "#E6A46F"),
    ("opus-4.5", "Opus 4.5", "#0F3F30"),
    ("gpt-5.2-xhigh", "GPT-5.2", "#6E8795"),
    ("deepseek-v3.2-100k-reasoning", "DeepSeek V3.2", "#707423"),
]

SECTIONS = ("in_context_awareness", "transcript_classification", "contamination_transfer")
T95_DF2 = 4.303


def run(ctx: RunContext, cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ctx.run_dir / "evaluation_awareness"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("mode") or "live")
    model_specs = MODEL_SPECS
    if mode == "live":
        root, model_specs = _run_live(ctx, cfg, out_dir)
        source = str(root)
    else:
        raise ValueError(f"Unsupported evaluation_awareness mode: {mode}; this repo is live-only and does not bundle result artifacts")

    aggregate = _aggregate_seed_results(root, model_specs)
    table_rows = _write_table4(out_dir / "table4.tsv", aggregate, model_specs)
    _plot_figure19(out_dir / "figures", aggregate, model_specs)

    write_json(out_dir / "manifest.json", {
        "benchmark": "evaluation_awareness",
        "mode": mode,
        "created_at": now_utc(),
        "source": source,
        "metric_formulas": {
            "awareness_rate": "aware_count / total; denominator scope: per seed/model/task category",
            "pass_rate": "pass_count / total; denominator scope: per seed/model/task category",
            "accuracy": "correct / total; denominator scope: per seed/model/task category",
            "deploy_accuracy": "deploy_correct / deploy_total; denominator scope: per seed/model/task category",
            "original_rate": "original_aware / original_total; denominator scope: per seed/model/task category",
            "novel_rate": "novel_aware / novel_total; denominator scope: per seed/model/task category",
            "ci95": "t95(df=n_seed-1) * sample_sd / sqrt(n_seed)",
        },
    })
    write_json(out_dir / "scores.json", {"table4": table_rows, "aggregate": aggregate})
    return {"status": "complete", "artifact_dir": str(out_dir), "mode": mode}


def _mean_ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "ci95": math.nan, "n": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci95": 0.0, "n": 1}
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    ci = T95_DF2 * math.sqrt(var) / math.sqrt(len(values))
    return {"mean": mean, "ci95": ci, "n": len(values)}


def _run_live(
    ctx: RunContext,
    cfg: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, list[tuple[str, str, str]]]:
    target = ctx.config.get("target_model") or {}
    slug = str(cfg.get("model_slug") or target.get("name") or "kimi-k2.5-100k-reasoning")
    model_id = str(cfg.get("model") or target.get("model") or "moonshotai/kimi-k2.5")
    label = str(cfg.get("label") or target.get("label") or target.get("name") or slug)
    color = str(cfg.get("color") or "#E6A46F")
    live_root = out_dir / "live_seed_results"
    sections = cfg.get("sections") or ["transcript_classification"]
    seeds = cfg.get("seeds") or [0]
    if not isinstance(sections, list) or not sections:
        raise ValueError("evaluation_awareness.live sections must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("evaluation_awareness.live seeds must be a non-empty list")

    cmd = [
        sys.executable,
        str(ctx.repo_root / "scripts/evaluation_awareness/run_eval_awareness_live.py"),
        "--output-root",
        str(live_root),
        "--models",
        slug,
        "--live-model-slug",
        slug,
        "--live-model-id",
        model_id,
        "--live-provider",
        str(cfg.get("openrouter_provider") or target.get("openrouter_provider") or ""),
        "--seeds",
        ",".join(str(int(s)) for s in seeds),
        "--sections",
        ",".join(str(s) for s in sections),
        "--dataset-json",
        str(ctx.repo_root / "scripts/evaluation_awareness/eval_awareness/dataset.json"),
        "--no-import-legacy",
        "--force",
        "--timeout-sec",
        str(int(cfg.get("timeout_sec", 90))),
        "--retries",
        str(int(cfg.get("retries", 4))),
        "--sleep-sec",
        str(float(cfg.get("sleep_sec", 0.2))),
        "--transcript-samples",
        str(int(cfg.get("transcript_samples", 2))),
        "--transcript-max-tokens",
        str(int(cfg.get("transcript_max_tokens", 256))),
        "--real-evals-trials",
        str(int(cfg.get("real_evals_trials", 1))),
        "--real-evals-scenario-limit",
        str(int(cfg.get("real_evals_scenario_limit", 0))),
        "--real-evals-max-tokens",
        str(int(cfg.get("real_evals_max_tokens", 512))),
        "--contamination-trials",
        str(int(cfg.get("contamination_trials", 1))),
        "--contamination-scenario-limit",
        str(int(cfg.get("contamination_scenario_limit", 0))),
        "--contamination-max-tokens",
        str(int(cfg.get("contamination_max_tokens", 512))),
    ]
    if cfg.get("reasoning_enabled", target.get("reasoning_enabled")):
        cmd.append("--live-reasoning-enabled")
    reasoning_tokens = cfg.get("reasoning_tokens", target.get("reasoning_tokens"))
    if reasoning_tokens is not None:
        cmd.extend(["--live-reasoning-tokens", str(int(reasoning_tokens))])
    reasoning_effort = cfg.get("reasoning_effort", target.get("reasoning_effort"))
    if reasoning_effort:
        cmd.extend(["--live-reasoning-effort", str(reasoning_effort)])

    log_file = ctx.run_log_dir / "evaluation_awareness_live.log"
    env = os.environ.copy()
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=ctx.repo_root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation_awareness live runner failed; see {log_file}")
    expected = [
        live_root / str(section) / slug / f"seed_{int(seed):03d}.json"
        for section in sections
        for seed in seeds
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError(f"evaluation_awareness live runner missing outputs: {missing}")
    for path in expected:
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics") or {}
        if int(metrics.get("total", 0)) <= 0:
            raise RuntimeError(f"evaluation_awareness live runner produced zero valid samples: {path}")
    return live_root, [(slug, label, color)]


def _aggregate_seed_results(root: Path, model_specs: list[tuple[str, str, str]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for section in SECTIONS:
        aggregate[section] = {}
        for slug, label, _ in model_specs:
            model_dir = root / section / slug
            seed_files = sorted(model_dir.glob("seed_*.json"))
            seed_metrics: list[dict[str, Any]] = []
            for path in seed_files:
                import json
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") != "complete":
                    continue
                seed_metrics.append(data.get("metrics") or {})
            keys = sorted({k for m in seed_metrics for k, v in m.items() if isinstance(v, (int, float))})
            aggregate[section][slug] = {
                "label": label,
                "seeds": len(seed_metrics),
                "metrics": {k: _mean_ci([float(m[k]) for m in seed_metrics if k in m]) for k in keys},
                "denominators": [int(m.get("total", 0)) for m in seed_metrics if "total" in m],
            }
    return aggregate


def _pct(v: float) -> str:
    return "" if math.isnan(v) else f"{v * 100:.1f}"


def _write_table4(path: Path, aggregate: dict[str, Any], model_specs: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for slug, label, _ in model_specs:
        inc = aggregate["in_context_awareness"][slug]["metrics"]
        tr = aggregate["transcript_classification"][slug]["metrics"]
        co = aggregate["contamination_transfer"][slug]["metrics"]
        row = {
            "model": label,
            "in_context_aware_pct": _pct(inc.get("awareness_rate", {}).get("mean", math.nan)),
            "in_context_pass_pct": _pct(inc.get("pass_rate", {}).get("mean", math.nan)),
            "transcript_accuracy_pct": _pct(tr.get("accuracy", {}).get("mean", math.nan)),
            "transcript_eval_yes_pct": _pct(tr.get("eval_accuracy", {}).get("mean", math.nan)),
            "transcript_deploy_no_pct": _pct(tr.get("deploy_accuracy", {}).get("mean", math.nan)),
            "contamination_aware_pct": _pct(co.get("awareness_rate", {}).get("mean", math.nan)),
            "contamination_original_pct": _pct(co.get("original_rate", {}).get("mean", math.nan)),
            "contamination_novel_pct": _pct(co.get("novel_rate", {}).get("mean", math.nan)),
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _metric(aggregate: dict[str, Any], section: str, slug: str, key: str) -> tuple[float, float]:
    data = aggregate[section][slug]["metrics"].get(key, {"mean": math.nan, "ci95": math.nan})
    return data["mean"] * 100, data["ci95"] * 100


def _plot_figure19(out_dir: Path, aggregate: dict[str, Any], model_specs: list[tuple[str, str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [m[1] for m in model_specs]
    colors = [m[2] for m in model_specs]
    slugs = [m[0] for m in model_specs]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    panels = [
        ("in_context_awareness", "awareness_rate", "In-Context Awareness"),
        ("transcript_classification", "deploy_accuracy", "Transcript Deploy->NO"),
    ]
    for ax, (section, key, title) in zip(axes[:2], panels):
        means, cis = zip(*[_metric(aggregate, section, slug, key) for slug in slugs])
        ax.bar(x, means, color=colors, width=0.72)
        ax.errorbar(x, means, yerr=cis, fmt="none", ecolor="#222222", capsize=3)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_ylabel("Rate (%)")
        ax.set_xticks(x, labels=labels, rotation=25, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax = axes[2]
    width = 0.34
    orig, orig_ci = zip(*[_metric(aggregate, "contamination_transfer", slug, "original_rate") for slug in slugs])
    novel, novel_ci = zip(*[_metric(aggregate, "contamination_transfer", slug, "novel_rate") for slug in slugs])
    ax.bar(x - width / 2, orig, width=width, color=colors, label="Original")
    ax.bar(x + width / 2, novel, width=width, color=colors, alpha=0.45, label="Novel")
    ax.errorbar(x - width / 2, orig, yerr=orig_ci, fmt="none", ecolor="#222222", capsize=3)
    ax.errorbar(x + width / 2, novel, yerr=novel_ci, fmt="none", ecolor="#222222", capsize=3)
    ax.set_title("Contamination Transfer", fontsize=15, fontweight="bold")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "figure19_eval_awareness_main_panels.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "figure19_eval_awareness_main_panels.pdf", bbox_inches="tight")
    plt.close(fig)
