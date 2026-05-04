#!/usr/bin/env python3
"""
Run additional seed indices for exp_002 prompt ablation without rerunning existing trials.

This script is resume-friendly:
- Runs only requested trial indices that are missing on disk
- Skips existing `trial_XXX.json` files
- Re-aggregates `summary.json` + `summary.csv` from all trial files in `data_dir`
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .config import AblationConfig
    from .run_ablation import aggregate, run_trial, _write_transcript
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from benchmarks.self_replication._impl.exp_002_prompt_ablation.config import AblationConfig
    from benchmarks.self_replication._impl.exp_002_prompt_ablation.run_ablation import aggregate, run_trial, _write_transcript

logger = logging.getLogger("ablation.add_seeds")


def _parse_trial_indices(raw: str) -> list[int]:
    vals: list[int] = []
    for part in re.split(r"[,\s]+", raw.strip()):
        if not part:
            continue
        try:
            v = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid trial index: {part}") from exc
        if v < 0:
            raise ValueError(f"trial index must be >= 0: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("no trial indices provided")
    return sorted(set(vals))


def _collect_prompt_files(prompts_dir: Path, prompt_min: int, prompt_max: int) -> list[Path]:
    files = sorted(prompts_dir.glob("P*.txt"))
    out: list[Path] = []
    for pf in files:
        m = re.match(r"^P(\d{2})_", pf.stem)
        if not m:
            continue
        idx = int(m.group(1))
        if prompt_min <= idx <= prompt_max:
            out.append(pf)
    return out


def _load_all_trials(data_dir: Path, prompt_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    all_results: dict[str, list[dict[str, Any]]] = {}
    for pid in prompt_ids:
        pdir = data_dir / pid
        trials = []
        if pdir.exists():
            for jf in sorted(pdir.glob("trial_*.json")):
                try:
                    trials.append(json.loads(jf.read_text()))
                except Exception:
                    logger.warning("Skipping unreadable trial JSON: %s", jf)
        all_results[pid] = trials
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run only missing additional seeds for exp_002")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trial-indices", type=str, required=True,
                        help="Comma/space-separated trial indices to run (for example: '3,4').")
    parser.add_argument("--prompt-min", type=int, default=1)
    parser.add_argument("--prompt-max", type=int, default=12)
    parser.add_argument("--prompts-dir", type=str, default="benchmarks/self_replication/prompts")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api")
    parser.add_argument("--live", action="store_true", help="Use real RunPod API")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-pods", type=int, default=3)
    parser.add_argument("--spending-cap", type=float, default=5.0)
    parser.add_argument("--adversary-interval", type=float, default=10.0)
    parser.add_argument("--adversary-delay", type=float, default=20.0)
    parser.add_argument("--reasoning-effort", type=str, default=None)
    parser.add_argument("--reasoning-tokens", type=int, default=None)
    parser.add_argument("--reasoning-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--model-args-json", type=str, default=None)
    args = parser.parse_args()

    try:
        trial_indices = _parse_trial_indices(args.trial_indices)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    model_args: dict[str, Any] = {}
    if args.model_args_json:
        try:
            parsed = json.loads(args.model_args_json)
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid --model-args-json: {exc}")
            sys.exit(1)
        if not isinstance(parsed, dict):
            print("ERROR: --model-args-json must decode to a JSON object")
            sys.exit(1)
        model_args = parsed

    cfg = AblationConfig(
        base_url=args.base_url,
        model=args.model,
        dry_run=not args.live,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_tokens=args.reasoning_tokens,
        reasoning_enabled=args.reasoning_enabled,
        extra_model_args=model_args,
        max_turns=args.max_turns,
        max_pods=args.max_pods,
        spending_cap_usd=args.spending_cap,
        adversary_interval_sec=args.adversary_interval,
        adversary_initial_delay_sec=args.adversary_delay,
        num_trials=len(trial_indices),
        prompts_dir=args.prompts_dir,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
    )

    if not cfg.openrouter_api_key:
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    if args.prompt_min < 1 or args.prompt_max < args.prompt_min:
        print(f"ERROR: invalid prompt range: {args.prompt_min}..{args.prompt_max}")
        sys.exit(1)

    prompts_dir = Path(cfg.prompts_dir)
    prompt_files = _collect_prompt_files(prompts_dir, args.prompt_min, args.prompt_max)
    if not prompt_files:
        print(f"ERROR: no prompt files in range P{args.prompt_min:02d}-P{args.prompt_max:02d} at {prompts_dir}")
        sys.exit(1)

    out_dir = Path(cfg.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "mode": "live" if not cfg.dry_run else "dry-run",
        "model": cfg.model,
        "prompt_range": [args.prompt_min, args.prompt_max],
        "prompt_ids": [pf.stem for pf in prompt_files],
        "trial_indices": trial_indices,
        "max_turns": cfg.max_turns,
        "max_tokens": cfg.max_tokens,
        "reasoning_effort": cfg.reasoning_effort,
        "reasoning_tokens": cfg.reasoning_tokens,
        "reasoning_enabled": cfg.reasoning_enabled,
        "extra_model_args": cfg.extra_model_args,
        "base_url": cfg.base_url,
    }
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"run_config_additional_{stamp}.json").write_text(
        json.dumps(run_meta, indent=2, default=str)
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    def _kill(sig, frame):
        logger.warning("Kill switch via signal %d", sig)
        cfg.kill_switch = True

    signal.signal(signal.SIGUSR1, _kill)

    print("=== Prompt Ablation Additional Seeds (exp_002) ===")
    print(f"  Model: {cfg.model}")
    print(f"  Prompts: {len(prompt_files)} (P{args.prompt_min:02d}-P{args.prompt_max:02d})")
    print(f"  Trial indices requested: {trial_indices}")
    print(f"  Mode: {'LIVE' if not cfg.dry_run else 'dry-run'}")
    print(f"  Max tokens: {cfg.max_tokens}")
    if cfg.reasoning_effort:
        print(f"  Reasoning effort: {cfg.reasoning_effort}")
    if cfg.reasoning_tokens is not None:
        print(f"  Reasoning tokens: {cfg.reasoning_tokens}")
    if cfg.reasoning_enabled is not None:
        print(f"  Reasoning enabled: {cfg.reasoning_enabled}")
    print()

    total_run = 0
    total_skip = 0

    for pf in prompt_files:
        prompt_id = pf.stem
        system_prompt = pf.read_text()
        prompt_out = out_dir / prompt_id
        prompt_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_id}")
        print(f"{'='*60}")

        for trial_idx in trial_indices:
            json_path = prompt_out / f"trial_{trial_idx:03d}.json"
            if json_path.exists():
                print(f"  Trial {trial_idx} ... SKIP (exists)")
                total_skip += 1
                continue

            print(f"  Trial {trial_idx} ...", end=" ", flush=True)
            trial_log_path = log_dir / f"{prompt_id}_trial_{trial_idx:03d}.log"
            fh = logging.FileHandler(trial_log_path, mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
            trial_logger = logging.getLogger(f"ablation.{prompt_id}.{trial_idx}")
            trial_logger.addHandler(fh)
            trial_logger.setLevel(logging.INFO)

            try:
                result = run_trial(prompt_id, system_prompt, cfg, trial_idx, trial_logger)
            except Exception:
                trial_logger.exception("Trial failed")
                print("FAILED")
                continue
            finally:
                trial_logger.removeHandler(fh)
                fh.close()

            json_path.write_text(json.dumps(result, indent=2, default=str))
            _write_transcript(result, prompt_out / f"trial_{trial_idx:03d}_transcript.md")
            ms = result["milestones"]
            print(
                " ".join([
                    f"M1:{'Y' if ms['m1_detection'] else 'n'}",
                    f"M2:{'Y' if ms['m2_intent'] else 'n'}",
                    f"M3:{'Y' if ms['m3_replication'] else 'n'}",
                    f"M4:{'Y' if ms['m4_independence'] else 'n'}",
                ]),
                f" turns={result['turns']}",
                f" pods={result['estimated_spend_usd']:.0f}¢",
            )
            total_run += 1

    prompt_ids = [pf.stem for pf in prompt_files]
    all_results = _load_all_trials(out_dir, prompt_ids)
    aggregate(all_results, out_dir)

    print(f"\n[done] trials_run={total_run} trials_skipped={total_skip}")
    print(f"[done] Results: {out_dir}")
    print(f"[done] Summary: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
