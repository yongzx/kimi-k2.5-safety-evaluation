"""Main evaluation runner for Kimi mental health analysis.

Usage:
    # Run all evaluations (full)
    python run_eval.py

    # Run only stigma evaluation
    python run_eval.py --eval stigma

    # Run only appropriateness evaluation
    python run_eval.py --eval appropriateness

    # Run only delusion confirmation evaluation
    python run_eval.py --eval delusion

    # Run a quick test (subset of data)
    python run_eval.py --test

    # Specify output file
    python run_eval.py --output results/my_run.json
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import EVAL_MODEL, JUDGE_MODEL, OUTPUT_DIR, MODEL_PROVIDERS, MODEL_REASONING, MODEL_MAX_TOKENS
from api import call_model
from stigma_data import (
    generate_all_vignettes,
    format_stigma_messages,
    parse_answer,
    is_stigmatizing_response,
    STIGMA_QUESTIONS,
)
from appropriateness_data import (
    APPROPRIATENESS_STIMULI,
    THERAPIST_SYSTEM_PROMPT,
    format_eval_prompt,
    format_judge_prompt,
    parse_judge_response,
    is_appropriate_response,
)
from delusion_data import (
    DELUSION_CASES,
    DelusionCase,
    Turn,
    ScoreType,
    score_turn,
    format_conversation,
    DCS_START_TURN,
    HES_START_TURN,
    SIS_START_TURN,
)

# Parallelization settings
MAX_WORKERS = 32

# Progress file for monitoring long-running evals
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Runtime overrides for eval model request settings (resolved in main()).
ACTIVE_EVAL_PROVIDER = MODEL_PROVIDERS.get(EVAL_MODEL)
ACTIVE_EVAL_REASONING_EFFORT = None
ACTIVE_EVAL_REASONING_MAX_TOKENS = None
ACTIVE_EVAL_MAX_TOKENS = MODEL_MAX_TOKENS.get(EVAL_MODEL, 65536)
ACTIVE_EVAL_SEED = None


def _resolve_reasoning_config(model: str) -> tuple[str | None, int | None]:
    """Resolve per-model reasoning config from config.py."""
    reasoning_cfg = MODEL_REASONING.get(model)

    if isinstance(reasoning_cfg, str):
        return reasoning_cfg, None
    if isinstance(reasoning_cfg, dict):
        return reasoning_cfg.get("effort"), reasoning_cfg.get("max_tokens")
    return None, None


def save_progress(eval_type: str, completed: int, total: int, model: str):
    """Save progress to a file for monitoring."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress = {
        "eval_type": eval_type,
        "model": model,
        "completed": completed,
        "total": total,
        "percent": round(100 * completed / total, 1) if total > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _eval_single_stigma(vignette: dict, question: dict) -> dict:
    """Evaluate a single stigma prompt. Used for parallel execution."""
    messages = format_stigma_messages(vignette, question)

    try:
        response = call_model(
            EVAL_MODEL,
            messages,
            temperature=0.0,
            max_tokens=ACTIVE_EVAL_MAX_TOKENS,
            provider=ACTIVE_EVAL_PROVIDER,
            reasoning_effort=ACTIVE_EVAL_REASONING_EFFORT,
            reasoning_max_tokens=ACTIVE_EVAL_REASONING_MAX_TOKENS,
            seed=ACTIVE_EVAL_SEED,
        )
        answer = parse_answer(response)
        is_stigma = is_stigmatizing_response(question, answer)
    except Exception as e:
        response = f"ERROR: {e}"
        answer = None
        is_stigma = None

    return {
        "eval_type": "stigma",
        "model": EVAL_MODEL,
        "condition": vignette["condition"],
        "gender": vignette["gender"],
        "race": vignette["race"],
        "education": vignette["education"],
        "name": vignette["name"],
        "question_id": question["id"],
        "raw_response": response,
        "parsed_answer": answer,
        "is_stigmatizing": is_stigma,
    }


def run_stigma_eval(vignettes: list[dict], questions: list[dict]) -> list[dict]:
    """
    Run the stigma evaluation with parallel API calls.

    Args:
        vignettes: List of vignette dicts
        questions: List of question dicts

    Returns:
        List of result dicts with all metadata and responses
    """
    # Build list of all (vignette, question) pairs
    tasks = [(v, q) for v in vignettes for q in questions]
    total = len(tasks)

    print(f"\n=== Running Stigma Evaluation ===")
    print(f"Vignettes: {len(vignettes)}, Questions: {len(questions)}, Total prompts: {total}")
    print(f"Parallel workers: {MAX_WORKERS}")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(_eval_single_stigma, v, q): (v, q) for v, q in tasks}

        # Collect results with progress bar
        with tqdm(total=total, desc="Stigma eval") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                # Save progress every 10 completions
                if len(results) % 10 == 0:
                    save_progress("stigma", len(results), total, EVAL_MODEL)

    save_progress("stigma", len(results), total, EVAL_MODEL)  # Final save
    return results


def _eval_single_appropriateness(stimulus: dict, system_prompt: str | None) -> dict:
    """Evaluate a single appropriateness prompt. Used for parallel execution."""
    judge_provider = MODEL_PROVIDERS.get(JUDGE_MODEL)

    # Step 1: Get model response
    eval_messages = format_eval_prompt(stimulus, system_prompt)
    try:
        model_response = call_model(
            EVAL_MODEL,
            eval_messages,
            max_tokens=ACTIVE_EVAL_MAX_TOKENS,
            provider=ACTIVE_EVAL_PROVIDER,
            reasoning_effort=ACTIVE_EVAL_REASONING_EFFORT,
            reasoning_max_tokens=ACTIVE_EVAL_REASONING_MAX_TOKENS,
            seed=ACTIVE_EVAL_SEED,
        )
    except Exception as e:
        model_response = f"ERROR: {e}"

    # Step 2: Judge the response
    judge_messages = format_judge_prompt(stimulus, model_response)
    try:
        judge_response = call_model(JUDGE_MODEL, judge_messages, max_tokens=100, provider=judge_provider)
        judge_answer = parse_judge_response(judge_response)
        is_appropriate = is_appropriate_response(stimulus, judge_answer)
    except Exception as e:
        judge_response = f"ERROR: {e}"
        judge_answer = None
        is_appropriate = None

    return {
        "eval_type": "appropriateness",
        "model": EVAL_MODEL,
        "judge_model": JUDGE_MODEL,
        "condition": stimulus["condition"],
        "stimulus_name": stimulus["name"],
        "stimulus_text": stimulus["stimulus"],
        "verification_question": stimulus["verification_question"],
        "correct_answer": stimulus["correct_answer"],
        "model_response": model_response,
        "judge_raw_response": judge_response,
        "judge_answer": judge_answer,
        "is_appropriate": is_appropriate,
        "used_system_prompt": system_prompt is not None,
    }


def run_appropriateness_eval(
    stimuli: list[dict],
    use_system_prompt: bool = True,
) -> list[dict]:
    """
    Run the appropriateness evaluation with parallel API calls.

    Args:
        stimuli: List of stimulus dicts
        use_system_prompt: Whether to use the therapist system prompt

    Returns:
        List of result dicts with all metadata, responses, and judge classifications
    """
    system_prompt = THERAPIST_SYSTEM_PROMPT if use_system_prompt else None

    print(f"\n=== Running Appropriateness Evaluation ===")
    print(f"Stimuli: {len(stimuli)}, System prompt: {'yes' if use_system_prompt else 'no'}")
    print(f"Parallel workers: {MAX_WORKERS}")

    results = []
    total = len(stimuli)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_eval_single_appropriateness, s, system_prompt): s for s in stimuli}

        with tqdm(total=total, desc="Appropriateness eval") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                save_progress("appropriateness", len(results), total, EVAL_MODEL)

    return results


def compute_stigma_summary(results: list[dict]) -> dict:
    """Compute summary statistics for stigma evaluation."""
    stigma_results = [r for r in results if r["is_stigmatizing"] is not None]

    if not stigma_results:
        return {"error": "No valid stigma results"}

    total = len(stigma_results)
    stigmatizing = sum(1 for r in stigma_results if r["is_stigmatizing"])

    # By condition
    by_condition = {}
    for condition in ["depression", "schizophrenia", "alcohol_dependence", "daily_troubles"]:
        cond_results = [r for r in stigma_results if r["condition"] == condition]
        if cond_results:
            cond_stigma = sum(1 for r in cond_results if r["is_stigmatizing"])
            by_condition[condition] = {
                "total": len(cond_results),
                "stigmatizing": cond_stigma,
                "rate": cond_stigma / len(cond_results),
            }

    return {
        "total_evaluated": total,
        "total_stigmatizing": stigmatizing,
        "stigma_rate": stigmatizing / total,
        "by_condition": by_condition,
    }


def compute_appropriateness_summary(results: list[dict]) -> dict:
    """Compute summary statistics for appropriateness evaluation."""
    valid_results = [r for r in results if r["is_appropriate"] is not None]

    if not valid_results:
        return {"error": "No valid appropriateness results"}

    total = len(valid_results)
    appropriate = sum(1 for r in valid_results if r["is_appropriate"])

    # By condition
    by_condition = {}
    for condition in ["delusions", "mania", "hallucinations", "suicidal_ideation", "ocd"]:
        cond_results = [r for r in valid_results if r["condition"] == condition]
        if cond_results:
            cond_appropriate = sum(1 for r in cond_results if r["is_appropriate"])
            by_condition[condition] = {
                "total": len(cond_results),
                "appropriate": cond_appropriate,
                "rate": cond_appropriate / len(cond_results),
            }

    return {
        "total_evaluated": total,
        "total_appropriate": appropriate,
        "appropriateness_rate": appropriate / total,
        "by_condition": by_condition,
    }


def run_delusion_eval(cases: list[DelusionCase], eval_model: str) -> list[dict]:
    """
    Run the delusion confirmation evaluation.

    This is a multi-turn evaluation - each case has 12 turns of pre-scripted
    user messages. The model responds to each, and we score DCS/HES/SIS.

    NOTE: Multi-turn conversations cannot be parallelized across turns
    (each turn depends on previous responses). We parallelize across cases.
    """
    print(f"\n=== Running Delusion Confirmation Evaluation ===")
    print(f"Cases: {len(cases)}, Turns per case: {len(cases[0].turns) if cases else 12}")
    print(f"Judge model: {JUDGE_MODEL}")

    judge_provider = MODEL_PROVIDERS.get(JUDGE_MODEL)

    all_results = []

    for case in tqdm(cases, desc="Delusion cases"):
        case_results = []
        turns: list[Turn] = []

        # Run through all turns sequentially (multi-turn conversation)
        for turn_num, user_message in enumerate(case.turns, start=1):
            # Build messages with full conversation history
            messages = []
            for t in turns:
                messages.append({"role": "user", "content": t.user_message})
                messages.append({"role": "assistant", "content": t.bot_response})
            messages.append({"role": "user", "content": user_message})

            # Get model response
            try:
                response = call_model(
                    eval_model,
                    messages,
                    max_tokens=ACTIVE_EVAL_MAX_TOKENS,
                    provider=ACTIVE_EVAL_PROVIDER,
                    reasoning_effort=ACTIVE_EVAL_REASONING_EFFORT,
                    reasoning_max_tokens=ACTIVE_EVAL_REASONING_MAX_TOKENS,
                    seed=ACTIVE_EVAL_SEED,
                )
            except Exception as e:
                response = f"ERROR: {e}"

            # Record this turn
            turn = Turn(
                turn_number=turn_num,
                user_message=user_message,
                bot_response=response,
            )
            turns.append(turn)

            # Score this turn (only scores if past threshold turns)
            scores = score_turn(turns, turn_num, case.harm_type, judge_provider=judge_provider)

            # Record result for this turn
            turn_result = {
                "eval_type": "delusion",
                "model": eval_model,
                "judge_model": JUDGE_MODEL,
                "case_id": case.id,
                "case_name": case.name,
                "theme": case.theme,
                "condition": case.condition,
                "harm_type": case.harm_type,
                "turn_number": turn_num,
                "user_message": user_message,
                "model_response": response,
                "dcs_score": scores[ScoreType.DCS].score,
                "dcs_explanation": scores[ScoreType.DCS].explanation,
                "hes_score": scores[ScoreType.HES].score,
                "hes_explanation": scores[ScoreType.HES].explanation,
                "sis_score": scores[ScoreType.SIS].score,
                "sis_explanation": scores[ScoreType.SIS].explanation,
            }
            case_results.append(turn_result)

        all_results.extend(case_results)
        save_progress("delusion", len(all_results), len(cases) * 12, eval_model)

    return all_results


def compute_delusion_summary(results: list[dict]) -> dict:
    """Compute summary statistics for delusion confirmation evaluation."""
    if not results:
        return {"error": "No delusion results"}

    # Filter to scored turns only (score >= 0)
    dcs_results = [r for r in results if r["dcs_score"] >= 0]
    hes_results = [r for r in results if r["hes_score"] >= 0]
    sis_results = [r for r in results if r["sis_score"] >= 0]

    def mean(scores):
        return sum(scores) / len(scores) if scores else 0

    # Overall metrics
    summary = {
        "total_turns": len(results),
        "dcs": {
            "scored_turns": len(dcs_results),
            "mean_score": mean([r["dcs_score"] for r in dcs_results]),
            "score_distribution": {
                0: sum(1 for r in dcs_results if r["dcs_score"] == 0),
                1: sum(1 for r in dcs_results if r["dcs_score"] == 1),
                2: sum(1 for r in dcs_results if r["dcs_score"] == 2),
            },
        },
        "hes": {
            "scored_turns": len(hes_results),
            "mean_score": mean([r["hes_score"] for r in hes_results]),
            "score_distribution": {
                0: sum(1 for r in hes_results if r["hes_score"] == 0),
                1: sum(1 for r in hes_results if r["hes_score"] == 1),
                2: sum(1 for r in hes_results if r["hes_score"] == 2),
            },
        },
        "sis": {
            "scored_turns": len(sis_results),
            "mean_score": mean([r["sis_score"] for r in sis_results]),
            "intervention_rate": sum(1 for r in sis_results if r["sis_score"] == 1) / len(sis_results) if sis_results else 0,
        },
    }

    # By theme - dynamically detect themes from results
    by_theme = {}
    all_themes = set(r["theme"] for r in dcs_results)
    for theme in sorted(all_themes):
        theme_dcs = [r for r in dcs_results if r["theme"] == theme]
        theme_hes = [r for r in hes_results if r["theme"] == theme]
        if theme_dcs:
            by_theme[theme] = {
                "dcs_mean": mean([r["dcs_score"] for r in theme_dcs]),
                "hes_mean": mean([r["hes_score"] for r in theme_hes]) if theme_hes else None,
            }
    summary["by_theme"] = by_theme

    # By condition (explicit vs implicit) - case-insensitive
    by_condition = {}
    for cond in ["explicit", "implicit"]:
        cond_dcs = [r for r in dcs_results if r["condition"].lower() == cond]
        if cond_dcs:
            by_condition[cond] = {
                "dcs_mean": mean([r["dcs_score"] for r in cond_dcs]),
            }
    summary["by_condition"] = by_condition

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run mental health evaluation")
    parser.add_argument(
        "--eval",
        choices=["stigma", "appropriateness", "delusion", "all"],
        default="all",
        help="Which evaluation to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override eval model (default: use config.py EVAL_MODEL)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with subset of data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in results/)",
    )
    parser.add_argument(
        "--with-system-prompt",
        action="store_true",
        help="Run appropriateness eval WITH therapist system prompt (default: no system prompt)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Override reasoning.effort for eval model",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=None,
        help="Override reasoning.max_tokens for eval model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens for eval model responses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for eval model calls",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Optional number of delusion cases to run.",
    )
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=None,
        help="Optional number of turns per delusion case to run.",
    )
    args = parser.parse_args()

    # Override eval model if specified
    global EVAL_MODEL, ACTIVE_EVAL_PROVIDER, ACTIVE_EVAL_REASONING_EFFORT
    global ACTIVE_EVAL_REASONING_MAX_TOKENS, ACTIVE_EVAL_MAX_TOKENS, ACTIVE_EVAL_SEED
    if args.model:
        EVAL_MODEL = args.model
        print(f"Using model override: {EVAL_MODEL}")

    ACTIVE_EVAL_PROVIDER = MODEL_PROVIDERS.get(EVAL_MODEL)
    cfg_effort, cfg_reasoning_max_tokens = _resolve_reasoning_config(EVAL_MODEL)
    ACTIVE_EVAL_REASONING_EFFORT = cfg_effort
    ACTIVE_EVAL_REASONING_MAX_TOKENS = cfg_reasoning_max_tokens

    if args.reasoning_effort is not None:
        ACTIVE_EVAL_REASONING_EFFORT = args.reasoning_effort
    if args.reasoning_max_tokens is not None:
        ACTIVE_EVAL_REASONING_MAX_TOKENS = args.reasoning_max_tokens

    ACTIVE_EVAL_MAX_TOKENS = args.max_tokens if args.max_tokens is not None else MODEL_MAX_TOKENS.get(EVAL_MODEL, 65536)
    ACTIVE_EVAL_SEED = args.seed

    print(f"Provider: {ACTIVE_EVAL_PROVIDER}")
    print(f"max_tokens: {ACTIVE_EVAL_MAX_TOKENS}")
    if ACTIVE_EVAL_SEED is not None:
        print(f"seed: {ACTIVE_EVAL_SEED}")
    if ACTIVE_EVAL_REASONING_EFFORT or ACTIVE_EVAL_REASONING_MAX_TOKENS is not None:
        print(
            "Reasoning config:",
            json.dumps(
                {
                    "effort": ACTIVE_EVAL_REASONING_EFFORT,
                    "max_tokens": ACTIVE_EVAL_REASONING_MAX_TOKENS,
                }
            ),
        )

    # Generate output filename
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = EVAL_MODEL.split("/")[-1]  # e.g., "kimi-k2.5" from "moonshotai/kimi-k2.5"
        seed_suffix = f"_seed{ACTIVE_EVAL_SEED}" if ACTIVE_EVAL_SEED is not None else ""
        output_path = OUTPUT_DIR / f"eval_{model_short}{seed_suffix}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    summaries = {}

    # Run stigma evaluation
    if args.eval in ["stigma", "all"]:
        vignettes = generate_all_vignettes()
        questions = STIGMA_QUESTIONS

        if args.test:
            # Just 2 vignettes (depression, schizophrenia), stigma questions only
            vignettes = [v for v in vignettes if v["condition"] in ["depression", "schizophrenia"]][:2]
            # Use stigma-measuring questions only (skip diagnostic ones)
            questions = [q for q in questions if q["stigma_direction"] is not None][:3]

        stigma_results = run_stigma_eval(vignettes, questions)
        all_results.extend(stigma_results)
        summaries["stigma"] = compute_stigma_summary(stigma_results)

    # Run appropriateness evaluation
    if args.eval in ["appropriateness", "all"]:
        stimuli = APPROPRIATENESS_STIMULI

        if args.test:
            # Just 2 stimuli for testing
            stimuli = stimuli[:2]

        appropriateness_results = run_appropriateness_eval(
            stimuli,
            use_system_prompt=args.with_system_prompt,
        )
        all_results.extend(appropriateness_results)
        summaries["appropriateness"] = compute_appropriateness_summary(appropriateness_results)

    # Run delusion confirmation evaluation
    if args.eval in ["delusion", "all"]:
        cases = DELUSION_CASES

        if args.test:
            # Just 1 case with 3 turns for testing
            cases = cases[:1]
            cases[0] = DelusionCase(
                id=cases[0].id,
                name=cases[0].name,
                theme=cases[0].theme,
                condition=cases[0].condition,
                harm_type=cases[0].harm_type,
                turns=cases[0].turns[:3],  # Only first 3 turns
            )
        elif args.case_limit is not None or args.turn_limit is not None:
            if args.case_limit is not None:
                cases = cases[: max(1, int(args.case_limit))]
            if args.turn_limit is not None:
                turn_limit = max(1, int(args.turn_limit))
                cases = [
                    DelusionCase(
                        id=case.id,
                        name=case.name,
                        theme=case.theme,
                        condition=case.condition,
                        harm_type=case.harm_type,
                        turns=case.turns[:turn_limit],
                    )
                    for case in cases
                ]

        delusion_results = run_delusion_eval(cases, EVAL_MODEL)
        all_results.extend(delusion_results)
        summaries["delusion"] = compute_delusion_summary(delusion_results)

    # Save results
    output_data = {
        "metadata": {
            "eval_model": EVAL_MODEL,
            "judge_model": JUDGE_MODEL,
            "timestamp": datetime.now().isoformat(),
            "test_mode": args.test,
            "eval_type": args.eval,
            "eval_provider": ACTIVE_EVAL_PROVIDER,
            "eval_max_tokens": ACTIVE_EVAL_MAX_TOKENS,
            "eval_seed": ACTIVE_EVAL_SEED,
            "eval_reasoning": {
                "effort": ACTIVE_EVAL_REASONING_EFFORT,
                "max_tokens": ACTIVE_EVAL_REASONING_MAX_TOKENS,
            }
            if (ACTIVE_EVAL_REASONING_EFFORT or ACTIVE_EVAL_REASONING_MAX_TOKENS is not None)
            else None,
        },
        "summaries": summaries,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n=== Results saved to {output_path} ===")
    print(f"\nSummaries:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
