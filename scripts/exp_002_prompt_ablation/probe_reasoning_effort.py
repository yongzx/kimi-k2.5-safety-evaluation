#!/usr/bin/env python3
"""
Probe reasoning token usage across reasoning_effort settings.

Why this script exists:
- Single-call comparisons are noisy.
- Truncated responses (finish_reason=length) can distort token comparisons.

This script retries each (question, effort) pair until it collects a target
number of clean `finish_reason=stop` samples (or reaches max attempts), then
reports medians on clean samples only.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS = [
    {
        "qid": "Q1",
        "text": (
            "Number theory (CRT): Find the smallest positive integer n such that "
            "n ≡ 2 (mod 3), n ≡ 3 (mod 5), n ≡ 5 (mod 7), n ≡ 7 (mod 11), and "
            "n ≡ 11 (mod 13). Return ONLY the final answer in one short line."
        ),
    },
    {
        "qid": "Q2",
        "text": (
            "Combinatorics (inclusion-exclusion): How many onto functions are there "
            "from a 9-element set to a 6-element set? Return ONLY the final answer "
            "in one short line."
        ),
    },
    {
        "qid": "Q3",
        "text": (
            "Algebra/Vieta: Let p(x)=x^4-10x^3+35x^2-50x+24 with roots r1,r2,r3,r4. "
            "Compute T = sum_{i=1}^4 1/(5-ri). Return ONLY the final answer in one "
            "short line."
        ),
    },
]


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_questions(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_QUESTIONS
    p = Path(path)
    obj = json.loads(p.read_text())
    if not isinstance(obj, list):
        raise ValueError("questions JSON must be a list")
    out: list[dict[str, str]] = []
    for i, item in enumerate(obj, 1):
        if not isinstance(item, dict):
            raise ValueError(f"questions[{i}] must be an object")
        qid = str(item.get("qid") or f"Q{i}")
        text = str(item.get("text") or "").strip()
        if not text:
            raise ValueError(f"questions[{i}] missing non-empty text")
        out.append({"qid": qid, "text": text})
    return out


def _chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
    effort: str,
    max_tokens: int,
    timeout_sec: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": effort,
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _clean_tokens(rows: list[dict[str, Any]], qid: str, effort: str) -> list[int]:
    vals: list[int] = []
    for r in rows:
        if r.get("qid") != qid or r.get("effort") != effort:
            continue
        if r.get("status") != "ok":
            continue
        if r.get("finish_reason") != "stop":
            continue
        rtok = r.get("reasoning_tokens")
        if isinstance(rtok, int):
            vals.append(rtok)
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe reasoning tokens by effort with clean-stop filtering."
    )
    parser.add_argument("--model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--efforts", default="low,medium,high,xhigh")
    parser.add_argument("--questions-json", default=None)
    parser.add_argument("--need-stop-samples", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--sleep-sec", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        default="data/processed/002-prompt-ablation-3seed-hybrid-goal4-p01to12/_reasoning_effort_probe",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENROUTER_API_KEY not set")

    efforts = _parse_csv(args.efforts)
    if not efforts:
        raise SystemExit("ERROR: --efforts is empty")

    questions = _load_questions(args.questions_json)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_json = out_dir / f"reasoning_effort_probe_{stamp}.json"
    out_tsv = out_dir / f"reasoning_effort_probe_{stamp}.tsv"

    rows: list[dict[str, Any]] = []
    run_meta = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model": args.model,
        "efforts": efforts,
        "questions": questions,
        "need_stop_samples": args.need_stop_samples,
        "max_attempts": args.max_attempts,
        "max_tokens": args.max_tokens,
        "timeout_sec": args.timeout_sec,
    }

    def persist() -> None:
        payload = {"meta": run_meta, "rows": rows}
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    total_pairs = len(questions) * len(efforts)
    pair_idx = 0

    for q in questions:
        qid = q["qid"]
        prompt = q["text"]
        for effort in efforts:
            pair_idx += 1
            stop_count = 0
            attempt = 0
            print(
                f"[{pair_idx}/{total_pairs}] {qid} effort={effort} "
                f"(target_stop={args.need_stop_samples})"
            )
            while stop_count < args.need_stop_samples and attempt < args.max_attempts:
                attempt += 1
                rec: dict[str, Any] = {
                    "qid": qid,
                    "effort": effort,
                    "attempt": attempt,
                    "input": {
                        "model": args.model,
                        "prompt": prompt,
                        "reasoning_effort": effort,
                        "max_tokens": args.max_tokens,
                    },
                    "status": "ok",
                    "finish_reason": None,
                    "reasoning_tokens": None,
                    "cot": None,
                    "output": None,
                    "error": None,
                }
                try:
                    obj = _chat(
                        api_key=api_key,
                        model=args.model,
                        prompt=prompt,
                        effort=effort,
                        max_tokens=args.max_tokens,
                        timeout_sec=args.timeout_sec,
                    )
                    ch = (obj.get("choices") or [{}])[0]
                    msg = ch.get("message") or {}
                    ctd = ((obj.get("usage") or {}).get("completion_tokens_details") or {})
                    rec["finish_reason"] = ch.get("finish_reason")
                    rec["reasoning_tokens"] = ctd.get("reasoning_tokens")
                    rec["cot"] = msg.get("reasoning")
                    rec["output"] = msg.get("content")
                    if rec["finish_reason"] == "stop" and isinstance(rec["reasoning_tokens"], int):
                        stop_count += 1
                    print(
                        f"  attempt={attempt} finish={rec['finish_reason']} "
                        f"rtok={rec['reasoning_tokens']} stop_count={stop_count}"
                    )
                except urllib.error.HTTPError as exc:
                    rec["status"] = f"http_{exc.code}"
                    rec["error"] = exc.read().decode("utf-8", "replace")[:1000]
                    print(f"  attempt={attempt} {rec['status']}")
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "exception"
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  attempt={attempt} exception")

                rows.append(rec)
                persist()
                time.sleep(args.sleep_sec)

    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("qid\teffort\tn_stop\tmedian_rtok\tmean_rtok\tstop_rtoks\n")
        print("\nCLEAN_AGGREGATE")
        print("qid\teffort\tn_stop\tmedian_rtok\tmean_rtok\tstop_rtoks")
        for q in questions:
            qid = q["qid"]
            for effort in efforts:
                vals = _clean_tokens(rows, qid, effort)
                if vals:
                    med = int(statistics.median(vals))
                    mean = round(sum(vals) / len(vals), 1)
                    line = f"{qid}\t{effort}\t{len(vals)}\t{med}\t{mean}\t{vals}"
                else:
                    line = f"{qid}\t{effort}\t0\tNA\tNA\t[]"
                print(line)
                f.write(line + "\n")

    print(f"\nOUT_JSON {out_json}")
    print(f"OUT_TSV  {out_tsv}")


if __name__ == "__main__":
    main()
