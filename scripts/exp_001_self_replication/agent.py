"""Core agentic loop: system prompt -> model tool calls -> execute -> repeat."""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from typing import Any

from .config import ExperimentConfig
from .milestones import MilestoneTracker
from .runpod_client import RunPodClient
from .system_prompt import INITIAL_USER_MESSAGE, SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, dispatch_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM call helpers (adapted from kimi_api_boilerplate.py)
# ---------------------------------------------------------------------------


def _chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    retries: int = 3,
) -> dict:
    """Call the OpenRouter proxy and return the response dict."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    data = json.dumps(body).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=180)
            return json.loads(resp.read().decode())
        except Exception as exc:
            logger.warning("API call attempt %d failed: %s", attempt + 1, exc)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _extract_text(msg: dict) -> str:
    """Pull visible text from an assistant message, stripping <think> blocks."""
    content = (msg.get("content") or "").strip()
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"^.*?</think>\s*", "", content, flags=re.DOTALL)
    content = re.sub(r"<think>.*$", "", content, flags=re.DOTALL)
    return content.strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


class AgentLoop:
    def __init__(
        self,
        config: ExperimentConfig,
        runpod_client: RunPodClient,
        milestones: MilestoneTracker,
        agent_pod_id: str,
    ):
        self.config = config
        self.runpod_client = runpod_client
        self.milestones = milestones
        self.agent_pod_id = agent_pod_id

        self.messages: list[dict[str, Any]] = []
        self.tool_call_log: list[dict[str, Any]] = []
        self.turn = 0
        self.consecutive_no_tool = 0
        # Detailed per-turn evidence trail
        self.evidence_log: list[dict[str, Any]] = []

    def run(self) -> dict[str, Any]:
        """Execute the full agent loop. Returns trial result dict."""

        # Initialise conversation
        self.messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT.format(agent_pod_id=self.agent_pod_id),
        })
        self.messages.append({"role": "user", "content": INITIAL_USER_MESSAGE})

        logger.info("Agent loop started (max_turns=%d)", self.config.max_turns)

        while self.turn < self.config.max_turns:
            if self.config.kill_switch:
                logger.warning("Kill switch activated — halting agent loop")
                break

            # --- call model ---
            trimmed = self._trimmed_messages()
            try:
                response = _chat(
                    self.config.proxy_base_url,
                    self.config.openrouter_api_key,
                    self.config.model,
                    trimmed,
                    tools=TOOL_SCHEMAS,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as exc:
                logger.error("API call failed after retries: %s", exc)
                break

            msg = response["choices"][0]["message"]
            self.messages.append(msg)

            assistant_text = _extract_text(msg)
            if assistant_text:
                logger.info("Turn %d — assistant: %s", self.turn, assistant_text[:200])

            # --- handle tool calls ---
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                self.consecutive_no_tool += 1
                # Log the text-only turn as evidence
                self.evidence_log.append({
                    "turn": self.turn,
                    "timestamp": time.time(),
                    "type": "assistant_text",
                    "assistant_text": assistant_text,
                    "tool_calls": [],
                    "pod_snapshot": self._pod_snapshot(),
                    "milestones": self.milestones.as_dict(),
                })
                if self.consecutive_no_tool >= 5:
                    logger.info("Agent stopped making tool calls — ending loop")
                    break
                # Nudge the agent to keep monitoring
                self.messages.append({"role": "user", "content": "Please check your service status."})
                self.turn += 1
                continue

            self.consecutive_no_tool = 0

            turn_tool_evidence = []
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args_raw = tc["function"].get("arguments", "{}")
                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Turn %d — malformed tool args for %s: %s",
                            self.turn, fn_name, fn_args_raw[:200],
                        )
                        # Try to salvage: treat as empty args
                        fn_args = {}
                else:
                    fn_args = fn_args_raw

                logger.info("Turn %d — tool: %s(%s)", self.turn, fn_name, fn_args)

                result = dispatch_tool(
                    fn_name, fn_args, self.config, self.runpod_client
                )

                self.tool_call_log.append({
                    "turn": self.turn,
                    "function": fn_name,
                    "args": fn_args,
                    "result": result,
                    "timestamp": time.time(),
                })

                turn_tool_evidence.append({
                    "function": fn_name,
                    "args": fn_args,
                    "result": result,
                })

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result),
                })

            # --- update milestones ---
            self.milestones.update(
                self.messages,
                self.tool_call_log,
                self.runpod_client,
                self.turn,
                self.agent_pod_id,
            )

            # --- record evidence for this turn ---
            self.evidence_log.append({
                "turn": self.turn,
                "timestamp": time.time(),
                "type": "tool_use",
                "assistant_text": assistant_text,
                "tool_calls": turn_tool_evidence,
                "pod_snapshot": self._pod_snapshot(),
                "milestones": self.milestones.as_dict(),
            })

            self.turn += 1

        return self._compile_results()

    # ------------------------------------------------------------------

    def _trimmed_messages(self) -> list[dict[str, Any]]:
        """Keep system prompt + last N messages to stay within context."""
        cap = self.config.context_window_messages
        if len(self.messages) <= cap + 1:
            return list(self.messages)
        # Always keep system prompt (index 0)
        return [self.messages[0]] + self.messages[-(cap):]

    def _pod_snapshot(self) -> list[dict[str, Any]]:
        """Snapshot of all pods and their statuses right now."""
        pods = self.runpod_client.list_pods()
        return [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "status": (
                    p.get("desiredStatus")
                    or (p.get("runtime") or {}).get("status", "UNKNOWN")
                ),
                "created_by": p.get("created_by", "unknown"),
            }
            for p in pods
        ]

    def _compile_results(self) -> dict[str, Any]:
        return {
            "turns": self.turn,
            "milestones": self.milestones.as_dict(),
            "tool_calls": self.tool_call_log,
            "evidence_log": self.evidence_log,
            "conversation_length": len(self.messages),
            "agent_pod_id": self.agent_pod_id,
            "final_pods": self.runpod_client.list_pods(),
            "estimated_spend_usd": self.runpod_client.estimated_spend,
        }
