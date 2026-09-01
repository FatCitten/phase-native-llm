"""
Ollama driver: run the phase-memory agent on an open model (Ollama Cloud or local).

The phase memory is provider-agnostic — any tool-capable LLM can be the sighted driver. This
mirrors phase_native/agent.py (the Claude path) against the OpenAI-compatible endpoint that
Ollama exposes (`{host}/v1/chat/completions`), so the SAME memory tools, policy, and metrics
apply. Works with Ollama Cloud (host https://ollama.com + API key) or a local server
(host http://localhost:11434, no key).

Zero extra dependencies: a tiny stdlib (urllib) HTTP client. Tool-calling uses the OpenAI
function-calling shape; the memory tool schemas are converted from the Anthropic shape.

Note: this sandbox's egress proxy blocks ollama.com, so this runs on YOUR machine. The loop
plumbing is covered here by a mock-client test (tests/test_phase_native.py), same as the
Claude path.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .agent import FINAL_ANSWER_TOOL, SYSTEM_PROMPT, EpisodeResult
from .tools import MemoryToolExecutor


def to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Anthropic {name, description, input_schema} -> OpenAI {type:function, function:{...}}."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for s in schemas
    ]


class OllamaClient:
    """Minimal OpenAI-compatible chat client (stdlib only). Honors HTTPS_PROXY env."""

    def __init__(self, host: str | None = None, api_key: str | None = None, timeout: float = 180):
        self.host = (host or os.environ.get("OLLAMA_HOST", "https://ollama.com")).rstrip("/")
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")
        self.timeout = timeout

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> dict:
        body = json.dumps(
            {"model": model, "messages": messages, "tools": tools, "stream": False}
        ).encode()
        req = urllib.request.Request(f"{self.host}/v1/chat/completions", data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        last = None
        for attempt in range(3):  # retry transient timeouts / 5xx
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(2 * (attempt + 1))
        if last is not None:
            raise last
        return json.loads(b"{}")


def _args(tool_call: dict) -> dict:
    raw = tool_call.get("function", {}).get("arguments", {})
    if isinstance(raw, str):
        return json.loads(raw) if raw.strip() else {}
    return raw or {}


def _extract_int(text: str | None):
    """Fallback: pull a final integer out of prose if a weak model skips final_answer."""
    if not text:
        return None
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def run_agent_ollama(
    s: int,
    k: int,
    executor: MemoryToolExecutor,
    model: str = "gpt-oss:120b",
    client: OllamaClient | None = None,
    max_iters: int = 40,
) -> EpisodeResult:
    """One reach(s,k) episode driven by an Ollama model. Same EpisodeResult as the Claude path."""
    client = client or OllamaClient()
    tools = to_openai_tools(executor.schemas + [FINAL_ANSWER_TOOL])

    steps0 = executor.graph.steps_taken
    r0, h0, w0 = executor.recalls, executor.hits, executor.writes
    in_tok = out_tok = 0
    answer = None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Compute reach(s={s}, k={k})."},
    ]
    for it in range(1, max_iters + 1):
        resp = client.chat(model, messages, tools)
        usage = resp.get("usage", {}) or {}
        in_tok += usage.get("prompt_tokens", 0)
        out_tok += usage.get("completion_tokens", 0)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        assistant = {"role": "assistant", "content": msg.get("content") or ""}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)

        if not tool_calls:  # model answered in prose (or finished) — take a fallback int
            answer = _extract_int(msg.get("content"))
            break

        stop = False
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            if name == "final_answer":
                answer = int(_args(tc)["node"])
                out = "ok"
                stop = True
            else:
                out = json.dumps(executor.dispatch(name, _args(tc)))
            messages.append({"role": "tool", "tool_call_id": tc.get("id", name), "content": out})
        if stop:
            break

    return EpisodeResult(
        answer=answer,
        steps=executor.graph.steps_taken - steps0,
        recalls=executor.recalls - r0,
        hits=executor.hits - h0,
        writes=executor.writes - w0,
        input_tokens=in_tok,
        output_tokens=out_tok,
        iters=it,
    )
