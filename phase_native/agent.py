"""
The real driver: an LLM that guides its own consolidation and calls the phase memory.

`run_agent` runs a manual Claude tool-use loop. The model is told the task (answer
reach(s,k) on a hidden graph, traversable only via the costly `step` tool) and the
consolidation policy (recall jumps first; verify low-confidence recalls; compute misses by
binary lifting; solidify each established jump with memory_write). The phase memory is the
model's O(1) "self-agent" — it decides what to store and relies on it across problems.

This is the solution the project is about. It requires Anthropic API credentials; where they
are absent (e.g. this sandbox — see MEMORY-AGENT.md) the identical tool surface is exercised
offline by driver.ScriptedDriver, and this file is ready to run wherever creds exist.

SDK usage follows the bundled `claude-api` skill: zero-arg client (credentials resolved from
the environment), adaptive thinking, `usage` captured as the compute-to-answer metric.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .domain import RelationGraph
from .tools import MemoryToolExecutor

SYSTEM_PROMPT = """\
You answer reach(s, k) = "apply the hidden successor function f to node s, k times".

RULES
- The graph is hidden. The ONLY way to learn an edge is the `step` tool: step(node) -> f(node).
- `step` is the costly operation. Your goal is a CORRECT answer with as FEW step calls as possible.

METHOD (binary lifting + your phase memory)
- Write k in binary. reach(s,k) composes jumps f^(2^j) over the set bits j (order does not matter).
- A jump nugget has cue exactly "jump(node=N,level=J)" meaning f^(2^J)(N).
- jump at level 0 is one `step`. jump at level J = jump(jump(N,J-1), J-1).
- BEFORE computing any jump, call memory_recall(cue). Its confidence is in [0,1]:
    * confidence >= 0.45 and found=true  -> trust it, use the conclusion (NO step calls).
    * otherwise                          -> it is a miss or corrupted; DERIVE it yourself.
- After you DERIVE a jump, call memory_write(cue, conclusion) to solidify it for next time.
- memory_recall does not get slower as you store more — rely on it freely.

When you have the final node, call final_answer(node). Think briefly; prefer tool calls over prose.
"""

FINAL_ANSWER_TOOL = {
    "name": "final_answer",
    "description": "Submit the final reached node and end the episode.",
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"node": {"type": "integer"}},
        "required": ["node"],
    },
}


def credentials_available() -> bool:
    """True if the anthropic SDK has a resolved credential (does not make a call).

    The client constructor does NOT raise on missing creds — it fails only at request
    time — so inspect the resolved api_key / auth_token instead.
    """
    try:
        import anthropic

        c = anthropic.Anthropic()
        return bool(getattr(c, "api_key", None) or getattr(c, "auth_token", None))
    except Exception:
        return False


@dataclass
class EpisodeResult:
    answer: int | None
    steps: int
    recalls: int
    hits: int
    writes: int
    input_tokens: int
    output_tokens: int
    iters: int


def run_agent(
    s: int,
    k: int,
    executor: MemoryToolExecutor,
    model: str = "claude-opus-5",
    max_iters: int = 40,
    max_tokens: int = 8192,
    client=None,
) -> EpisodeResult:
    """Drive one reach(s,k) episode with the LLM. Returns answer + compute usage."""
    import anthropic

    client = client or anthropic.Anthropic()
    tools = executor.schemas + [FINAL_ANSWER_TOOL]

    steps0 = executor.graph.steps_taken
    r0, h0, w0 = executor.recalls, executor.hits, executor.writes
    in_tok = out_tok = 0
    answer = None

    messages = [{"role": "user", "content": f"Compute reach(s={s}, k={k})."}]
    for it in range(1, max_iters + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        in_tok += resp.usage.input_tokens
        out_tok += resp.usage.output_tokens
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        stop = False
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "final_answer":
                answer = int(block.input["node"])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "ok"}
                )
                stop = True
            else:
                out = executor.dispatch(block.name, dict(block.input))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out),
                    }
                )
        messages.append({"role": "user", "content": tool_results})
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
