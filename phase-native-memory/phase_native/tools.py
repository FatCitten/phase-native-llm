"""
The memory as a set of tools an LLM drives itself.

`MemoryToolExecutor` exposes Anthropic tool schemas (`.schemas`) and a `.dispatch()` that
runs a tool call against a PhaseNuggetMemory + RelationGraph. The same executor backs both
the live ClaudeDriver (agent.py) and the offline ScriptedDriver (driver.py), so the O(1)
memory the LLM relies on and the compute it saves are measured identically either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain import RelationGraph
from .memory import PhaseNuggetMemory

MEMORY_TOOL_SCHEMAS = [
    {
        "name": "memory_recall",
        "description": (
            "O(1) associative seek into your phase memory. Returns the conclusion you "
            "previously solidified for this cue, with a confidence in [0,1]. High "
            "confidence (>=0.45) is a reliable hit; low confidence means the cue was "
            "never stored OR the recall is corrupted by interference — re-derive it. "
            "Retrieval time does NOT grow with how much you have stored."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"cue": {"type": "string", "description": "Exact cue string."}},
            "required": ["cue"],
        },
    },
    {
        "name": "memory_write",
        "description": (
            "Solidify an established result into your phase memory as a low-byte nugget: "
            "memory[cue] = conclusion. Use for sub-results you may need again. Writing an "
            "existing cue replaces it."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "cue": {"type": "string"},
                "conclusion": {"type": "string", "description": "The established value."},
            },
            "required": ["cue", "conclusion"],
        },
    },
    {
        "name": "step",
        "description": (
            "Traverse ONE hidden graph edge: returns the successor of `node`. This is the "
            "only way to learn edges, and it is the costly operation — minimize step calls "
            "by recalling jumps you have already solidified."
        ),
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"node": {"type": "integer"}},
            "required": ["node"],
        },
    },
    {
        "name": "memory_stats",
        "description": "Report memory occupancy and size (for introspection).",
        "input_schema": {"type": "object", "additionalProperties": False, "properties": {}},
    },
]


@dataclass
class MemoryToolExecutor:
    graph: RelationGraph
    memory: PhaseNuggetMemory | None = None  # None => memory-disabled control
    recalls: int = field(default=0, init=False)
    hits: int = field(default=0, init=False)
    writes: int = field(default=0, init=False)

    @property
    def schemas(self) -> list[dict]:
        # hide memory tools when running the memory-disabled control
        if self.memory is None:
            return [s for s in MEMORY_TOOL_SCHEMAS if s["name"] == "step"]
        return MEMORY_TOOL_SCHEMAS

    def dispatch(self, name: str, tool_input: dict) -> dict:
        if name == "step":
            return {"next": self.graph.step(int(tool_input["node"]))}
        if name == "memory_recall":
            if self.memory is None:
                return {"found": False, "confidence": 0.0}
            self.recalls += 1
            r = self.memory.recall(str(tool_input["cue"]))
            if r.hit:
                self.hits += 1
            return {
                "found": r.hit,
                "conclusion": r.payload if r.hit else None,
                "confidence": round(r.confidence, 3),
            }
        if name == "memory_write":
            if self.memory is None:
                return {"ok": False}
            self.writes += 1
            self.memory.write(str(tool_input["cue"]), str(tool_input["conclusion"]))
            return {"ok": True}
        if name == "memory_stats":
            return self.memory.stats() if self.memory else {"memory": "disabled"}
        return {"error": f"unknown tool {name}"}
