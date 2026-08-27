"""
Drivers: the thing that decides WHEN to recall, compute, and solidify.

The real driver is the LLM (agent.py). `ScriptedDriver` is offline test scaffolding — a
deterministic stand-in that follows the same consolidation policy through the same tools,
so the loop plumbing and the compute-offload effect are provable without API access. It is
explicitly NOT the solution ("blindness is the enemy"); it exists to test the mechanics and
to give the live agent a baseline to beat.

Policy (shared with the agent's system prompt):
  1. recall-first: query memory before doing work;
  2. compute-on-miss: derive via binary lifting, taking base `step`s only when needed;
  3. solidify: write each established jump;
  4. verify: a low-confidence recall counts as a miss and is re-derived (sightedness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .domain import bits, jump_cue
from .tools import MemoryToolExecutor


class Driver(Protocol):
    def solve(self, s: int, k: int) -> int: ...


@dataclass
class ScriptedDriver:
    ex: MemoryToolExecutor

    def _recall(self, cue: str):
        r = self.ex.dispatch("memory_recall", {"cue": cue})
        return int(r["conclusion"]) if r.get("found") else None

    def _write(self, cue: str, val: int) -> None:
        self.ex.dispatch("memory_write", {"cue": cue, "conclusion": str(val)})

    def _step(self, node: int) -> int:
        return int(self.ex.dispatch("step", {"node": node})["next"])

    def _jump(self, node: int, j: int) -> int:
        """f^{2^j}(node): recall if solidified, else derive from two (j-1) jumps."""
        cue = jump_cue(node, j)
        hit = self._recall(cue)
        if hit is not None:
            return hit
        if j == 0:
            t = self._step(node)
        else:
            mid = self._jump(node, j - 1)
            t = self._jump(mid, j - 1)
        self._write(cue, t)
        return t

    def solve(self, s: int, k: int) -> int:
        node = s
        for j in bits(k):  # powers of f commute, so any order is fine
            node = self._jump(node, j)
        return node
