"""
Composition: chain established nuggets into multi-hop conclusions the memory was never told.

This is what a *geometric* store gives you over a dict: knowledge you can combine, not just
retrieve. The LLM solidifies only ATOMIC facts (one-step edges); any multi-hop answer is built
by iterated O(1) recall — no new writes, no new compute (`step`s). Each hop is a phase unbind,
so a chain of H hops is H O(1) recalls.

Honest limit: fidelity compounds per hop. If per-hop recall is p, an H-hop linear chain is ~p^H,
so the reliable depth is set by memory load (p rises as dim/N rises). `recall_chain` returns the
minimum confidence along the path so the driver/LLM can re-derive a broken hop. Binary-lifting
composition (`lifted_reach`) needs only ~log2(k) hops, extending the reliable k-range under load
at the cost of storing the composed jumps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .domain import RelationGraph, bits, jump_cue
from .memory import PhaseNuggetMemory


def edge_cue(node: int) -> str:
    """Cue for the atomic one-step fact f(node)."""
    return f"edge({node})"


@dataclass
class ChainResult:
    node: int  # reached node (best effort even if a hop broke)
    ok: bool  # every hop was a confident hit
    min_confidence: float  # weakest hop — the LLM re-derives if this is low
    hops: int  # hops actually taken
    broke_at: int | None = None  # hop index where recall missed, if any


def recall_chain(mem: PhaseNuggetMemory, start: int, hops: int, cue_fn=edge_cue) -> ChainResult:
    """Compose `hops` atomic nuggets by iterated O(1) recall. No writes, no compute."""
    node, min_conf = start, 1.0
    for h in range(hops):
        r = mem.recall(cue_fn(node))
        if not r.hit:
            return ChainResult(node, False, min_conf, h, broke_at=h)
        min_conf = min(min_conf, r.confidence)
        node = int(r.payload)
    return ChainResult(node, True, min_conf, hops)


def learn_atomic(mem: PhaseNuggetMemory, graph: RelationGraph, node: int) -> int:
    """Return f(node), taking (and solidifying) the atomic edge only if not already known."""
    r = mem.recall(edge_cue(node))
    if r.hit:
        return int(r.payload)
    nxt = graph.step(node)  # the one costly op — paid at most once per node, ever
    mem.write(edge_cue(node), nxt)
    return nxt


def compose_reach(mem: PhaseNuggetMemory, graph: RelationGraph, s: int, k: int) -> ChainResult:
    """Answer reach(s,k) by linear composition, learning atomic edges lazily on miss."""
    node, min_conf = s, 1.0
    for h in range(k):
        r = mem.recall(edge_cue(node))
        if r.hit:
            min_conf = min(min_conf, r.confidence)
            node = int(r.payload)
        else:
            node = learn_atomic(mem, graph, node)  # first visit: one step
    return ChainResult(node, True, min_conf, k)


def lifted_reach(mem: PhaseNuggetMemory, graph: RelationGraph, s: int, k: int):
    """Binary-lifting composition: ~log2(k) hops instead of k, caching composed jumps.

    Robust to large k under heavier load (fewer hops => less compounded error), at the cost of
    storing the composed jump nuggets. Returns (node, min_confidence).
    """

    def jump(node: int, j: int) -> tuple[int, float]:
        r = mem.recall(jump_cue(node, j))
        if r.hit:
            return int(r.payload), r.confidence
        if j == 0:
            t = learn_atomic(mem, graph, node)
            c = 1.0
        else:
            mid, c1 = jump(node, j - 1)
            t, c2 = jump(mid, j - 1)
            c = min(c1, c2)
        mem.write(jump_cue(node, j), t)
        return t, c

    node, min_conf = s, 1.0
    for j in bits(k):
        node, c = jump(node, j)
        min_conf = min(min_conf, c)
    return node, min_conf
