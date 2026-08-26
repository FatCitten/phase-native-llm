"""
A recurring-subproblem reasoning domain: reach(s, k) = f^k(s) on a hidden graph.

Each node has one hidden outgoing edge. The ONLY way to traverse an edge is the costly
`step` primitive (the graph is not visible) — so re-deriving a path always costs base
steps unless a result was solidified into memory. Queries share sub-structure: reach(s,k)
decomposes by binary lifting into jumps f^{2^j}, and jump(node, j) = jump(jump(node,j-1),
j-1). Solidifying jump nuggets lets later queries be answered by O(1) recalls instead of
re-stepping — so compute-to-answer falls as the memory fills. This is the measurable
"specialization" signal, and it is exactly the kind of established-substructure reuse the
project is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RelationGraph:
    n_nodes: int = 256
    seed: int = 0
    bijective: bool = False  # True -> successor is a permutation (long cycles, no convergence)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # functional graph (default) or a permutation. The permutation avoids the coincidental
        # "wrong path lands on the right node" matches that a convergent functional graph
        # produces at large depth, so composition-depth curves read cleanly.
        self._next = (
            rng.permutation(self.n_nodes) if self.bijective
            else rng.integers(0, self.n_nodes, size=self.n_nodes)
        )
        self.steps_taken = 0  # global compute counter (base hops actually traversed)

    def step(self, node: int) -> int:
        """Traverse one hidden edge. This is the unit of COMPUTE we count."""
        self.steps_taken += 1
        return int(self._next[node])

    # ground truth (free; used only for scoring, never counted as compute)
    def truth_pow(self, node: int, k: int) -> int:
        for _ in range(k):
            node = int(self._next[node])
        return node

    def reset_counter(self) -> None:
        self.steps_taken = 0


def bits(k: int) -> list[int]:
    """Set bit positions of k (which 2^j jumps compose to k)."""
    return [j for j in range(k.bit_length()) if (k >> j) & 1]


def jump_cue(node: int, j: int) -> str:
    """Stable cue string for the nugget 'f^{2^j}(node)'."""
    return f"jump(node={node},level={j})"


@dataclass
class QueryStream:
    """A stream of reach(s,k) problems that deliberately share jump sub-structure.

    Anchors (the recurring start nodes) are revealed GRADUALLY: the stream begins with a
    few and adds one every `reveal_every` queries. Each freshly revealed anchor needs its
    jumps built (a compute spike) that then amortizes to O(1) recall — so compute-per-query
    declines progressively rather than in a single warm-up cliff, giving a readable
    specialization curve.
    """

    graph: RelationGraph
    max_k: int = 1023
    n_queries: int = 200
    seed: int = 1
    n_anchors: int = 24
    start_anchors: int = 3
    reveal_every: int = 6

    def _anchors(self):
        rng = np.random.default_rng(self.seed + 999)
        return rng.integers(0, self.graph.n_nodes, size=self.n_anchors)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        pool = self._anchors()
        for q in range(self.n_queries):
            active = min(self.n_anchors, self.start_anchors + q // self.reveal_every)
            s = int(pool[rng.integers(0, active)])
            k = int(rng.integers(1, self.max_k + 1))
            yield s, k

    def truth(self, s: int, k: int) -> int:
        return self.graph.truth_pow(s, k)
