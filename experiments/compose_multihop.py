"""
Compositional multi-hop recall: chain atomic nuggets into answers never stored.

Store only ATOMIC one-step edges (learned once). Then answer arbitrary reach(s,k) by iterated
O(1) recall — no new writes, no new steps. Two results, saved to results/:

A. COMPOSITION WORKS, WITH AN HONEST HORIZON.
   Pure linear composition (chained O(1) recalls, 0 steps, 0 writes) is 100% accurate to deep
   chains when the memory is lightly loaded (dim >> nuggets); heavier load lowers per-hop
   fidelity so deep chains break (~p^depth). The horizon is set by load, and it is measured.

B. COMPUTE IS BOUNDED BY THE WORLD, NOT THE WORKLOAD.
   With memory, `reach(s,k)` learns each atomic edge at most once — total costly `step`s plateau
   at <= n_nodes and then every future multi-hop query is FREE (pure composition), regardless of
   how many queries or how many hops. Without memory, compute grows without bound (k per query).

Uses a bijective (permutation) graph so composition-depth curves read cleanly.
Run: python experiments/compose_multihop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_native.compose import compose_reach, edge_cue, recall_chain
from phase_native.domain import RelationGraph
from phase_native.memory import PhaseNuggetMemory

RESULTS = Path("results")
REPS = 2048  # dim = 8192


def warm_atomic(n_nodes, seed=7):
    """A memory holding every atomic edge of a permutation graph; returns (mem, graph)."""
    g = RelationGraph(n_nodes=n_nodes, seed=seed, bijective=True)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=REPS)
    for n in range(n_nodes):
        mem.write(edge_cue(n), g.step(n))
    return mem, g


def depth_horizon(loads, depths, trials=40):
    rows = {}
    for n_nodes in loads:
        mem, g = warm_atomic(n_nodes)
        rng = np.random.default_rng(0)
        accs = []
        for d in depths:
            ok = 0
            for _ in range(trials):
                s = int(rng.integers(0, n_nodes))
                res = recall_chain(mem, s, d)  # 0 steps, 0 writes
                ok += res.ok and res.node == g.truth_pow(s, d)
            accs.append(ok / trials)
        rows[n_nodes] = {"acc": accs, "dim": mem.dim}
    return rows


def compute_vs_queries(n_nodes=160, n_queries=150, max_k=200, seed=3):
    """Cumulative costly steps over a query stream: with memory (plateaus) vs without (grows)."""
    g = RelationGraph(n_nodes=n_nodes, seed=seed, bijective=True)
    mem = PhaseNuggetMemory(moduli=(8, 9, 5, 7), reps=REPS)
    rng = np.random.default_rng(seed)
    with_mem, no_mem, correct = [], [], 0
    for _ in range(n_queries):
        s = int(rng.integers(0, n_nodes))
        k = int(rng.integers(1, max_k + 1))
        g.reset_counter()
        res = compose_reach(mem, g, s, k)  # composes; learns each atomic edge at most once
        with_mem.append(g.steps_taken)
        no_mem.append(k)  # brute force would take k steps every time
        correct += res.node == g.truth_pow(s, k)
    return np.cumsum(with_mem), np.cumsum(no_mem), correct / n_queries, mem.stats()["n_nuggets"]


def main():
    RESULTS.mkdir(exist_ok=True)

    loads = [64, 160, 320]
    depths = [1, 5, 10, 20, 40, 80, 160]
    print("=== A: composition depth horizon vs load (dim=8192, permutation graph) ===")
    A = depth_horizon(loads, depths)
    for n_nodes, r in A.items():
        print(f"  load={n_nodes:3d} (dim/N={r['dim']//n_nodes:3d})  acc@depth " +
              " ".join(f"{d}:{a:.2f}" for d, a in zip(depths, r["acc"])))
    print("  (every query above used 0 steps and 0 writes — pure composition)")

    print("\n=== B: cumulative compute over a multi-hop query stream ===")
    wm, nm, acc, nug = compute_vs_queries()
    print(f"  with memory:  {int(wm[-1])} total steps (plateaus at <= n_nodes), acc={acc:.3f}, "
          f"{nug} atomic nuggets")
    print(f"  no memory:    {int(nm[-1])} total steps (grows with every query)")
    print(f"  => {int(nm[-1])/max(int(wm[-1]),1):.0f}x less compute; extra queries are free")

    json.dump({"depths": depths, "horizon": {str(k): v for k, v in A.items()},
               "with_mem_cumsteps": int(wm[-1]), "no_mem_cumsteps": int(nm[-1]),
               "stream_acc": acc, "atomic_nuggets": nug},
              open(RESULTS / "compose_multihop.json", "w"), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n_nodes, r in A.items():
        ax1.plot(depths, r["acc"], "o-", label=f"{n_nodes} nuggets (dim/N={r['dim']//n_nodes})")
    ax1.set_xlabel("composition depth (chained O(1) recalls)")
    ax1.set_ylabel("multi-hop accuracy")
    ax1.set_title("Compose atomic facts into deep answers — free, honest horizon")
    ax1.set_ylim(-0.03, 1.03)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    q = np.arange(1, len(wm) + 1)
    ax2.plot(q, nm, label="no memory (re-derive: k steps/query)")
    ax2.plot(q, wm, label=f"with memory (learn each edge once -> plateau)")
    ax2.set_xlabel("multi-hop queries answered")
    ax2.set_ylabel("cumulative costly steps")
    ax2.set_title("Compute bounded by the world, not the workload")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "compose_multihop.png", dpi=140)
    print("\nSaved results/compose_multihop.{png,json}")


if __name__ == "__main__":
    main()
