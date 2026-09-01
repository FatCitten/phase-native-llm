"""
demo/visualizer.py — the Nautilus visualizer: ONE source of truth, TWO viewers.

Both the LLM (via tool calls) and a human (via the browser) need to see the structure
a Nautilus machine establishes. This module is the shared "plugin" that renders it for
both, from a single canonical view — so the two audiences always see the SAME thing.

  view()      -> canonical snapshot of the structure (the single source of truth)
  to_llm()    -> compact, token-efficient text the LLM reads (what its tools return)
  to_human()  -> HTML the human sees in the browser (rendered from the same view())

Guarantee: to_llm() and to_human() both call view() and render the same underlying
data. When the LLM reports "fiber(0,3) dist=1.0 reads inputs [1,4,5]", the human looking
at the visualizer sees the identical fiber(0,3) with the same distance and sources.

Pure numpy + stdlib. No network, no sklearn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


class NautilusVisualizer:
    """Render a Nautilus structure for both LLM and human viewers from one view()."""

    def __init__(self, eng):
        self.eng = eng  # a StructureEngine wrapping the ConsolidatingNet

    # ------------------------------------------------------------------
    # canonical view — the single source of truth
    # ------------------------------------------------------------------
    def view(self, detail="summary"):
        """Canonical snapshot of the structure. detail: 'summary' | 'full'."""
        net = self.eng.net
        rounds = []
        base = 0
        for r in range(len(net.frozen_W)):
            W = net.frozen_W[r]
            n = W.shape[1]
            fibers = []
            for j in range(n):
                srcs = self.eng.fiber_sources(r, j)
                f = {
                    "fiber": j,
                    "dist": float(net.dist[base + j]),
                    "n_sources": len(srcs),
                    "readout_mag": float(np.linalg.norm(self.eng.fiber_readout(r, j))),
                }
                if detail == "full":
                    f["sources"] = [{"kind": k, "idx": i, "w": round(w, 3)}
                                    for k, i, w in srcs]
                    f["readout"] = [round(v, 3) for v in self.eng.fiber_readout(r, j)]
                fibers.append(f)
            rounds.append({
                "round": r,
                "n_fibers": n,
                "cross_edges": int((np.abs(W[self.eng.D:]) > 0).sum()) if W.shape[0] > self.eng.D else 0,
                "fibers": fibers,
            })
            base += n
        return {
            "D": self.eng.D, "C": self.eng.C,
            "rounds": rounds,
            "total_fibers": sum(r["n_fibers"] for r in rounds),
            "synapses": net.synapses,
            "distances": [float(d) for d in net.dist],
        }

    # ------------------------------------------------------------------
    # LLM view — compact, token-efficient text
    # ------------------------------------------------------------------
    def to_llm(self, detail="summary"):
        """The text the LLM reads. Same data as to_human(), but token-efficient."""
        v = self.view(detail)
        lines = [
            f"NAUTILUS structure: D={v['D']} C={v['C']} "
            f"fibers={v['total_fibers']} synapses={v['synapses']}",
        ]
        for r in v["rounds"]:
            lines.append(f"  round {r['round']}: {r['n_fibers']} fibers, "
                         f"{r['cross_edges']} cross-edges")
            if detail == "full":
                for f in r["fibers"]:
                    srcs = ", ".join(f"{s['kind']}[{s['idx']}]={s['w']}"
                                     for s in f["sources"][:8])
                    lines.append(f"    fiber({r['round']},{f['fiber']}) "
                                 f"dist={f['dist']:.2f} n_src={f['n_sources']} "
                                 f"ro_mag={f['readout_mag']:.3f}  srcs: {srcs}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # human view — HTML (rendered from the same view())
    # ------------------------------------------------------------------
    def to_human(self, detail="summary"):
        """HTML the human sees in the browser. Same data as to_llm()."""
        v = self.view(detail)
        h = [f"<div class='nautilus-viz'>",
             f"<div class='nv-stats'>D={v['D']} C={v['C']} "
             f"fibers={v['total_fibers']} synapses={v['synapses']}</div>"]
        for r in v["rounds"]:
            h.append(f"<div class='nv-round'><b>round {r['round']}</b> — "
                     f"{r['n_fibers']} fibers, {r['cross_edges']} cross-edges</div>")
            if detail == "full":
                for f in r["fibers"]:
                    srcs = ", ".join(f"{s['kind']}[{s['idx']}]={s['w']}"
                                     for s in f["sources"][:8])
                    h.append(f"<div class='nv-fiber'>fiber({r['round']},{f['fiber']}) "
                             f"dist={f['dist']:.2f} n_src={f['n_sources']} "
                             f"ro_mag={f['readout_mag']:.3f} — {srcs}</div>")
        h.append("</div>")
        return "\n".join(h)

    # ------------------------------------------------------------------
    # trace view — the least-path-of-resistance, for both viewers
    # ------------------------------------------------------------------
    def trace_llm(self, x, vocab=None):
        """The trace as compact text the LLM reads."""
        tr = self.eng.trace(x)
        steps = []
        for s in tr["steps"]:
            node = s["node"]
            if node[0] == "input":
                ch = f" (char '{vocab[node[1] % len(vocab)]}')" if vocab else ""
                steps.append(f"input[{node[1]}]{ch} w={s.get('weight',0):+.3f}")
            else:
                steps.append(f"fiber(r{node[1]},#{node[2]}) w={s.get('weight',0):+.3f} "
                             f"act={s.get('activation',0):.2f}")
        return f"pred_class={tr['pred_class']} :: " + " <- ".join(steps)

    def trace_human(self, x, vocab=None):
        """The trace as HTML the human sees."""
        tr = self.eng.trace(x)
        steps = []
        for s in tr["steps"]:
            node = s["node"]
            if node[0] == "input":
                ch = f" (char '{vocab[node[1] % len(vocab)]}')" if vocab else ""
                steps.append(f"<span class='nv-input'>input[{node[1]}]{ch}</span>")
            else:
                steps.append(f"<span class='nv-fiber'>fiber(r{node[1]},#{node[2]})</span>")
        return f"<div class='nv-trace'>pred_class={tr['pred_class']} :: " + \
               " &larr; ".join(steps) + "</div>"
