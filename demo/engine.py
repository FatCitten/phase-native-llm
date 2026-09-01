"""
demo/engine.py — the NAUTILUS engine: observe, trace, edit, and PERSIST a structure.

NAUTILUS: a consolidation network whose internal structure is legible, editable, and
durable. The architecture's whole value is that its structure is LEGIBLE: every fiber
has a distance-from-axiom, a set of incoming connections (inputs + established fibers),
and a readout. This engine makes that legibility — and now durability — first-class:

  OBSERVE   — a growth log of every consolidation round: which fibers were born, what
              they read, their distance-from-axiom, whether they survived pruning, and
              their readout contribution. Watch structure being ESTABLISHED.
  TRACE     — for any input, walk the least-path-of-resistance from the output pulse
              back through fibers to inputs, showing the actual weights and activations
              along the path (not just node names).
  EDIT      — surgically modify the structure (prune / zero / rewire / add fibers, set
              readouts) and re-evaluate to measure the effect.
  PERSIST   — SAVE the whole structure to disk, LOAD it back, READ it, WRITE it: the
              structure is bytes on disk as much as it is a live object. Because it can
              be edited AND saved, it is a new kind of machine — one that grows outward
              in waves, freezes axioms, is engineered by LLMs, and survives reboots.

The engine wraps a ConsolidatingNet and operates directly on its frozen arrays
(frozen_W / frozen_V / frozen_b / dist). Edits and saves are applied to the same arrays.

Pure numpy, CPU. No network, no sklearn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits


class StructureEngine:
    """Observe, trace, edit, and persist the structure a ConsolidatingNet establishes."""

    def __init__(self, net):
        self.net = net
        self.D = net.D
        self.C = net.C
        self._growth_log = []

    # ------------------------------------------------------------------
    # OBSERVE
    # ------------------------------------------------------------------
    def fiber_sources(self, round_idx, fiber_idx):
        W = self.net.frozen_W[round_idx]
        col = W[:, fiber_idx]
        nz = np.nonzero(col)[0]
        out = []
        for i in nz:
            if i < self.D:
                out.append(("input", int(i), float(col[i])))
            else:
                out.append(("fiber", int(i - self.D), float(col[i])))
        return out

    def fiber_readout(self, round_idx, fiber_idx):
        V = self.net.frozen_V[round_idx]
        return V[fiber_idx].copy()

    def fiber_profile(self, round_idx, fiber_idx):
        base = sum(len(w[0]) for w in self.net.frozen_W[:round_idx])
        return {
            "round": round_idx,
            "fiber": fiber_idx,
            "distance": float(self.net.dist[base + fiber_idx]) if base + fiber_idx < len(self.net.dist) else None,
            "bias": float(self.net.frozen_b[round_idx][fiber_idx]),
            "sources": self.fiber_sources(round_idx, fiber_idx),
            "readout": self.fiber_readout(round_idx, fiber_idx).tolist(),
            "n_sources": len(self.fiber_sources(round_idx, fiber_idx)),
        }

    def observe_growth(self):
        log = []
        for r in range(len(self.net.frozen_W)):
            W = self.net.frozen_W[r]
            n_fibers = W.shape[1]
            base = sum(len(w[0]) for w in self.net.frozen_W[:r])
            fibers = []
            for j in range(n_fibers):
                srcs = self.fiber_sources(r, j)
                n_input = sum(1 for k, _, _ in srcs if k == "input")
                n_fiber = sum(1 for k, _, _ in srcs if k == "fiber")
                fibers.append({
                    "fiber": j,
                    "distance": float(self.net.dist[base + j]),
                    "n_input_sources": n_input,
                    "n_fiber_sources": n_fiber,
                    "readout_mag": float(np.linalg.norm(self.fiber_readout(r, j))),
                })
            log.append({
                "round": r,
                "n_fibers": n_fibers,
                "fibers": fibers,
                "cross_edges": int((np.abs(W[self.D:]) > 0).sum()) if W.shape[0] > self.D else 0,
            })
        self._growth_log = log
        return log

    def structure_summary(self):
        log = self.observe_growth()
        return {
            "rounds": len(log),
            "total_fibers": sum(r["n_fibers"] for r in log),
            "distances": [float(d) for d in self.net.dist],
            "cross_edges": sum(r["cross_edges"] for r in log),
            "synapses": self.net.synapses,
        }

    # ------------------------------------------------------------------
    # TRACE
    # ------------------------------------------------------------------
    def trace(self, x, pred_class=None):
        net = self.net
        F = np.zeros((1, 0))
        acts = []
        for Wr, br in zip(net.frozen_W, net.frozen_b):
            A = np.maximum(np.concatenate([x.reshape(1, -1), F], 1) @ Wr + br, 0)
            acts.append(A[0])
            F = np.concatenate([F, A], 1)
        logits = forward_logits(net, x.reshape(1, -1))[0]
        if pred_class is None:
            pred_class = int(logits.argmax())

        last = len(net.frozen_W) - 1
        V = net.frozen_V[last]
        Alast = acts[last]
        kf = int(np.argmax(Alast * V[:, pred_class]))
        steps = [{"node": ("fiber", last, kf), "activation": float(Alast[kf]),
                  "readout_to_pred": float(V[kf, pred_class])}]

        Wlast = net.frozen_W[last]
        col = Wlast[:, kf]
        F_prev = np.concatenate([x, np.concatenate(acts[:-1])]) if len(acts) > 1 else x
        src = np.abs(F_prev)
        s = int(np.argmax(np.abs(col) * src))
        for _ in range(20):
            if s < self.D:
                steps.append({"node": ("input", s), "weight": float(col[s]),
                              "activation": float(x[s])})
                break
            j = s - self.D
            r_owner = self._fiber_round(j)
            steps.append({"node": ("fiber", r_owner, j - sum(len(w[0]) for w in net.frozen_W[:r_owner])),
                          "weight": float(col[s]), "activation": float(F_prev[s])})
            Wj = net.frozen_W[r_owner]
            jj = j - sum(len(w[0]) for w in net.frozen_W[:r_owner])
            colj = Wj[:, jj]
            F_prev_j = np.concatenate([x, np.concatenate(acts[:r_owner])]) if r_owner > 0 else x
            srcj = np.abs(F_prev_j)
            s = int(np.argmax(np.abs(colj) * srcj))
        return {"pred_class": pred_class, "steps": steps}

    def _fiber_round(self, global_fiber_idx):
        acc = 0
        for r, W in enumerate(self.net.frozen_W):
            if global_fiber_idx < acc + W.shape[1]:
                return r
            acc += W.shape[1]
        return len(self.net.frozen_W) - 1

    # ------------------------------------------------------------------
    # EDIT
    # ------------------------------------------------------------------
    def _fiber_referenced_by_later(self, round_idx, fiber_idx):
        global_idx = sum(len(w[0]) for w in self.net.frozen_W[:round_idx]) + fiber_idx
        for r in range(round_idx + 1, len(self.net.frozen_W)):
            W = self.net.frozen_W[r]
            pos = self.D + global_idx
            if pos < W.shape[0] and np.any(np.abs(W[pos]) > 0):
                return True
        return False

    def prune_fiber(self, round_idx, fiber_idx):
        if self._fiber_referenced_by_later(round_idx, fiber_idx):
            raise ValueError(
                f"fiber ({round_idx},{fiber_idx}) is read by a later round; "
                f"rewire its dependents first, or use zero_fiber to disable it safely")
        net = self.net
        W = net.frozen_W[round_idx]
        V = net.frozen_V[round_idx]
        b = net.frozen_b[round_idx]
        net.frozen_W[round_idx] = np.delete(W, fiber_idx, axis=1)
        net.frozen_V[round_idx] = np.delete(V, fiber_idx, axis=0)
        net.frozen_b[round_idx] = np.delete(b, fiber_idx)
        base = sum(len(w[0]) for w in net.frozen_W[:round_idx])
        del net.dist[base + fiber_idx]
        return self

    def zero_fiber(self, round_idx, fiber_idx):
        net = self.net
        net.frozen_V[round_idx][fiber_idx] = 0.0
        return self

    def rewire_fiber(self, round_idx, fiber_idx, new_weights):
        net = self.net
        W = net.frozen_W[round_idx]
        n_in = W.shape[0]
        col = np.zeros(n_in)
        for src, w in new_weights.items():
            if 0 <= src < n_in:
                col[src] = w
        W[:, fiber_idx] = col
        return self

    def set_readout(self, round_idx, fiber_idx, class_idx, value):
        net = self.net
        net.frozen_V[round_idx][fiber_idx, class_idx] = value
        return self

    def add_fiber(self, round_idx, sources, readout, bias=0.0):
        net = self.net
        W = net.frozen_W[round_idx]
        n_in = W.shape[0]
        col = np.zeros(n_in)
        for src, w in sources.items():
            if 0 <= src < n_in:
                col[src] = w
        net.frozen_W[round_idx] = np.concatenate([W, col[:, None]], axis=1)
        net.frozen_V[round_idx] = np.concatenate([net.frozen_V[round_idx],
                                                  np.array(readout)[None, :]], axis=0)
        net.frozen_b[round_idx] = np.concatenate([net.frozen_b[round_idx], [bias]])
        base = sum(len(w[0]) for w in net.frozen_W[:round_idx])
        net.dist.insert(base + W.shape[1], 1.0)
        return self

    def freeze(self, round_idx, fiber_idx):
        if not hasattr(self, "_axioms"):
            self._axioms = set()
        self._axioms.add((round_idx, fiber_idx))
        return (round_idx, fiber_idx)

    def is_axiom(self, round_idx, fiber_idx):
        return hasattr(self, "_axioms") and (round_idx, fiber_idx) in self._axioms

    # ------------------------------------------------------------------
    # PERSIST — the machine survives: store / load / read / write
    # ------------------------------------------------------------------
    def save_structure(self, path):
        """WRITE the entire structure to disk as a .json 'machine file'."""
        data = {
            "D": int(self.D), "C": int(self.C),
            "bias": self.net.bias.tolist(),
            "synapses": int(self.net.synapses),
            "dist": [float(d) for d in self.net.dist],
            "rounds": [{"W": w.tolist(), "V": v.tolist(), "b": b.tolist()}
                       for w, v, b in zip(self.net.frozen_W, self.net.frozen_V, self.net.frozen_b)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))
        return path

    # `write` is the imperative alias for save_structure
    write = save_structure

    @classmethod
    def load_structure(cls, path, D=None, C=None):
        """READ a saved structure from disk and rebuild a live StructureEngine around it."""
        data = json.loads(Path(path).read_text())
        D = data.get("D", D)
        C = data.get("C", C)
        net = ConsolidatingNet(D, C, seed=0)
        net.bias = np.array(data["bias"])
        net.dist = list(data["dist"])
        net.frozen_W = [np.array(r["W"]) for r in data["rounds"]]
        net.frozen_V = [np.array(r["V"]) for r in data["rounds"]]
        net.frozen_b = [np.array(r["b"]) for r in data["rounds"]]
        net.synapses = int(data.get("synapses", 0))
        return cls(net)

    # `read` is the imperative alias for load_structure
    @staticmethod
    def read(path, D=None, C=None):
        return StructureEngine.load_structure(path, D, C)

    def evaluate(self, X, y):
        """Accuracy of the (possibly edited) net on (X, y)."""
        return float((forward_logits(self.net, X).argmax(1) == y).mean())

    def forward(self, X):
        return forward_logits(self.net, X)
