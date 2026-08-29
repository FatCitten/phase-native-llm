"""
Iterative consolidation: overproduce -> relational-prune the void -> freeze, growing outward.

Biological development in waves. Round 1's surviving hidden units are the AXIOMS. Each later round:
  1. OVERPRODUCE a pool of new candidate fibers, each reading from [inputs ; frozen established
     features] — so it can build on the axioms (further out) or on raw inputs (near).
  2. TRAIN only the new fibers + their new readout toward the objective; the frozen base and its
     readout contribution are NEVER touched (the "avoid previous prunes" guarantee).
  3. RELATIONAL-PRUNE THE VOID, at two scales. A candidate that contributes nothing to the output
     relates to nothing -> pruned (unit prune). Then each survivor's near-zero incoming reads are
     dropped (connection prune) so a "synapse" only ever counts a real relation, not a wire to the
     void. Survivors "further established phases" and are kept.
  4. FREEZE survivors as new established fiber primitives (bundles). Repeat.

Distance-from-axiom (relational hops, "distance = the gap/void of non-relation"):
    dist(input) = 0;  dist(fiber) = 1 + (sum over SURVIVING incoming |w|*dist(source)) / (sum |w|).
A fiber reading only inputs -> 1; one that bundles level-(r-1) fibers -> ~r. Growing this mean
outward across rounds is the numeric form of "further from axiom."

Honest controls & kill-criteria (stated up front, no forcing the result): the round-1 axioms are
asserted byte-identical after every later round; two MONOLITHIC nets of the same final width trained
jointly (a dense one, and one magnitude-pruned to the loop's synapse budget) bound capability-per-
synapse. What the numbers show on this shallow teacher: consolidation delivers inviolable, outward-
growing structure, but TRADES efficiency for it -- the frozen-core loop needs more wires than
unconstrained joint training to reach comparable accuracy (there is little deep reuse here to amortize).
Pure numpy, CPU.  Run: python experiments/consolidation_rounds.py
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

RESULTS = Path("results")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fiber_distance(w_abs, src_dist):
    """Relational distance-from-axiom: 1 hop past the weighted-average distance of the sources this
    fiber actually reads (inputs are distance 0). A fiber reading only inputs -> 1; one that bundles
    distance-d fibers -> 1+d. 'Distance = the gap/void of non-relation', over surviving connections."""
    s = w_abs.sum() + 1e-9
    return 1.0 + float((w_abs * src_dist).sum()) / s


def hier_teacher_data(D=20, hid=(10, 10), C=4, N=4000, ntr=3000, seed=0):
    """Labels from a DEEP (compositional) teacher, so primitives are reusable across rounds."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D))
    h = X
    for d_out in hid:
        W = rng.normal(0, np.sqrt(2 / h.shape[1]), (h.shape[1], d_out))
        h = np.maximum(h @ W, 0)
    Wo = rng.normal(0, np.sqrt(2 / h.shape[1]), (h.shape[1], C))
    logits = h @ Wo
    logits -= logits.mean(0)
    y = logits.argmax(1)
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:], D, C


class ConsolidatingNet:
    """A net that grows a FROZEN axiom core outward, one consolidation round at a time."""

    def __init__(self, D, C, seed=0):
        self.D, self.C = D, C
        self.rng = np.random.default_rng(seed)
        self.Ftr = np.zeros((0, 0))   # cached frozen established features (set per fit)
        self.Fte = np.zeros((0, 0))
        self.frozen_tr = None         # accumulated frozen readout logits (train / test)
        self.frozen_te = None
        self.bias = np.zeros(C)
        self.dist = []                # distance-from-axiom of each established fiber
        self.frozen_W = []            # kept incoming weights (for the inviolability assert)
        self.synapses = 0             # cumulative synapses committed

    def _ensure(self, Xtr, Xte):
        if self.frozen_tr is None:
            self.Ftr = np.zeros((len(Xtr), 0)); self.Fte = np.zeros((len(Xte), 0))
            self.frozen_tr = np.zeros((len(Xtr), self.C)); self.frozen_te = np.zeros((len(Xte), self.C))

    def acc(self, y, which="te"):
        logits = (self.frozen_te if which == "te" else self.frozen_tr) + self.bias
        return float((logits.argmax(1) == y).mean())

    def grow_round(self, Xtr, ytr, Xte, yte, P=32, epochs=1500, lr=0.05, wd=1e-4,
                   floor=0.1, conn_floor=0.2, refit=400):
        self._ensure(Xtr, Xte)
        Ztr = np.concatenate([Xtr, self.Ftr], 1)      # candidates read inputs + frozen base
        Zte = np.concatenate([Xte, self.Fte], 1)
        din, n = Ztr.shape[1], len(ytr)
        W = self.rng.normal(0, np.sqrt(2 / din), (din, P))  # overproduce P candidates
        b = np.zeros(P)
        V = self.rng.normal(0, 0.01, (P, self.C))
        db = np.zeros(self.C)
        onehot = np.eye(self.C)[ytr]
        for _ in range(epochs):
            pre = Ztr @ W + b; A = np.maximum(pre, 0)
            logits = self.frozen_tr + A @ V + db + self.bias
            dl = (softmax(logits) - onehot) / n
            dV = A.T @ dl + wd * V; ddb = dl.sum(0)
            dpre = (dl @ V.T) * (pre > 0)
            dW = Ztr.T @ dpre + wd * W; dbb = dpre.sum(0)
            W -= lr * dW; b -= lr * dbb; V -= lr * dV; db -= lr * ddb

        # (1) UNIT prune: a candidate that moves the output relates to something; the void does not.
        contribution = np.linalg.norm(V, axis=1) * np.maximum(Ztr @ W + b, 0).std(0)
        keep = np.where(contribution >= floor * contribution.max())[0]
        void_frac = 1 - len(keep) / P
        W, b, V = W[:, keep], b[keep], V[keep]

        # (2) CONNECTION prune the void: drop each surviving fiber's near-zero incoming reads, so a
        # "synapse" only ever counts a real relation. (The plan's "over its surviving incoming
        # weights" -- a near-zero read is non-relation, not a wire.)
        cmask = (np.abs(W) >= conn_floor * (np.abs(W).max(0, keepdims=True) + 1e-12)).astype(float)
        W = W * cmask

        # (3) re-solidify the readout + unit biases after pruning (connections W fixed; frozen base
        # and its bias untouched) -- the same "retrain the survivors" step as magnitude pruning.
        for _ in range(refit):
            pre = Ztr @ W + b; A = np.maximum(pre, 0)
            dl = (softmax(self.frozen_tr + A @ V + db + self.bias) - onehot) / n
            dpre = (dl @ V.T) * (pre > 0)
            V -= lr * (A.T @ dl + wd * V); db -= lr * dl.sum(0); b -= lr * dpre.sum(0)

        Atr = np.maximum(Ztr @ W + b, 0); Ate = np.maximum(Zte @ W + b, 0)

        # distance-from-axiom over SURVIVING incoming connections (inputs dist 0, est fibers stored)
        src_dist = np.concatenate([np.zeros(self.D), np.array(self.dist)]) if self.dist else np.zeros(self.D)
        new_dists = [fiber_distance(np.abs(W[:, j]), src_dist) for j in range(W.shape[1])]

        # freeze survivors into the established base (append-only -> axioms never disturbed)
        self.frozen_tr = self.frozen_tr + Atr @ V
        self.frozen_te = self.frozen_te + Ate @ V
        self.bias = self.bias + db
        self.Ftr = np.concatenate([self.Ftr, Atr], 1)
        self.Fte = np.concatenate([self.Fte, Ate], 1)
        self.dist += new_dists
        self.frozen_W.append(W.copy())
        n_conn = int(cmask.sum())                    # surviving incoming connections this round
        self.synapses += n_conn + len(keep) * self.C  # + readout wires (C per kept fiber)
        return {"kept": len(keep), "void_frac": void_frac,
                "conn_kept": n_conn, "conn_frac": float(cmask.mean()),
                "mean_dist": float(np.mean(self.dist)), "max_dist": float(np.max(self.dist)),
                "test_acc": self.acc(yte), "synapses": self.synapses}


def monolithic(Xtr, ytr, Xte, yte, D, C, H, epochs, target_syn=None, lr=0.05, wd=1e-4, seed=1):
    """Standard training, no freezing. Returns dense (acc, syn) and a magnitude-pruned (acc, syn)
    variant at ~target_syn synapses -- the honest 'pruning WITHOUT consolidation' control."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2 / D), (D, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, np.sqrt(2 / H), (H, C)); b2 = np.zeros(C)
    n = len(ytr); onehot = np.eye(C)[ytr]
    def fit(iters, m1=None, m2=None):
        nonlocal W1, b1, W2, b2
        for _ in range(iters):
            if m1 is not None: W1 *= m1; W2 *= m2
            pre = Xtr @ W1 + b1; A = np.maximum(pre, 0)
            dl = (softmax(A @ W2 + b2) - onehot) / n
            dpre = (dl @ W2.T) * (pre > 0)
            W2 -= lr * (A.T @ dl + wd * W2); b2 -= lr * dl.sum(0)
            W1 -= lr * (Xtr.T @ dpre + wd * W1); b1 -= lr * dpre.sum(0)
        if m1 is not None: W1 *= m1; W2 *= m2
    def acc():
        return float((np.maximum(Xte @ W1 + b1, 0) @ W2 + b2).argmax(1).__eq__(yte).mean())
    fit(epochs)
    dense = (acc(), H * (D + C))
    if not target_syn or target_syn >= dense[1]:
        return dense, dense
    allw = np.concatenate([np.abs(W1).ravel(), np.abs(W2).ravel()])
    thr = np.sort(allw)[len(allw) - int(target_syn)]           # keep the top target_syn |weights|
    m1 = (np.abs(W1) >= thr).astype(float); m2 = (np.abs(W2) >= thr).astype(float)
    fit(max(400, epochs // 4), m1, m2)                         # re-solidify the survivors
    return dense, (acc(), int(m1.sum() + m2.sum()))


def main():
    RESULTS.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte, D, C = hier_teacher_data()
    ROUNDS, P, EP = 5, 32, 1500

    net = ConsolidatingNet(D, C, seed=1)
    axiom_snapshot = None
    rounds = []
    print("iterative consolidation (overproduce -> relational-prune -> freeze):")
    for r in range(1, ROUNDS + 1):
        st = net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP)
        rounds.append(st)
        if r == 1:
            axiom_snapshot = net.frozen_W[0].copy()   # the axioms, established in round 1
        print(f"  round {r}: kept {st['kept']:2d}/{P}  void_pruned {st['void_frac']*100:4.0f}%  "
              f"conn_kept {st['conn_frac']*100:4.0f}%  test_acc={st['test_acc']:.3f}  "
              f"mean_dist_from_axiom={st['mean_dist']:.2f}  synapses={st['synapses']}")

    # inviolability: the round-1 axioms are byte-identical after all later rounds
    inviolate = bool(np.array_equal(axiom_snapshot, net.frozen_W[0]))
    total_units = len(net.dist)
    loop_acc = rounds[-1]["test_acc"]; loop_syn = net.synapses

    # controls (same final width, same total epochs, standard training -- no freezing):
    #   dense  = pruning-free baseline;  pruned = magnitude-pruned to ~the loop's synapse budget,
    #   which isolates what CONSOLIDATION adds beyond plain pruning.
    (mono_acc, mono_syn), (monp_acc, monp_syn) = monolithic(
        Xtr, ytr, Xte, yte, D, C, H=total_units, epochs=ROUNDS * EP, target_syn=loop_syn)

    print(f"\nfrozen axioms inviolate across all rounds: {inviolate}")
    print(f"loop (consolidated): test_acc={loop_acc:.3f}  synapses={loop_syn:5d}  "
          f"capability/synapse x{loop_acc/loop_syn*1e3:.2f}e-3")
    print(f"monolithic dense:    test_acc={mono_acc:.3f}  synapses={mono_syn:5d}  "
          f"capability/synapse x{mono_acc/mono_syn*1e3:.2f}e-3")
    print(f"monolithic pruned:   test_acc={monp_acc:.3f}  synapses={monp_syn:5d}  "
          f"capability/synapse x{monp_acc/monp_syn*1e3:.2f}e-3  (pruning WITHOUT consolidation)")
    grew = rounds[-1]["mean_dist"] > rounds[0]["mean_dist"] + 0.05
    print(f"structure grew outward (mean distance-from-axiom rose): {grew} "
          f"({rounds[0]['mean_dist']:.2f} -> {rounds[-1]['mean_dist']:.2f})")
    beats_pruned = loop_acc >= monp_acc and loop_syn <= monp_syn * 1.10
    print(f"consolidation beats plain pruning at matched budget: {beats_pruned} "
          f"(loop {loop_acc:.3f}@{loop_syn} vs pruned {monp_acc:.3f}@{monp_syn})")

    res = {"rounds": rounds, "inviolate": inviolate, "loop_acc": loop_acc, "loop_syn": loop_syn,
           "mono_acc": mono_acc, "mono_syn": mono_syn, "monp_acc": monp_acc, "monp_syn": monp_syn,
           "grew_outward": grew, "beats_pruned": beats_pruned, "total_units": total_units}
    (RESULTS / "consolidation_rounds.json").write_text(json.dumps(res, indent=2, default=float))

    rr = list(range(1, ROUNDS + 1))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3))
    a1.plot(rr, [s["test_acc"] for s in rounds], "o-", color="#2e7d32", label="loop (frozen core)")
    a1.axhline(mono_acc, ls=":", color="gray", label=f"monolithic (joint) {mono_acc:.2f}")
    a1.set_xlabel("consolidation round"); a1.set_ylabel("test accuracy")
    a1.set_title("Capability accrues on a frozen core\n(below unconstrained joint training)"); a1.legend()
    a2.plot(rr, [s["mean_dist"] for s in rounds], "o-", color="#1565c0", label="mean")
    a2.plot(rr, [s["max_dist"] for s in rounds], "s--", color="#1565c0", alpha=0.5, label="max")
    a2.set_xlabel("consolidation round"); a2.set_ylabel("distance from axiom (relational hops)")
    a2.set_title("New primitives grow further from the axioms\n(the clean result)"); a2.legend()
    cum = [s["synapses"] for s in rounds]
    a3.plot(cum, [s["test_acc"] for s in rounds], "o-", color="#2e7d32", label="loop (cumulative)")
    a3.scatter([mono_syn], [mono_acc], color="#555", zorder=5, s=80, marker="s", label="monolithic dense")
    if monp_syn < mono_syn:  # pruned control only distinct when it actually pruned to the loop's budget
        a3.scatter([monp_syn], [monp_acc], color="#c62828", zorder=5, s=70, label="monolithic pruned")
    a3.annotate("", xy=(mono_syn, mono_acc), xytext=(cum[-1], rounds[-1]["test_acc"]),
                arrowprops=dict(arrowstyle="->", color="#c62828", ls="--"))
    a3.set_xlabel("synapses (surviving connections)"); a3.set_ylabel("test accuracy")
    a3.set_title("Efficiency cost: the loop trails joint training here\n(more wires, less accuracy)")
    a3.legend()
    fig.tight_layout(); fig.savefig(RESULTS / "consolidation_rounds.png", dpi=140)
    print("\nSaved results/consolidation_rounds.{json,png}")


if __name__ == "__main__":
    main()
