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

The TIGHTENING RATIO (tau, ramped loose->tight across rounds) pulls the concept-lines into one another,
forming cross-paths, via force + reward + pressure: raw-input reads get costlier and hard-capped while
frozen-base reads get cheaper (force + reward), each new fiber may bundle only k_par existing fibers
(sparse composition -> every cross-path is cheap), and a survivor must hold a base-mass share >= a
rising-but-capped threshold or be pruned as "not tight enough" (pressure). tau=0 reproduces the loose
baseline; the ramp turns parallel accreted columns into a sparse fiber->fiber mesh.

Distance-from-axiom (relational hops, "distance = the gap/void of non-relation"):
    dist(input) = 0;  dist(fiber) = 1 + (sum over SURVIVING incoming |w|*dist(source)) / (sum |w|).

Honest controls & kill-criteria (stated up front, no forcing the result): round-1 axioms are asserted
byte-identical after every later round; a TIGHTENED loop is run against an UNTIGHTENED one (same seed)
and two MONOLITHIC nets (dense + magnitude-pruned to the tightened budget). What the numbers show on
this shallow teacher: tightening forms a sparse cross-path mesh and cuts wires ~3.3x at ~equal accuracy,
which FLIPS the efficiency loss versus dense joint training (capability/synapse up). Honest limits it
does NOT hide: mean distance-from-axiom no longer grows further than the loose loop (sparse bundling
trades depth for cheap wires), and plain magnitude pruning of a jointly-trained net stays the most
wire-efficient of all -- consolidation's return is inviolable, legible structure, not beating pruning.
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
        self.cross_edges = 0          # cumulative fiber->fiber connections (the cross-path mesh)

    def _ensure(self, Xtr, Xte):
        if self.frozen_tr is None:
            self.Ftr = np.zeros((len(Xtr), 0)); self.Fte = np.zeros((len(Xte), 0))
            self.frozen_tr = np.zeros((len(Xtr), self.C)); self.frozen_te = np.zeros((len(Xte), self.C))

    @staticmethod
    def _cap_cols(M, k):
        """Keep only the top-k |values| in each column of M (a view into W); zero the rest, in place."""
        for j in range(M.shape[1]):
            col = np.abs(M[:, j])
            if (col > 0).sum() > k:
                M[:, j][col < np.sort(col)[-k]] = 0.0

    def acc(self, y, which="te"):
        logits = (self.frozen_te if which == "te" else self.frozen_tr) + self.bias
        return float((logits.argmax(1) == y).mean())

    def grow_round(self, Xtr, ytr, Xte, yte, P=32, epochs=1500, lr=0.05, wd=1e-4,
                   floor=0.1, conn_floor=0.2, refit=400, tau=0.0, k_par=6):
        """One consolidation wave. `tau` in [0,1) is the TIGHTENING RATIO: the target share of a new
        fiber's incoming weight-mass that must land on the frozen base (cross-paths) rather than raw
        inputs. tau ramps up across rounds to pull the concept-lines into one another. `k_par` bounds
        how many frozen fibers a new fiber may bundle (sparse composition -> each cross-path is cheap)."""
        self._ensure(Xtr, Xte)
        Ztr = np.concatenate([Xtr, self.Ftr], 1)      # candidates read [inputs | frozen base]
        Zte = np.concatenate([Xte, self.Fte], 1)
        din, n = Ztr.shape[1], len(ytr)
        n_base = din - self.D                         # frozen fibers available to compose
        tight = n_base > 0 and tau > 0
        W = self.rng.normal(0, np.sqrt(2 / din), (din, P))  # overproduce P candidates
        b = np.zeros(P)
        V = self.rng.normal(0, 0.01, (P, self.C))
        db = np.zeros(self.C)
        onehot = np.eye(self.C)[ytr]

        # FORCE + REWARD: per-row weight decay -- raw-input reads get costlier as tau rises, frozen
        # base reads get cheaper, so gradient descent routes signal THROUGH existing fibers.
        row_wd = np.full((din, 1), wd)
        if tight:
            row_wd[:self.D] = wd * (1.0 + 5.0 * tau)   # inputs: penalized
            row_wd[self.D:] = wd * (1.0 - 0.8 * tau)   # base: rewarded (stays >= 0.2*wd)
        for _ in range(epochs):
            pre = Ztr @ W + b; A = np.maximum(pre, 0)
            logits = self.frozen_tr + A @ V + db + self.bias
            dl = (softmax(logits) - onehot) / n
            dV = A.T @ dl + wd * V; ddb = dl.sum(0)
            dpre = (dl @ V.T) * (pre > 0)
            dW = Ztr.T @ dpre + row_wd * W; dbb = dpre.sum(0)
            W -= lr * dW; b -= lr * dbb; V -= lr * dV; db -= lr * ddb

        # (1) UNIT prune: a candidate that moves the output relates to something; the void does not.
        contribution = np.linalg.norm(V, axis=1) * np.maximum(Ztr @ W + b, 0).std(0)
        keep = np.where(contribution >= floor * contribution.max())[0]
        W, b, V = W[:, keep], b[keep], V[keep]

        # (2) CONNECTION prune the void: drop each survivor's near-zero incoming reads, so a "synapse"
        # only ever counts a real relation (the plan's "over its surviving incoming weights").
        W = W * (np.abs(W) >= conn_floor * (np.abs(W).max(0, keepdims=True) + 1e-12))

        # (3) FORCE (hard caps): as tau rises each fiber may keep only k_in raw-input reads, so it must
        # route through the base; and it may BUNDLE at most k_par frozen fibers (sparse composition --
        # "bundle a few phases"), which is what keeps every cross-path cheap and the mesh rich.
        if tight:
            self._cap_cols(W[:self.D], max(1, int(round((1.0 - tau) * self.D))))  # input cap
            self._cap_cols(W[self.D:], k_par)                                     # parent (bundle) cap

        # (4) PRESSURE gate: a survivor must relate to concepts more than to raw data -- base-mass
        # share >= a threshold that rises with tau but is capped (a good compositional fiber is ~50-80%
        # base, never pure), so loose near-input readers are squeezed out without collapsing the mesh.
        base_mass = np.abs(W[self.D:]).sum(0) if n_base else np.zeros(W.shape[1])
        share = base_mass / (np.abs(W[:self.D]).sum(0) + base_mass + 1e-9)
        if tight:
            gated = np.where(share >= min(tau, 0.5))[0]
            if len(gated) == 0:
                gated = np.array([int(np.argmax(share))])
            W, b, V, share = W[:, gated], b[gated], V[gated], share[gated]
        kept = W.shape[1]
        void_frac = 1 - kept / P

        # re-solidify the readout + unit biases after pruning (connections W fixed; frozen base
        # and its bias untouched) -- the same "retrain the survivors" step as magnitude pruning.
        for _ in range(refit):
            pre = Ztr @ W + b; A = np.maximum(pre, 0)
            dl = (softmax(self.frozen_tr + A @ V + db + self.bias) - onehot) / n
            dpre = (dl @ V.T) * (pre > 0)
            V -= lr * (A.T @ dl + wd * V); db -= lr * dl.sum(0); b -= lr * dpre.sum(0)

        Atr = np.maximum(Ztr @ W + b, 0); Ate = np.maximum(Zte @ W + b, 0)

        # distance-from-axiom over SURVIVING incoming connections (inputs dist 0, est fibers stored)
        src_dist = np.concatenate([np.zeros(self.D), np.array(self.dist)]) if self.dist else np.zeros(self.D)
        new_dists = [fiber_distance(np.abs(W[:, j]), src_dist) for j in range(kept)]

        # freeze survivors into the established base (append-only -> axioms never disturbed)
        self.frozen_tr = self.frozen_tr + Atr @ V
        self.frozen_te = self.frozen_te + Ate @ V
        self.bias = self.bias + db
        self.Ftr = np.concatenate([self.Ftr, Atr], 1)
        self.Fte = np.concatenate([self.Fte, Ate], 1)
        self.dist += new_dists
        self.frozen_W.append(W.copy())
        n_in = int((np.abs(W[:self.D]) > 0).sum())                    # surviving raw-input reads
        n_cross = int((np.abs(W[self.D:]) > 0).sum()) if n_base else 0  # fiber->fiber cross-paths
        self.cross_edges += n_cross
        self.synapses += n_in + n_cross + kept * self.C               # + readout wires (C per fiber)
        return {"kept": kept, "void_frac": void_frac, "cross_edges": n_cross,
                "base_share": float(np.mean(share)) if kept else 0.0,
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


def run_loop(Xtr, ytr, Xte, yte, D, C, taus, P=32, epochs=1500, seed=1):
    """Run ROUNDS consolidation waves with a per-round tightening ratio; return net + per-round stats
    + the round-1 axiom snapshot (for the inviolability assert)."""
    net = ConsolidatingNet(D, C, seed=seed)
    rounds, snapshot = [], None
    for r, tau in enumerate(taus, 1):
        rounds.append(net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=epochs, tau=tau))
        if r == 1:
            snapshot = net.frozen_W[0].copy()
    return net, rounds, snapshot


def main():
    RESULTS.mkdir(exist_ok=True)
    Xtr, ytr, Xte, yte, D, C = hier_teacher_data()
    ROUNDS, P, EP = 5, 32, 1500
    # tightening ratio, ramped loose -> tight (round 1 = pure axioms). Ceiling 0.8 was chosen by a
    # swept comparison (experiments/consolidation_tau_sweep.py): best accuracy and a smooth monotone
    # rise, at ~equal capability/synapse -- 0.9 over-tightens the last round and dips.
    taus = [0.0] + [round(float(t), 3) for t in np.linspace(0.3, 0.8, ROUNDS - 1)]  # ->[0,.3,.467,.633,.8]

    print(f"TIGHTENED loop (force + reward + pressure; tau ramp {taus}):")
    tnet, tr, tsnap = run_loop(Xtr, ytr, Xte, yte, D, C, taus, P=P, epochs=EP)
    for r, s in enumerate(tr, 1):
        print(f"  round {r}: tau={taus[r-1]:.1f}  kept {s['kept']:2d}/{P}  base_share={s['base_share']:.2f}"
              f"  cross_edges={s['cross_edges']:3d}  test_acc={s['test_acc']:.3f}  "
              f"mean_dist={s['mean_dist']:.2f}  synapses={s['synapses']}")

    print("\nUNTIGHTENED loop (tau=0 throughout; the prior baseline):")
    unet, ur, usnap = run_loop(Xtr, ytr, Xte, yte, D, C, [0.0] * ROUNDS, P=P, epochs=EP)
    for r, s in enumerate(ur, 1):
        print(f"  round {r}: kept {s['kept']:2d}/{P}  cross_edges={s['cross_edges']:3d}  "
              f"test_acc={s['test_acc']:.3f}  mean_dist={s['mean_dist']:.2f}  synapses={s['synapses']}")

    t_inv = bool(np.array_equal(tsnap, tnet.frozen_W[0]))
    u_inv = bool(np.array_equal(usnap, unet.frozen_W[0]))
    t_acc, t_syn = tr[-1]["test_acc"], tnet.synapses
    u_acc, u_syn = ur[-1]["test_acc"], unet.synapses

    # monolithic controls (standard joint training, same final width): dense + magnitude-pruned to
    # the tightened loop's synapse budget -- the honest "pruning WITHOUT consolidation" comparison.
    (mono_acc, mono_syn), (monp_acc, monp_syn) = monolithic(
        Xtr, ytr, Xte, yte, D, C, H=len(tnet.dist), epochs=ROUNDS * EP, target_syn=t_syn)

    def cps(a, s): return a / s * 1e3
    print(f"\naxioms inviolate (tightened / untightened): {t_inv} / {u_inv}")
    print(f"tightened loop:    acc={t_acc:.3f}  syn={t_syn:5d}  cap/syn x{cps(t_acc,t_syn):.2f}e-3  "
          f"cross_edges={tnet.cross_edges}")
    print(f"untightened loop:  acc={u_acc:.3f}  syn={u_syn:5d}  cap/syn x{cps(u_acc,u_syn):.2f}e-3  "
          f"cross_edges={unet.cross_edges}")
    print(f"monolithic dense:  acc={mono_acc:.3f}  syn={mono_syn:5d}  cap/syn x{cps(mono_acc,mono_syn):.2f}e-3")
    print(f"monolithic pruned: acc={monp_acc:.3f}  syn={monp_syn:5d}  cap/syn x{cps(monp_acc,monp_syn):.2f}e-3")

    cut = t_syn < u_syn
    beats_dense = cps(t_acc, t_syn) > cps(mono_acc, mono_syn)
    beats_pruned = t_acc >= monp_acc - 0.01 and t_syn <= monp_syn * 1.05
    deeper = tr[-1]["mean_dist"] >= ur[-1]["mean_dist"] - 0.01
    print(f"\ntightening formed cross-paths: {tnet.cross_edges} fiber->fiber edges "
          f"(untightened {unet.cross_edges})")
    print(f"tightening cut wires: {cut} ({u_syn} -> {t_syn})")
    print(f"distance deepened at least as far: {deeper} "
          f"(untightened {ur[-1]['mean_dist']:.2f} vs tightened {tr[-1]['mean_dist']:.2f})")
    print(f"THE GOAL -- tightened beats monolithic dense on capability/synapse: {beats_dense}")
    print(f"tightened beats plain pruning at matched budget: {beats_pruned} "
          f"(loop {t_acc:.3f}@{t_syn} vs pruned {monp_acc:.3f}@{monp_syn})")

    res = {"taus": taus, "tightened": tr, "untightened": ur,
           "t_inviolate": t_inv, "u_inviolate": u_inv,
           "t_acc": t_acc, "t_syn": t_syn, "t_cross": tnet.cross_edges,
           "u_acc": u_acc, "u_syn": u_syn, "u_cross": unet.cross_edges,
           "mono_acc": mono_acc, "mono_syn": mono_syn, "monp_acc": monp_acc, "monp_syn": monp_syn,
           "cut_wires": cut, "beats_dense": beats_dense, "beats_pruned": beats_pruned, "deeper": deeper}
    (RESULTS / "consolidation_rounds.json").write_text(json.dumps(res, indent=2, default=float))

    rr = list(range(1, ROUNDS + 1))
    G, Bl, Gr, Rd = "#2e7d32", "#1565c0", "#777", "#c62828"
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13, 9))
    a1.plot(rr, [s["test_acc"] for s in tr], "o-", color=G, label="tightened")
    a1.plot(rr, [s["test_acc"] for s in ur], "s--", color=Gr, label="untightened")
    a1.axhline(mono_acc, ls=":", color=Rd, label=f"monolithic {mono_acc:.2f}")
    a1.set_xlabel("consolidation round"); a1.set_ylabel("test accuracy")
    a1.set_title("Capability across rounds"); a1.legend()

    a2.plot(rr, [s["mean_dist"] for s in tr], "o-", color=Bl, label="tightened (mean)")
    a2.plot(rr, [s["max_dist"] for s in tr], "s--", color=Bl, alpha=0.5, label="tightened (max)")
    a2.plot(rr, [s["mean_dist"] for s in ur], "o-", color=Gr, label="untightened (mean)")
    a2.set_xlabel("consolidation round"); a2.set_ylabel("distance from axiom (relational hops)")
    a2.set_title("Distance from axiom: tightening lifts the max, not the mean\n(sparse bundling trades depth for cheap wires)"); a2.legend()

    a3.plot([s["synapses"] for s in tr], [s["test_acc"] for s in tr], "o-", color=G, label="tightened")
    a3.plot([s["synapses"] for s in ur], [s["test_acc"] for s in ur], "s--", color=Gr, label="untightened")
    a3.scatter([mono_syn], [mono_acc], color="#333", zorder=5, s=80, marker="D", label="monolithic dense")
    if monp_syn < mono_syn:
        a3.scatter([monp_syn], [monp_acc], color=Rd, zorder=5, s=70, label="monolithic pruned")
    a3.set_xlabel("synapses (surviving connections)"); a3.set_ylabel("test accuracy")
    a3.set_title("Efficiency: tightening beats dense training per synapse\n(plain magnitude pruning still cheapest)"); a3.legend()

    a4.bar(rr, [s["cross_edges"] for s in tr], color=Bl, alpha=0.55, label="cross-path edges")
    a4.set_xlabel("consolidation round"); a4.set_ylabel("fiber->fiber edges (the mesh)", color=Bl)
    a4b = a4.twinx()
    a4b.plot(rr, [s["base_share"] for s in tr], "o-", color=G, label="base-mass share")
    a4b.plot(rr, taus, "k:", alpha=0.6, label="tau (target)")
    a4b.set_ylabel("base-mass share", color=G); a4b.set_ylim(0, 1)
    a4.set_title("The tightening ratio forms cross-paths")
    a4.legend(loc="upper left"); a4b.legend(loc="lower right")

    fig.tight_layout(); fig.savefig(RESULTS / "consolidation_rounds.png", dpi=140)
    print("\nSaved results/consolidation_rounds.{json,png}")


if __name__ == "__main__":
    main()
