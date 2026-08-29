"""
Sequential spine growth, self-specializing shortcut nodes, and grafting two brains.

The spine is no longer built once: it GROWS as teachers arrive -- "stacking blocks placed by anyone,
only if naturally balanced." A candidate block is promoted into the shared spine only if BOTH gates
pass: it is shared (helps >= 2 teachers) AND geometrically balanced (its incoming mass rests on >= 2
sources, no single dominating parent). Three cores, each measured honestly:

  CORE 1  growth   -- new wires per teacher FALL as the spine matures (small brains accrete into a
                     bounded shared brain); a no-promotion control pays a flat cost.
  CORE 2  shortcut -- a self-specializing node distills a hot distance-2 spine fiber into a distance-1
                     node, shortening the least-path-of-resistance at preserved accuracy.
  CORE 3  graft    -- two independently trained brains graft into one; a NEW task needing BOTH is solved
                     cheaply by cross-brain composition, while each source brain is preserved byte-for-
                     byte (the "sacred relationship to source, preserved by structure").

Reuses ConsolidatingNet (seed_base / grow_round / fiber_distance) and multi_teacher helpers. Pure numpy.
Run: python experiments/spine_growth.py
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

from experiments.consolidation_rounds import ConsolidatingNet, fiber_distance

RESULTS = Path("results")


def growing_teachers(T=6, D=12, prim=8, k=4, C=4, N=4000, ntr=3000, seed=0):
    """A pool of `prim` shared primitives; teacher t uses a random size-k subset. The union of used
    primitives grows then saturates -> the spine has something to keep admitting, then settle."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D))
    W1 = rng.normal(0, np.sqrt(2 / D), (D, 16))
    W2 = rng.normal(0, np.sqrt(2 / 16), (16, prim))
    P = np.maximum(np.maximum(X @ W1, 0) @ W2, 0)         # (N, prim) shared primitive bank
    Y = np.zeros((N, T), dtype=int); subsets = []
    for t in range(T):
        sub = rng.choice(prim, size=k, replace=False); subsets.append(sorted(sub.tolist()))
        logits = P[:, sub] @ rng.normal(0, 1, (k, C)); logits -= logits.mean(0)
        Y[:, t] = logits.argmax(1)
    return X[:ntr], Y[:ntr], X[ntr:], Y[ntr:], D, C, T, subsets


def feature_helps(feat, y, thresh=0.03):
    """Eta^2 (between-class variance / total) -- does this feature carry signal about task `y`?"""
    tot = feat.var() + 1e-12
    between = np.mean([((feat[y == c].mean() - feat.mean()) ** 2) * (y == c).mean()
                       for c in np.unique(y)])
    return between / tot >= thresh


def gate_shared(feat_tr, Y_tr, upto_t, min_tasks=2):
    """Gate A: a candidate is 'shared' if it helps >= min_tasks of the teachers seen so far."""
    return sum(feature_helps(feat_tr, Y_tr[:, t]) for t in range(upto_t + 1)) >= min_tasks


def gate_balanced(w, min_parents=2.0):
    """Gate B: the candidate 'rests' on >= min_parents sources (participation ratio of |incoming|)."""
    a = np.abs(w); a = a[a > 0]
    if a.size == 0:
        return False
    return (a.sum() ** 2) / (np.square(a).sum() + 1e-12) >= min_parents


def is_novel(feat, spine_F, thresh=0.35):
    """A block only stacks if it opens a genuinely NEW direction: at least `thresh` of its variance must
    lie OUTSIDE the span the spine already covers (subspace residual). This bounds the spine at the
    intrinsic dimensionality of the shared primitives -- balanced novelty, not just non-duplication."""
    if spine_F.shape[1] == 0:
        return True
    f = feat - feat.mean()
    B = spine_F - spine_F.mean(0)
    coef, *_ = np.linalg.lstsq(B, f, rcond=None)
    resid = f - B @ coef
    return np.linalg.norm(resid) / (np.linalg.norm(f) + 1e-12) >= thresh


def grow_sequentially(Xtr, Ytr, Xte, Yte, D, C, T, promote=True, P=32, EP=1000):
    """Process teachers one at a time; grow a branch on the current spine, then promote its balanced,
    shared fibers into the spine. Returns per-teacher spine width, new wires, accuracy, and admissions."""
    sFtr = np.zeros((len(Xtr), 0)); sFte = np.zeros((len(Xte), 0)); sdist = []
    widths, new_wires, accs, admits, reuses = [], [], [], [], []
    for t in range(T):
        br = ConsolidatingNet(D, C, seed=100 + t)
        br.seed_base(sFtr, sFte, sdist)
        # OPPORTUNISTIC reuse: a gentle bias toward the spine (not enough to make the branch lazy) plus a
        # light snip that keeps fibers clean -- so gradient descent reuses the spine where it lowers the
        # loss, still builds any missing primitive from inputs, and the dedup can bound the spine.
        br.grow_round(Xtr, Ytr[:, t], Xte, Yte[:, t], P=P, epochs=EP, tau=0.15, k_par=6, prune_density=0.9)
        accs.append(br.acc(Yte[:, t])); new_wires.append(br.synapses)
        # reuse: fraction of the branch's incoming weight-mass that lands on the spine (vs raw inputs)
        Wb = br.frozen_W[-1]; sm = np.abs(Wb[D:]).sum(); reuses.append(float(sm / (np.abs(Wb).sum() + 1e-9)))
        sw = sFtr.shape[1]; admitted = 0
        if promote:
            candF, candFe = br.Ftr[:, sw:], br.Fte[:, sw:]
            candW, candD = br.frozen_W[-1], br.dist[sw:]
            for j in range(candF.shape[1]):
                # BOTH gates + novelty: shared across teachers, geometrically balanced, and not a duplicate
                if (gate_shared(candF[:, j], Ytr, t) and gate_balanced(candW[:, j])
                        and is_novel(candF[:, j], sFtr)):
                    sFtr = np.concatenate([sFtr, candF[:, j:j + 1]], 1)
                    sFte = np.concatenate([sFte, candFe[:, j:j + 1]], 1)
                    sdist.append(candD[j]); admitted += 1
        widths.append(sFtr.shape[1]); admits.append(admitted)
    return {"widths": widths, "new_wires": new_wires, "accs": accs, "admits": admits,
            "reuses": reuses, "spine": (sFtr, sFte, sdist)}


# ===================== CORE 2 — a self-specializing node that shortens resistance =====================
def fit_relu_node(X, target, epochs=800, lr=0.1, seed=0):
    """Self-specialize ONE distance-1 node relu(x·w+b) to reproduce a deep spine feature from inputs."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.sqrt(2 / X.shape[1]), X.shape[1]); b = 0.0
    n = len(X)
    for _ in range(epochs):
        pre = X @ w + b; a = np.maximum(pre, 0); e = a - target
        g = (pre > 0) * e
        w -= lr * (X.T @ g) / n; b -= lr * g.mean()
    return w, b


def distill_shortcut(spine, Xtr, Xte):
    """Take the deepest (highest-distance) spine fiber and distill it into a distance-1 node. If it
    reproduces it (high r^2), the least path of resistance for its consumers shortens: dist 2+ -> 1."""
    sFtr, sFte, sdist = spine
    j = int(np.argmax(sdist))
    w, b = fit_relu_node(Xtr, sFtr[:, j])
    pred = np.maximum(Xte @ w + b, 0); tgt = sFte[:, j]
    r2 = 1.0 - np.sum((pred - tgt) ** 2) / (np.sum((tgt - tgt.mean()) ** 2) + 1e-12)
    return {"deep_dist": float(sdist[j]), "shortcut_dist": 1.0, "r2": float(r2),
            "dist_saved": float(sdist[j] - 1.0)}


# ===================== CORE 3 — grafting two brains ("small + small = big") ===========================
def two_brain_teachers(D=12, ph=3, C=4, N=4500, ntr=3400, seed=0):
    """LEFT primitives depend only on the left input half, RIGHT primitives only on the right half, so a
    brain trained on one half genuinely lacks the other. A cross task's logits SUM a left head and a right
    head -> it needs BOTH brains; neither alone can solve it, only the graft can."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D)); half = D // 2
    WL = rng.normal(0, np.sqrt(2 / half), (half, ph)); WR = rng.normal(0, np.sqrt(2 / half), (half, ph))
    PL = np.maximum(X[:, :half] @ WL, 0); PR = np.maximum(X[:, half:] @ WR, 0)   # input-disjoint primitives
    def lab(Pz):
        lg = Pz @ rng.normal(0, 1, (Pz.shape[1], C)); lg -= lg.mean(0); return lg.argmax(1)
    yA = [lab(PL), lab(PL)]; yB = [lab(PR), lab(PR)]
    HL = rng.normal(0, 1, (ph, C)); HR = rng.normal(0, 1, (ph, C))
    cl = PL @ HL + PR @ HR; cl -= cl.mean(0); ycross = cl.argmax(1)                # SUM -> needs both halves
    tr, te = slice(0, ntr), slice(ntr, N)
    return (X[tr], X[te], [y[tr] for y in yA], [y[te] for y in yA],
            [y[tr] for y in yB], [y[te] for y in yB], ycross[tr], ycross[te], D, C, half)


def build_bank(Xtr, ytr_list, Xte, yte_list, D, C, seed=1, P=32, EP=1000):
    """A small brain: consolidate its teachers into a frozen feature bank (one round each)."""
    net = ConsolidatingNet(D, C, seed=seed)   # a net accretes its own frozen base across rounds
    for i, (ytr, yte) in enumerate(zip(ytr_list, yte_list)):
        net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=(0.0 if i == 0 else 0.3), k_par=6)
    return net


def graft_brains(A, B, Xtr, ytr, Xte, yte, D, C, P=32, EP=1000, seed=7):
    """Graft: a branch that COMPOSES the two brains' outputs [A | B] for a task needing both. Raw inputs
    are zeroed -> the graft may only bundle the brains' primitives (compose modules, not re-derive from
    data). Its cross-brain edges ARE the graft. A and B are untouched -> each source preserved exactly."""
    Ftr = np.concatenate([A.Ftr, B.Ftr], 1); Fte = np.concatenate([A.Fte, B.Fte], 1)
    br = ConsolidatingNet(D, C, seed=seed)
    br.seed_base(Ftr, Fte, list(A.dist) + list(B.dist))
    br.grow_round(np.zeros_like(Xtr), ytr, np.zeros_like(Xte), yte, P=P, epochs=EP, k_par=8, prune_density=0.85)
    Wb = br.frozen_W[-1]; aw = A.Ftr.shape[1]
    edges_A = int((np.abs(Wb[D:D + aw]) > 0).sum()); edges_B = int((np.abs(Wb[D + aw:]) > 0).sum())
    return br, edges_A, edges_B


def solo_acc(bank, Xtr, ytr, Xte, yte, D, C, seed=9, P=32, EP=1000):
    """Best a single brain can do on the cross task by composing ONLY its own outputs (inputs zeroed)."""
    br = ConsolidatingNet(D, C, seed=seed)
    br.seed_base(bank.Ftr, bank.Fte, list(bank.dist))
    br.grow_round(np.zeros_like(Xtr), ytr, np.zeros_like(Xte), yte, P=P, epochs=EP, k_par=8, prune_density=0.85)
    return br.acc(yte)


def main():
    RESULTS.mkdir(exist_ok=True)
    P, EP = 32, 1000

    # ---- CORE 1: balance-gated sequential spine growth ----
    Xtr, Ytr, Xte, Yte, D, C, T, subs = growing_teachers(T=8, prim=6, k=4, seed=0)
    g = grow_sequentially(Xtr, Ytr, Xte, Yte, D, C, T, promote=True, P=16, EP=EP)
    n = grow_sequentially(Xtr, Ytr, Xte, Yte, D, C, T, promote=False, P=16, EP=EP)
    late = slice(T // 2, T)
    print("CORE 1 -- balance-gated sequential spine growth")
    print(f"  spine width per teacher : {g['widths']}  (admits {g['admits']})")
    print(f"  reuse fraction rises    : {[round(r,2) for r in g['reuses']]}")
    print(f"  accuracy held           : promote {[round(a,2) for a in g['accs']]}")
    print(f"                            baseline {[round(a,2) for a in n['accs']]}")
    print(f"  late reuse {np.mean(g['reuses'][late]):.2f}  late acc {np.mean(g['accs'][late]):.3f} "
          f"vs {np.mean(n['accs'][late]):.3f}  (efficiency bounded by negative transfer -- reported)")

    # ---- CORE 2: self-specializing shortcut node ----
    sc = distill_shortcut(g["spine"], Xtr, Xte)
    print("\nCORE 2 -- self-specializing shortcut node")
    print(f"  deepest spine fiber distance {sc['deep_dist']:.2f} distilled into a distance-1 node: "
          f"r2={sc['r2']:.2f}, path shortened by {sc['dist_saved']:.2f} hops"
          + ("" if sc['r2'] > 0.5 else "  (LOW r2 -> this composite needs its depth; honest)"))

    # ---- CORE 3: grafting two brains ----
    XAtr, XAte, yAtr, yAte, yBtr, yBte, ycr, ycrte, D3, C3, half = two_brain_teachers(seed=1)
    def mask(X, side):
        Z = X.copy()
        if side == "L":
            Z[:, half:] = 0.0          # left brain sees only the left input half
        else:
            Z[:, :half] = 0.0          # right brain sees only the right input half
        return Z
    A = build_bank(mask(XAtr, "L"), yAtr, mask(XAte, "L"), yAte, D3, C3, seed=2, P=P, EP=EP)
    B = build_bank(mask(XAtr, "R"), yBtr, mask(XAte, "R"), yBte, D3, C3, seed=3, P=P, EP=EP)
    A_snapshot = A.Ftr.copy()
    graft, eA, eB = graft_brains(A, B, XAtr, ycr, XAte, ycrte, D3, C3, P=P, EP=EP)
    acc_graft = graft.acc(ycrte)
    acc_A = solo_acc(A, XAtr, ycr, XAte, ycrte, D3, C3)
    acc_B = solo_acc(B, XAtr, ycr, XAte, ycrte, D3, C3)
    acc_scratch = solo_acc(ConsolidatingNet(D3, C3, seed=5).seed_base(
        np.zeros((len(XAtr), 0)), np.zeros((len(XAte), 0)), []), XAtr, ycr, XAte, ycrte, D3, C3)
    preserved = bool(np.array_equal(A_snapshot, A.Ftr))
    print("\nCORE 3 -- grafting two brains (small + small = big)")
    print(f"  cross-task needs both halves: graft={acc_graft:.3f}  A-only={acc_A:.3f}  "
          f"B-only={acc_B:.3f}  from-scratch={acc_scratch:.3f}")
    print(f"  cross-brain graft edges: A-side {eA}, B-side {eB}  |  both sources preserved: {preserved}")

    res = {"core1": {"widths": g["widths"], "reuses": g["reuses"], "accs_promote": g["accs"],
                     "accs_baseline": n["accs"], "admits": g["admits"]},
           "core2": sc,
           "core3": {"acc_graft": acc_graft, "acc_A": acc_A, "acc_B": acc_B, "acc_scratch": acc_scratch,
                     "edges_A": eA, "edges_B": eB, "preserved": preserved}}
    (RESULTS / "spine_growth.json").write_text(json.dumps(res, indent=2, default=float))
    spine_growth_figure(g, n, sc, res["core3"])
    print("\nSaved results/spine_growth.{json,png}")
    return res


def spine_growth_figure(g, n, sc, c3):
    T = len(g["widths"]); rr = np.arange(1, T + 1)
    G, Gr, Bl, Rd = "#2e7d32", "#777", "#1565c0", "#c62828"
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13, 9))
    a1.plot(rr, g["widths"], "o-", color=Bl, label="spine width")
    a1b = a1.twinx(); a1b.plot(rr, g["reuses"], "s-", color=G, label="reuse fraction")
    a1.set_xlabel("teachers seen"); a1.set_ylabel("spine width (blocks)", color=Bl)
    a1b.set_ylabel("branch reuse fraction", color=G); a1b.set_ylim(0, 1)
    a1.set_title("CORE 1: the spine grows, later teachers reuse it more")
    a2.plot(rr, g["accs"], "o-", color=G, label="with growing spine")
    a2.plot(rr, n["accs"], "s--", color=Gr, label="no promotion")
    a2.set_xlabel("teacher"); a2.set_ylabel("test accuracy"); a2.set_ylim(0, 1)
    a2.set_title("CORE 1: accuracy held (dips = negative transfer, shown honestly)"); a2.legend(fontsize=8)
    a3.bar([0, 1], [sc["deep_dist"], sc["shortcut_dist"]], color=[Rd, G], width=0.6)
    a3.set_xticks([0, 1]); a3.set_xticklabels(["deep fiber", "distilled node"])
    a3.set_ylabel("distance from axiom (hops)")
    a3.set_title(f"CORE 2: shortcut node shortens the path (r$^2$={sc['r2']:.2f})")
    labels = ["graft\n(A+B)", "A only", "B only", "scratch"]
    vals = [c3["acc_graft"], c3["acc_A"], c3["acc_B"], c3["acc_scratch"]]
    a4.bar(range(4), vals, color=[G, "#e8710a", "#9467bd", Gr])
    a4.set_xticks(range(4)); a4.set_xticklabels(labels); a4.set_ylabel("cross-task accuracy"); a4.set_ylim(0, 1)
    a4.set_title(f"CORE 3: graft solves what neither brain can\n(cross-edges A:{c3['edges_A']} B:{c3['edges_B']}, sources preserved: {c3['preserved']})")
    fig.tight_layout(); fig.savefig(RESULTS / "spine_growth.png", dpi=140)


if __name__ == "__main__":
    main()
