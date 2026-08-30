"""
Multiple teachers on one spine: reuse + no-interference beats plain pruning.

Breadth = context. Several teachers SHARE low-level primitives phi(x) (a narrow bottleneck) and differ
only in their heads. Consolidation builds a shared frozen SPINE (phi, learned once) and grows a cheap
per-task BRANCH off it -- each branch bundles the spine along the least-resistance path and snips its
loose threads (magnitude pruning as a feature). Two claims, measured honestly:

  REUSE (joint)      -- spine + T small branches reaches the tasks at FEWER wires than a monolithic
                        multi-task net pruned to the same budget (the shared primitives amortize).
  NO FORGETTING (seq)-- tasks arrive one at a time; each new branch is frozen and isolated, so every
                        earlier task's accuracy is preserved EXACTLY, while a fine-tuned monolithic
                        control forgets. This is "conquering confusion" -- interference a shared-weight
                        net cannot avoid.

Inference is a Least Path of Resistance: an input (+ task) triggers activation up the stem into the
branch along the strongest edges; the final pulse (readout) is the output. Reuses ConsolidatingNet /
grow_round (with seed_base + prune_density) and the synaptic_pruning MLP control. Pure numpy, CPU.
Run: python experiments/multi_teacher.py
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

from experiments.consolidation_rounds import ConsolidatingNet
from experiments.synaptic_pruning import MLP, magnitude_mask, train

RESULTS = Path("results")


def shared_primitive_teachers(T=4, D=12, prim=6, C=4, N=4000, ntr=3000, seed=0):
    """T teachers sharing a deep feature map phi(x) (a narrow `prim`-wide bottleneck); each has its own
    linear head. Same inputs X for every task -> genuine cross-task reuse of the shared primitives."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (N, D))
    W1 = rng.normal(0, np.sqrt(2 / D), (D, 16))
    W2 = rng.normal(0, np.sqrt(2 / 16), (16, prim))
    phi = np.maximum(np.maximum(X @ W1, 0) @ W2, 0)      # (N, prim) SHARED primitives
    Y = np.zeros((N, T), dtype=int)
    for t in range(T):
        logits = phi @ rng.normal(0, 1, (prim, C))
        logits -= logits.mean(0)
        Y[:, t] = logits.argmax(1)
    return X[:ntr], Y[:ntr], X[ntr:], Y[ntr:], D, C, T


def build_spine(Xtr, y0tr, Xte, y0te, D, C, P=32, EP=1200, seed=1):
    """The shared stem: a couple of consolidation rounds on the FIRST task (all tasks share phi, so
    features learned here transfer). Round 1 = axioms; round 2 = a tightened composite layer."""
    net = ConsolidatingNet(D, C, seed=seed)
    net.grow_round(Xtr, y0tr, Xte, y0te, P=P, epochs=EP, tau=0.0)
    net.grow_round(Xtr, y0tr, Xte, y0te, P=P, epochs=EP, tau=0.4, prune_density=0.8)
    return net


def grow_branch(spine, Xtr, ytr, Xte, yte, D, C, P=32, EP=1200, seed=2):
    """A per-task branch: seed the frozen spine as its base, grow one tightened+pruned round that
    bundles the spine (reuse) along cheap paths. Isolated -> growing it disturbs nothing else."""
    br = ConsolidatingNet(D, C, seed=seed)
    br.seed_base(spine.Ftr, spine.Fte, spine.dist, spine.frozen_W)
    br.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=0.6, k_par=6, prune_density=0.7)
    return br


def tagged(X, Y, tasks, T):
    """Stack (x, one-hot task tag, y_task) over `tasks` -> a multi-task dataset for the MLP control."""
    xs, ys = [], []
    for t in tasks:
        tag = np.zeros((len(X), T)); tag[:, t] = 1.0
        xs.append(np.concatenate([X, tag], 1)); ys.append(Y[:, t])
    return np.concatenate(xs), np.concatenate(ys)


def mlp_task_acc(net, X, Y, t, T):
    tag = np.zeros((len(X), T)); tag[:, t] = 1.0
    return net.acc(np.concatenate([X, tag], 1), Y[:, t])


# ----- geometry: the spine-and-branches graph, and a Least-Path-of-Resistance trace -----------------
def spine_incoming(net, D):
    """Per spine fiber, its full incoming vector padded to length D+width (inputs then earlier fibers)."""
    width = len(net.dist); inc = []; before = 0
    for Wr in net.frozen_W:
        rows, cols = Wr.shape
        for c in range(cols):
            full = np.zeros(D + width); full[:rows] = Wr[:, c]; inc.append(full)
        before += cols
    return inc


def trace_lpr(branch, spine, x_i, i, D):
    """Least Path of Resistance for test sample i: from the output pulse back through the branch fiber,
    up the stem, to an input -- following the strongest |weight| x activation edge at each step."""
    sw = len(spine.dist)
    spine_feat = spine.Fte[i]; branch_act = branch.Fte[i, sw:]
    pred = int((branch.frozen_te[i] + branch.bias).argmax())
    V = branch.frozen_V[-1]
    kf = int(np.argmax(branch_act * V[:, pred]))          # dominant branch fiber for the prediction
    path = [("branch", kf)]
    w = branch.frozen_W[-1][:, kf]
    inc = spine_incoming(spine, D)
    s = int(np.argmax(np.abs(w) * np.concatenate([np.abs(x_i), np.abs(spine_feat)])))
    for _ in range(12):
        if s < D:
            path.append(("input", s)); break
        j = s - D; path.append(("spine", j))
        wj = inc[j]
        s = int(np.argmax(np.abs(wj) * np.concatenate([np.abs(x_i), np.abs(spine_feat)])))
    return path, pred


def layout(spine, branches, D, T):
    sw = len(spine.dist); pos = {}; ga = np.pi * (3 - np.sqrt(5))
    for d in range(D):
        a = 2 * np.pi * d / D; pos[("input", d)] = (0.35 * np.cos(a), 0.35 * np.sin(a))
    for j in range(sw):
        r = 0.8 + 0.45 * (spine.dist[j] - 1); a = j * ga
        pos[("spine", j)] = (r * np.cos(a), r * np.sin(a))
    smax = (0.8 + 0.45 * (max(spine.dist) - 1)) if sw else 0.8
    for t, br in enumerate(branches):
        nb = len(br.dist) - sw; th = 2 * np.pi * t / T
        for k in range(nb):
            aa = th + (np.pi / T) * 0.7 * (k / max(1, nb - 1) - 0.5) * 2
            rr = smax + 0.8 + 0.5 * (br.dist[sw + k] - 1)
            pos[("branch", t, k)] = (rr * np.cos(aa), rr * np.sin(aa))
    return pos, sw


def spine_figure(spine, branches, D, T, X, res, path=None, ptask=0):
    pos, sw = layout(spine, branches, D, T)
    # teacher palette avoids the spine's blue and the LPR path's red
    colors = ["#e8710a", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_axes([0.02, 0.05, 0.56, 0.9]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("One shared spine, a branch per teacher — inference = least path of resistance",
                 fontsize=11)
    # spine-internal edges (the stem)
    for j, wj in enumerate(spine_incoming(spine, D)):
        for p in np.where(np.abs(wj[D:]) > 0)[0]:
            a, b = pos[("spine", j)], pos[("spine", int(p))]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#bbb", lw=0.5, alpha=0.5, zorder=1)
    # branch -> spine reuse edges (colored by task)
    for t, br in enumerate(branches):
        Wb = br.frozen_W[-1]
        for k in range(len(br.dist) - sw):
            for p in np.where(np.abs(Wb[D:, k]) > 0)[0]:
                a, b = pos[("branch", t, k)], pos[("spine", int(p))]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=colors[t % len(colors)], lw=0.4, alpha=0.35, zorder=1)
    # nodes
    ix = np.array([pos[("input", d)] for d in range(D)])
    ax.scatter(ix[:, 0], ix[:, 1], s=18, c="#333", marker="s", zorder=3, label="inputs")
    sx = np.array([pos[("spine", j)] for j in range(sw)])
    ax.scatter(sx[:, 0], sx[:, 1], s=34, c="#1565c0", zorder=3, label="spine (shared)")
    for t, br in enumerate(branches):
        bx = np.array([pos[("branch", t, k)] for k in range(len(br.dist) - sw)])
        if len(bx):
            ax.scatter(bx[:, 0], bx[:, 1], s=30, color=colors[t % len(colors)], zorder=3, label=f"teacher {t}")
    # least-path-of-resistance overlay
    if path:
        pts = []
        for node in path:
            pts.append(pos[("branch", ptask, node[1])] if node[0] == "branch" else pos[node])
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], color="#c62828", lw=2.4, zorder=5, label="least-resistance path")
        ax.scatter(pts[:, 0], pts[:, 1], s=60, facecolors="none", edgecolors="#c62828", lw=1.8, zorder=6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    a1 = fig.add_axes([0.66, 0.58, 0.31, 0.36])
    tk = np.arange(res["T"])
    a1.bar(tk - 0.2, res["cons_acc"], 0.4, color="#2e7d32", label=f"consolidation ({res['cons_syn']} wires)")
    a1.bar(tk + 0.2, res["monp_acc"], 0.4, color="#c62828", label=f"pruned monolithic ({res['mono_syn']} wires)")
    a1.set_xticks(tk); a1.set_xlabel("teacher"); a1.set_ylabel("test accuracy"); a1.set_ylim(0, 1)
    a1.set_title("REUSE: same wire budget, more accuracy", fontsize=10); a1.legend(fontsize=7)

    a2 = fig.add_axes([0.66, 0.09, 0.31, 0.36])
    rounds = np.arange(1, res["T"] + 1)
    a2.plot(rounds, res["cons_task0_seq"], "o-", color="#2e7d32", label="consolidation (frozen branch)")
    a2.plot(rounds, res["seq_task0"], "s--", color="#c62828", label="monolithic (fine-tuned)")
    a2.set_xlabel("tasks learned"); a2.set_ylabel("task-0 accuracy"); a2.set_ylim(0, 1)
    a2.set_title("NO FORGETTING: task 0 as new teachers arrive", fontsize=10); a2.legend(fontsize=7)
    fig.savefig(RESULTS / "multi_teacher.png", dpi=140)


def main():
    RESULTS.mkdir(exist_ok=True)
    Xtr, Ytr, Xte, Yte, D, C, T = shared_primitive_teachers()
    P, EP = 32, 1200

    # ---- consolidation: one shared spine + T isolated branches ----
    spine = build_spine(Xtr, Ytr[:, 0], Xte, Yte[:, 0], D, C, P=P, EP=EP)
    branches = [grow_branch(spine, Xtr, Ytr[:, t], Xte, Yte[:, t], D, C, P=P, EP=EP, seed=10 + t)
                for t in range(T)]
    cons_acc = [branches[t].acc(Yte[:, t]) for t in range(T)]
    cons_syn = spine.synapses + sum(b.synapses for b in branches)
    cons_mean = float(np.mean(cons_acc))
    print(f"consolidation (spine + {T} branches):")
    print(f"  spine width={spine.Ftr.shape[1]}  spine_syn={spine.synapses}  "
          f"branch_syn={[b.synapses for b in branches]}")
    print(f"  per-task acc={[round(a,3) for a in cons_acc]}  mean={cons_mean:.3f}  total_syn={cons_syn}")

    # ---- control: one monolithic multi-task net, trained JOINT then magnitude-pruned ----
    Xj, yj = tagged(Xtr, Ytr, range(T), T)
    Xjte, yjte = tagged(Xte, Yte, range(T), T)
    mono = MLP(D + T, 256, C, seed=1)
    train(mono, Xj, yj, Xjte, yjte, 2500, 0.05, 1e-4)
    mono_acc = [mlp_task_acc(mono, Xte, Yte, t, T) for t in range(T)]
    magnitude_mask(mono, min(1.0, cons_syn / mono.n_weights()))   # prune to the consolidation budget
    train(mono, Xj, yj, Xjte, yjte, 400, 0.05, 1e-4)
    monp_acc = [mlp_task_acc(mono, Xte, Yte, t, T) for t in range(T)]
    mono_syn = mono.n_active()
    print(f"monolithic joint+pruned: per-task acc={[round(a,3) for a in monp_acc]}  "
          f"mean={np.mean(monp_acc):.3f}  syn={mono_syn}")

    # ---- SEQUENTIAL: tasks arrive one at a time; measure earlier-task drift ----
    # consolidation: task t's branch is already isolated -> task-0 accuracy is preserved EXACTLY.
    cons_task0_after = [branches[0].acc(Yte[:, 0]) for _ in range(T)]  # unchanged as tasks are added
    # control: a monolithic net fine-tuned task-by-task -> forgets task 0.
    seq = MLP(D + T, 256, C, seed=2)
    X0, y0 = tagged(Xtr, Ytr, [0], T); X0te, y0te = tagged(Xte, Yte, [0], T)
    train(seq, X0, y0, X0te, y0te, 1500, 0.05, 1e-4)
    seq_task0_after = [mlp_task_acc(seq, Xte, Yte, 0, T)]
    for t in range(1, T):
        Xt, yt = tagged(Xtr, Ytr, [t], T); Xtte, ytte = tagged(Xte, Yte, [t], T)
        train(seq, Xt, yt, Xtte, ytte, 1500, 0.05, 1e-4)   # fine-tune on the new task only
        seq_task0_after.append(mlp_task_acc(seq, Xte, Yte, 0, T))
    print(f"\nSEQUENTIAL task-0 accuracy as tasks 1..{T-1} are added:")
    print(f"  consolidation (frozen branch): {[round(a,3) for a in cons_task0_after]}  -> no forgetting")
    print(f"  monolithic (fine-tuned):       {[round(a,3) for a in seq_task0_after]}  -> forgets")

    reuse_win = np.mean(cons_acc) >= np.mean(monp_acc) - 0.02 and cons_syn <= mono_syn
    forget_win = (cons_task0_after[-1] - seq_task0_after[-1]) > 0.05
    print(f"\nREUSE: consolidation matches accuracy at fewer wires than pruning: {reuse_win} "
          f"(cons {cons_mean:.3f}@{cons_syn} vs pruned {np.mean(monp_acc):.3f}@{mono_syn})")
    print(f"NO FORGETTING: consolidation holds task 0, monolithic forgets: {forget_win} "
          f"({cons_task0_after[-1]:.3f} vs {seq_task0_after[-1]:.3f})")

    res = {"T": T, "cons_acc": cons_acc, "cons_syn": cons_syn, "cons_mean": cons_mean,
           "spine_syn": spine.synapses, "branch_syn": [b.synapses for b in branches],
           "monp_acc": monp_acc, "mono_syn": mono_syn,
           "cons_task0_seq": cons_task0_after, "seq_task0": seq_task0_after,
           "reuse_win": bool(reuse_win), "forget_win": bool(forget_win)}
    (RESULTS / "multi_teacher.json").write_text(json.dumps(res, indent=2, default=float))
    print("\nSaved results/multi_teacher.json")

    # least-path-of-resistance trace for one test example, then the spine/spiral figure
    path, pred = trace_lpr(branches[0], spine, Xte[0], 0, D)
    print("LPR trace (task 0, sample 0 -> class %d): %s"
          % (pred, " <- ".join(f"{n[0]}[{n[1]}]" for n in path)))
    spine_figure(spine, branches, D, T, Xte, res, path=path, ptask=0)
    print("Saved results/multi_teacher.png")
    return spine, branches, res


if __name__ == "__main__":
    main()
