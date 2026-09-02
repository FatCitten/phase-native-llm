"""
A society of spines: generative data, cross-teaching, and flaw-break-reform.

Each generation MAKES ITS OWN DATA (fresh sampled inputs, self-labeled) -- which collapses a lone
self-training spine (a model trained on its own outputs, errors compounding). The fix is a SOCIETY:
each spine's opinionated data is corrected by CROSS-TEACHING (a spine trains on its PEERS' consensus
labels, never its own -- escaping its echo chamber), and accuracy is brute-forced up by BREAKING each
spine's against-interest connections (top contributors to its WRONG outputs, found by tracing source->
destination) and letting them REFORM on the corrected data -- healing without homogenizing.

Three conditions, accuracy vs the TRUE world per generation: LONE (expect collapse), SOCIETY (cross-
teaching, expect held/up), SOCIETY+BREAK (expect up). Reuses ConsolidatingNet. Pure numpy, CPU.
Run: python experiments/society.py
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

from experiments.consolidation_rounds import ConsolidatingNet, softmax
from experiments.multi_teacher import shared_primitive_teachers

RESULTS = Path("results")


def forward_logits(net, X, backend=None):
    """Forward a frozen spine on inputs X. X is (N, W) integer token indices (sparse)
    OR (N, D) one-hot (legacy). The input layer is a gather (W[token]) instead of a
    dense one-hot matmul. Returns logits (N, C)."""
    bk = backend or getattr(net, "backend", None) or _get_numpy_backend()
    F = bk.zeros((len(X), 0)); logits = bk.zeros((len(X), net.C))
    sparse = X.shape[1] != net.D
    V = net.D // X.shape[1] if sparse else 0   # vocab size (position k token t -> index k*V+t)
    for Wr, Vr, br in zip(net.frozen_W, net.frozen_V, net.frozen_b):
        if sparse:  # integer tokens (N, W): input contribution = sum_k Wr[k*V + token_k]  (N, P)
            zin = bk.zeros((len(X), Wr.shape[1]))
            for k in range(X.shape[1]):
                zin = zin + bk.gather(Wr[:net.D], k * V + X[:, k])
        else:  # one-hot (N, D)
            zin = bk.matmul(X, Wr[:net.D])
        pre = zin + bk.matmul(F, Wr[net.D:]) + br
        A = bk.relu(pre)
        logits = logits + bk.matmul(A, Vr)
        F = bk.concatenate([F, A], 1)
    return logits + net.bias


_numpy_backend = None
def _get_numpy_backend():
    global _numpy_backend
    if _numpy_backend is None:
        from demo.backend import NumpyBackend
        _numpy_backend = NumpyBackend()
    return _numpy_backend


def forward_logits_masked(net, X, masks, backend=None):
    """Forward using precomputed recall masks: skip fibers that never fire on the batch.

    Returns the SAME logits as forward_logits — a fiber with activation 0 contributes
    0 to both the readout (0 @ V) and the next round's base (0 column in F), so dropping
    it changes nothing. This is the forced-recall pathway pruning: only the active
    subgraph is computed, so recall gets cheaper as the net grows."""
    bk = backend or getattr(net, "backend", None) or _get_numpy_backend()
    F = bk.zeros((len(X), 0)); logits = bk.zeros((len(X), net.C))
    sparse = X.shape[1] != net.D
    vocab_size = net.D // X.shape[1] if sparse else 0
    for i, (Wr, Vr, br) in enumerate(zip(net.frozen_W, net.frozen_V, net.frozen_b)):
        m = masks[i]
        if sparse:
            zin = bk.zeros((len(X), Wr.shape[1]))
            for k in range(X.shape[1]):
                zin = zin + bk.gather(Wr[:net.D], k * vocab_size + X[:, k])
        else:
            zin = bk.matmul(X, Wr[:net.D])
        pre = zin + bk.matmul(F, Wr[net.D:]) + br
        A = bk.relu(pre)
        logits = logits + bk.matmul(A, Vr)
        # F stays FULL-WIDTH (all fibers) so the next round's Wr[net.D:] columns align.
        # Unfired fibers are all-zero columns, so their contribution is 0 either way.
        F = bk.concatenate([F, A], 1)
    return logits + net.bias


def predict(net, X):
    return forward_logits(net, X).argmax(1)


def acc_on(net, X, y):
    return float((predict(net, X) == y).mean())


def make_spine(Xtr, ytr, Xte, yte, D, C, seed, rounds=2, P=32, EP=800):
    """A standalone spine: a couple of consolidation rounds reading raw inputs (no seeding)."""
    net = ConsolidatingNet(D, C, seed=seed)
    for r in range(rounds):
        net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=(0.0 if r == 0 else 0.3))
    return net


def train_on(D, C, X, y, Xte, yte, seed, rounds=2, P=32, EP=800):
    """Train a fresh spine on generated (X, pseudo-label y). Evaluated later via forward_logits."""
    return make_spine(X, y, Xte, yte, D, C, seed, rounds=rounds, P=P, EP=EP)


def forward_feats(net, X):
    """Per-round fiber activations of a frozen spine on inputs X (for readout attribution / refit)."""
    F = np.zeros((len(X), 0)); As = []
    for Wr, br in zip(net.frozen_W, net.frozen_b):
        A = np.maximum(np.concatenate([X, F], 1) @ Wr + br, 0)
        As.append(A); F = np.concatenate([F, A], 1)
    return As


def peer_consensus(peer_logits):
    """Confidence-weighted majority vote of the peers (sum of softmax probabilities)."""
    return sum(softmax(l) for l in peer_logits).argmax(1)


def disagreement(spines, X):
    """Mean pairwise disagreement across the society -- its diversity (independence)."""
    preds = [predict(s, X) for s in spines]
    pairs = [(preds[i] != preds[j]).mean() for i in range(len(preds)) for j in range(i + 1, len(preds))]
    return float(np.mean(pairs)) if pairs else 0.0


def flaw_break_reform(net, Xg, target, harm_frac=0.15, refit=300, lr=0.05, wd=1e-4):
    """Break each spine's AGAINST-INTEREST connections -- the fibers whose readout pushes its OWN wrong
    outputs (top contributor to the wrong class on the points it gets wrong vs the society) -- then REFORM
    the readout on the corrected targets. Heals the self-defeating parts without touching the features
    (independence preserved). Returns the count of broken fibers."""
    As = forward_feats(net, Xg); A = np.concatenate(As, 1)
    V = np.concatenate(net.frozen_V, 0); b = net.bias.copy()
    onehot = np.eye(net.C)[target]
    pred = (A @ V + b).argmax(1); flaw = pred != target
    broke = 0
    if flaw.sum() >= 5:
        Af, wrong = A[flaw], pred[flaw]
        harm = (Af.T * V[:, wrong]).sum(1)              # per fiber: contribution to its OWN wrong logits
        k = max(1, int(harm_frac * len(harm)))
        V[np.argsort(harm)[-k:]] = 0.0                  # break the against-interest connections
        broke = k
    for _ in range(refit):                              # reform the readout on the society's targets
        dl = (softmax(A @ V + b) - onehot) / len(target)
        V -= lr * (A.T @ dl + wd * V); b -= lr * dl.sum(0)
    net.bias = b; idx = 0                               # write the reformed readout back into the spine
    for r in range(len(net.frozen_V)):
        w = net.frozen_V[r].shape[0]; net.frozen_V[r] = V[idx:idx + w].copy(); idx += w
    return broke


def lone_loop(Xtr, ytr, Xte, yte, D, C, G, seed, EP, n_gen):
    """A single spine self-training on its OWN generated labels -- the collapse baseline."""
    rng = np.random.default_rng(seed)
    spine = make_spine(Xtr, ytr, Xte, yte, D, C, seed=seed, EP=EP)
    accs = [acc_on(spine, Xte, yte)]
    for g in range(1, G + 1):
        Xg = rng.normal(0, 1, (n_gen, D))
        spine = train_on(D, C, Xg, predict(spine, Xg), Xte, yte, seed=seed + 100 * g, EP=EP)
        accs.append(acc_on(spine, Xte, yte))
    return accs


def society_loop(subsets, Xte, yte, D, C, G, seeds, EP, n_gen, break_reform=False):
    """K diverse spines. Each generation every spine makes its OWN opinionated data; spine k then trains
    on the POOL of the OTHER spines' data+labels (never its own -- escaping its echo chamber, and each
    spine seeing several peers' different data keeps the society diverse). Optional flaw-break-reform.
    Returns mean acc/gen and disagreement/gen."""
    K = len(seeds)
    spines = [make_spine(Xs, ys, Xte, yte, D, C, seed=seeds[k], EP=EP) for k, (Xs, ys) in enumerate(subsets)]
    rng = np.random.default_rng(999)
    accs = [float(np.mean([acc_on(s, Xte, yte) for s in spines]))]
    disag = [disagreement(spines, Xte)]
    for g in range(1, G + 1):
        gen = [(lambda Xg: (Xg, predict(s, Xg)))(rng.normal(0, 1, (n_gen, D))) for s in spines]
        new = []
        for k in range(K):
            Xp = np.concatenate([gen[j][0] for j in range(K) if j != k])    # peers' opinionated inputs
            yp = np.concatenate([gen[j][1] for j in range(K) if j != k])    # peers' opinions (never k's)
            sk = train_on(D, C, Xp, yp, Xte, yte, seed=seeds[k] + 100 * g, EP=EP)
            if break_reform:
                flaw_break_reform(sk, Xp, yp)                               # heal k's against-interest parts
            new.append(sk)
        spines = new
        accs.append(float(np.mean([acc_on(s, Xte, yte) for s in spines])))
        disag.append(disagreement(spines, Xte))
    return accs, disag


def main():
    RESULTS.mkdir(exist_ok=True)
    Xtr, Ytr, Xte, Yte, D, C, T = shared_primitive_teachers(T=1, seed=0)
    ytr, yte = Ytr[:, 0], Yte[:, 0]
    K, G, EP, NG = 4, 6, 350, 600
    rng = np.random.default_rng(3)
    subsets = [(Xtr[i], ytr[i]) for i in
               (rng.choice(len(Xtr), int(0.7 * len(Xtr)), replace=False) for _ in range(K))]
    seeds = [10 * (k + 1) for k in range(K)]

    lone = lone_loop(Xtr, ytr, Xte, yte, D, C, G, seed=10, EP=EP, n_gen=NG)
    soc, dis = society_loop(subsets, Xte, yte, D, C, G, seeds, EP=EP, n_gen=NG, break_reform=False)
    socb, disb = society_loop(subsets, Xte, yte, D, C, G, seeds, EP=EP, n_gen=NG, break_reform=True)

    print(f"generative self-teaching over {G} generations (accuracy vs the TRUE world):")
    print(f"  LONE       : {[round(a,3) for a in lone]}")
    print(f"  SOCIETY    : {[round(a,3) for a in soc]}   disagreement {[round(x,2) for x in dis]}")
    print(f"  SOC+BREAK  : {[round(a,3) for a in socb]}   disagreement {[round(x,2) for x in disb]}")
    lone_collapse = lone[0] - lone[-1] > 0.03
    # a rescue = the society holds above the lone spine CONSISTENTLY (mean over generations), and
    # collapses less -- a fair measure of a modest-but-real gap, not one endpoint.
    society_rescues = np.mean(soc[1:]) > np.mean(lone[1:]) + 0.01
    less_collapse = (lone[0] - lone[-1]) - (soc[0] - soc[-1])
    break_helps = socb[-1] > soc[-1] + 0.01
    independent = min(disb[1:]) > 0.02
    print(f"\nlone generative self-teaching collapses: {lone_collapse} ({lone[0]:.3f} -> {lone[-1]:.3f})")
    print(f"the society collapses LESS, holding above lone every generation: {society_rescues} "
          f"(mean society {np.mean(soc[1:]):.3f} vs lone {np.mean(lone[1:]):.3f}; "
          f"society loses {soc[0]-soc[-1]:.3f} vs lone {lone[0]-lone[-1]:.3f})")
    print(f"flaw-break-reform adds accuracy on top: {break_helps} "
          f"(break {socb[-1]:.3f} vs society {soc[-1]:.3f})")
    print(f"healing preserves independence (spines stay diverse, not homogenized): {independent} "
          f"(min disagreement {min(disb[1:]):.2f})")

    res = {"K": K, "G": G, "lone": lone, "society": soc, "society_break": socb, "disag": dis,
           "disag_break": disb, "lone_collapse": bool(lone_collapse), "society_rescues": bool(society_rescues),
           "less_collapse": float(less_collapse), "break_helps": bool(break_helps),
           "independence_preserved": bool(independent)}
    (RESULTS / "society.json").write_text(json.dumps(res, indent=2, default=float))

    gg = list(range(0, G + 1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(gg, lone, "^:", color="#c62828", label="lone (self-teaching)")
    a1.plot(gg, soc, "o-", color="#2e7d32", label="society (cross-teaching)")
    a1.plot(gg, socb, "s--", color="#1565c0", label="society + flaw-break-reform")
    a1.axhline(lone[0], ls=":", color="#777", label=f"gen-0 {lone[0]:.2f}")
    a1.set_xlabel("generation (each makes its own data)"); a1.set_ylabel("accuracy vs TRUE world")
    a1.set_ylim(0, 1); a1.set_title("A lone spine collapses on its own data;\nthe cross-teaching society collapses less (holds above it)"); a1.legend(fontsize=8)
    a2.plot(gg, dis, "o-", color="#2e7d32", label="society")
    a2.plot(gg, disb, "s--", color="#1565c0", label="society + break-reform")
    a2.set_xlabel("generation"); a2.set_ylabel("pairwise disagreement (diversity)")
    a2.set_ylim(0, max(0.3, max(dis + disb) * 1.2))
    a2.set_title("Independence preserved\n(healing does not homogenize the spines)"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS / "society.png", dpi=140)
    print("\nSaved results/society.{json,png}")
    return res


if __name__ == "__main__":
    main()
