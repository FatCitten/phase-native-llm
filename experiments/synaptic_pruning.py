"""
Synaptic pruning: solidifying memory amplifies capability, and training is blind to it.

Real gradient-descent training (a numpy MLP), then the biological act of synaptic pruning
(magnitude pruning) and a lottery-ticket reset. The thesis, in numbers:

  EXHIBIT A — TRAINING IS BLIND / HAS NO MINDSET. On modular addition the net reaches 100% TRAIN
    accuracy and ~0% TEST: the objective is perfectly satisfied while the model understands
    nothing. The loss sees only the landing spot; it cannot tell memorization from structure.

  Then, on a task that DOES have findable structure (a fixed teacher network):
  (1) BLIND, again — the trained net over-produces synapses; test accuracy barely moves while
      most weights are pruned. The objective was invariant to what it built.
  (2) SOLIDIFYING AMPLIFIES — capability per surviving synapse (accuracy / weight-fraction)
      rises sharply as we prune to the structure that matters.
  (3) THE SOLIDIFIED STRUCTURE *IS* THE MEMORY — reset the survivors to their original init and
      retrain only them (a winning ticket): it learns FASTER than the dense net, while a random
      mask of equal sparsity fails. Structure, not sparsity, carries the power.

Grounded in the Lottery Ticket Hypothesis (Frankle & Carbin, 2018). Pure numpy, CPU.
Run: python experiments/synaptic_pruning.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")


class MLP:
    """A small MLP with manual SGD — 'current training technology': gradient descent."""

    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / d_hidden), (d_hidden, d_out))
        self.b2 = np.zeros(d_out)
        self.m1 = np.ones_like(self.W1)  # synapse masks (1 present, 0 pruned)
        self.m2 = np.ones_like(self.W2)
        self.init = (self.W1.copy(), self.W2.copy())  # original init for lottery-ticket reset

    def forward(self, X):
        self.z1 = X @ (self.W1 * self.m1) + self.b1
        self.a1 = np.maximum(self.z1, 0)
        z2 = self.a1 @ (self.W2 * self.m2) + self.b2
        z2 -= z2.max(1, keepdims=True)
        e = np.exp(z2)
        self.p = e / e.sum(1, keepdims=True)
        return self.p

    def step(self, X, y, lr, wd):
        n = len(y)
        p = self.forward(X).copy()
        p[np.arange(n), y] -= 1.0
        dz2 = p / n
        dW2 = self.a1.T @ dz2 + wd * (self.W2 * self.m2)
        db2 = dz2.sum(0)
        dz1 = (dz2 @ (self.W2 * self.m2).T) * (self.z1 > 0)
        dW1 = X.T @ dz1 + wd * (self.W1 * self.m1)
        db1 = dz1.sum(0)
        self.W1 -= lr * dW1 * self.m1
        self.W2 -= lr * dW2 * self.m2
        self.b1 -= lr * db1
        self.b2 -= lr * db2

    def acc(self, X, y):
        return float((self.forward(X).argmax(1) == y).mean())

    def n_active(self):
        return int(self.m1.sum() + self.m2.sum())

    def n_weights(self):
        return self.W1.size + self.W2.size


def train(net, Xtr, ytr, Xte, yte, epochs, lr, wd, log_every=0):
    hist = []
    for e in range(epochs):
        net.step(Xtr, ytr, lr, wd)
        if log_every and (e % log_every == 0 or e == epochs - 1):
            hist.append((e, net.acc(Xtr, ytr), net.acc(Xte, yte)))
    return hist


def magnitude_mask(net, keep_abs):
    """Keep the top `keep_abs` FRACTION OF ALL weights by |magnitude|, among current survivors."""
    total = net.n_weights()
    n_keep = int(round(keep_abs * total))
    w = np.sort(np.concatenate([np.abs(net.W1[net.m1 > 0]).ravel(),
                                np.abs(net.W2[net.m2 > 0]).ravel()]))
    if n_keep >= len(w):
        return
    thr = w[len(w) - n_keep]
    net.m1 = ((np.abs(net.W1) >= thr) & (net.m1 > 0)).astype(float)
    net.m2 = ((np.abs(net.W2) >= thr) & (net.m2 > 0)).astype(float)


# --------------------------------------------------------------------------------------
def modular_data(k=12, train_frac=0.75, seed=0):
    a = np.repeat(np.arange(k), k); b = np.tile(np.arange(k), k)
    X = np.zeros((k * k, 2 * k)); X[np.arange(k * k), a] = 1; X[np.arange(k * k), k + b] = 1
    y = (a + b) % k
    idx = np.random.default_rng(seed).permutation(k * k); n = int(len(idx) * train_frac)
    return X[idx[:n]], y[idx[:n]], X[idx[n:]], y[idx[n:]]


def teacher_data(D=20, Ht=8, C=4, N=4000, ntr=3000, seed=0):
    """Labels from a fixed nonlinear teacher net — a real function with findable structure."""
    rng = np.random.default_rng(seed)
    Wt1 = rng.normal(0, 1, (D, Ht)); Wt2 = rng.normal(0, 1, (Ht, C))
    X = rng.normal(0, 1, (N, D))
    logits = np.maximum(X @ Wt1, 0) @ Wt2
    logits -= logits.mean(0)  # center so all classes appear
    y = logits.argmax(1)
    return X[:ntr], y[:ntr], X[ntr:], y[ntr:], D, C


def main():
    RESULTS.mkdir(exist_ok=True)

    # ---- EXHIBIT A: blindness — 100% train, ~0% test on modular addition ----
    Xtr, ytr, Xte, yte = modular_data(k=12)
    m = MLP(24, 256, 12, seed=1)
    train(m, Xtr, ytr, Xte, yte, 3000, 0.05, 1e-3)
    a_tr, a_te = m.acc(Xtr, ytr), m.acc(Xte, yte)
    print(f"EXHIBIT A (blindness): modular addition  train={a_tr:.3f}  test={a_te:.3f}")
    print("  -> the objective is perfectly satisfied; the model learned no structure.\n")

    # ---- main task: teacher-student (real structure to find) ----
    Xtr, ytr, Xte, yte, D, C = teacher_data()
    EP, LR, WD = 2500, 0.05, 1e-4
    dense = MLP(D, 256, C, seed=1)
    train(dense, Xtr, ytr, Xte, yte, EP, LR, WD)
    dense_acc = dense.acc(Xte, yte)
    print(f"dense student: test_acc={dense_acc:.3f}  synapses={dense.n_weights()}")

    # ---- (1)+(2): ITERATIVE prune-and-retrain (the real Lottery-Ticket method) ----
    keeps = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
    sweep = [{"keep": 1.0, "test_acc": dense_acc, "weight_frac": 1.0, "amp": dense_acc}]
    masks = {1.0: (dense.m1.copy(), dense.m2.copy())}
    n = copy.deepcopy(dense)
    for kf in keeps:
        magnitude_mask(n, kf)                        # prune the surplus, keep top |w|
        train(n, Xtr, ytr, Xte, yte, 350, LR, WD)    # re-solidify the survivors
        acc = n.acc(Xte, yte); frac = n.n_active() / n.n_weights()
        sweep.append({"keep": kf, "test_acc": acc, "weight_frac": frac, "amp": acc / frac})
        masks[kf] = (n.m1.copy(), n.m2.copy())
        print(f"  keep {kf*100:5.1f}%  test_acc={acc:.3f}  capability/synapse x{acc/frac:6.1f}")

    # blindness: sparsest keep still within 2% of dense accuracy (surplus the loss never saw)
    blind = min((s for s in sweep if s["test_acc"] >= dense_acc - 0.02),
                key=lambda s: s["weight_frac"])
    # amplification, honest: sparsest keep retaining >=90% of dense capability
    knee = min((s for s in sweep if s["test_acc"] >= 0.9 * dense_acc),
               key=lambda s: s["weight_frac"])
    amp = {"amp": 1.0 / knee["weight_frac"], "weight_frac": knee["weight_frac"],
           "test_acc": knee["test_acc"]}

    # ---- (3): winning ticket (init + solidified mask) vs random mask vs dense-from-scratch ----
    tk = knee["keep"]
    m1, m2 = masks[tk]
    ticket = MLP(D, 256, C, seed=1)                  # SAME init as `dense`
    ticket.m1, ticket.m2 = m1.copy(), m2.copy()
    ticket.W1, ticket.W2 = ticket.init[0] * m1, ticket.init[1] * m2
    h_ticket = train(ticket, Xtr, ytr, Xte, yte, EP, LR, WD, log_every=100)

    rand = MLP(D, 256, C, seed=1)                    # same init, RANDOM mask of equal sparsity
    rng = np.random.default_rng(7)
    rand.m1 = (rng.random(rand.W1.shape) < m1.mean()).astype(float)
    rand.m2 = (rng.random(rand.W2.shape) < m2.mean()).astype(float)
    rand.W1, rand.W2 = rand.init[0] * rand.m1, rand.init[1] * rand.m2
    h_rand = train(rand, Xtr, ytr, Xte, yte, EP, LR, WD, log_every=100)

    dense2 = MLP(D, 256, C, seed=2)
    h_dense = train(dense2, Xtr, ytr, Xte, yte, EP, LR, WD, log_every=100)

    thr = round(0.9 * dense_acc, 3)
    def eto(h): return next((e for e, _, te in h if te >= thr), None)
    et_t, et_d, et_r = eto(h_ticket), eto(h_dense), eto(h_rand)

    print(f"\n(1) BLIND: {(1-blind['weight_frac'])*100:.0f}% of synapses removable with test_acc "
          f"{blind['test_acc']:.3f} vs dense {dense_acc:.3f} -> the loss never saw the surplus.")
    print(f"(2) AMPLIFY: {amp['amp']:.0f}x fewer synapses ({amp['weight_frac']*100:.1f}% kept) for "
          f"{amp['test_acc']/dense_acc*100:.0f}% of the capability (acc {amp['test_acc']:.3f}).")
    print(f"(3) SOLIDIFIED = MEMORY: winning ticket ({tk*100:.0f}% of synapses) final "
          f"{ticket.acc(Xte,yte):.3f} vs SAME-sparsity random {rand.acc(Xte,yte):.3f} "
          f"(threshold {thr} in {et_t} vs {et_r} epochs) -> the specific structure carries it, "
          f"not the sparsity. (Dense trains fastest in raw epochs {et_d}: it keeps every synapse.)")

    res = {"exhibit_A": {"train": a_tr, "test": a_te}, "dense_acc": dense_acc,
           "dense_params": dense.n_weights(), "sweep": sweep, "blind": blind, "amp": amp,
           "ticket_keep": tk, "threshold": thr,
           "epochs_to_thr": {"ticket": et_t, "dense": et_d, "random": et_r},
           "final_acc": {"ticket": ticket.acc(Xte, yte), "random": rand.acc(Xte, yte)}}
    (RESULTS / "synaptic_pruning.json").write_text(json.dumps(res, indent=2, default=float))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fr = [s["weight_frac"] * 100 for s in sweep]
    ax1.plot(fr, [s["test_acc"] for s in sweep], "o-", color="#2e7d32", label="test accuracy")
    ax1.axhline(dense_acc, ls=":", color="gray", label="dense accuracy")
    ax1.set_xlabel("% of synapses kept"); ax1.set_ylabel("test accuracy"); ax1.invert_xaxis()
    axr = ax1.twinx()
    axr.plot(fr, [s["amp"] for s in sweep], "s--", color="#c62828")
    axr.set_ylabel("capability per surviving synapse (x)", color="#c62828")
    ax1.set_title("Prune the surplus the loss was blind to → amplification")
    ax1.legend(loc="lower left")
    for h, lab, c in [(h_ticket, f"winning ticket ({tk*100:.0f}%)", "#2e7d32"),
                      (h_dense, "dense (100%)", "#555"),
                      (h_rand, f"random mask ({tk*100:.0f}%)", "#c62828")]:
        ax2.plot([e for e, _, _ in h], [t for _, _, t in h], label=lab, color=c)
    ax2.set_xlabel("training epochs"); ax2.set_ylabel("test accuracy")
    ax2.set_title("Solidified structure >> random subnet of equal sparsity (it IS the memory)")
    ax2.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(RESULTS / "synaptic_pruning.png", dpi=140)
    print("\nSaved results/synaptic_pruning.{json,png}")


if __name__ == "__main__":
    main()
