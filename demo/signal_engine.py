"""demo/signal_engine.py — the LLM guides, the structure decides.

The teacher (a frontier LLM) emits SIGNALS: soft probability distributions over the
next word for a batch of contexts (the "twinge"). The child grows a consolidation
round with soft=... — its own overproduce->prune->freeze dynamics decide which of
those signals STICK (become frozen fibers) and which are pruned as void.

This is the difference between "LLM forces structure" and "LLM suggests, structure
decides." The teacher guides; the child's consolidation is the arbiter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from demo import metrics, wordlm
from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits


class SignalEngine:
    """Emit teacher signals, grow the child, measure what stuck."""

    def __init__(self, net, vocab, W, X_old, y_old, X_new, y_new):
        self.net = net
        self.vocab = vocab
        self.W = W
        self.X_old, self.y_old = X_old, y_old   # "old" data (no-forgetting baseline)
        self.X_new, self.y_new = X_new, y_new   # "new" data (what we grow on)
        self.signal_log = []                    # every twinge + what stuck

    def teacher_twinge(self, client, model, contexts, top_k=20):
        """Ask the teacher for soft next-word distributions on `contexts`.
        Returns a list of (context, dist|None)."""
        from demo.foster import soft_targets_for_contexts
        return soft_targets_for_contexts(client, model, contexts, self.vocab, top_k=top_k)

    def grow_on_signals(self, X, y, P, epochs, tau=0.0, floor=0.05, soft=None):
        """Grow a consolidation round where the training target is the teacher's
        soft distribution (the twinge). The child's prune/freeze decides what sticks.
        `y` is only used for its length (the soft matrix is the real target).
        Grows on X (train) and evaluates on the engine's test set (X_new).
        Returns the round stats (kept/void_frac/cross_edges)."""
        # recompute the frozen-base activations for the NEW input sets. grow_round
        # caches Ftr/Fte from the original training; a new round on different-sized
        # data must recompute them by running the frozen weights forward.
        self._recompute_base(X, self.X_new)
        stats = self.net.grow_round(X, y, self.X_new, self.y_new, P=P, epochs=epochs,
                                    tau=tau, floor=floor, soft=soft)
        self.signal_log.append(stats)
        return stats

    def refine_on_signals(self, X, y, soft, epochs=200, lr=0.05, wd=1e-4):
        """Refine the EXISTING readout (frozen_V + bias) toward the teacher's soft
        targets. Does NOT add a new round — the frozen structure (W) stays intact,
        only the readout adjusts. This is the gentle path: it cannot add a dominating
        noise layer, so it should preserve accuracy (unlike grow_on_signals, which
        collapsed the child last run by adding a whole new round on 10 contexts).

        `soft` is (n, C) teacher probabilities; `y` is only used for its length.
        Returns {"refined": True, "epochs": epochs}."""
        net = self.net
        self._recompute_base(X, self.X_new)
        A = net.Ftr  # (n, total_fibers) activations of the frozen base
        V = np.concatenate(net.frozen_V, 0)  # (total_fibers, C)
        b = net.bias.copy()
        n = len(X)
        soft = np.asarray(soft, float)
        if soft.shape != (n, net.C):
            raise ValueError(f"soft targets shape {soft.shape} != (n={n}, C={net.C})")
        for _ in range(epochs):
            logits = A @ V + b
            lsm = logits - logits.max(1, keepdims=True)
            lsm = lsm - np.log(np.exp(lsm).sum(1, keepdims=True))
            sm = np.exp(lsm)
            dl = (sm - soft) / n
            V -= lr * (A.T @ dl + wd * V)
            b -= lr * dl.sum(0)
        # write V back into frozen_V (split by round)
        idx = 0
        for r in range(len(net.frozen_V)):
            w = net.frozen_V[r].shape[0]
            net.frozen_V[r] = V[idx:idx + w].copy()
            idx += w
        net.bias = b
        return {"refined": True, "epochs": epochs}

    def digest_round(self, X, y, keep_frac=0.5):
        """DIGEST-THEN-FORGET: after a signal solidifies, prune the non-load-bearing
        scaffolding fibers — those whose removal doesn't change the output. This is
        the efficiency driver: the child keeps only the load-bearing structure and
        discards the scaffolding that helped digest the signal but is no longer needed.

        Load-bearing score = each fiber's contribution to the CORRECT-class logits on
        samples where it fires: (A * V[:, y]).sum(0). Task-aligned — a fiber that fires
        rarely but on the hard samples is still load-bearing.

        SAFETY: only prunes the LAST round (append-only — earlier rounds are read by
        later ones, so pruning them would break the input width). The last round is
        where the most recent signal was digested anyway.

        Returns {"kept": n, "pruned": n}."""
        net = self.net
        self._recompute_base(X, self.X_new)
        A = net.Ftr  # (n, total_fibers) activations
        V = np.concatenate(net.frozen_V, 0)  # (total_fibers, C)
        # per-fiber contribution to the correct class, summed over samples:
        # correct[f, i] = V[f, y[i]] (readout of fiber f to sample i's correct class)
        y = np.asarray(y)
        correct = V[:, y]  # (total_fibers, n)
        load = (A * correct.T).sum(0)  # (total_fibers,) task-aligned load-bearing score
        last = len(net.frozen_W) - 1
        base = sum(len(w[0]) for w in net.frozen_W[:last])
        n_last = net.frozen_W[last].shape[1]
        local_load = load[base:base + n_last]
        n_keep = max(1, int(keep_frac * n_last))
        local_keep = np.argsort(local_load)[-n_keep:]  # keep the most load-bearing
        # prune the scaffolding from the last round
        net.frozen_W[last] = net.frozen_W[last][:, local_keep]
        net.frozen_V[last] = net.frozen_V[last][local_keep]
        net.frozen_b[last] = net.frozen_b[last][local_keep]
        net.dist = net.dist[:base] + [net.dist[base + j] for j in local_keep]
        # rebuild the activation caches (frozen_tr/te are now stale)
        self._recompute_base(X, self.X_new)
        return {"kept": int(len(local_keep)), "pruned": int(n_last - len(local_keep))}

    def _recompute_base(self, Xtr, Xte):
        """Run the frozen weights forward on new inputs to rebuild Ftr/Fte/frozen_tr/te.
        Preserves the frozen WEIGHTS (the established structure); only the cached
        activations are recomputed for the new input set."""
        from experiments.society import forward_logits
        net = self.net
        # recompute per-round activations
        def _acts(X):
            F = np.zeros((len(X), 0)); As = []
            for Wr, br in zip(net.frozen_W, net.frozen_b):
                if X.shape[1] == net.D:
                    zin = X @ Wr[:net.D]
                else:
                    zin = np.zeros((len(X), Wr.shape[1]))
                    for k in range(X.shape[1]):
                        zin = zin + Wr[:net.D][k * (net.D // X.shape[1]) + X[:, k]]
                A = np.maximum(zin + F @ Wr[net.D:] + br, 0)
                As.append(A); F = np.concatenate([F, A], 1)
            return As
        Atr = _acts(Xtr); Ate = _acts(Xte)
        net.Ftr = np.concatenate(Atr, 1) if Atr else np.zeros((len(Xtr), 0))
        net.Fte = np.concatenate(Ate, 1) if Ate else np.zeros((len(Xte), 0))
        # frozen readout logits = sum of A @ V over rounds
        net.frozen_tr = np.zeros((len(Xtr), net.C))
        net.frozen_te = np.zeros((len(Xte), net.C))
        for r, (A, V) in enumerate(zip(Atr, net.frozen_V)):
            net.frozen_tr = net.frozen_tr + A @ V
        for r, (A, V) in enumerate(zip(Ate, net.frozen_V)):
            net.frozen_te = net.frozen_te + A @ V

    def measure(self):
        """cps + no-forgetting of the current child."""
        cps = metrics.capability_per_synapse(self.net, self.X_new, self.y_new)
        old_acc, new_acc = metrics.no_forgetting(self.net, self.X_old, self.y_old,
                                                 self.X_new, self.y_new)
        return {"cps": cps, "old_acc": old_acc, "new_acc": new_acc}
