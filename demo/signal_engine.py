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
