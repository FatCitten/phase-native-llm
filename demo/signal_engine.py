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
        stats = self.net.grow_round(X, y, self.X_new, self.y_new, P=P, epochs=epochs,
                                    tau=tau, floor=floor, soft=soft)
        self.signal_log.append(stats)
        return stats

    def measure(self):
        """cps + no-forgetting of the current child."""
        cps = metrics.capability_per_synapse(self.net, self.X_new, self.y_new)
        old_acc, new_acc = metrics.no_forgetting(self.net, self.X_old, self.y_old,
                                                 self.X_new, self.y_new)
        return {"cps": cps, "old_acc": old_acc, "new_acc": new_acc}
