"""demo/metrics.py — the two headline metrics that prove Nautilus's value.

capability_per_synapse: accuracy per synapse (the honest efficiency win).
no_forgetting: accuracy on OLD data after growing on NEW data (the thing
transformers can't do — they catastrophically forget).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.society import forward_logits


def capability_per_synapse(net, X, y):
    """Accuracy per synapse, scaled to a readable number (acc/synapse * 1e3)."""
    acc = float((forward_logits(net, X).argmax(1) == y).mean())
    syn = max(net.synapses, 1)
    return acc / syn * 1e3


def no_forgetting(net, X_old, y_old, X_new, y_new):
    """Accuracy on OLD data after the net has grown on NEW data.
    Returns (old_acc_after, new_acc). A transformer would drop old_acc_after
    catastrophically; Nautilus's frozen branches should preserve it."""
    old_acc = float((forward_logits(net, X_old).argmax(1) == y_old).mean())
    new_acc = float((forward_logits(net, X_new).argmax(1) == y_new).mean())
    return old_acc, new_acc
