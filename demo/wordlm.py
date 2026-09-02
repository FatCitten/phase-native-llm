"""
demo/wordlm.py — word-level next-token prediction with the Nautilus architecture.

Tokenizes text into words (lowercased, punctuation-stripped), builds a word vocab,
and trains a ConsolidatingNet as a fixed-context next-WORD predictor. This is the
retargeted goal: next-token prediction, where the architecture's structural wins
(no-forgetting, interpretability, capability-per-synapse) are meaningful.

Honest framing: this is a TINY model on a ~15k-word corpus. It will NOT match
frontier next-token accuracy (those are 100B+ params on trillions of tokens). But
next-token prediction is a legitimate, measurable task where we can track real
progress and where Nautilus's structural advantages matter.

Pure numpy, CPU. No network, no sklearn, no torch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.consolidation_rounds import ConsolidatingNet
from experiments.society import forward_logits


# ---------------------------------------------------------------------------
# word tokenization
# ---------------------------------------------------------------------------
def tokenize(text):
    """Split text into lowercased word tokens (strip punctuation)."""
    import re
    return re.findall(r"[a-z']+", text.lower())


def build_vocab(texts, min_freq=1):
    """Word vocab from a list of texts, sorted by frequency (most common first)."""
    from collections import Counter
    counts = Counter()
    for t in texts:
        counts.update(tokenize(t))
    # keep words above min_freq; index 0 = <UNK>
    words = [w for w, c in counts.most_common() if c >= min_freq]
    return ["<UNK>"] + words


def window_words(tokens, W, vocab):
    """Slide a window of W words over `tokens`; return (X [N, W] integer token indices, y [N] next-word index).

    Sparse: X holds integer vocab indices (one per window position), NOT one-hot vectors.
    The input layer becomes a gather (W[token]) instead of a dense one-hot matmul.
    """
    V = len(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    n = max(0, len(tokens) - W)
    X = np.zeros((n, W), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for k in range(W):
            X[i, k] = idx.get(tokens[i + k], 0)
        y[i] = idx.get(tokens[i + W], 0)
    return X, y


def one_hot(X, V):
    """Convert integer token indices (N, W) back to one-hot (N, W*V) for callers that need it."""
    N, W = X.shape
    oh = np.zeros((N, W * V), dtype=float)
    for i in range(N):
        for k in range(W):
            oh[i, k * V + X[i, k]] = 1.0
    return oh


def split(X, y, frac=0.8, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    ntr = int(frac * len(y))
    return X[perm[:ntr]], y[perm[:ntr]], X[perm[ntr:]], y[perm[ntr:]]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_wordlm(Xtr, ytr, Xte, yte, D, C, seed=1, rounds=3, P=48, EP=400, tau_ramp=0.6):
    """Train a word-level next-token Nautilus model."""
    net = ConsolidatingNet(D, C, seed=seed)
    for r in range(rounds):
        tau = 0.0 if r == 0 else min(tau_ramp, 0.3 + 0.15 * r)
        net.grow_round(Xtr, ytr, Xte, yte, P=P, epochs=EP, tau=tau,
                       prune_density=(0.8 if r else None))
    return net


def next_word_acc(net, X, y):
    return float((forward_logits(net, X).argmax(1) == y).mean())


def generate(net, prompt_words, n, temp, vocab, W):
    """Autoregressive next-word generation."""
    V = len(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    ctx = list(prompt_words)
    out = []
    for _ in range(n):
        tail = ctx[-W:]
        x = np.zeros((1, W * V), dtype=float)
        for k, w in enumerate(tail):
            x[0, k * V + idx.get(w, 0)] = 1.0
        logits = forward_logits(net, x)[0]
        logits = logits / max(temp, 1e-6)
        logits -= logits.max()
        e = np.exp(logits)
        p = e / e.sum()
        w = vocab[int(np.random.choice(V, p=p))]
        out.append(w)
        ctx.append(w)
    return out
